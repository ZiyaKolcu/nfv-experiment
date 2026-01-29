import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import warnings

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════════════════

MODEL_PAIRS = {
    "claude": ("claude-haiku-4.5", "claude-sonnet-4.5"),
    "gemini": ("gemini-3-flash", "gemini-3-pro"),
    "openai": ("gpt-5-mini", "gpt-5.1"),
}

CONDITIONS = ["Baseline", "Method A", "Method B"]
BOOTSTRAP_RESAMPLES = 10000
CONFIDENCE_LEVEL = 0.95

# ════════════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════


def find_model_in_filename(filename, provider):
    """Extract model name from filename."""
    lower_name = filename.lower()

    if provider.lower() == "claude":
        if "haiku" in lower_name:
            return "claude-haiku-4.5"
        elif "sonnet" in lower_name:
            return "claude-sonnet-4.5"
    elif provider.lower() == "gemini":
        if "flash" in lower_name:
            return "gemini-3-flash"
        elif "pro" in lower_name:
            return "gemini-3-pro"
    elif provider.lower() == "openai":
        if "mini" in lower_name:
            return "gpt-5-mini"
        elif "5.1" in lower_name or "51" in lower_name:
            return "gpt-5.1"

    return None


def parse_h1_files(provider, condition):
    """Parse H1 evaluation JSON files organized by model."""
    directory = f"hypothesis_1/results and evaluation/{provider}/{condition}"

    if not os.path.exists(directory):
        return {}, {}

    all_ingredients = {}
    all_nutrition = {}

    try:
        for filename in os.listdir(directory):
            if not filename.endswith("_evaluation.json"):
                continue

            model = find_model_in_filename(filename, provider)
            if model is None:
                continue

            if model not in all_ingredients:
                all_ingredients[model] = {}
                all_nutrition[model] = {}

            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "detailed_evaluations" not in data:
                    continue

                evaluations = data["detailed_evaluations"]
                if not isinstance(evaluations, list):
                    continue

                for evaluation in evaluations:
                    if isinstance(evaluation, str):
                        continue

                    if not isinstance(evaluation, dict):
                        continue

                    product_id = evaluation.get("product_id")
                    if not product_id:
                        continue

                    if "ingredients" in evaluation and isinstance(
                        evaluation["ingredients"], dict
                    ):
                        if "f1_score" in evaluation["ingredients"]:
                            all_ingredients[model][product_id] = evaluation[
                                "ingredients"
                            ]["f1_score"]

                    if "nutrition" in evaluation and isinstance(
                        evaluation["nutrition"], dict
                    ):
                        if "f1_score" in evaluation["nutrition"]:
                            all_nutrition[model][product_id] = evaluation["nutrition"][
                                "f1_score"
                            ]

            except Exception as e:
                print(f"  Warning: Error parsing {filepath}: {e}")
                continue

    except Exception as e:
        print(f"  Warning: Error accessing directory {directory}: {e}")
        return {}, {}

    return all_ingredients, all_nutrition


def parse_h2_files(provider, condition):
    """Parse H2 evaluation JSON files organized by model with profile-based grouping."""
    directory = f"hypothesis_2/results and evaluation/{provider}/{condition}"

    if not os.path.exists(directory):
        return {}

    all_risk = {}

    try:
        for filename in os.listdir(directory):
            if not filename.endswith("_h2_evaluation.json"):
                continue

            model = find_model_in_filename(filename, provider)
            if model is None:
                continue

            if model not in all_risk:
                all_risk[model] = {}

            filepath = os.path.join(directory, filename)

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "detailed_evaluations" not in data:
                    continue

                evaluations = data["detailed_evaluations"]

                # H2 structure: detailed_evaluations is a dict with profile names as keys
                if isinstance(evaluations, dict):
                    for profile_name, profile_evals in evaluations.items():
                        if not isinstance(profile_evals, list):
                            continue

                        for evaluation in profile_evals:
                            if not isinstance(evaluation, dict):
                                continue

                            product_id = evaluation.get("product_id")
                            if not product_id:
                                continue

                            # Use profile-aware key
                            key = f"{product_id}_{profile_name}"

                            if "risk_detection" in evaluation and isinstance(
                                evaluation["risk_detection"], dict
                            ):
                                if "f1_score" in evaluation["risk_detection"]:
                                    f1_score = evaluation["risk_detection"]["f1_score"]
                                    if f1_score is not None:
                                        all_risk[model][key] = float(f1_score)

                # Also handle old format if needed (list of dicts)
                elif isinstance(evaluations, list):
                    for evaluation in evaluations:
                        if not isinstance(evaluation, dict):
                            continue

                        product_id = evaluation.get("product_id")
                        profile_name = evaluation.get("profile_name")

                        if not product_id or not profile_name:
                            continue

                        key = f"{product_id}_{profile_name}"

                        if "risk_detection" in evaluation and isinstance(
                            evaluation["risk_detection"], dict
                        ):
                            if "f1_score" in evaluation["risk_detection"]:
                                f1_score = evaluation["risk_detection"]["f1_score"]
                                if f1_score is not None:
                                    all_risk[model][key] = float(f1_score)

            except Exception as e:
                print(f"  Warning: Error parsing {filepath}: {e}")
                continue

    except Exception as e:
        print(f"  Warning: Error accessing directory {directory}: {e}")
        return {}

    return all_risk


def pair_data(model_a_data, model_b_data):
    """Match data points between two models based on common keys."""
    if not model_a_data or not model_b_data:
        return None, None

    common_keys = set(model_a_data.keys()) & set(model_b_data.keys())

    if not common_keys:
        return None, None

    sorted_keys = sorted(list(common_keys))
    model_a_values = np.array([model_a_data[key] for key in sorted_keys])
    model_b_values = np.array([model_b_data[key] for key in sorted_keys])

    return model_a_values, model_b_values


def calculate_bootstrap_ci(values, n_resamples=10000, ci=0.95):
    """Calculate bootstrap confidence interval for the mean of differences."""
    if len(values) < 2:
        return None, None

    bootstrap_means = []

    np.random.seed(42)  # For reproducibility
    for _ in range(n_resamples):
        resample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))

    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return lower, upper


def get_significance_stars(p_value):
    """Convert p-value to significance stars."""
    if p_value is None or np.isnan(p_value):
        return "N/A"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def perform_statistical_test(model_a_values, model_b_values):
    """Perform Wilcoxon signed-rank test and bootstrap CI."""
    if model_a_values is None or model_b_values is None:
        return None, None, None, None, None

    if len(model_a_values) < 3:
        return None, None, None, None, None

    differences = model_a_values - model_b_values
    mean_diff = np.mean(differences)

    # Check if all differences are zero
    if np.all(differences == 0):
        return mean_diff, 1.0, 0.0, 0.0, len(model_a_values)

    try:
        statistic, p_value = wilcoxon(model_a_values, model_b_values)
    except Exception as e:
        print(f"Warning: Wilcoxon test failed: {e}")
        return mean_diff, None, None, None, len(model_a_values)

    try:
        lower_ci, upper_ci = calculate_bootstrap_ci(
            differences, BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL
        )
    except Exception as e:
        print(f"Warning: Bootstrap CI calculation failed: {e}")
        lower_ci, upper_ci = None, None

    return mean_diff, p_value, lower_ci, upper_ci, len(model_a_values)


# ════════════════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════════════════════


def main():
    """Main execution function."""
    results = []

    print("=" * 140)
    print("STATISTICAL SIGNIFICANCE ANALYSIS FOR LLM COMPARISON STUDY")
    print("=" * 140)
    print()

    # H1 Analysis
    print("Processing Hypothesis 1 (OCR & Extraction)...")
    print("-" * 140)

    for provider, (model_a, model_b) in MODEL_PAIRS.items():
        for condition in CONDITIONS:
            ingredients_data, nutrition_data = parse_h1_files(provider, condition)

            if not ingredients_data or not nutrition_data:
                print(f"  Warning: Incomplete data for {provider} - {condition}")
                continue

            # Ingredients comparison
            if model_a in ingredients_data and model_b in ingredients_data:
                a_vals, b_vals = pair_data(
                    ingredients_data[model_a], ingredients_data[model_b]
                )

                if a_vals is not None:
                    mean_diff, p_val, ci_lower, ci_upper, n = perform_statistical_test(
                        a_vals, b_vals
                    )

                    if p_val is not None:
                        results.append(
                            {
                                "Hypothesis": "H1-Ingredients",
                                "Provider": provider.capitalize(),
                                "Condition": condition,
                                "Model A": model_a,
                                "Model B": model_b,
                                "Mean Difference": f"{mean_diff:.4f}",
                                "p-value": (
                                    f"{p_val:.6f}" if p_val is not None else "N/A"
                                ),
                                "Significance": get_significance_stars(p_val),
                                "95% CI Lower": (
                                    f"{ci_lower:.4f}" if ci_lower is not None else "N/A"
                                ),
                                "95% CI Upper": (
                                    f"{ci_upper:.4f}" if ci_upper is not None else "N/A"
                                ),
                                "N": n,
                            }
                        )

            # Nutrition comparison
            if model_a in nutrition_data and model_b in nutrition_data:
                a_vals, b_vals = pair_data(
                    nutrition_data[model_a], nutrition_data[model_b]
                )

                if a_vals is not None:
                    mean_diff, p_val, ci_lower, ci_upper, n = perform_statistical_test(
                        a_vals, b_vals
                    )

                    if p_val is not None:
                        results.append(
                            {
                                "Hypothesis": "H1-Nutrition",
                                "Provider": provider.capitalize(),
                                "Condition": condition,
                                "Model A": model_a,
                                "Model B": model_b,
                                "Mean Difference": f"{mean_diff:.4f}",
                                "p-value": (
                                    f"{p_val:.6f}" if p_val is not None else "N/A"
                                ),
                                "Significance": get_significance_stars(p_val),
                                "95% CI Lower": (
                                    f"{ci_lower:.4f}" if ci_lower is not None else "N/A"
                                ),
                                "95% CI Upper": (
                                    f"{ci_upper:.4f}" if ci_upper is not None else "N/A"
                                ),
                                "N": n,
                            }
                        )

    # H2 Analysis
    print("Processing Hypothesis 2 (Risk Assessment)...")
    print("-" * 140)

    h2_count = 0
    for provider, (model_a, model_b) in MODEL_PAIRS.items():
        for condition in CONDITIONS:
            risk_data = parse_h2_files(provider, condition)

            if not risk_data:
                print(f"  ℹ {provider}-{condition}: No H2 data loaded")
                continue

            if model_a not in risk_data:
                print(f"  ✗ {provider}-{condition}: Model A ({model_a}) not found")
                continue

            if model_b not in risk_data:
                print(f"  ✗ {provider}-{condition}: Model B ({model_b}) not found")
                continue

            print(f"  ✓ {provider}-{condition}: Found both models")
            print(
                f"    Model A keys: {len(risk_data[model_a])}, Model B keys: {len(risk_data[model_b])}"
            )

            a_vals, b_vals = pair_data(risk_data[model_a], risk_data[model_b])

            if a_vals is None:
                print(f"    ✗ No common keys between models")
                continue

            print(f"    ✓ Paired {len(a_vals)} data points")

            mean_diff, p_val, ci_lower, ci_upper, n = perform_statistical_test(
                a_vals, b_vals
            )

            if p_val is not None:
                h2_count += 1
                results.append(
                    {
                        "Hypothesis": "H2-Risk",
                        "Provider": provider.capitalize(),
                        "Condition": condition,
                        "Model A": model_a,
                        "Model B": model_b,
                        "Mean Difference": f"{mean_diff:.4f}",
                        "p-value": f"{p_val:.6f}" if p_val is not None else "N/A",
                        "Significance": get_significance_stars(p_val),
                        "95% CI Lower": (
                            f"{ci_lower:.4f}" if ci_lower is not None else "N/A"
                        ),
                        "95% CI Upper": (
                            f"{ci_upper:.4f}" if ci_upper is not None else "N/A"
                        ),
                        "N": n,
                    }
                )
                print(f"    ✓ Result added: p={p_val:.6f}")
            else:
                print(f"    ✗ Statistical test failed")

    print(f"\n  ✓ H2 analysis complete: {h2_count} comparisons generated")

    # Output results
    print()
    print("=" * 140)
    print("RESULTS TABLE")
    print("=" * 140)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print()

        # Save to CSV
        csv_filename = "statistical_test/statistical_significance_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Results saved to {csv_filename}")
        print()

        # Print significance legend
        print("=" * 140)
        print("SIGNIFICANCE LEGEND")
        print("=" * 140)
        print("*** : p < 0.001 (highly significant)")
        print("**  : p < 0.01  (very significant)")
        print("*   : p < 0.05  (significant)")
        print("ns  : p >= 0.05 (not significant)")
        print("N/A : Test could not be performed")
        print()
    else:
        print(
            "No results were generated. Please check that data files exist in the expected directories."
        )
        print(
            "Ensure JSON files contain 'detailed_evaluations' as a list of dictionaries."
        )
        print()


if __name__ == "__main__":
    main()
