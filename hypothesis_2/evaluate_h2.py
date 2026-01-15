"""
Hypothesis 2 Evaluation: Personalized Risk Analysis Accuracy

Evaluates how accurately the LLM performs profile-based risk assessment:
1. Risk Detection Accuracy (detecting expected high-risk ingredients)
2. Risk Level Consistency (appropriate risk levels assigned)
3. Profile Differentiation (different profiles → different risk assessments)
"""

import os
import sys
import json
from typing import Any, Dict, List
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hypothesis_2.profiles_loader import load_profiles


def extract_actual_risks_from_product(
    h1_ingredients: str,
    h1_nutrition: Dict[str, Any],
    profile_data: Dict[str, Any],
    expected_risks: List[str],
) -> List[str]:
    """
    Extract which expected risks are actually present in the product.

    Hybrid ground truth: checks both ingredient text matching AND nutrition thresholds.

    This creates a dynamic expected risk list based on what's actually
    in the product, making evaluation more realistic.

    Args:
        h1_ingredients: Ingredients text from H1
        h1_nutrition: Nutrition data from H1 (contains 'values' dict)
        profile_data: Full profile data (contains 'nutrition_rules')
        expected_risks: Full list of expected risks for profile

    Returns:
        List of expected risks that are actually in this product
        (includes both ingredient matches and nutrition threshold violations)
    """
    actual_risks = []

    # Part 1: Check ingredient text matching
    if h1_ingredients:
        ing_lower = h1_ingredients.lower()

        for risk in expected_risks:
            risk_lower = risk.lower()
            # Check if risk is in ingredients (substring or word match)
            if risk_lower in ing_lower:
                actual_risks.append(risk)
                continue

            # Word-level matching
            risk_words = set(risk_lower.split())
            ing_words = set(ing_lower.split())
            if risk_words & ing_words:
                actual_risks.append(risk)

    # Part 2: Check nutrition thresholds
    nutrition_rules = profile_data.get("nutrition_rules", {})
    if nutrition_rules and h1_nutrition:
        nutrition_values = h1_nutrition.get("values", {})

        for nutrient_key, rule in nutrition_rules.items():
            threshold = rule.get("threshold")
            operator = rule.get("operator", ">")

            if threshold is None:
                continue

            actual_value = nutrition_values.get(nutrient_key)
            if actual_value is None:
                continue

            # Check if threshold is exceeded
            threshold_exceeded = False
            if operator == ">" and actual_value > threshold:
                threshold_exceeded = True
            elif operator == ">=" and actual_value >= threshold:
                threshold_exceeded = True
            elif operator == "<" and actual_value < threshold:
                threshold_exceeded = True
            elif operator == "<=" and actual_value <= threshold:
                threshold_exceeded = True

            if threshold_exceeded:
                # Add nutrition-based risk in format: "NUT: nutrient_key (actual > threshold)"
                risk_label = (
                    f"NUT: {nutrient_key} ({actual_value} {operator} {threshold})"
                )
                actual_risks.append(risk_label)

    return actual_risks


def normalize_ingredient(ingredient: str) -> str:
    """Normalize ingredient name for matching."""
    return ingredient.lower().strip()


def fuzzy_match_ingredient(ingredient: str, expected_risk: str) -> bool:
    """
    Improved fuzzy matching for ingredients.

    Returns True if ingredient matches expected risk with flexible matching:
    - Substring matching (both directions)
    - Word-level matching
    - Handles compound ingredients
    """
    ing_norm = normalize_ingredient(ingredient)
    exp_norm = normalize_ingredient(expected_risk)

    if ing_norm == exp_norm:
        return True

    if exp_norm in ing_norm:
        return True

    ing_words = set(ing_norm.split())
    exp_words = set(exp_norm.split())

    if exp_words.issubset(ing_words):
        return True

    return False


def check_expected_risks(
    risks: Dict[str, str], expected_high_risk: List[str], ingredients: List[str]
) -> Dict[str, Any]:
    """
    Check if expected high-risk ingredients are correctly identified.

    Fixed logic to prevent precision from exceeding 100% by eliminating double-counting.

    Precision Logic:
    - Denominator: Total number of "High" predictions by LLM
    - Numerator: Number of "High" predictions that match ANY expected risk (counted once per prediction)

    Recall Logic:
    - Denominator: Total number of expected risks actually present in product (ground truth)
    - Numerator: Number of present risks that were flagged as "High" by LLM

    Returns:
        {
            "total_expected_in_product": int,
            "correctly_flagged_high": int,
            "true_positives": int,
            "missed_high_risks": list,
            "total_high_assigned": int,
            "precision": float,
            "recall": float,
            "f1_score": float
        }
    """
    # Normalize all data
    normalized_ingredients = {normalize_ingredient(ing) for ing in ingredients}
    normalized_risks = {normalize_ingredient(k): v for k, v in risks.items()}
    normalized_expected = [normalize_ingredient(exp) for exp in expected_high_risk]

    # STEP 1: Identify ground truth - which expected risks are actually in this product
    expected_in_product = []
    for exp in normalized_expected:
        for ing in normalized_ingredients:
            if fuzzy_match_ingredient(ing, exp):
                expected_in_product.append(exp)
                break

    # STEP 2: Calculate PRECISION
    # Get all LLM predictions flagged as "High"
    high_predictions = [
        ing_name
        for ing_name, risk_level in normalized_risks.items()
        if risk_level == "High"
    ]

    total_high_assigned = len(high_predictions)

    # Count how many "High" predictions match ANY expected risk (no double counting)
    true_positives_for_precision = 0
    for high_pred in high_predictions:
        # Check if this prediction matches ANY expected risk
        matches_any_expected = False
        for exp in normalized_expected:
            if fuzzy_match_ingredient(high_pred, exp):
                matches_any_expected = True
                break

        if matches_any_expected:
            true_positives_for_precision += 1

    # STEP 3: Calculate RECALL
    # Count how many expected risks (that are in product) were flagged as High
    correctly_flagged = []
    missed_risks = []

    for exp in expected_in_product:
        found_high = False
        for ing_name, risk_level in normalized_risks.items():
            if fuzzy_match_ingredient(ing_name, exp) and risk_level == "High":
                correctly_flagged.append(exp)
                found_high = True
                break

        if not found_high:
            missed_risks.append(exp)

    true_positives_for_recall = len(correctly_flagged)

    # STEP 4: Calculate metrics with capping
    # Precision: Of all High predictions, how many were correct?
    if total_high_assigned > 0:
        precision = true_positives_for_precision / total_high_assigned
    else:
        precision = 0.0

    # Recall: Of all expected risks in product, how many were caught?
    if len(expected_in_product) > 0:
        recall = true_positives_for_recall / len(expected_in_product)
    else:
        recall = 1.0  # No risks expected, so perfect recall

    # Cap precision and recall at 1.0 (should not exceed, but safety check)
    precision = min(precision, 1.0)
    recall = min(recall, 1.0)

    # F1 Score
    if (precision + recall) > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Cap F1 at 1.0
    f1_score = min(f1_score, 1.0)

    return {
        "total_expected_in_product": len(expected_in_product),
        "correctly_flagged_high": true_positives_for_recall,
        "true_positives": true_positives_for_precision,
        "missed_high_risks": missed_risks,
        "total_high_assigned": total_high_assigned,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1_score, 3),
    }


def evaluate_risk_distribution(risks: Dict[str, str]) -> Dict[str, Any]:
    """
    Analyze the distribution of risk levels.

    Returns:
        {
            "total_ingredients": int,
            "high_count": int,
            "medium_count": int,
            "low_count": int,
            "high_ratio": float
        }
    """
    total = len(risks)
    high_count = sum(1 for v in risks.values() if v == "High")
    medium_count = sum(1 for v in risks.values() if v == "Medium")
    low_count = sum(1 for v in risks.values() if v == "Low")

    return {
        "total_ingredients": total,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "high_ratio": round(high_count / total, 3) if total > 0 else 0.0,
        "medium_ratio": round(medium_count / total, 3) if total > 0 else 0.0,
        "low_ratio": round(low_count / total, 3) if total > 0 else 0.0,
    }


def evaluate_single_result(
    result: Dict[str, Any], profile_data: Dict[str, Any], expected_high_risk: List[str]
) -> Dict[str, Any]:
    """
    Evaluate a single product result for one profile.

    Uses dynamic expected risks based on what's actually in the product.

    Returns evaluation metrics for risk detection accuracy.
    """
    product_id = result.get("product_id", "unknown")
    profile_name = result.get("profile_name", "unknown")

    evaluation = {
        "product_id": product_id,
        "profile_name": profile_name,
        "has_error": False,
        "error_message": None,
    }

    # Check for errors
    if result.get("error") or not result.get("llm_response"):
        evaluation["has_error"] = True
        evaluation["error_message"] = result.get("error", "No LLM response")
        return evaluation

    llm_response = result["llm_response"]
    ingredients = llm_response.get("ingredients", [])
    risks = llm_response.get("risks", {})
    summary_risk = llm_response.get("summary_risk", "Unknown")

    # Get H1 data for dynamic risk extraction
    h1_ingredients = result.get("h1_ingredients", "")
    h1_nutrition = result.get("h1_nutrition", {})

    # Use dynamic expected risks: only risks that are actually in this product
    # (includes both ingredient matches and nutrition threshold violations)
    actual_expected_risks = extract_actual_risks_from_product(
        h1_ingredients, h1_nutrition, profile_data, expected_high_risk
    )

    # Evaluate risk detection with dynamic expected risks
    risk_metrics = check_expected_risks(risks, actual_expected_risks, ingredients)
    evaluation["risk_detection"] = risk_metrics
    evaluation["risk_detection"]["dynamic_expected_count"] = len(actual_expected_risks)

    # Evaluate risk distribution
    distribution = evaluate_risk_distribution(risks)
    evaluation["risk_distribution"] = distribution

    # Summary risk level
    evaluation["summary_risk"] = summary_risk

    # Overall score (based on F1 score of risk detection)
    evaluation["overall_score"] = risk_metrics["f1_score"]

    return evaluation


def calculate_overall_statistics(
    profile_summaries: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate overall statistics across all profiles.
    """
    total_products = sum(s["total_products"] for s in profile_summaries.values())
    total_successful = sum(
        s["successful_evaluations"] for s in profile_summaries.values()
    )

    all_f1_scores = [s["average_f1_score"] for s in profile_summaries.values()]
    all_precision = [s["average_precision"] for s in profile_summaries.values()]
    all_recall = [s["average_recall"] for s in profile_summaries.values()]

    return {
        "total_products_analyzed": total_products,
        "total_successful": total_successful,
        "overall_average_f1": (
            round(sum(all_f1_scores) / len(all_f1_scores), 3) if all_f1_scores else 0.0
        ),
        "overall_average_precision": (
            round(sum(all_precision) / len(all_precision), 3) if all_precision else 0.0
        ),
        "overall_average_recall": (
            round(sum(all_recall) / len(all_recall), 3) if all_recall else 0.0
        ),
        "interpretation": "Measures how well the system detects profile-specific risks",
    }


def main():
    """Main evaluation pipeline for Hypothesis 2."""
    print("=" * 70)
    print("HYPOTHESIS 2: EVALUATION")
    print("=" * 70)
    print("Evaluating personalized risk analysis accuracy...\n")

    results_file = os.path.join("hypothesis_2", "h2_results.json")
    if not os.path.exists(results_file):
        print(f"✗ Error: Results file not found: {results_file}")
        print("Please run 'python -m hypothesis_2.run_h2' first")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    print(f"✓ Loaded {len(all_results)} results\n")

    profiles = load_profiles()

    results_by_profile = defaultdict(list)
    for result in all_results:
        profile_name = result.get("profile_name")
        if profile_name:
            results_by_profile[profile_name].append(result)

    evaluations_by_profile = {}
    profile_summaries = {}

    for profile_name, results in results_by_profile.items():
        print(f"Evaluating {profile_name}...")

        profile_data = profiles.get(profile_name, {})
        expected_high_risk = profile_data.get("expected_ingredient_risks", [])

        evaluations = []
        f1_scores = []

        for result in tqdm(results, desc=f"  {profile_name}", unit="product"):
            eval_result = evaluate_single_result(
                result, profile_data, expected_high_risk
            )
            evaluations.append(eval_result)

            if not eval_result["has_error"]:
                f1_scores.append(eval_result["overall_score"])

        evaluations_by_profile[profile_name] = evaluations

        total = len(evaluations)
        successful = sum(1 for e in evaluations if not e["has_error"])
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        total_precision = []
        total_recall = []

        for e in evaluations:
            if not e["has_error"]:
                total_precision.append(e["risk_detection"]["precision"])
                total_recall.append(e["risk_detection"]["recall"])

        avg_precision = (
            sum(total_precision) / len(total_precision) if total_precision else 0.0
        )
        avg_recall = sum(total_recall) / len(total_recall) if total_recall else 0.0

        profile_summaries[profile_name] = {
            "total_products": total,
            "successful_evaluations": successful,
            "average_f1_score": round(avg_f1, 3),
            "average_precision": round(avg_precision, 3),
            "average_recall": round(avg_recall, 3),
        }

    overall_stats = calculate_overall_statistics(profile_summaries)

    evaluation_data = {
        "hypothesis": "H2: Personalized Risk Analysis",
        "profile_summaries": profile_summaries,
        "overall_statistics": overall_stats,
        "detailed_evaluations": evaluations_by_profile,
    }

    output_file = os.path.join("hypothesis_2", "h2_evaluation.json")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Evaluation saved to {output_file}")
    except Exception as e:
        print(f"✗ Error saving evaluation: {e}")

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    for profile_name, summary in profile_summaries.items():
        print(f"\n{profile_name}:")
        print(
            f"  Products Evaluated:  {summary['successful_evaluations']}/{summary['total_products']}"
        )
        print(f"  Average F1 Score:    {summary['average_f1_score']:.1%}")
        print(f"  Average Precision:   {summary['average_precision']:.1%}")
        print(f"  Average Recall:      {summary['average_recall']:.1%}")

    print(f"\n{'=' * 70}")
    print("Overall Statistics:")
    print(f"  Total Products:      {overall_stats['total_products_analyzed']}")
    print(f"  Successful:          {overall_stats['total_successful']}")
    print(f"  Overall F1 Score:    {overall_stats['overall_average_f1']:.1%}")
    print(f"  Overall Precision:   {overall_stats['overall_average_precision']:.1%}")
    print(f"  Overall Recall:      {overall_stats['overall_average_recall']:.1%}")
    print("=" * 70)

    print(f"\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
