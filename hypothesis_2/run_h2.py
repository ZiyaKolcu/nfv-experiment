"""
Hypothesis 2: Personalized Risk Analysis Test

Tests whether the system can perform dynamic, profile-based risk assessment.
Tests same products with 3 different health profiles.
"""

import os
import sys
import time
import json
from typing import Any, Dict, List
from tqdm import tqdm
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.nutrition.nutrition_analyzer import analyze_label_with_profile
from app.services.nutrition.llm_client import get_model_name, get_llm_provider
from hypothesis_2.profiles_loader import load_profiles

load_dotenv()

TARGET_PRODUCT_IDS = ["045", "019", "003", "015", "035"]

H1_RESULTS_JSON_PATH = "hypothesis_1/results and evaluation/claude/Method B/claude-sonnet-4.5_h1_results.json"


def extract_profile_health_info(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract only allergies, dietary_preferences, and health_conditions from profile.

    Args:
        profile_data: Full profile data from JSON

    Returns:
        Filtered profile with only health-related fields
    """
    profile = profile_data.get("profile", {})
    return {
        "allergies": profile.get("allergies", []),
        "dietary_preferences": profile.get("dietary_preferences", []),
        "health_conditions": profile.get("health_conditions", []),
    }


def load_products_from_h1_results(
    json_path: str, target_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    Load products from H1 results JSON file and filter by target product IDs.

    Args:
        json_path: Path to H1 results JSON file
        target_ids: List of product IDs to filter (e.g., ["045", "019"])

    Returns:
        List of filtered product data
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"H1 results file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        all_products = json.load(f)

    filtered_products = []
    for product in all_products:
        product_id = product.get("product_id", "")
        for target_id in target_ids:
            if target_id in product_id:
                filtered_products.append(product)
                break

    return filtered_products


def process_single_product_profile(
    product: Dict[str, Any], profile_name: str, health_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process a single product with a single profile using nutrition service.

    Args:
        product: Product data from H1 results
        profile_name: Name of the profile
        health_profile: Health profile data (allergies, dietary_preferences, health_conditions)

    Returns:
        Result dictionary with LLM analysis
    """
    product_id = product.get("product_id", "unknown")
    h1_llm_response = product.get("llm_response", {})

    h1_ingredients_text = h1_llm_response.get("ingredients_text", "")
    h1_nutrition_data = h1_llm_response.get("nutrition_data", {})

    result = {
        "product_id": product_id,
        "profile_name": profile_name,
        "h1_ingredients": h1_ingredients_text,
        "h1_nutrition": h1_nutrition_data,
        "llm_response": None,
        "error": None,
    }

    if product.get("error"):
        result["error"] = f"H1 Error: {product['error']}"
        return result

    if not h1_ingredients_text:
        result["error"] = "No ingredients from H1"
        return result

    try:
        structured_text = f"İçindekiler: {h1_ingredients_text}\n\n"
        structured_text += "Besin Değerleri (100g/ml):\n"

        nutrition_values = h1_nutrition_data.get("values", {})
        if nutrition_values.get("energy_kcal"):
            structured_text += f"Enerji: {nutrition_values['energy_kcal']} kcal\n"
        if nutrition_values.get("fat_total_g") is not None:
            structured_text += f"Yağ: {nutrition_values['fat_total_g']} g\n"
        if nutrition_values.get("carbohydrate_g") is not None:
            structured_text += f"Karbonhidrat: {nutrition_values['carbohydrate_g']} g\n"
        if nutrition_values.get("sugar_g") is not None:
            structured_text += f"Şeker: {nutrition_values['sugar_g']} g\n"
        if nutrition_values.get("protein_g") is not None:
            structured_text += f"Protein: {nutrition_values['protein_g']} g\n"
        if nutrition_values.get("salt_g") is not None:
            structured_text += f"Tuz: {nutrition_values['salt_g']} g\n"

        (ingredients, nutrition_data, risks, summary_explanation, summary_risk) = (
            analyze_label_with_profile(
                raw_text=structured_text, health_profile=health_profile, language="tr"
            )
        )

        result["llm_response"] = {
            "ingredients": ingredients,
            "nutrition_data": nutrition_data,
            "risks": risks,
            "summary_explanation": summary_explanation,
            "summary_risk": summary_risk,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n✗ Error processing {product_id} with {profile_name}: {error_msg}")
        result["error"] = error_msg

    return result


def main():
    """Main processing pipeline for Hypothesis 2."""
    print("=" * 70)
    print("HYPOTHESIS 2: Personalized Risk Analysis")
    print("=" * 70)

    llm_provider = get_llm_provider()
    model_name = get_model_name()
    print(f"LLM Provider: {llm_provider}")
    print(f"Model: {model_name}")
    print(f"Target Products: {TARGET_PRODUCT_IDS}")
    print()

    if not H1_RESULTS_JSON_PATH:
        print("✗ Error: H1_RESULTS_JSON_PATH is not set!")
        print("Please set the path to your H1 results JSON file in the code.")
        print(
            "Example: H1_RESULTS_JSON_PATH = '/path/to/hypothesis_1/results and evaluation/openai/Baseline/gpt-5.1_h1_results.json'"
        )
        return

    print(f"Loading products from: {H1_RESULTS_JSON_PATH}")

    try:
        products = load_products_from_h1_results(
            H1_RESULTS_JSON_PATH, TARGET_PRODUCT_IDS
        )
        print(f"✓ Loaded {len(products)} products matching target IDs\n")
    except Exception as e:
        print(f"✗ Error loading products: {e}")
        return

    profiles = load_profiles()
    print(f"✓ Loaded {len(profiles)} profiles\n")

    all_results = []

    for profile_name, profile_data in profiles.items():
        print(f"\n{'='*70}")
        print(f"Profile: {profile_data.get('name', profile_name)}")
        print(f"{'='*70}")

        health_profile = extract_profile_health_info(profile_data)

        for product in tqdm(products, desc=f"{profile_name}", unit="product"):
            result = process_single_product_profile(
                product, profile_name, health_profile
            )
            all_results.append(result)
            time.sleep(1)

    output_file = os.path.join("hypothesis_2", "h2_results.json")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to {output_file}")

        print(f"\n{'='*70}")
        print("Processing Summary:")
        print(f"{'='*70}")
        total = len(all_results)
        successful = sum(1 for r in all_results if r.get("llm_response") is not None)
        print(f"Total analyses: {total}")
        print(f"Successful: {successful}/{total} ({successful/total*100:.1f}%)")
        print(f"Products: {len(products)}")
        print(f"Profiles: {len(profiles)}")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"✗ Error saving results: {e}")

    print("✓ Hypothesis 2 processing complete!")
    print("\nNext: Run 'python -m hypothesis_2.evaluate_h2' to evaluate")


if __name__ == "__main__":
    main()
