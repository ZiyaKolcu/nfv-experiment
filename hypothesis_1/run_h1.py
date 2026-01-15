"""
Hypothesis 1: OCR Correction Accuracy Test

In mobile food label analysis, the quality of the OCR engine and text pre-processing strategies directly affect the information extraction performance and hallucination rates of Large Language Models (LLMs).
"""

import os
import sys
import time
from typing import Any, Dict
from tqdm import tqdm
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data_loader import load_raw_ocr_data, combine_product_texts
from common.output_saver import save_results
from app.services.nutrition.label_parser import parse_ocr_raw_text
from app.services.nutrition.llm_client import get_model_name

load_dotenv()


def process_single_product_h1(product_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single product for Hypothesis 1 (parsing only).

    Returns:
        {
            "product_id": str,
            "llm_response": {
                "ingredients_text": str,  # Comma-separated string
                "nutrition_data": dict
            },
            "error": str | None
        }
    """
    product_id = product_data.get("folder_name", "unknown")
    combined_text = combine_product_texts(product_data)

    result = {"product_id": product_id, "llm_response": None, "error": None}

    try:
        # Call ONLY the parser - returns (ingredients_list, nutrition_data)
        ingredients_list, nutrition_data = parse_ocr_raw_text(combined_text)

        # Convert ingredients list to comma-separated string (like ground truth)
        ingredients_text = ", ".join(ingredients_list)

        # Format response - ONLY parsing results
        result["llm_response"] = {
            "ingredients_text": ingredients_text,
            "nutrition_data": nutrition_data,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n✗ Error processing {product_id}: {error_msg}")
        result["error"] = error_msg

    return result


def main():
    """Main processing pipeline for Hypothesis 1."""
    print("=" * 70)
    print("HYPOTHESIS 1: OCR Correction Accuracy")
    print("=" * 70)
    print(f"Model: {get_model_name()}")
    print(f"Test: Parsing accuracy (ingredients + nutrition extraction)")
    print()

    # Load data
    raw_data = load_raw_ocr_data("ocr texts/cloud_vision_ocr_texts.json")
    if not raw_data:
        print("No data to process. Exiting.")
        return

    results = []

    # Process each product
    print("\nProcessing products...")
    for product_data in tqdm(raw_data, desc="H1 Progress", unit="product"):
        result = process_single_product_h1(product_data)
        results.append(result)
        time.sleep(1)  # Rate limiting

    # Save results
    output_file = os.path.join("hypothesis_1", "h1_results.json")
    save_results(results, output_file, "Hypothesis 1")

    print("✓ Hypothesis 1 processing complete!")
    print("\nNext: Run 'python -m hypothesis_1.evaluate_h1' to evaluate")


if __name__ == "__main__":
    main()
