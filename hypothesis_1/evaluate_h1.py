"""
Hypothesis 1 Evaluation: Comprehensive Metrics (Precision, Recall, F1)

Evaluates OCR Correction and Extraction with academic-standard metrics.
Includes specific logic for both Textual (Ingredients) and Numerical (Nutrition) data.
"""

import os
import sys
import json
import re
import difflib
from typing import Any, Dict, List, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data_loader import load_ground_truth
from common.output_saver import save_evaluation


def normalize_turkish_text(text: str) -> str:
    """Aggressive normalization for Turkish ingredients."""
    if not text:
        return ""
    text = text.lower().strip()
    replacements = {
        "  ": " ",
        " .": "",
        ".": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return clean_parentheses(text.strip())


def clean_parentheses(text: str) -> str:
    """Removes content in parentheses for base comparison."""
    return re.sub(r"\([^)]*\)", "", text).strip()


def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_ingredient_metrics(predicted: str, ground_truth: str) -> Dict[str, Any]:
    """
    Calculate Precision, Recall, F1 for Ingredients using Fuzzy Matching.
    """
    pred_list = [normalize_turkish_text(i) for i in predicted.split(",") if i.strip()]
    gt_list = [normalize_turkish_text(i) for i in ground_truth.split(",") if i.strip()]

    if not gt_list:
        return {"precision": 0, "recall": 0, "f1_score": 0, "details": []}

    tp = 0  
    fp = 0  
    fn = 0  

    matched_gt_indices = set()
    match_details = []

    # 1. Check Predictions (Calculate TP and FP)
    for pred_item in pred_list:
        best_ratio = 0.0
        best_gt_idx = -1

        for idx, gt_item in enumerate(gt_list):
            # Fuzzy match score
            ratio = difflib.SequenceMatcher(None, gt_item, pred_item).ratio()
            # Substring bonus
            if len(gt_item) > 3 and (gt_item in pred_item or pred_item in gt_item):
                ratio = max(ratio, 0.9)

            if ratio > best_ratio:
                best_ratio = ratio
                best_gt_idx = idx

        # Threshold 0.7
        if best_ratio >= 0.7:
            tp += 1
            matched_gt_indices.add(best_gt_idx)
            match_details.append(
                {"pred": pred_item, "match": "TP", "score": round(best_ratio, 2)}
            )
        else:
            fp += 1
            match_details.append(
                {"pred": pred_item, "match": "FP", "score": round(best_ratio, 2)}
            )

    # 2. Calculate FN (Items in GT that were never matched)
    fn = len(gt_list) - len(matched_gt_indices)

    # 3. Calculate Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1(precision, recall)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "match_details": match_details,
    }


def calculate_nutrition_metrics(
    predicted: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate Precision, Recall, F1 for Nutrition Values.
    Treats extraction as a retrieval task with tolerance.
    """
    pred_values = predicted.get("values", {})
    gt_values = ground_truth.get("values", {})

    keys = [
        "energy_kcal",
        "fat_total_g",
        "carbohydrate_g",
        "sugar_g",
        "protein_g",
        "salt_g",
    ]

    tp = 0
    fp = 0
    fn = 0
    details = []

    for key in keys:
        gt_val = gt_values.get(key)
        pred_val = pred_values.get(key)

        # Case 1: GT is missing (We don't expect anything)
        if gt_val is None:
            if pred_val is not None:
                fp += 1  # System hallucinated a value
                details.append(
                    {
                        "field": key,
                        "status": "FP (Hallucination)",
                        "gt": None,
                        "pred": pred_val,
                    }
                )
            continue

        # Case 2: GT exists, but System missed it
        if pred_val is None:
            fn += 1
            details.append(
                {"field": key, "status": "FN (Missed)", "gt": gt_val, "pred": None}
            )
            continue

        # Case 3: Both exist, compare values
        try:
            diff = abs(float(pred_val) - float(gt_val))
            is_match = False

            # Tolerances
            if key == "energy_kcal":
                is_match = diff <= 5
            elif key == "salt_g":
                is_match = diff <= 0.1
            else:
                is_match = diff <= 1.0 or (gt_val > 0 and diff / gt_val < 0.1)

            if is_match:
                tp += 1
                details.append(
                    {
                        "field": key,
                        "status": "TP (Correct)",
                        "gt": gt_val,
                        "pred": pred_val,
                    }
                )
            else:
                # Value is wrong -> Count as False Positive (Wrong Prediction) AND False Negative (Missed Correct)
                # Strict extraction penalty
                fp += 1
                fn += 1
                details.append(
                    {"field": key, "status": "Mismatch", "gt": gt_val, "pred": pred_val}
                )

        except (ValueError, TypeError):
            fp += 1
            fn += 1

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1(precision, recall)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "details": details,
    }


def evaluate_single_product(
    result: Dict[str, Any], ground_truth: Dict[str, Any]
) -> Dict[str, Any]:
    product_id = result.get("product_id", "unknown")
    llm_response = result.get("llm_response")
    error = result.get("error")

    eval_res = {"product_id": product_id, "has_error": False}

    if error or not llm_response:
        eval_res["has_error"] = True
        return eval_res

    # Ingredients Evaluation
    ing_metrics = calculate_ingredient_metrics(
        llm_response.get("ingredients_text", ""),
        ground_truth.get("ingredients_text", ""),
    )

    # Nutrition Evaluation
    nutr_metrics = calculate_nutrition_metrics(
        llm_response.get("nutrition_data", {}), ground_truth.get("nutrition_data", {})
    )

    eval_res["ingredients"] = ing_metrics
    eval_res["nutrition"] = nutr_metrics

    # Combined F1
    eval_res["overall_f1"] = round(
        (ing_metrics["f1_score"] + nutr_metrics["f1_score"]) / 2, 3
    )

    return eval_res


def main():
    print("=" * 70)
    print("HYPOTHESIS 1: ACADEMIC METRICS EVALUATION")
    print("=" * 70)

    # Load Data
    results_file = os.path.join("hypothesis_1", "h1_results.json")
    if not os.path.exists(results_file):
        print("Error: h1_results.json not found.")
        return

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    ground_truths = load_ground_truth("Dataset")

    evaluations = []

    print("Calculating Precision, Recall, F1...")
    for res in tqdm(results):
        pid = res.get("product_id")
        gt = ground_truths.get(pid)
        if gt:
            evaluations.append(evaluate_single_product(res, gt))

    # --- AGGREGATE STATISTICS (MACRO AVERAGE) ---
    valid_evals = [e for e in evaluations if not e["has_error"]]
    count = len(valid_evals)

    if count == 0:
        print("No valid evaluations found.")
        return

    # Ingredients Averages
    avg_ing_prec = sum(e["ingredients"]["precision"] for e in valid_evals) / count
    avg_ing_rec = sum(e["ingredients"]["recall"] for e in valid_evals) / count
    avg_ing_f1 = sum(e["ingredients"]["f1_score"] for e in valid_evals) / count

    # Nutrition Averages
    avg_nutr_prec = sum(e["nutrition"]["precision"] for e in valid_evals) / count
    avg_nutr_rec = sum(e["nutrition"]["recall"] for e in valid_evals) / count
    avg_nutr_f1 = sum(e["nutrition"]["f1_score"] for e in valid_evals) / count

    # Overall
    avg_overall_f1 = sum(e["overall_f1"] for e in valid_evals) / count

    # Summary Object for Saving
    summary_data = {
        "total_products": len(results),
        "valid_evaluations": count,
        "ingredients": {
            "precision": round(avg_ing_prec, 3),
            "recall": round(avg_ing_rec, 3),
            "f1_score": round(avg_ing_f1, 3),
        },
        "nutrition": {
            "precision": round(avg_nutr_prec, 3),
            "recall": round(avg_nutr_rec, 3),
            "f1_score": round(avg_nutr_f1, 3),
        },
        "overall_f1": round(avg_overall_f1, 3),
    }

    save_evaluation(
        {"summary": summary_data, "detailed_evaluations": evaluations},
        os.path.join("hypothesis_1", "h1_evaluation.json"),
        "H1 Academic Eval",
    )

    # --- PRINT FINAL REPORT ---
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS (n={count})")
    print("=" * 70)
    print(f"INGREDIENTS EXTRACTION:")
    print(f"  Precision: {avg_ing_prec:.1%}")
    print(f"  Recall:    {avg_ing_rec:.1%}")
    print(f"  F1-Score:  {avg_ing_f1:.1%}")
    print("-" * 30)
    print(f"NUTRITION DATA EXTRACTION:")
    print(f"  Precision: {avg_nutr_prec:.1%}")
    print(f"  Recall:    {avg_nutr_rec:.1%}")
    print(f"  F1-Score:  {avg_nutr_f1:.1%}")
    print("=" * 70)
    print(f"OVERALL SYSTEM F1-SCORE: {avg_overall_f1:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
