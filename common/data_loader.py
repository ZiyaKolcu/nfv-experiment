"""Common data loading utilities for both hypotheses."""

import json
from typing import Any, Dict, List
from pathlib import Path


def load_raw_ocr_data(filepath: str = "raw_ocr_texts.json") -> List[Dict[str, Any]]:
    """
    Load raw OCR texts from Flutter app output.

    Returns:
        List of products with their OCR texts
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} products from {filepath}")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File '{filepath}' not found.")
        return []
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in '{filepath}': {e}")
        return []


def load_ground_truth(dataset_dir: str = "Dataset") -> Dict[str, Dict[str, Any]]:
    """
    Load all ground truth files from the dataset directory.

    Returns:
        Dict mapping product_id -> ground_truth_data
    """
    ground_truths = {}
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"Warning: Dataset directory '{dataset_dir}' not found")
        return {}

    for product_dir in dataset_path.iterdir():
        if not product_dir.is_dir():
            continue

        gt_file = product_dir / "ground_truth.json"
        if gt_file.exists():
            try:
                with open(gt_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    product_id = product_dir.name
                    ground_truths[product_id] = data
            except Exception as e:
                print(f"Error loading {gt_file}: {e}")

    print(f"✓ Loaded {len(ground_truths)} ground truth files")
    return ground_truths


def combine_product_texts(product_data: Dict[str, Any]) -> str:
    """
    Combine all image texts for a product into a single string.

    Args:
        product_data: Product dict with 'images' list

    Returns:
        Combined text separated by double newlines
    """
    images = product_data.get("images", [])
    combined_texts = []

    for img in images:
        raw_text = img.get("raw_text", "").strip()
        if raw_text:
            combined_texts.append(raw_text)

    return "\n\n".join(combined_texts)
