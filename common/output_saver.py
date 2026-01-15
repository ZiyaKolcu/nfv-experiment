"""Common output saving utilities."""

import json
from typing import Any, Dict, List


def save_results(
    results: List[Dict[str, Any]], output_file: str, hypothesis_name: str = "Test"
) -> None:
    """
    Save processing results to JSON file with summary statistics.

    Args:
        results: List of result dictionaries
        output_file: Output filename
        hypothesis_name: Name for display (e.g., "Hypothesis 1")
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Results saved to {output_file}")

        # Print summary statistics
        total = len(results)
        successful = sum(1 for r in results if r.get("llm_response") is not None)
        failed = total - successful

        print(f"\n{'='*60}")
        print(f"{hypothesis_name} - Processing Summary:")
        print(f"{'='*60}")
        print(f"Total products:         {total}")
        print(f"Successfully processed: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed:                 {failed} ({failed/total*100:.1f}%)")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"✗ Error saving results: {e}")


def save_evaluation(
    evaluation_data: Dict[str, Any],
    output_file: str,
    hypothesis_name: str = "Hypothesis 1",
) -> None:
    """
    Save evaluation results to JSON file.

    Args:
        evaluation_data: Evaluation metrics dictionary
        output_file: Output filename
        hypothesis_name: Name for display
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
        print(f"\n✓ {hypothesis_name} evaluation saved to {output_file}")
    except Exception as e:
        print(f"✗ Error saving evaluation: {e}")
