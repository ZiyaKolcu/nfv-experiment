"""Parsing helpers for OCR text and nutrition normalization."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from app.services.nutrition.llm_client import call_llm_json
from app.services.nutrition.prompt_templates import (
    build_system_prompt_parse,
    build_user_prompt_parse,
)

EXPECTED_MACRO_KEYS = [
    "energy_kcal",
    "fat_total_g",
    "fat_saturated_g",
    "fat_trans_g",
    "carbohydrate_g",
    "sugar_g",
    "fiber_g",
    "protein_g",
    "salt_g",
]


def normalize_nutrition(nutrition_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean the nutrition_data structure returned by LLM.
    Since LLM now handles the 'Priority Rule' (100g vs Portion),
    we just ensure the structure is correct.
    """
    # Handle case where nutrition_data might be a JSON string
    if isinstance(nutrition_data, str):
        try:
            nutrition_data = json.loads(nutrition_data)
        except (json.JSONDecodeError, ValueError):
            return {}

    if not isinstance(nutrition_data, dict):
        return {}

    # Default structure
    normalized = {
        "basis": nutrition_data.get("basis", "Unknown"),
        "is_normalized_100g": nutrition_data.get("is_normalized_100g", False),
        "values": {},
    }

    raw_values = nutrition_data.get("values") or {}

    # Process Macros (Fixed Keys)
    clean_values = {}
    for key in EXPECTED_MACRO_KEYS:
        val = raw_values.get(key)
        # Ensure it's a number or None
        if isinstance(val, (int, float)):
            clean_values[key] = val
        else:
            clean_values[key] = None

    # Process Micros (Dynamic Keys) - Pass through if exists
    micros = raw_values.get("micros")
    if isinstance(micros, dict) and micros:
        clean_values["micros"] = micros
    else:
        clean_values["micros"] = None

    normalized["values"] = clean_values
    return normalized


def parse_ocr_raw_text(
    raw_text: str,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Parse OCR raw text and return (ingredients_list, nutrition_data).
    nutrition_data follows the schema:
    { "basis": str, "is_normalized_100g": bool, "values": {...} }
    """
    system_prompt = build_system_prompt_parse()
    user_prompt = build_user_prompt_parse(raw_text)
    data = call_llm_json(system_prompt, user_prompt)

    # Handle ingredients - could be a string or already a list
    ingredients_data = data.get("ingredients_plain_text", "")
    if isinstance(ingredients_data, list):
        # Already a list, just clean it
        ingredients_list = [str(item).strip() for item in ingredients_data if item]
    elif isinstance(ingredients_data, str):
        # String, split by comma
        ingredients_plain = ingredients_data.strip()
        ingredients_list = [
            token.strip()
            for token in ingredients_plain.split(",")
            if token and token.strip()
        ]
    else:
        # Unexpected type, return empty list
        ingredients_list = []

    # LLM returns "nutrition_data" object directly now
    nutrition_raw = data.get("nutrition_data") or {}
    normalized = normalize_nutrition(nutrition_raw)

    return ingredients_list, normalized
