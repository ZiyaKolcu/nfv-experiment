"""Unified label analysis orchestrating parsing, risk assessment, and summary."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from app.services.nutrition.llm_client import call_llm_json
from app.services.nutrition.prompt_templates import (
    build_system_prompt_unified,
    build_user_prompt_unified,
)
from app.services.nutrition.label_parser import normalize_nutrition
from app.services.nutrition.health_risk_assessor import _is_allergen_match


def _profile_to_text(profile: Optional[Dict[str, Any]]) -> str:
    if not profile:
        return ""
    allergies = profile.get("allergies") or []
    health_conditions = profile.get("health_conditions") or []
    prefs = profile.get("dietary_preferences") or []
    return (
        "User health profile:\n"
        f"Allergies: {', '.join(allergies) if allergies else 'None'}\n"
        f"Health conditions: {', '.join(health_conditions) if health_conditions else 'None'}\n"
        f"Dietary preferences: {', '.join(prefs) if prefs else 'None'}\n\n"
    )


def analyze_label_with_profile(
    raw_text: str, health_profile: Optional[Dict[str, Any]], language: str = "en"
) -> Tuple[
    List[str],
    Dict[str, Any],
    Dict[str, str],
    str,
    str,
]:
    """Single-call analysis that returns ingredients, normalized nutrition map,
    ingredient risk labels, a short explanation, and an overall risk.
    """
    system_prompt = build_system_prompt_unified(language=language)
    user_prompt = build_user_prompt_unified(
        raw_text, _profile_to_text(health_profile), language=language
    )
    data = call_llm_json(system_prompt, user_prompt)

    # Handle ingredients - could be a list, string, or other type
    ingredients_list: List[str] = []
    ingredients_data = data.get("ingredients")

    if isinstance(ingredients_data, list):
        ingredients_list = [
            str(token).strip()
            for token in ingredients_data
            if isinstance(token, (str, int, float)) and str(token).strip()
        ]
    elif isinstance(ingredients_data, str):
        # If it's a string, try to parse it as JSON first, then fall back to comma-split
        try:
            parsed = json.loads(ingredients_data)
            if isinstance(parsed, list):
                ingredients_list = [str(item).strip() for item in parsed if item]
            else:
                # Not a list after parsing, split by comma
                ingredients_list = [
                    token.strip()
                    for token in ingredients_data.split(",")
                    if token.strip()
                ]
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, split by comma
            ingredients_list = [
                token.strip() for token in ingredients_data.split(",") if token.strip()
            ]

    # Fetch 'nutrition_data'
    nutrition_raw = data.get("nutrition_data") or {}
    normalized = normalize_nutrition(nutrition_raw)

    risks: Dict[str, str] = {}
    risks_data = data.get("risks")

    # Handle case where risks might be a JSON string
    if isinstance(risks_data, str):
        try:
            risks_data = json.loads(risks_data)
        except (json.JSONDecodeError, ValueError):
            risks_data = {}

    if isinstance(risks_data, dict):
        for name, label in risks_data.items():
            if (
                isinstance(name, str)
                and isinstance(label, str)
                and label in {"Low", "Medium", "High"}
            ):
                risks[name.strip()] = label

    allergy_list: List[str] = []
    if health_profile and isinstance(health_profile.get("allergies"), list):
        allergy_list = health_profile.get("allergies")
    if allergy_list:
        for ing in ingredients_list:
            if _is_allergen_match(ing, allergy_list):
                risks[ing] = "High"
    for ing in ingredients_list:
        risks.setdefault(ing, "Low")

    summary_explanation = str(data.get("summary_explanation") or "").strip()
    summary_risk = str(data.get("summary_risk") or "").strip()
    if summary_risk not in {"Low", "Medium", "High"}:
        summary_risk = "Low"

    # Re-evaluate summary_risk based on actual ingredient risks
    # If any ingredient is High risk, summary should be High
    # If any ingredient is Medium risk (and no High), summary should be Medium
    risk_values = list(risks.values())
    if "High" in risk_values:
        summary_risk = "High"
    elif "Medium" in risk_values:
        summary_risk = "Medium"
    # else keep the AI's assessment or default "Low"

    return ingredients_list, normalized, risks, summary_explanation, summary_risk
