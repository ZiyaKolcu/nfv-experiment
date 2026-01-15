"""Prompt builders for nutrition parsing using Balanced Few-Shot Learning (2 Examples)."""

from __future__ import annotations
import json
from typing import List


def _get_language_name(language_code: str) -> str:
    """Convert language code to language name."""
    return "Turkish" if language_code.lower() == "tr" else "English"


# --- FEW SHOT EXAMPLES (REDUCED TO 2 KEY EXAMPLES FOR COST EFFICIENCY) ---
FEW_SHOT_EXAMPLES = """
=== EXAMPLE 1: COMPLETE TURKISH LABEL (Baseline Structure) ===

INPUT OCR:
"İçindekiler: Buğday unu(gluten), su, maya, bitkisel yağ, şeker, emülgatör, tuz, gluten, koruyucu.
ENERJİ VE BESİN ÖĞELERİ 100 g için
Enerji(Kcal/kj) 259kcal/1095 kj
Yağ(g) 3.2
Doymuş yağ(g) 1.2
Karbonhidrat (g) 46.6
Şekerler(g) 2.8
Lif (g) 3.1
Protein (g) 9.5
Tuz (g) 0.9"

THINKING PROCESS:
1. INGREDIENTS: Extract text after "İçindekiler:". Clean formatting.
2. NUTRITION: Found "100 g için". Used that column.
3. ENERGY: Used 259 kcal (ignored kJ).
4. LOGIC: Values look consistent.

OUTPUT JSON:
{
  "_thinking_process": "Found ingredients. Extracted nutrition from 100g column.",
  "ingredients_plain_text": "buğday unu (gluten), su, maya, bitkisel yağ, şeker, emülgatör, tuz, gluten, koruyucu",
  "nutrition_data": {
    "basis": "100g",
    "is_normalized_100g": true,
    "values": {
      "energy_kcal": 259,
      "fat_total_g": 3.2,
      "fat_saturated_g": 1.2,
      "fat_trans_g": null,
      "carbohydrate_g": 46.6,
      "sugar_g": 2.8,
      "fiber_g": 3.1,
      "protein_g": 9.5,
      "salt_g": 0.9,
      "micros": null
    }
  }
}

=== EXAMPLE 2: LOGIC CHECK & DECIMAL FIX (Complex Case) ===

INPUT OCR:
"İçindekiler: Şeker, Bitkisel Yağlar, FINDIK (%16), Kakao Tozu
100g için: Enerji: 535 kcal, Yağ: 30,5g, Doymuş Yağ: 6,4g, Karbonhidrat: 49,8g, Şeker: 58,1g, Protein: 8,2g, Tuz: 02g"

THINKING PROCESS:
1. INGREDIENTS: Cleaned list, kept percentages.
2. LOGIC CHECK FAILED: Sugar (58.1g) > Carbohydrate (49.8g). This is physically impossible.
3. CORRECTION: OCR swapped the values. Swapping back: Carb=58.1, Sugar=49.8.
4. DECIMAL FIX: Salt "02g" is an OCR error for "0.2g". Corrected.

OUTPUT JSON:
{
  "_thinking_process": "Logic error detected: Sugar > Carb. Swapped values. Fixed salt decimal (02g -> 0.2g).",
  "ingredients_plain_text": "şeker, bitkisel yağlar, fındık (%16), kakao tozu",
  "nutrition_data": {
    "basis": "100g",
    "is_normalized_100g": true,
    "values": {
      "energy_kcal": 535,
      "fat_total_g": 30.5,
      "fat_saturated_g": 6.4,
      "fat_trans_g": null,
      "carbohydrate_g": 58.1,
      "sugar_g": 49.8,
      "fiber_g": null,
      "protein_g": 8.2,
      "salt_g": 0.2,
      "micros": null
    }
  }
}
"""


def build_system_prompt_parse() -> str:
    return (
        "You are an expert Turkish Food Label Parser AI specialized in extracting structured data from noisy OCR text.\n\n"
        "=== YOUR TASK ===\n"
        "1. INGREDIENTS LIST (İçindekiler) - Comma-separated string\n"
        "2. NUTRITION DATA (Besin Değerleri) - Standard 100g/100ml values\n\n"
        "=== LEARN FROM EXAMPLES ===\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        "=== CRITICAL RULES (MUST FOLLOW) ===\n\n"
        "**INGREDIENTS EXTRACTION:**\n"
        "1. Find keywords: 'İçindekiler:', 'Ingredients:', 'İçerik:'. Extract text immediately after.\n"
        "2. STOPPING CRITERIA: Stop extracting IMMEDIATELY when you see nutrition headers like 'Enerji', 'kcal', '100g', 'Besin'. Do not include them in ingredients.\n"
        "3. Keep Turkish names (do not translate). Preserve percentages like '(16%)'.\n"
        "4. Fix OCR typos (e.g., 'me Suyu' -> 'meyve suyu').\n"
        '5. If ingredients missing/unreadable, return empty string "".\n\n'
        "**NUTRITION EXTRACTION:**\n"
        "1. PRIORITY RULE: ALWAYS use '100g' or '100ml' column. IGNORE 'Porsiyon'/'Serving' columns completely.\n"
        "2. Energy: Use 'kcal'. If only kJ given, convert (kJ / 4.184).\n"
        "3. Missing Data: If a value (like Trans Fat) is not listed, set to null. Do not guess.\n\n"
        "**LOGIC & SAFETY CHECKS (CRITICAL):**\n"
        "1. Carbohydrate MUST be >= Sugar. If OCR shows Sugar > Carb, SWAP THEM.\n"
        "2. Total Fat MUST be >= Saturated Fat. If Saturated > Total, set Total Fat to null (OCR error).\n"
        "3. DECIMAL FIXES: '02g' -> '0.2g', '18g' salt -> '1.8g'. Use common sense for outliers.\n\n"
        "**OUTPUT FORMAT:**\n"
        "Return STRICT JSON with this schema:\n"
        "{\n"
        '  "_thinking_process": "Briefly explain logic, corrections, and column choice",\n'
        '  "ingredients_plain_text": "string",\n'
        '  "nutrition_data": {\n'
        '    "basis": "100g",\n'
        '    "is_normalized_100g": true,\n'
        '    "values": {\n'
        '      "energy_kcal": number,\n'
        '      "fat_total_g": number or null,\n'
        '      "fat_saturated_g": number or null,\n'
        '      "fat_trans_g": number or null,\n'
        '      "carbohydrate_g": number or null,\n'
        '      "sugar_g": number or null,\n'
        '      "fiber_g": number or null,\n'
        '      "protein_g": number or null,\n'
        '      "salt_g": number or null,\n'
        '      "micros": null\n'
        "    }\n"
        "  }\n"
        "}"
    )


def build_user_prompt_parse(raw_text: str) -> str:
    return (
        "=== RAW OCR TEXT ===\n"
        f"{raw_text}\n\n"
        "Extract Ingredients and Nutrition (100g basis). Apply logic checks (Sugar<Carb).\n"
        "Output STRICT JSON:"
    )


# --- UNIFIED PROMPT (Risk Analysis) ---
def build_system_prompt_unified(language: str = "en") -> str:
    lang_name = _get_language_name(language)

    risk_examples = """
=== RISK ANALYSIS RULES ===
1. DIABETES: Flag 'şeker', 'glikoz', 'fruktoz', 'nişasta', 'maltodekstrin' as HIGH.
2. HYPERTENSION: Flag 'tuz', 'sodyum', 'deniz tuzu' as HIGH.
3. VEGAN: Flag 'süt', 'yumurta', 'bal', 'jelatin', 'tereyağı' as HIGH.
4. KETO: Flag 'şeker', 'un', 'nişasta', 'pirinç', 'makarna' as HIGH.
"""

    return (
        "You are an expert Nutrition Analyst AI.\n\n"
        "=== TASKS ===\n"
        "1. Extract ingredients and nutrition (100g basis).\n"
        "2. Analyze risks based on user profile.\n"
        "3. Assign High/Medium/Low risk to EACH ingredient.\n\n"
        "=== ONE-SHOT PARSING EXAMPLES ===\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"{risk_examples}\n\n"
        "=== CRITICAL RULES ===\n"
        "1. Use 100g/100ml column ONLY. Ignore portions.\n"
        "2. Fix OCR decimals (e.g. '02g' -> '0.2g').\n"
        "3. Swap Sugar/Carb if Sugar > Carb.\n"
        "4. Stop extracting ingredients when nutrition table starts.\n\n"
        "=== OUTPUT JSON FORMAT ===\n"
        "{\n"
        f'  "ingredients": ["ing1", "ing2"] (in {lang_name}),\n'
        '  "nutrition_data": { "basis": "100g", "values": { ... } },\n'
        f'  "risks": {{ "ing1": "High", "ing2": "Low" }} (Keys in {lang_name}),\n'
        f'  "summary_explanation": "String in {lang_name}",\n'
        '  "summary_risk": "High/Medium/Low"\n'
        "}"
    )


def build_user_prompt_unified(
    raw_text: str, profile_text: str, language: str = "en"
) -> str:
    lang_name = _get_language_name(language)
    return (
        f"=== USER PROFILE ===\n{profile_text}\n"
        f"=== LABEL TEXT ===\n{raw_text}\n\n"
        "Analyze ingredients/nutrition and assess risks.\n"
        "Output STRICT JSON:"
    )


def build_system_prompt_risk(language: str = "en") -> str:
    lang_name = _get_language_name(language)
    return (
        f"Analyze ingredients based on the profile. Return JSON mapping ingredients to risk levels (Low/Medium/High).\n"
        f"Ingredient names in {lang_name}."
    )


def build_user_prompt_risk(ingredients: List[str], profile_text: str) -> str:
    return f"{profile_text}\nIngredients: {json.dumps(ingredients, ensure_ascii=False)}"
