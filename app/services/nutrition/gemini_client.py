"""Centralized Gemini client helpers for JSON-only chat completions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def call_gemini_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call Gemini and return parsed JSON dict."""

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        generate_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            system_instruction=system_prompt,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=user_prompt, config=generate_config
        )

        if not response.text:
            raise ValueError("Gemini returned empty response")

        content = response.text.strip()

    except Exception as exc:
        raise ValueError(f"Gemini call failed: {exc}") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        snippet = content[:200]
        raise ValueError(
            f"Model did not return valid JSON: {exc}; content={snippet}..."
        ) from exc
