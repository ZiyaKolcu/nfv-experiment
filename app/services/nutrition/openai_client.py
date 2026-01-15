"""Centralized OpenAI client helpers for JSON-only chat completions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from openai import (
    OpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from dotenv import load_dotenv

load_dotenv()


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_openai_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call OpenAI and return parsed JSON dict. Raises ValueError on failure.

    The model must return STRICT JSON (no markdown fencing, no prose).
    """
    try:
        client = OpenAI()
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or ""
    except (
        APIError,
        APIConnectionError,
        RateLimitError,
        AuthenticationError,
        BadRequestError,
    ) as exc:
        raise ValueError(f"OpenAI call failed: {exc}") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        snippet = content[:200]
        raise ValueError(
            f"Model did not return valid JSON: {exc}; content={snippet}..."
        ) from exc
