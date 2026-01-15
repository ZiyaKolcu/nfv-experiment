"""Centralized Anthropic Claude client helpers for JSON-only chat completions."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from anthropic import (
    Anthropic,
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError,
    BadRequestError,
)
from dotenv import load_dotenv

load_dotenv()


CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def call_claude_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call Anthropic Claude and return parsed JSON dict.

    Uses the prefill technique to ensure valid JSON output:
    - Adds a prefilled assistant message with "{"
    - Prepends the missing "{" to the response before parsing

    Args:
        system_prompt: System-level instructions for the model
        user_prompt: User query/request

    Returns:
        Parsed JSON dictionary from the model response

    Raises:
        ValueError: If the call fails or returns invalid JSON
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "{"},
            ],
        )

        content = response.content[0].text if response.content else ""

    except (
        APIError,
        APIConnectionError,
        RateLimitError,
        AuthenticationError,
        BadRequestError,
    ) as exc:
        raise ValueError(f"Claude call failed: {exc}") from exc

    try:
        full_json = "{" + content

        json_decoder = json.JSONDecoder()
        parsed_data, end_idx = json_decoder.raw_decode(full_json)

        if "_thinking_process" in parsed_data:
            del parsed_data["_thinking_process"]

        return parsed_data
    except json.JSONDecodeError as exc:
        snippet = content[:500]
        raise ValueError(
            f"Model did not return valid JSON: {exc}; content={snippet}..."
        ) from exc
