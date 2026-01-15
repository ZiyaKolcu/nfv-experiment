"""Unified LLM client abstraction that supports both OpenAI and Gemini."""

from __future__ import annotations

import os
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()


def get_llm_provider() -> str:
    """Determine which LLM provider to use based on environment variables.

    Returns:
        'openai', 'gemini', or 'claude'
    """
    provider = os.getenv("LLM_PROVIDER", "").lower()

    if provider in ["openai", "gemini", "claude", "anthropic"]:
        if provider == "anthropic":
            return "claude"
        return provider

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GOOGLE_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if openai_key:
        return "openai"
    elif gemini_key:
        return "gemini"
    elif anthropic_key:
        return "claude"
    else:
        raise ValueError(
            "No LLM provider configured. Set LLM_PROVIDER to 'openai', 'gemini', or 'claude', "
            "or provide OPENAI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY"
        )


def call_llm_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Call the configured LLM provider and return parsed JSON dict.

    This function automatically routes to the appropriate provider based on
    environment configuration, ensuring consistent behavior across experiments.

    Args:
        system_prompt: System-level instructions for the model
        user_prompt: User query/request

    Returns:
        Parsed JSON dictionary from the model response

    Raises:
        ValueError: If the call fails or returns invalid JSON
    """
    provider = get_llm_provider()

    if provider == "openai":
        from app.services.nutrition.openai_client import call_openai_json

        return call_openai_json(system_prompt, user_prompt)
    elif provider == "gemini":
        from app.services.nutrition.gemini_client import call_gemini_json

        return call_gemini_json(system_prompt, user_prompt)
    elif provider == "claude":
        from app.services.nutrition.claude_client import call_claude_json

        return call_claude_json(system_prompt, user_prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_model_name() -> str:
    """Get the name of the currently configured model.

    Returns:
        Model name string (e.g., 'gpt-4o-mini', 'gemini-1.5-flash', or 'claude-3-5-sonnet-latest')
    """
    provider = get_llm_provider()

    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "gemini":
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    elif provider == "claude":
        return os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
    else:
        return "unknown"
