import json
import os
from typing import Dict, Any


def load_profiles() -> Dict[str, Any]:
    """Load profiles from JSON file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "profiles.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Profile file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile_by_name(profile_name: str) -> Dict[str, Any]:
    profiles = load_profiles()
    if profile_name not in profiles:
        raise ValueError(f"Profile {profile_name} not found")
    return profiles[profile_name]
