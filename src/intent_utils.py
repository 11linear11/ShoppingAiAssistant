"""
Intent normalization utilities.

Normalizes intent labels from different components into a stable canonical set.
"""

from __future__ import annotations


CANONICAL_INTENTS = {
    "browse",
    "find_cheapest",
    "find_best_value",
    "compare",
}


def normalize_intent(raw_intent: str | None) -> str:
    """Normalize incoming intent label to canonical values."""
    value = (raw_intent or "").strip().lower()
    if not value:
        return "browse"

    aliases = {
        "find_best": "find_best_value",
        "best": "find_best_value",
        "find_high_quality": "find_best_value",
        "find_best_quality": "find_best_value",
        "high_quality": "find_best_value",
        "cheapest": "find_cheapest",
        "lowest_price": "find_cheapest",
        "find_cheap": "find_cheapest",
        "cheap": "find_cheapest",
        "normal": "browse",
        "default": "browse",
        "search": "browse",
    }

    normalized = aliases.get(value, value)
    if normalized in CANONICAL_INTENTS:
        return normalized
    return "browse"
