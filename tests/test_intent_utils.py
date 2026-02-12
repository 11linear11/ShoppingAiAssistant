from src.intent_utils import normalize_intent


def test_normalize_intent_maps_best_aliases():
    assert normalize_intent("find_best") == "find_best_value"
    assert normalize_intent("find_high_quality") == "find_best_value"


def test_normalize_intent_maps_cheapest_aliases():
    assert normalize_intent("cheapest") == "find_cheapest"
    assert normalize_intent("lowest_price") == "find_cheapest"


def test_normalize_intent_fallbacks_to_browse():
    assert normalize_intent(None) == "browse"
    assert normalize_intent("") == "browse"
    assert normalize_intent("unknown_intent") == "browse"
