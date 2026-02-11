import asyncio
from typing import Any

from src.agent_cache import AgentResponseCache, make_cache_key, normalize_query


def test_normalize_query_normalizes_digits_and_punctuation():
    raw = "  ارزان\u200cترین   شامپو!! ۱۲۳  "
    normalized = normalize_query(raw)
    assert normalized == "ارزان\u200cترین شامپو 123"


def test_make_cache_key_is_stable_for_equivalent_queries():
    q1 = "من   دوغ میخوام"
    q2 = "من دوغ میخوام!!!"
    assert make_cache_key(q1) == make_cache_key(q2)


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def ping(self):
        return True

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ex: int | None = None):
        self.store[key] = value
        return True

    async def delete(self, key: str):
        self.store.pop(key, None)
        return 1

    async def scan(self, cursor: int, match: str, count: int = 100):
        prefix = match.replace("*", "")
        keys = [k for k in self.store.keys() if k.startswith(prefix)]
        return 0, keys

    async def close(self):
        return None


def _run(coro: Any):
    return asyncio.run(coro)


def test_agent_cache_set_and_get_roundtrip():
    cache = AgentResponseCache(default_ttl=60)
    cache._redis = FakeRedis()

    response = {
        "success": True,
        "response": "ok",
        "session_id": "abc",
        "products": [{"id": "1", "name": "x", "price": 10, "has_discount": False}],
        "metadata": {"took_ms": 50, "query_type": "direct"},
    }

    stored = _run(cache.set("دوغ", response))
    assert stored is True

    cached = _run(cache.get("دوغ"))
    assert cached is not None
    assert cached["success"] is True
    assert cached["metadata"]["from_agent_cache"] is True
    assert "cached_at" in cached["metadata"]


def test_agent_cache_does_not_store_unsuccessful_response():
    cache = AgentResponseCache(default_ttl=60)
    cache._redis = FakeRedis()

    stored = _run(cache.set("query", {"success": False, "metadata": {}}))
    assert stored is False


def test_agent_cache_stats_counts_keys():
    cache = AgentResponseCache(default_ttl=60)
    cache._redis = FakeRedis()

    _run(cache.set("query 1", {"success": True, "metadata": {"took_ms": 1}, "products": []}))
    _run(cache.set("query 2", {"success": True, "metadata": {"took_ms": 1}, "products": []}))

    stats = _run(cache.get_stats())
    assert stats["available"] is True
    assert stats["cached_responses"] == 2
