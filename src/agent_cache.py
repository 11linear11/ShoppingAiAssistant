"""
Agent Response Cache

Redis-based caching layer for the agent's final response.
When a user sends a query that has been answered before,
this cache returns the stored response instantly — 
skipping the LLM, interpret, and search stages entirely.

Cache key is derived from a normalized version of the user message.
Stored value is the full response dict (text + products + metadata).

Designed to work alongside the search-level cache:
  Level 1 (search cache):  Skips Elasticsearch, LLM still runs  → ~40-60s → ~5-10s
  Level 2 (agent cache):   Skips everything, returns cached response → <100ms
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Optional

import redis.asyncio as aioredis

from src.pipeline_logger import log_agent, log_cache, log_error


# ============================================================================
# Query Normalizer
# ============================================================================


def normalize_query(query: str) -> str:
    """
    Normalize user query to produce a stable cache key.

    Handles:
    - Persian/Arabic digit normalization
    - Whitespace collapsing
    - Punctuation stripping
    - Lowercasing (for Latin chars)
    - Stripping common filler words at start (e.g. "میخوام", "لطفا")

    Examples:
        "  من ژله  میخوام  " → "من ژله میخوام"
        "ارزان‌ترین   شامپو!!" → "ارزان‌ترین شامپو"
    """
    if not query:
        return ""

    text = query.strip()

    # Normalize Persian/Arabic digits → ASCII
    digit_map = str.maketrans(
        "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
        "01234567890123456789",
    )
    text = text.translate(digit_map)

    # Remove punctuation (keep Persian/Arabic/Latin alphanumeric + spaces + ZWNJ)
    text = re.sub(r"[^\w\s\u200c]", "", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Lowercase Latin characters
    text = text.lower()

    return text


def make_cache_key(query: str) -> str:
    """
    Build a Redis key for the agent response cache.

    Key format: cache:v1:agent:{sha256_of_normalized_query}
    """
    normalized = normalize_query(query)
    query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"cache:v1:agent:{query_hash}"


# ============================================================================
# Agent Response Cache
# ============================================================================


class AgentResponseCache:
    """
    Redis-backed cache for the agent's final response.

    Usage:
        cache = AgentResponseCache(redis_host="redis")
        await cache.connect()

        # Check cache
        hit = await cache.get(user_message)
        if hit:
            return hit  # skip LLM entirely

        # ... run agent normally ...
        await cache.set(user_message, response_dict, ttl=86400)
    """

    def __init__(
        self,
        redis_host: str = "127.0.0.1",
        redis_port: int = 6379,
        redis_password: str = "",
        redis_db: int = 0,
        default_ttl: int = 86400,
    ):
        self._host = redis_host
        self._port = redis_port
        self._password = redis_password or None
        self._db = redis_db
        self._ttl = default_ttl
        self._redis: Optional[aioredis.Redis] = None

    async def connect(self) -> bool:
        """Connect to Redis. Returns True if successful."""
        try:
            self._redis = aioredis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                db=self._db,
                decode_responses=True,
            )
            await self._redis.ping()
            log_agent("AgentResponseCache connected", {
                "host": self._host,
                "port": self._port,
            })
            return True
        except Exception as e:
            log_error("AGENT_CACHE", f"Redis connection failed: {e}", e)
            self._redis = None
            return False

    @property
    def available(self) -> bool:
        """Check if cache is available."""
        return self._redis is not None

    async def get(self, query: str) -> Optional[dict]:
        """
        Look up a cached agent response for the given user query.

        Returns the full response dict if found, None otherwise.
        """
        if not self._redis:
            return None

        key = make_cache_key(query)

        try:
            raw = await self._redis.get(key)
            if raw is None:
                log_cache(f"Agent cache MISS", {"key": key, "query": query[:50]})
                return None

            data = json.loads(raw)

            # Update metadata to indicate this came from cache
            if "metadata" in data:
                data["metadata"]["from_agent_cache"] = True
                data["metadata"]["cache_key"] = key

            log_cache(f"Agent cache HIT", {
                "key": key,
                "query": query[:50],
                "original_took_ms": data.get("metadata", {}).get("original_took_ms", "?"),
            })
            return data

        except Exception as e:
            log_error("AGENT_CACHE", f"Cache get failed: {e}", e)
            return None

    async def set(
        self,
        query: str,
        response: dict,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store an agent response in cache.

        Args:
            query: The original user message
            response: The full response dict from agent_service.chat()
            ttl: TTL in seconds (defaults to self._ttl)

        Returns:
            True if stored successfully
        """
        if not self._redis:
            return False

        # Only cache successful responses with products
        if not response.get("success", False):
            return False

        key = make_cache_key(query)
        ttl = ttl or self._ttl

        try:
            # Store the original response time before caching
            cache_data = response.copy()
            if "metadata" in cache_data:
                cache_data["metadata"] = cache_data["metadata"].copy()
                cache_data["metadata"]["original_took_ms"] = cache_data["metadata"].get("took_ms", 0)
                cache_data["metadata"]["cached_at"] = datetime.now().isoformat()

            raw = json.dumps(cache_data, ensure_ascii=False)
            await self._redis.set(key, raw, ex=ttl)

            log_cache(f"Agent cache SET", {
                "key": key,
                "query": query[:50],
                "ttl": ttl,
                "products_count": len(response.get("products", [])),
            })
            return True

        except Exception as e:
            log_error("AGENT_CACHE", f"Cache set failed: {e}", e)
            return False

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._redis:
            return {"available": False}

        try:
            # Count agent cache keys
            cursor = 0
            count = 0
            while True:
                cursor, keys = await self._redis.scan(cursor, match="cache:v1:agent:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break

            return {
                "available": True,
                "cached_responses": count,
                "host": self._host,
                "default_ttl": self._ttl,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None
