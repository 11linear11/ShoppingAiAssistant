"""
Unified Search MCP Server (MCP Protocol Version)
Port: 5002

Handles DSL generation, Elasticsearch search, and result re-ranking
for the Shopping AI Assistant.

Uses GitHub Models API (Llama-3.3-70B-Instruct) for DSL generation.

This is the MCP protocol version using FastMCP SDK.
"""

import asyncio
import hashlib
import json
import re
import sys
import uuid
from time import perf_counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
import redis.asyncio as aioredis
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.pipeline_logger import (
    TraceContext,
    log_search,
    log_cache,
    log_embed,
    log_error,
    log_latency_summary,
    reset_current_trace,
    set_current_trace,
    trace_stage,
)

# Load .env
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


# ============================================================================
# Configuration
# ============================================================================


class Settings(BaseSettings):
    """Application settings."""

    # GitHub API settings (same as interpret server)
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_base_url: str = Field(
        default="https://models.inference.ai.azure.com",
        alias="GITHUB_BASE_URL",
    )
    github_model: str = Field(
        default="Llama-3.3-70B-Instruct",
        alias="GITHUB_MODEL",
    )

    # Mixtral DSL generation via OpenRouter
    openrouter_api_key: str = Field(default="", alias="OPEN_ROUTERS_API_KEY")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    mixtral_model: str = Field(default="mistralai/mixtral-8x7b-instruct", alias="MIXTRAL_MODEL")
    mixtral_dsl_timeout: float = Field(default=10.0, alias="MIXTRAL_DSL_TIMEOUT")

    # Elasticsearch settings
    es_host: str = Field(default="localhost", alias="ELASTICSEARCH_HOST")
    es_port: int = Field(default=9200, alias="ELASTICSEARCH_PORT")
    es_scheme: str = Field(default="http", alias="ELASTICSEARCH_SCHEME")
    es_user: str = Field(default="elastic", alias="ELASTICSEARCH_USER")
    es_password: str = Field(default="", alias="ELASTICSEARCH_PASSWORD")
    es_index: str = Field(default="shopping_products", alias="ELASTICSEARCH_INDEX")

    # MCP service URLs
    embedding_url: str = Field(default="http://localhost:5003", alias="MCP_EMBEDDING_URL")

    # Redis settings (direct connection for caching)
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    cache_search_ttl: int = Field(default=3600, alias="CACHE_SEARCH_TTL")
    cache_dsl_ttl: int = Field(default=3600, alias="CACHE_DSL_TTL")

    # Search settings
    search_timeout: int = Field(default=30, alias="SEARCH_TIMEOUT")
    search_size: int = Field(default=50, alias="SEARCH_SIZE")
    result_limit: int = Field(default=10, alias="RESULT_LIMIT")

    # Debug mode
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")
    use_semantic_search: bool = Field(default=False, alias="USE_SEMANTIC_SEARCH")

    model_config = {"extra": "ignore"}


settings = Settings()


# ============================================================================
# LangChain LLM Client
# ============================================================================

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class LangChainLLM:
    """LangChain-based LLM client using ChatOpenAI with GitHub Models endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.1,
            max_tokens=2000,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ) -> str:
        """Generate text using LangChain with Groq API."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            log_error("SEARCH", f"LangChain LLM error: {e}", e)
            raise

    async def close(self):
        """Cleanup (no-op for LangChain)."""
        pass


# ============================================================================
# Mixtral DSL Generator
# ============================================================================

# All known categories in the database
KNOWN_CATEGORIES = [
    "خانه و سبک زندگی", "آرایشی و بهداشتی", "تنقلات", "نوشیدنی",
    "مد و پوشاک", "لوازم برقی و دیجیتال", "دستمال و شوینده",
    "چاشنی و افزودنی", "لبنیات", "خواربار و نان",
    "کنسرو و غذای آماده", "مواد پروتئینی", "خشکبار، دسر و شیرینی",
    "کودک و نوزاد", "کالاهای اساسی", "شوینده و مواد ضد عفونی کننده",
    "بهداشت و سلامت", "صبحانه", "طلا", "میوه و سبزیجات تازه",
    "محصولات سلولزی", "کنسرو و غذاهای آماده", "خانه و آشپزخانه",
    "دسر و شیرینی پزی", "آجیل و خشکبار", "لوازم تحریر و اداری",
    "نان و شیرینی", "لوازم الکترونیکی",
]

MIXTRAL_DSL_SYSTEM_PROMPT = """You are an Elasticsearch DSL query generator for a Persian e-commerce product database.

## Elasticsearch Index: shopping_products

### Fields:
1. **product_name** (text, searchable): Product name in Persian. Examples: "خامه نیم چرب هراز ۲۰۰ میلی لیتری", "بسته ۳ عددی پنیر سفید نیم چرب پگاه", "شامپو سر ضد شوره هد اند شولدرز"
2. **brand_name** (text + keyword): Brand name in Persian. Examples: "هراز", "پگاه", "سامسونگ", "ال جی"
3. **category_name** (text + keyword): Product category in Persian. Must be one of the known categories.
4. **price** (float): Product price in Toman (Iranian currency). Range: ~10,000 to ~500,000,000
5. **discount_price** (float): Discounted price. Same as price if no discount.
6. **has_discount** (boolean): Whether product has a discount.
7. **discount_percentage** (float): Discount percentage (0-100).
8. **price_range** (keyword): "budget", "mid-range", "luxury", "premium"
9. **product_id** (keyword): Unique product identifier.
10. **product_embedding** (dense_vector, 768 dims): For semantic search. Use {{EMBEDDING_PLACEHOLDER}} as placeholder.

### Sample Records:
```json
{"product_name": "خامه نیم چرب هراز ۲۰۰ میلی لیتری", "brand_name": "هراز", "price": 67900.0, "discount_price": 54320.0, "category_name": "لبنیات", "price_range": "budget", "has_discount": true, "discount_percentage": 20.0}
{"product_name": "بسته ۳ عددی پنیر سفید نیم چرب حاوی ویتامین D۳ پگاه ۴۰۰ گرمی", "brand_name": "پگاه", "price": 232500.0, "category_name": "لبنیات", "price_range": "mid-range", "has_discount": false}
{"product_name": "گوشی موبایل سامسونگ Galaxy A54", "brand_name": "سامسونگ", "price": 15500000.0, "category_name": "لوازم برقی و دیجیتال", "price_range": "luxury", "has_discount": true, "discount_percentage": 10.0}
```

### Known Categories:
""" + ", ".join(KNOWN_CATEGORIES) + """

## RULES:
1. Return ONLY valid JSON (Elasticsearch DSL query). No explanation, no markdown.
2. Use "multi_match" on product_name (boost ^3) and brand_name (boost ^2) for text search.
3. Set fuzziness to "AUTO" for Persian text tolerance.
4. If categories are provided, ONLY keep those with ≥70% relevance to the searched product. Remove irrelevant ones.
5. If a brand is provided, add a "match" filter on brand_name.
6. If price range is provided, add a "range" filter on price.
7. For intent "find_cheapest", sort by price ASC then _score.
8. For intent "find_best" or "find_high_quality", sort by _score then price ASC.
9. Default sort: ["_score"].
10. Always set "size" to the provided size value.
11. If you want to include semantic/embedding search, add a "knn" section with field "product_embedding" and use the keyword {{EMBEDDING_PLACEHOLDER}} as the query_vector value. We will replace it with the real vector later.
12. When categories are passed, evaluate each one: if it's clearly unrelated to the product being searched (less than 70% likely), REMOVE it. Only keep highly relevant categories.

## Output Format:
Return ONLY the JSON DSL query object. Example:
{"query": {"bool": {"must": [...], "filter": [...]}}, "size": 50, "sort": [...]}
"""


class MixtralDSLGenerator:
    """Generate Elasticsearch DSL using Mixtral 8x7B via OpenRouter."""

    def __init__(self):
        self._llm: Optional[ChatOpenAI] = None
        self._dsl_cache_redis: Optional[aioredis.Redis] = None

    async def init(self):
        """Initialize the Mixtral client and DSL cache."""
        if settings.openrouter_api_key:
            self._llm = ChatOpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                model=settings.mixtral_model,
                temperature=0.05,
                max_tokens=2000,
            )
            log_search("MixtralDSLGenerator initialized", {"model": settings.mixtral_model})
        else:
            log_search("MixtralDSLGenerator: No OpenRouter API key, will use fallback DSL only")

    async def init_cache(self, redis_instance: Optional[aioredis.Redis]):
        """Use the same Redis instance as the search engine for DSL cache."""
        self._dsl_cache_redis = redis_instance

    def _make_dsl_cache_key(self, params_str: str) -> str:
        """Build cache key for DSL generation."""
        key_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"cache:v1:dsl:{key_hash}"

    async def _get_cached_dsl(self, params_str: str) -> Optional[dict]:
        """Check DSL cache."""
        if not self._dsl_cache_redis or settings.debug_mode:
            return None
        try:
            key = self._make_dsl_cache_key(params_str)
            val = await self._dsl_cache_redis.get(key)
            if val:
                log_cache("DSL cache HIT", {"key": key})
                return json.loads(val)
        except Exception as e:
            log_error("SEARCH", f"DSL cache GET failed: {e}", e)
        return None

    async def _store_dsl(self, params_str: str, dsl: dict) -> None:
        """Store DSL in cache."""
        if not self._dsl_cache_redis or settings.debug_mode:
            return
        try:
            key = self._make_dsl_cache_key(params_str)
            await self._dsl_cache_redis.set(
                key,
                json.dumps(dsl, ensure_ascii=False),
                ex=settings.cache_dsl_ttl,
            )
            log_cache("DSL cache SET", {"key": key, "ttl": settings.cache_dsl_ttl})
        except Exception as e:
            log_error("SEARCH", f"DSL cache SET failed: {e}", e)

    async def generate(self, search_params: dict[str, Any]) -> Optional[dict]:
        """
        Generate Elasticsearch DSL using Mixtral.
        
        Returns DSL dict on success, None on failure (caller should fallback).
        Has a 10-second timeout.
        """
        if not self._llm:
            return None

        product = search_params.get("product") or search_params.get("persian_full_query", "")
        intent = search_params.get("intent", "browse")
        brand = search_params.get("brand")
        categories = search_params.get("categories_fa", [])
        price_range = search_params.get("constraints", {}).get("price_range") or search_params.get("price_range", {})
        size = settings.search_size

        # Build cache key from params
        params_str = json.dumps({
            "product": product,
            "intent": intent,
            "brand": brand,
            "categories": sorted(categories) if categories else [],
            "price_range": price_range,
            "size": size,
        }, sort_keys=True, ensure_ascii=False)

        # Check DSL cache
        cached_dsl = await self._get_cached_dsl(params_str)
        if cached_dsl:
            return cached_dsl

        # Build prompt for Mixtral
        prompt_parts = [f"Generate an Elasticsearch DSL query for the following search:\n"]
        prompt_parts.append(f"Product: {product}")
        prompt_parts.append(f"Intent: {intent}")
        if brand:
            prompt_parts.append(f"Brand: {brand}")
        if categories:
            prompt_parts.append(f"Categories (evaluate relevance, keep only ≥70% relevant): {', '.join(str(c) for c in categories)}")
        if price_range:
            if price_range.get("min"):
                prompt_parts.append(f"Min price: {price_range['min']}")
            if price_range.get("max"):
                prompt_parts.append(f"Max price: {price_range['max']}")
        prompt_parts.append(f"Size: {size}")
        prompt_parts.append("\nReturn ONLY the JSON DSL query.")

        prompt = "\n".join(prompt_parts)

        try:
            # 10-second timeout
            response = await asyncio.wait_for(
                self._llm.ainvoke([
                    SystemMessage(content=MIXTRAL_DSL_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]),
                timeout=settings.mixtral_dsl_timeout,
            )

            raw_text = response.content.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', raw_text)
            if not json_match:
                log_error("SEARCH", f"Mixtral DSL: no JSON found in response: {raw_text[:200]}")
                return None

            dsl = json.loads(json_match.group())

            # Validate basic structure
            if "query" not in dsl:
                log_error("SEARCH", "Mixtral DSL: missing 'query' key")
                return None

            # Replace embedding placeholder with actual keyword marker
            dsl_str = json.dumps(dsl, ensure_ascii=False)
            if "{{EMBEDDING_PLACEHOLDER}}" in dsl_str:
                # Mark that embedding should be injected later
                dsl["_needs_embedding"] = True
                dsl["_embedding_query"] = product

            # Ensure size is set
            if "size" not in dsl:
                dsl["size"] = size

            log_search("✅ Mixtral DSL generated successfully", {
                "product": product,
                "has_knn": "knn" in dsl or dsl.get("_needs_embedding", False),
            })

            # Cache the DSL
            await self._store_dsl(params_str, dsl)

            return dsl

        except asyncio.TimeoutError:
            log_error("SEARCH", f"Mixtral DSL generation timed out after {settings.mixtral_dsl_timeout}s")
            return None
        except json.JSONDecodeError as e:
            log_error("SEARCH", f"Mixtral DSL: invalid JSON: {e}")
            return None
        except Exception as e:
            log_error("SEARCH", f"Mixtral DSL generation failed: {e}", e)
            return None


# ============================================================================
# Search Engine
# ============================================================================


class SearchEngine:
    """Unified search engine for e-commerce products."""

    def __init__(self, llm_client: LangChainLLM, mixtral_dsl: MixtralDSLGenerator):
        self.llm = llm_client
        self.mixtral_dsl = mixtral_dsl
        self.http_client = httpx.AsyncClient(timeout=settings.search_timeout)
        self._brand_scores: dict[str, float] = {}
        self._redis: Optional[aioredis.Redis] = None
        self._known_categories_map = {
            self._normalize_text(cat): cat for cat in KNOWN_CATEGORIES
        }

    async def init_redis(self):
        """Initialize direct Redis connection for caching."""
        try:
            self._redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                db=settings.redis_db,
                decode_responses=True,
            )
            await self._redis.ping()
            log_cache(
                "Search engine connected to Redis",
                {"host": settings.redis_host, "port": settings.redis_port},
            )
            # Share Redis with Mixtral DSL generator for DSL caching
            await self.mixtral_dsl.init_cache(self._redis)
        except Exception as e:
            log_error("SEARCH", f"Redis connection failed (caching disabled): {e}", e)
            self._redis = None

    async def load_brand_scores(self, filepath: str = "BrandScore.json"):
        """Load brand reputation scores."""
        try:
            project_root = Path(__file__).parent.parent.parent
            with open(project_root / filepath, "r", encoding="utf-8") as f:
                self._brand_scores = json.load(f)
            log_search("Brand scores loaded", {"count": len(self._brand_scores)})
        except FileNotFoundError:
            log_error("SEARCH", f"Brand scores file not found: {filepath}")

    async def search(
        self,
        search_params: dict[str, Any],
        session_id: str,
        use_cache: bool = True,
        use_semantic: bool = True,
    ) -> dict[str, Any]:
        """Execute full search pipeline."""
        pipeline_start = perf_counter()
        timings: dict[str, int] = {}
        search_params = dict(search_params or {})

        original_categories = search_params.get("categories_fa", []) or []
        guarded_categories = self._sanitize_categories(original_categories)
        if original_categories and guarded_categories != original_categories:
            log_search(
                "Category guard adjusted incoming categories",
                {"original": original_categories, "kept": guarded_categories},
            )
        search_params["categories_fa"] = guarded_categories

        if settings.debug_mode:
            use_cache = False

        product_key = search_params.get("product") or search_params.get("persian_full_query", "")
        intent = search_params.get("intent", "browse")
        display_query = search_params.get("persian_full_query", "")
        cache_lookup_keys = self._build_cache_lookup_keys(search_params, use_semantic=use_semantic)
        primary_cache_lookup_key = cache_lookup_keys[0] if cache_lookup_keys else ""
        lock_owner = ""
        lock_key = ""

        log_search("📥 Search request", {
            "product": search_params.get("product"),
            "query": display_query,
            "intent": intent,
        })

        def _build_result(
            *,
            total_hits: int,
            results: list[dict[str, Any]],
            from_cache: bool,
        ) -> dict[str, Any]:
            took_ms = int((perf_counter() - pipeline_start) * 1000)
            return {
                "success": True,
                "query": display_query,
                "total_hits": total_hits,
                "results": results,
                "took_ms": took_ms,
                "from_cache": from_cache,
                "latency_breakdown_ms": timings,
            }

        def _log_summary(result: dict[str, Any], cache_path: str):
            log_latency_summary(
                "SEARCH",
                "search.pipeline",
                result.get("took_ms", 0),
                breakdown_ms=timings,
                meta={
                    "session_id": session_id[:8],
                    "intent": intent,
                    "hits": result.get("total_hits", 0),
                    "from_cache": result.get("from_cache", False),
                    "cache_path": cache_path,
                },
            )

        # Check negative cache first (known no-results query)
        if use_cache:
            stage_start = perf_counter()
            is_negative = await self._check_negative_cache(product_key, intent)
            timings["negative_cache_check_ms"] = int((perf_counter() - stage_start) * 1000)
            if is_negative:
                result = _build_result(total_hits=0, results=[], from_cache=True)
                _log_summary(result, cache_path="negative_cache_hit")
                return result

        # Check cache
        if use_cache:
            cache_lookup_ms = 0
            for cache_lookup_key in cache_lookup_keys:
                stage_start = perf_counter()
                cached = await self._check_cache(cache_lookup_key)
                cache_lookup_ms += int((perf_counter() - stage_start) * 1000)
                if cached:
                    timings["search_cache_lookup_ms"] = cache_lookup_ms
                    result = _build_result(
                        total_hits=len(cached),
                        results=cached,
                        from_cache=True,
                    )
                    _log_summary(result, cache_path="search_cache_hit")
                    return result
            timings["search_cache_lookup_ms"] = cache_lookup_ms

            lock_owner = uuid.uuid4().hex
            stage_start = perf_counter()
            lock_key, lock_acquired = await self._acquire_cache_lock(primary_cache_lookup_key, lock_owner)
            timings["cache_lock_ms"] = int((perf_counter() - stage_start) * 1000)
            if not lock_acquired:
                stage_start = perf_counter()
                warmed = await self._wait_for_warm_cache(primary_cache_lookup_key)
                timings["warm_cache_wait_ms"] = int((perf_counter() - stage_start) * 1000)
                if warmed is not None:
                    result = _build_result(
                        total_hits=len(warmed),
                        results=warmed,
                        from_cache=True,
                    )
                    _log_summary(result, cache_path="warm_cache_hit")
                    return result

        try:
            # Generate DSL
            stage_start = perf_counter()
            dsl = await self.generate_dsl(search_params, use_semantic=use_semantic)
            timings["dsl_generation_ms"] = int((perf_counter() - stage_start) * 1000)

            # Execute search
            stage_start = perf_counter()
            raw_results = await self._execute_search(dsl)
            timings["es_search_ms"] = int((perf_counter() - stage_start) * 1000)

            # Protective guard: if category filters over-constrain to zero hits,
            # retry once without category filters.
            if not raw_results and search_params.get("categories_fa"):
                stage_start = perf_counter()
                relaxed_params = dict(search_params)
                relaxed_params["categories_fa"] = []
                relaxed_dsl = await self.generate_dsl(relaxed_params, use_semantic=use_semantic)
                raw_results = await self._execute_search(relaxed_dsl)
                timings["category_guard_retry_ms"] = int((perf_counter() - stage_start) * 1000)
                log_search(
                    "Category guard fallback search executed",
                    {
                        "original_categories": search_params.get("categories_fa", []),
                        "recovered_hits": len(raw_results),
                    },
                )

            if not raw_results:
                if use_cache:
                    stage_start = perf_counter()
                    await self._set_negative_cache(product_key, intent)
                    timings["negative_cache_set_ms"] = int((perf_counter() - stage_start) * 1000)
                result = _build_result(total_hits=0, results=[], from_cache=False)
                _log_summary(result, cache_path="miss_no_results")
                return result

            # Rerank results with search query for relevancy filtering
            search_query = search_params.get("product") or search_params.get("persian_full_query", "")
            stage_start = perf_counter()
            ranked_results = await self.rerank_results(
                raw_results,
                search_params.get("preferences", {}),
                intent,
                search_query=search_query,
            )
            timings["rerank_ms"] = int((perf_counter() - stage_start) * 1000)

            # Log rerank summary
            if ranked_results:
                top_relevancy = ranked_results[0].get("relevancy_score", 0) if ranked_results else 0
                log_search("📊 Rerank complete", {
                    "count": len(ranked_results),
                    "top_relevancy": top_relevancy,
                    "top_product": ranked_results[0].get("product_name", "")[:40] if ranked_results else "",
                })

            # Update cache
            if use_cache and ranked_results:
                stage_start = perf_counter()
                await self._update_cache(primary_cache_lookup_key, ranked_results)
                timings["search_cache_set_ms"] = int((perf_counter() - stage_start) * 1000)

            result = _build_result(
                total_hits=len(ranked_results),
                results=ranked_results[: settings.result_limit],
                from_cache=False,
            )
            _log_summary(result, cache_path="miss_fresh")
            return result
        except Exception as e:
            log_latency_summary(
                "SEARCH",
                "search.pipeline",
                int((perf_counter() - pipeline_start) * 1000),
                breakdown_ms=timings,
                meta={
                    "session_id": session_id[:8],
                    "intent": intent,
                    "success": False,
                    "error_type": e.__class__.__name__,
                },
            )
            raise
        finally:
            if use_cache and lock_key and lock_owner:
                await self._release_cache_lock(lock_key, lock_owner)

    async def generate_dsl(self, search_params: dict[str, Any], use_semantic: bool = True) -> dict:
        """
        Generate Elasticsearch DSL query.
        
        Strategy:
        1. Try Mixtral LLM-based DSL generation (10s timeout)
        2. If Mixtral fails or times out, fallback to rule-based DSL
        3. If semantic is enabled, ensure DSL includes a knn section with query embedding
        """
        search_params = dict(search_params or {})
        search_params["categories_fa"] = self._sanitize_categories(
            search_params.get("categories_fa", []) or []
        )
        semantic_query = search_params.get("product") or search_params.get("persian_full_query", "")

        async def _ensure_knn(dsl_obj: dict) -> dict:
            if not use_semantic or not isinstance(dsl_obj, dict):
                return dsl_obj
            if "knn" in dsl_obj:
                return dsl_obj
            if not semantic_query:
                return dsl_obj
            try:
                embedding = await self._get_embedding(semantic_query)
                if not embedding:
                    log_search("Semantic KNN skipped (embedding unavailable)", {"query": semantic_query[:80]})
                    return dsl_obj
                dsl_copy = dict(dsl_obj)
                dsl_copy["knn"] = {
                    "field": "product_embedding",
                    "query_vector": embedding,
                    "k": settings.search_size,
                    "num_candidates": max(settings.search_size * 4, 80),
                }
                log_search("Semantic KNN appended to DSL", {"query": semantic_query[:80], "k": settings.search_size})
                return dsl_copy
            except Exception as e:
                log_error("SEARCH", f"Failed to append semantic KNN: {e}", e)
                return dsl_obj

        # ── Try Mixtral first ──
        mixtral_start = perf_counter()
        mixtral_dsl = await self.mixtral_dsl.generate(search_params)
        mixtral_ms = int((perf_counter() - mixtral_start) * 1000)

        if mixtral_dsl is not None:
            log_search("Using Mixtral-generated DSL", {"took_ms": mixtral_ms})
            
            # Handle embedding placeholder injection
            if mixtral_dsl.pop("_needs_embedding", False):
                embedding_query = mixtral_dsl.pop("_embedding_query", "")
                if embedding_query:
                    try:
                        embedding = await self._get_embedding(embedding_query)
                        if embedding:
                            # Replace placeholder in knn section
                            dsl_str = json.dumps(mixtral_dsl, ensure_ascii=False)
                            dsl_str = dsl_str.replace('"{{EMBEDDING_PLACEHOLDER}}"', json.dumps(embedding))
                            mixtral_dsl = json.loads(dsl_str)
                    except Exception as e:
                        log_error("SEARCH", f"Embedding injection failed: {e}", e)
                        # Remove knn section if embedding failed
                        mixtral_dsl.pop("knn", None)
            
            # Clean up any remaining internal markers
            mixtral_dsl.pop("_embedding_query", None)
            guarded_dsl = self._prune_dsl_category_filters(
                mixtral_dsl,
                search_params.get("categories_fa", []),
            )
            return await _ensure_knn(guarded_dsl)

        log_search("Mixtral DSL failed/timed out, using fallback", {"mixtral_ms": mixtral_ms})

        # ── Fallback: rule-based DSL ──
        fallback_dsl = self._generate_fallback_dsl(search_params)
        guarded_fallback_dsl = self._prune_dsl_category_filters(
            fallback_dsl,
            search_params.get("categories_fa", []),
        )
        return await _ensure_knn(guarded_fallback_dsl)

    async def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Get embedding for text via the embedding MCP server."""
        try:
            resp = await self.http_client.post(
                f"{settings.embedding_url}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "generate_embedding",
                        "arguments": {"text": text, "normalize": True, "use_cache": True}
                    }
                },
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                timeout=5.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if "result" in data:
                    content = data["result"].get("content", [])
                    for item in content:
                        if item.get("type") == "text":
                            result = json.loads(item["text"])
                            return result.get("embedding")
        except Exception as e:
            log_error("SEARCH", f"Embedding fetch failed: {e}", e)
        return None

    def _generate_fallback_dsl(self, search_params: dict[str, Any]) -> dict:
        """Generate fallback rule-based Elasticsearch DSL query."""
        search_query = search_params.get("product") or search_params.get("persian_full_query", "")
        intent = search_params.get("intent", "browse")
        brand = search_params.get("brand")
        categories = self._sanitize_categories(search_params.get("categories_fa", []) or [])
        constraints = search_params.get("constraints", {})
        fallback_price_range = search_params.get("price_range", {})
        price_range = constraints.get("price_range") or fallback_price_range or {}

        # Build fallback DSL (simpler and more reliable)
        dsl = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": search_query,
                                "fields": ["product_name^3", "brand_name^2"],
                                "type": "best_fields",
                                "fuzziness": "AUTO",
                            }
                        }
                    ],
                    "filter": [],
                }
            },
            "size": settings.search_size,
            "sort": ["_score"],
        }

        # Add brand filter
        if brand:
            dsl["query"]["bool"]["filter"].append({"match": {"brand_name": brand}})

        # Add category filters (matched categories from interpret server)
        # Keep it inclusive: any matched category can pass.
        normalized_categories = [self._normalize_text(str(category)) for category in categories if str(category).strip()]
        if normalized_categories:
            category_should_filters = []
            for category in normalized_categories:
                category_should_filters.append({"match_phrase": {"category_name": category}})
                category_should_filters.append({"term": {"category_name.keyword": category}})

            dsl["query"]["bool"]["filter"].append({
                "bool": {
                    "should": category_should_filters,
                    "minimum_should_match": 1,
                }
            })

        # Add price filter
        if price_range.get("min") or price_range.get("max"):
            range_filter = {"range": {"price": {}}}
            if price_range.get("min"):
                range_filter["range"]["price"]["gte"] = price_range["min"]
            if price_range.get("max"):
                range_filter["range"]["price"]["lte"] = price_range["max"]
            dsl["query"]["bool"]["filter"].append(range_filter)

        # Adjust sort for intent
        if intent == "find_cheapest":
            dsl["sort"] = [{"price": "asc"}, "_score"]
        elif intent in ["find_high_quality", "find_best_value"]:
            dsl["sort"] = ["_score", {"price": "asc"}]

        return dsl

    def _calculate_relevancy_score(self, product_name: str, search_query: str) -> float:
        """
        Calculate how relevant a product name is to the search query.
        
        Core Logic: Products where the search query is the FIRST word(s) are
        more relevant than products where it appears later.
        
        Examples:
        - "یخچال فریزر جی‌پلاس" for query "یخچال" → HIGH (query is first word)
        - "اسپری پاک کننده یخچال" for query "یخچال" → LOW (query at end)
        - "قاب گوشی سامسونگ" for query "گوشی" → MEDIUM (query is 2nd word)
        
        Returns: 0.0 to 1.0 (higher = more relevant)
        """
        product_name_lower = product_name.lower().strip()
        search_query_lower = search_query.lower().strip()
        
        if not search_query_lower or not product_name_lower:
            return 0.5
        
        product_words = product_name_lower.split()
        query_words = search_query_lower.split()
        
        if not product_words or not query_words:
            return 0.5
        
        # BEST: Product name starts with the query (exact match at beginning)
        if product_name_lower.startswith(search_query_lower):
            return 1.0
        
        # Check if first word of query matches any of the first N words of product
        first_query_word = query_words[0]
        
        # Find position of first query word in product words
        try:
            word_position = product_words.index(first_query_word)
        except ValueError:
            # Word not found exactly, try substring match
            word_position = -1
            for i, pw in enumerate(product_words):
                if first_query_word in pw or pw in first_query_word:
                    word_position = i
                    break
        
        if word_position == -1:
            # Query word not found in product at all
            return 0.3
        
        # Calculate score based on word position
        # Position 0 (first word) → 1.0
        # Position 1 (second word) → 0.7
        # Position 2+ → decreasing
        total_words = len(product_words)
        if word_position == 0:
            return 1.0
        elif word_position == 1:
            return 0.65  # Second word - likely accessory like "قاب گوشی"
        else:
            # Later positions get progressively lower scores
            return max(0.3, 0.6 - (word_position * 0.1))

    async def rerank_results(
        self,
        results: list[dict],
        preferences: dict,
        intent: str = "browse",
        search_query: str = "",
    ) -> list[dict]:
        """
        Re-rank results based on relevancy and value formula.
        
        Key improvement: Filter out accessory products and prioritize
        products that are actually what the user asked for.
        """
        if not results:
            return []

        price_sensitivity = preferences.get("price_sensitivity", 0.5)
        quality_sensitivity = preferences.get("quality_sensitivity", 0.5)

        # Normalize scores
        scores = [r.get("_score") or 0 for r in results]
        max_score = max(scores) if scores else 1
        min_score = min(scores) if scores else 0
        score_range = max_score - min_score or 1

        # Normalize prices
        prices = [
            r.get("_source", {}).get("discount_price")
            or r.get("_source", {}).get("price", 0)
            for r in results
        ]
        max_price = max(prices) if prices else 1
        min_price = min(prices) if prices else 0
        price_range = max_price - min_price or 1

        ranked = []
        for r in results:
            source = r.get("_source", {})
            product_name = source.get("product_name", "")

            raw_score = r.get("_score") or 0
            es_score = (raw_score - min_score) / score_range

            brand = source.get("brand_name", "")
            brand_score = self._brand_scores.get(brand, 0.5)

            price = source.get("discount_price") or source.get("price", 0)
            price_norm = (price - min_price) / price_range

            discount = source.get("discount_percentage", 0) / 100
            
            # Calculate relevancy score (how well does product name match query)
            relevancy = self._calculate_relevancy_score(product_name, search_query) if search_query else 1.0

            # Updated value formula with relevancy
            value_score = (
                0.30 * es_score
                + 0.30 * relevancy  # NEW: Relevancy is important!
                + 0.15 * brand_score * quality_sensitivity
                + 0.15 * (1 - price_norm) * price_sensitivity
                + 0.10 * discount
            )

            ranked.append({
                "id": r.get("_id", ""),
                "product_name": product_name,
                "brand_name": source.get("brand_name"),
                "category_name": source.get("category_name"),
                "price": source.get("price", 0),
                "discount_price": source.get("discount_price"),
                "has_discount": source.get("has_discount", False),
                "discount_percentage": source.get("discount_percentage", 0),
                "image_url": source.get("image_url"),
                "product_url": source.get("product_url"),
                "score": r.get("_score") or 0,
                "relevancy_score": round(relevancy, 4),
                "value_score": round(value_score, 4),
            })

        # Sort based on intent
        if intent == "find_cheapest":
            # First sort by relevancy (high), then by price (low)
            ranked.sort(key=lambda x: (-x.get("relevancy_score", 0), x.get("discount_price") or x.get("price", 0)))
        elif intent == "find_high_quality":
            ranked.sort(
                key=lambda x: (
                    -x.get("relevancy_score", 0),
                    self._brand_scores.get(x.get("brand_name") or "", 0.5),
                    x.get("score", 0),
                ),
                reverse=True,
            )
        else:
            # Default: sort by value_score which now includes relevancy
            ranked.sort(key=lambda x: x.get("value_score", 0), reverse=True)

        return ranked

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def _execute_search(self, dsl: dict) -> list[dict]:
        """Execute search against Elasticsearch."""
        try:
            es_url = f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}/{settings.es_index}/_search"

            auth = None
            if settings.es_password:
                auth = (settings.es_user, settings.es_password)

            response = await self.http_client.post(es_url, json=dsl, auth=auth)

            if response.status_code != 200:
                log_error("SEARCH", f"Elasticsearch HTTP error: {response.status_code}")
                return []

            data = response.json()
            return data["hits"]["hits"]

        except Exception as e:
            log_error("SEARCH", f"Elasticsearch error: {e}", e)
            return []

    async def get_product_by_id(self, product_id: str) -> dict[str, Any]:
        """Get a single product by ID."""
        try:
            es_url = f"{settings.es_scheme}://{settings.es_host}:{settings.es_port}/{settings.es_index}/_doc/{product_id}"

            auth = None
            if settings.es_password:
                auth = (settings.es_user, settings.es_password)

            response = await self.http_client.get(es_url, auth=auth)

            if response.status_code != 200:
                return {"success": False, "error": f"Product not found: {product_id}"}

            data = response.json()
            source = data.get("_source", {})

            return {
                "success": True,
                "product": {
                    "id": product_id,
                    "product_name": source.get("product_name", ""),
                    "brand_name": source.get("brand_name", ""),
                    "price": source.get("price", 0),
                    "discount_price": source.get("discount_price"),
                    "has_discount": source.get("has_discount", False),
                    "discount_percentage": source.get("discount_percentage", 0),
                    "category_name": source.get("category_name", ""),
                    "product_url": source.get("product_url", ""),
                },
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for stable cache keys."""
        if not value:
            return ""
        normalized = value.strip().lower()
        normalized = normalized.replace("ي", "ی").replace("ك", "ک")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _sanitize_categories(self, categories: list[Any]) -> list[str]:
        """Keep only known/valid categories and canonicalize their labels."""
        cleaned: list[str] = []
        removed: list[str] = []
        seen: set[str] = set()

        for category in categories or []:
            raw = str(category or "").strip()
            if not raw:
                continue
            normalized = self._normalize_text(raw)
            canonical = self._known_categories_map.get(normalized)
            if not canonical:
                removed.append(raw)
                continue
            if canonical in seen:
                continue
            seen.add(canonical)
            cleaned.append(canonical)

        if removed:
            log_search("Category guard removed unknown categories", {"removed": removed[:10]})

        return cleaned

    @classmethod
    def _extract_category_values_from_filter(cls, node: Any) -> list[str]:
        """Extract category filter values from any nested DSL fragment."""
        found: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"category_name", "category_name.keyword"}:
                    if isinstance(value, str):
                        found.append(value)
                    elif isinstance(value, list):
                        found.extend([str(v) for v in value if str(v).strip()])
                else:
                    found.extend(cls._extract_category_values_from_filter(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(cls._extract_category_values_from_filter(item))
        return found

    def _prune_dsl_category_filters(self, dsl: dict, allowed_categories: list[str]) -> dict:
        """
        Remove category filters from DSL when they are not in allowed categories.
        This is a hard guard in code (not just prompt reliance).
        """
        if not isinstance(dsl, dict):
            return dsl

        bool_query = dsl.get("query", {}).get("bool", {})
        filters = bool_query.get("filter")
        if not isinstance(filters, list):
            return dsl

        allowed = {self._normalize_text(c) for c in (allowed_categories or []) if str(c).strip()}
        kept_filters = []
        removed_count = 0

        for f in filters:
            category_values = self._extract_category_values_from_filter(f)
            if not category_values:
                kept_filters.append(f)
                continue

            normalized_values = {
                self._normalize_text(v) for v in category_values if str(v).strip()
            }
            if allowed and normalized_values.intersection(allowed):
                kept_filters.append(f)
            else:
                removed_count += 1

        if removed_count:
            dsl = dict(dsl)
            dsl_query = dict(dsl.get("query", {}))
            dsl_bool = dict(dsl_query.get("bool", {}))
            dsl_bool["filter"] = kept_filters
            dsl_query["bool"] = dsl_bool
            dsl["query"] = dsl_query
            log_search(
                "Category guard pruned DSL filters",
                {"removed_filters": removed_count, "allowed_categories": allowed_categories},
            )

        return dsl

    def _build_cache_lookup_keys(
        self,
        search_params: dict[str, Any],
        use_semantic: bool,
    ) -> list[str]:
        """Build ordered cache lookup keys (exact first, semantic fallback second)."""
        base_payload = self._build_cache_payload(search_params)
        keys = [json.dumps({"v": "v2", "mode": "exact", **base_payload}, sort_keys=True, ensure_ascii=False)]

        if use_semantic:
            semantic_payload = {
                "query": base_payload["query"],
                "intent": base_payload["intent"],
                "mode_semantic": True,
            }
            keys.append(json.dumps({"v": "v2", "mode": "semantic_fallback", **semantic_payload}, sort_keys=True, ensure_ascii=False))

        return keys

    def _build_cache_payload(self, search_params: dict[str, Any]) -> dict[str, Any]:
        """Build deterministic payload for search cache keys."""
        query = self._normalize_text(search_params.get("product") or search_params.get("persian_full_query", ""))
        intent = search_params.get("intent", "browse")
        brand = self._normalize_text(search_params.get("brand", ""))
        categories = search_params.get("categories_fa", []) or []
        normalized_categories = sorted(
            [self._normalize_text(str(category)) for category in categories if str(category).strip()]
        )

        constraints = search_params.get("constraints", {}) or {}
        price_range = constraints.get("price_range") or search_params.get("price_range") or {}
        min_price = price_range.get("min")
        max_price = price_range.get("max")

        preferences = search_params.get("preferences", {}) or {}
        price_sensitivity = preferences.get("price_sensitivity")
        quality_sensitivity = preferences.get("quality_sensitivity")

        return {
            "query": query,
            "intent": intent,
            "brand": brand,
            "min_price": min_price,
            "max_price": max_price,
            "categories": normalized_categories,
            "price_sensitivity": price_sensitivity,
            "quality_sensitivity": quality_sensitivity,
        }

    def _make_cache_key(self, query: str) -> str:
        """Build Redis key for search cache."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"cache:v2:search:{query_hash}"

    def _make_negative_cache_key(self, query: str, intent: str) -> str:
        """Build Redis key for negative cache."""
        normalized = f"{self._normalize_text(query)}:{intent}"
        key_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"cache:v2:negative:{key_hash}"

    async def _check_negative_cache(self, query: str, intent: str) -> bool:
        """Check negative cache for known no-result queries."""
        if not self._redis:
            return False
        try:
            key = self._make_negative_cache_key(query, intent)
            exists = await self._redis.exists(key)
            return bool(exists)
        except Exception as e:
            log_error("SEARCH", f"Negative cache check error: {e}", e)
            return False

    async def _set_negative_cache(self, query: str, intent: str) -> bool:
        """Store known no-result query in negative cache."""
        if not self._redis:
            return False
        try:
            key = self._make_negative_cache_key(query, intent)
            await self._redis.setex(key, settings.cache_search_ttl, "1")
            return True
        except Exception as e:
            log_error("SEARCH", f"Negative cache set error: {e}", e)
            return False

    async def _acquire_cache_lock(self, query: str, owner: str) -> tuple[str, bool]:
        """Acquire a short lock to reduce cache stampede."""
        if not self._redis:
            return "", False
        lock_key = f"{self._make_cache_key(query)}:lock"
        try:
            acquired = await self._redis.set(lock_key, owner, ex=15, nx=True)
            return lock_key, bool(acquired)
        except Exception as e:
            log_error("SEARCH", f"Cache lock acquire error: {e}", e)
            return lock_key, False

    async def _release_cache_lock(self, lock_key: str, owner: str) -> None:
        """Release lock only if current owner matches."""
        if not self._redis or not lock_key:
            return
        try:
            current_owner = await self._redis.get(lock_key)
            if current_owner == owner:
                await self._redis.delete(lock_key)
        except Exception as e:
            log_error("SEARCH", f"Cache lock release error: {e}", e)

    async def _wait_for_warm_cache(self, query: str) -> Optional[list[dict]]:
        """Briefly wait for a concurrent request to warm cache."""
        for _ in range(3):
            await asyncio.sleep(0.15)
            cached = await self._check_cache(query)
            if cached is not None:
                return cached
        return None

    async def _check_cache(self, query: str) -> Optional[list[dict]]:
        """Check Redis cache for results."""
        if not self._redis:
            return None
        try:
            cache_key = self._make_cache_key(query)
            data = await self._redis.get(cache_key)
            if data:
                log_cache("Cache HIT", {"key": cache_key[:40], "query": query[:30]})
                return json.loads(data)
        except Exception as e:
            log_error("SEARCH", f"Cache check error: {e}", e)
        return None

    async def _update_cache(self, query: str, results: list[dict]) -> bool:
        """Store results in Redis cache."""
        if not self._redis:
            return False
        try:
            cache_key = self._make_cache_key(query)
            await self._redis.setex(
                cache_key,
                settings.cache_search_ttl,
                json.dumps(results, ensure_ascii=False),
            )
            log_cache("Cache SET", {"key": cache_key[:40], "query": query[:30], "ttl": settings.cache_search_ttl})
            return True
        except Exception as e:
            log_error("SEARCH", f"Cache update error: {e}", e)
            return False

    async def close(self):
        """Cleanup."""
        await self.http_client.aclose()
        if self._redis:
            await self._redis.close()
        await self.llm.close()
        # Mixtral DSL generator uses the same Redis, no separate cleanup needed


# ============================================================================
# MCP Server Setup
# ============================================================================


@dataclass
class AppContext:
    """Application context."""

    search_engine: SearchEngine


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle."""
    llm_client = LangChainLLM(
        api_key=settings.github_token,
        model=settings.github_model,
        base_url=settings.github_base_url,
    )
    
    # Initialize Mixtral DSL generator
    mixtral_dsl = MixtralDSLGenerator()
    await mixtral_dsl.init()
    
    search_engine = SearchEngine(llm_client, mixtral_dsl)
    await search_engine.load_brand_scores()
    await search_engine.init_redis()

    log_search(
        "Search server started",
        {
            "model": settings.github_model,
            "es": f"{settings.es_host}:{settings.es_port}",
        },
    )
    if settings.debug_mode:
        log_search("DEBUG mode active: caching disabled")
    else:
        log_cache(
            "Caching status",
            {"status": "enabled" if search_engine._redis else "disabled"},
        )

    try:
        yield AppContext(search_engine=search_engine)
    finally:
        await search_engine.close()
        log_search("Search server shutting down")


# Create MCP server
mcp = FastMCP(
    "Unified Search Server",
    lifespan=app_lifespan,
)

# Allow Docker internal hostnames for transport security
mcp.settings.transport_security.allowed_hosts.extend([
    "search:*",
    "host.docker.internal",
    "host.docker.internal:*",
    "localhost",
    "localhost:*",
    "127.0.0.1",
    "127.0.0.1:*",
    "0.0.0.0:*",
    "*",
])
mcp.settings.transport_security.allowed_origins.extend([
    "http://search:*",
    "http://backend:*",
    "http://host.docker.internal:*",
    "http://localhost:*",
    "http://127.0.0.1:*",
    "http://0.0.0.0:*",
])


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def search_products(
    search_params: dict[str, Any],
    session_id: str,
    use_cache: bool = True,
    use_semantic: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Execute full search pipeline.

    Args:
        search_params: Search parameters from interpretation
            - intent: browse/find_cheapest/find_best_value
            - product: Product name
            - brand: Brand filter
            - persian_full_query: Full query text
            - categories_fa: Category filters
            - constraints: Price range, etc.
        session_id: Session identifier
        use_cache: Whether to use cache
        use_semantic: Whether to use semantic search

    Returns:
        Dict with results, total_hits, took_ms, from_cache
    """
    engine = ctx.request_context.lifespan_context.search_engine
    incoming_trace_id = (search_params or {}).get("trace_id")
    trace_token = None

    if incoming_trace_id:
        trace_query = search_params.get("product") or search_params.get("persian_full_query", "")
        trace = TraceContext(query=trace_query, session_id=session_id)
        trace.trace_id = str(incoming_trace_id)
        trace_token = set_current_trace(trace)

    try:
        result = await engine.search(
            search_params=search_params,
            session_id=session_id,
            use_cache=use_cache,
            use_semantic=use_semantic,
        )
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if trace_token is not None:
            reset_current_trace(trace_token)


@mcp.tool()
async def generate_dsl(
    search_params: dict[str, Any],
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Generate Elasticsearch DSL only (no execution).

    Args:
        search_params: Search parameters

    Returns:
        Dict with DSL query
    """
    engine = ctx.request_context.lifespan_context.search_engine

    try:
        dsl = await engine.generate_dsl(search_params)
        return {"success": True, "dsl": dsl}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
async def get_product(
    product_id: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Get a single product by ID.

    Args:
        product_id: Product identifier

    Returns:
        Dict with product details
    """
    engine = ctx.request_context.lifespan_context.search_engine

    return await engine.get_product_by_id(product_id)


@mcp.tool()
async def rerank_results(
    results: list[dict],
    preferences: dict = None,
    intent: str = "browse",
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Re-rank existing results with value formula.

    Args:
        results: List of products to rerank
        preferences: User preferences (price_sensitivity, quality_sensitivity)
        intent: Search intent

    Returns:
        Dict with reranked results
    """
    engine = ctx.request_context.lifespan_context.search_engine

    # Convert to ES format
    es_results = [
        {"_id": r.get("id", ""), "_score": r.get("score", 1.0), "_source": r}
        for r in results
    ]

    try:
        ranked = await engine.rerank_results(es_results, preferences or {}, intent)
        return {"success": True, "results": ranked}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_search_info(ctx: Context = None) -> dict[str, Any]:
    """
    Get search server configuration.

    Returns:
        Dict with ES host, model, settings
    """
    return {
        "success": True,
        "es_host": f"{settings.es_host}:{settings.es_port}",
        "es_index": settings.es_index,
        "github_model": settings.github_model,
        "search_timeout": settings.search_timeout,
        "result_limit": settings.result_limit,
        "debug_mode": settings.debug_mode,
    }


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("search://config")
def search_config_resource() -> str:
    """Expose search configuration as a resource."""
    return json.dumps(
        {
            "es_host": settings.es_host,
            "es_port": settings.es_port,
            "es_index": settings.es_index,
            "search_timeout": settings.search_timeout,
            "result_limit": settings.result_limit,
            "debug_mode": settings.debug_mode,
        },
        indent=2,
    )


# ============================================================================
# Run Server
# ============================================================================


if __name__ == "__main__":
    transport = "streamable-http"

    if "--stdio" in sys.argv:
        transport = "stdio"
        log_search("Running with stdio transport")
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 5002
        log_search("Running with HTTP transport", {"url": "http://0.0.0.0:5002/mcp"})

    mcp.run(transport=transport)
