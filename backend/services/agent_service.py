"""
Agent Service

Wrapper around the ShoppingAgent for use in FastAPI.
Handles async operations and extracts structured data from responses.
Includes agent response caching (Level 2 cache) to skip LLM on repeat queries.
"""

import asyncio
import json
import re
import uuid
from time import perf_counter
from datetime import datetime
from typing import Any, Optional

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent import ShoppingAgent
from src.agent_cache import AgentResponseCache
from src.mcp_client import InterpretMCPClient, SearchMCPClient
from src.logging_config import get_logger
from src.pipeline_logger import log_latency_summary
from backend.api.schemas import ProductInfo, ChatMetadata
from backend.core.config import settings

logger = get_logger(__name__)


class AgentService:
    """
    Service wrapper for ShoppingAgent.
    
    Provides a clean interface for the API to interact with the agent,
    handling initialization, session management, and response parsing.
    """

    def __init__(self):
        self._agent: Optional[ShoppingAgent] = None
        self._initialized = False
        self._cache: Optional[AgentResponseCache] = None
        self._interpret_client: Optional[InterpretMCPClient] = None
        self._search_client: Optional[SearchMCPClient] = None

    async def initialize(self) -> None:
        """Initialize the agent and response cache (lazy loading)."""
        if not self._initialized:
            self._agent = ShoppingAgent()
            self._interpret_client = InterpretMCPClient(settings.mcp_interpret_url, timeout=30.0)
            self._search_client = SearchMCPClient(settings.mcp_search_url, timeout=60.0)
            self._initialized = True

            # Initialize agent response cache
            if settings.agent_cache_enabled:
                self._cache = AgentResponseCache(
                    redis_host=settings.redis_host,
                    redis_port=settings.redis_port,
                    redis_password=settings.redis_password,
                    redis_db=settings.redis_db,
                    default_ttl=settings.agent_cache_ttl,
                )
                await self._cache.connect()
            else:
                logger.info("Agent response cache is disabled")

    @property
    def agent(self) -> ShoppingAgent:
        """Get the agent instance."""
        if not self._initialized or self._agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._agent

    @staticmethod
    def _build_direct_search_params(interpret_result: dict[str, Any], fallback_query: str) -> dict[str, Any]:
        """Build normalized search params from interpret output."""
        search_params = interpret_result.get("search_params", {}) if isinstance(interpret_result, dict) else {}

        product = search_params.get("product") or fallback_query
        intent = search_params.get("intent", "browse")
        brand = search_params.get("brand")
        categories_fa = search_params.get("categories_fa", []) or []
        price_range = search_params.get("price_range", {}) or {}

        final_search_params: dict[str, Any] = {
            "product": product,
            "intent": intent,
            "persian_full_query": search_params.get("persian_full_query") or product,
        }

        if brand and str(brand).strip():
            final_search_params["brand"] = brand

        filtered_categories = [c for c in categories_fa if c and str(c).strip()]
        if filtered_categories:
            final_search_params["categories_fa"] = filtered_categories

        normalized_price_range: dict[str, int] = {}
        if isinstance(price_range, dict):
            if price_range.get("min") and price_range["min"] > 0:
                normalized_price_range["min"] = price_range["min"]
            if price_range.get("max") and price_range["max"] > 0:
                normalized_price_range["max"] = price_range["max"]
        if normalized_price_range:
            final_search_params["price_range"] = normalized_price_range

        return final_search_params

    @staticmethod
    def _format_direct_products(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Map search server results to API ProductInfo fields."""
        products: list[dict[str, Any]] = []
        for item in results or []:
            products.append(
                {
                    "id": str(item.get("id", "")),
                    "name": str(item.get("product_name", "")),
                    "brand": item.get("brand_name"),
                    "price": float(item.get("price", 0) or 0),
                    "discount_price": item.get("discount_price"),
                    "has_discount": bool(item.get("has_discount", False)),
                    "discount_percentage": float(item.get("discount_percentage", 0) or 0),
                    "product_url": item.get("product_url"),
                }
            )
        return products

    @staticmethod
    def _build_no_results_response(search_params: dict[str, Any]) -> str:
        """Build user-facing alternatives when search returns no products."""
        suggestions: list[str] = []
        if search_params.get("brand"):
            suggestions.append("همین محصول بدون محدودیت برند")
        price_range = search_params.get("price_range", {}) or {}
        if price_range.get("max"):
            suggestions.append("افزایش سقف بودجه")
        if price_range.get("min"):
            suggestions.append("کاهش حداقل بودجه")
        suggestions.extend(
            [
                "عبارت کوتاه‌تر از همین محصول",
                "یک دسته نزدیک به همین محصول",
            ]
        )
        suggestion_lines = "\n".join([f"🛒 {s}" for s in suggestions[:5]])
        return (
            "متأسفانه با این شرط‌ها نتیجه‌ای پیدا نشد. "
            "برای ادامه یکی از گزینه‌های زیر رو انتخاب کن:\n"
            f"{suggestion_lines}"
        )

    async def _try_direct_delivery(
        self,
        message: str,
        session_id: str,
        timings: dict[str, int],
    ) -> Optional[dict[str, Any]]:
        """
        Direct pipeline for direct queries: interpret -> search -> return.
        Returns None when query is not direct or if direct pipeline fails.
        """
        if not settings.direct_delivery_bypass_agent:
            return None
        if self._interpret_client is None or self._search_client is None:
            return None

        try:
            stage_start = perf_counter()
            interpret_result = await self._interpret_client.interpret_query(
                query=message,
                session_id=session_id,
                context={},
            )
            timings["direct_interpret_ms"] = int((perf_counter() - stage_start) * 1000)
        except Exception as e:
            logger.warning(f"Direct delivery interpret failed, fallback to agent: {e}")
            return None

        if not isinstance(interpret_result, dict):
            return None
        if not interpret_result.get("success", True):
            return None
        if interpret_result.get("query_type") != "direct" or not interpret_result.get("searchable"):
            return None

        search_params = self._build_direct_search_params(interpret_result, message)

        try:
            stage_start = perf_counter()
            search_result = await self._search_client.search_products(
                search_params=search_params,
                session_id=session_id,
                use_cache=True,
                use_semantic=True,
            )
            timings["direct_search_ms"] = int((perf_counter() - stage_start) * 1000)
        except Exception as e:
            logger.warning(f"Direct delivery search failed, fallback to agent: {e}")
            return None

        if not isinstance(search_result, dict) or not search_result.get("success", False):
            return None

        products = self._format_direct_products(search_result.get("results", []))
        total_hits = int(search_result.get("total_hits", len(products)) or len(products))

        if not products:
            return {
                "success": True,
                "response": self._build_no_results_response(search_params),
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "took_ms": int(search_result.get("took_ms", 0) or 0),
                    "query_type": "no_results",
                    "total_results": 0,
                    "from_agent_cache": False,
                    "latency_breakdown_ms": timings,
                },
            }

        return {
            "success": True,
            "response": "این محصولات رو پیدا کردم",
            "session_id": session_id,
            "products": products,
            "metadata": {
                "took_ms": int(search_result.get("took_ms", 0) or 0),
                "query_type": "direct",
                "total_results": total_hits,
                "from_agent_cache": False,
                "latency_breakdown_ms": timings,
            },
        }

    async def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        timeout: int = 120,
    ) -> dict:
        """
        Process a chat message and return structured response.
        
        Two-level caching:
          Level 1 (search cache): handled by search_server, saves ES lookup time
          Level 2 (agent cache):  handled here, skips LLM entirely for repeat queries
        
        Args:
            message: User message in Persian
            session_id: Optional session ID for continuity
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with response, products, and metadata
        """
        request_start = perf_counter()
        timings: dict[str, int] = {}

        stage_start = perf_counter()
        await self.initialize()
        timings["initialize_ms"] = int((perf_counter() - stage_start) * 1000)
        
        start_time = datetime.now()
        
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # ── Level 2: Agent Response Cache check ────────────────────
        if self._cache and self._cache.available:
            cache_lookup_start = perf_counter()
            cached = await self._cache.get(message)
            timings["agent_cache_lookup_ms"] = int((perf_counter() - cache_lookup_start) * 1000)
            if cached is not None:
                took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                # Overwrite timing with actual cache-hit time
                if "metadata" in cached:
                    cached["metadata"]["took_ms"] = took_ms
                    cached["metadata"]["from_agent_cache"] = True
                    cached["metadata"]["latency_breakdown_ms"] = timings
                cached["session_id"] = session_id
                log_latency_summary(
                    "AGENT",
                    "agent_service.chat",
                    int((perf_counter() - request_start) * 1000),
                    breakdown_ms=timings,
                    meta={"cache": "hit", "query_type": cached.get("metadata", {}).get("query_type")},
                )
                logger.info(f"Agent cache HIT ({took_ms}ms) for: {message[:50]}")
                return cached
        else:
            timings["agent_cache_lookup_ms"] = 0

        # ── Direct delivery bypass (interpret -> search -> return) ───────────
        if settings.direct_delivery_bypass_agent:
            stage_start = perf_counter()
            direct_result = await self._try_direct_delivery(message, session_id, timings)
            timings["direct_bypass_ms"] = int((perf_counter() - stage_start) * 1000)
            if direct_result is not None:
                took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                direct_result.setdefault("metadata", {})
                direct_result["metadata"]["took_ms"] = took_ms
                direct_result["metadata"]["from_agent_cache"] = False
                direct_result["metadata"]["latency_breakdown_ms"] = timings

                direct_query_type = direct_result.get("metadata", {}).get("query_type") or "direct"
                products = direct_result.get("products", []) or []
                if self._cache and self._cache.available and direct_query_type == "direct" and products:
                    cache_set_start = perf_counter()
                    await self._cache.set(message, direct_result)
                    timings["agent_cache_set_ms"] = int((perf_counter() - cache_set_start) * 1000)
                else:
                    timings["agent_cache_set_ms"] = 0

                log_latency_summary(
                    "AGENT",
                    "agent_service.chat",
                    int((perf_counter() - request_start) * 1000),
                    breakdown_ms=timings,
                    meta={
                        "cache": "direct_bypass",
                        "query_type": direct_query_type,
                        "success": True,
                        "products": len(products),
                    },
                )
                return direct_result
        
        # ── Cache miss → full pipeline ─────────────────────────────
        try:
            # Call agent with timeout
            stage_start = perf_counter()
            response, session_id = await asyncio.wait_for(
                self.agent.chat(message, session_id),
                timeout=timeout,
            )
            timings["agent_chat_ms"] = int((perf_counter() - stage_start) * 1000)

            if isinstance(response, str) and response.startswith("__AGENT_ERROR__:"):
                error_text = response.replace("__AGENT_ERROR__:", "", 1).strip()
                error_stage = "agent.chat"
                error_type = "AgentError"
                error_message = error_text
                try:
                    parsed_error = json.loads(error_text)
                    if isinstance(parsed_error, dict):
                        error_stage = str(parsed_error.get("stage") or error_stage)
                        error_type = str(parsed_error.get("error_type") or error_type)
                        error_message = str(parsed_error.get("message") or error_text)
                except Exception:
                    pass
                took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                log_latency_summary(
                    "AGENT",
                    "agent_service.chat",
                    int((perf_counter() - request_start) * 1000),
                    breakdown_ms=timings,
                    meta={
                        "cache": "miss",
                        "query_type": "error",
                        "success": False,
                        "error_stage": error_stage,
                        "error_type": error_type,
                    },
                )
                return {
                    "success": False,
                    "response": "متأسفانه مشکلی پیش اومد. لطفاً دوباره تلاش کنید.",
                    "session_id": session_id,
                    "products": [],
                    "metadata": {
                        "took_ms": took_ms,
                        "query_type": "error",
                        "total_results": 0,
                        "from_agent_cache": False,
                        "error_stage": error_stage,
                        "error_type": error_type,
                        "latency_breakdown_ms": timings,
                    },
                    "error": error_message,
                }
            
            took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Extract products from response (if any)
            stage_start = perf_counter()
            products = self._extract_products(response)
            timings["extract_products_ms"] = int((perf_counter() - stage_start) * 1000)
            
            # Clean response text (remove product details if extracted)
            stage_start = perf_counter()
            clean_response = self._clean_response_text(response, products)
            timings["clean_response_ms"] = int((perf_counter() - stage_start) * 1000)
            
            # Determine query type
            stage_start = perf_counter()
            query_type = self._detect_query_type(response, products)
            timings["detect_query_type_ms"] = int((perf_counter() - stage_start) * 1000)
            
            result = {
                "success": True,
                "response": clean_response,
                "session_id": session_id,
                "products": products,
                "metadata": {
                    "took_ms": took_ms,
                    "query_type": query_type,
                    "total_results": len(products) if products else None,
                    "from_agent_cache": False,
                    "latency_breakdown_ms": timings,
                },
            }
            
            # ── Store in agent cache (only real product-result responses) ──────
            # Keep this rule simple and data-driven (no keyword heuristics).
            is_cacheable = bool(products) and query_type == "direct"
            if self._cache and self._cache.available and is_cacheable:
                stage_start = perf_counter()
                await self._cache.set(message, result)
                timings["agent_cache_set_ms"] = int((perf_counter() - stage_start) * 1000)
                logger.info(f"Agent cache SET ({len(products)} products) for: {message[:50]}")
            elif self._cache and products and not is_cacheable:
                logger.info(f"Agent cache SKIP (query_type={query_type}) for: {message[:50]}")
                timings["agent_cache_set_ms"] = 0
            else:
                timings["agent_cache_set_ms"] = 0

            result["metadata"]["latency_breakdown_ms"] = timings
            log_latency_summary(
                "AGENT",
                "agent_service.chat",
                int((perf_counter() - request_start) * 1000),
                breakdown_ms=timings,
                meta={
                    "cache": "miss",
                    "query_type": query_type,
                    "success": True,
                    "products": len(products),
                },
            )
            
            return result
            
        except asyncio.TimeoutError:
            took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            log_latency_summary(
                "AGENT",
                "agent_service.chat",
                int((perf_counter() - request_start) * 1000),
                breakdown_ms=timings,
                meta={"cache": "miss", "query_type": "timeout", "success": False},
            )
            return {
                "success": False,
                "response": "متأسفانه پاسخگویی بیش از حد طول کشید. لطفاً دوباره تلاش کنید.",
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "took_ms": took_ms,
                    "query_type": "timeout",
                    "total_results": 0,
                    "from_agent_cache": False,
                    "latency_breakdown_ms": timings,
                },
            }
        except Exception as e:
            took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            error_stage = "agent_service.chat"
            error_type = e.__class__.__name__
            log_latency_summary(
                "AGENT",
                "agent_service.chat",
                int((perf_counter() - request_start) * 1000),
                breakdown_ms=timings,
                meta={
                    "cache": "miss",
                    "query_type": "error",
                    "success": False,
                    "error_stage": error_stage,
                    "error_type": error_type,
                },
            )
            return {
                "success": False,
                "response": f"متأسفانه مشکلی پیش اومد. لطفاً دوباره تلاش کنید.",
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "took_ms": took_ms,
                    "query_type": "error",
                    "total_results": 0,
                    "from_agent_cache": False,
                    "error_stage": error_stage,
                    "error_type": error_type,
                    "latency_breakdown_ms": timings,
                },
                "error": str(e),
            }

    def _normalize_json_text(self, text: str) -> str:
        """
        Normalize Persian/Arabic digits and number formatting in JSON text
        so json.loads can parse it.
        """
        # Map Persian/Arabic digits to ASCII
        digit_map = str.maketrans('۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩', '01234567890123456789')
        text = text.translate(digit_map)

        # Remove commas inside numbers (e.g. 5,999,000 → 5999000)
        # Match digits separated by commas that are NOT inside quotes
        text = re.sub(r'(?<=\d),(?=\d)', '', text)
        text = text.replace('،', ',')

        return text

    def _sanitize_json_like(self, text: str) -> str:
        """Repair common LLM JSON formatting issues."""
        text = self._normalize_json_text(text or "").strip()
        text = text.replace('\ufeff', '')
        text = re.sub(r'^\s*json\s*\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r',\s*([}\]])', r'\1', text)
        text = re.sub(r'\bTrue\b', 'true', text)
        text = re.sub(r'\bFalse\b', 'false', text)
        text = re.sub(r'\bNone\b', 'null', text)
        text = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
        text = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1"\2":', text)
        text = re.sub(r':\s*\'([^\']*)\'', r': "\1"', text)
        text = re.sub(r':\s*(-?\d+(?:\.\d+)?)"', r': \1', text)
        text = re.sub(r':\s*(true|false|null)"', r': \1', text, flags=re.IGNORECASE)
        return text

    def _parse_json_candidate(self, candidate: str):
        candidate = (candidate or "").strip()
        if not candidate:
            return None

        direct = re.sub(r'^\s*json\s*\n', '', candidate, flags=re.IGNORECASE)
        for attempt in (direct, self._sanitize_json_like(direct)):
            try:
                return json.loads(attempt)
            except Exception:
                continue
        return None

    @staticmethod
    def _to_number(value):
        if isinstance(value, (int, float)):
            return float(value)
        if not isinstance(value, str):
            return None
        normalized = value.replace(",", "").strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except Exception:
            return None

    def _normalize_product(self, item: dict, index: int) -> Optional[dict]:
        if not isinstance(item, dict):
            return None
        name = str(item.get("name") or item.get("product_name") or "").strip()
        if not name:
            return None

        price = self._to_number(item.get("price"))
        discount_price = self._to_number(item.get("discount_price"))
        discount_percentage = self._to_number(item.get("discount_percentage"))
        has_discount_raw = item.get("has_discount", False)
        has_discount = (
            has_discount_raw is True
            or str(has_discount_raw).strip().lower() == "true"
            or (
                price is not None
                and discount_price is not None
                and discount_price < price
            )
        )

        return {
            "id": str(index + 1),
            "name": name,
            "brand": str(item.get("brand") or item.get("brand_name") or "").strip(),
            "price": float(price) if price is not None else 0,
            "has_discount": bool(has_discount),
            "discount_percentage": float(discount_percentage) if discount_percentage is not None else 0,
            "discount_price": float(discount_price) if discount_price is not None else None,
            "product_url": str(item.get("product_url") or item.get("url") or "").strip(),
        }

    def _extract_products_from_fields(self, text: str) -> list[dict]:
        """Fallback parser for broken JSON snippets containing product fields."""
        text = self._sanitize_json_like(text or "")
        candidates = re.findall(r'\{[\s\S]*?\}', text) or [text]
        rows = []

        def read_field(block: str, key: str) -> str:
            pattern = re.compile(
                rf'["\']?{key}["\']?\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|([^,\n\r}}]+))',
                flags=re.IGNORECASE,
            )
            m = pattern.search(block)
            if not m:
                return ""
            return (m.group(1) or m.group(2) or m.group(3) or "").strip()

        for block in candidates:
            row = {
                "name": read_field(block, "name") or read_field(block, "product_name"),
                "brand": read_field(block, "brand") or read_field(block, "brand_name"),
                "price": read_field(block, "price"),
                "discount_price": read_field(block, "discount_price"),
                "has_discount": read_field(block, "has_discount"),
                "discount_percentage": read_field(block, "discount_percentage"),
                "product_url": read_field(block, "product_url") or read_field(block, "url"),
            }
            if row["name"]:
                rows.append(row)

        products = []
        for i, row in enumerate(rows):
            normalized = self._normalize_product(row, i)
            if normalized:
                products.append(normalized)
        return products

    def _extract_products(self, response: str) -> list[dict]:
        """
        Extract product information from agent response.
        
        The agent returns products as a JSON array inside a ```json code block.
        """
        products = []

        candidates = []
        code_blocks = re.findall(
            r'(?:```[\t ]*(?:json)?|json```)[\t ]*\n?([\s\S]*?)```',
            response or "",
            flags=re.IGNORECASE,
        )
        candidates.extend(code_blocks)

        if not candidates:
            array_match = re.search(r'\[[\s\S]*\]', response or "", re.DOTALL)
            if array_match:
                candidates.append(array_match.group(0))
            object_match = re.search(r'\{[\s\S]*\}', response or "", re.DOTALL)
            if object_match:
                candidates.append(object_match.group(0))

        for candidate in candidates:
            parsed = self._parse_json_candidate(candidate)
            if parsed is not None:
                payload = parsed.get("results") if isinstance(parsed, dict) else parsed
                if isinstance(parsed, dict) and payload is None:
                    payload = parsed.get("products") or parsed.get("data") or parsed
                rows = payload if isinstance(payload, list) else [payload]
                for row in rows:
                    normalized = self._normalize_product(row, len(products))
                    if normalized:
                        products.append(normalized)

            if not products:
                fallback_rows = self._extract_products_from_fields(candidate)
                if fallback_rows:
                    products.extend(fallback_rows)

        if not products:
            products = self._extract_products_from_fields(response or "")

        deduped = []
        seen = set()
        for p in products:
            key = f'{p.get("name","")}|{p.get("price","")}|{p.get("product_url","")}'
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)

        return deduped
    
    def _clean_response_text(self, response: str, products: list) -> str:
        """
        Remove JSON product block from response, keep only the intro text.
        """
        if not products:
            return response

        clean = response or ""
        clean = re.sub(r'```[\t ]*(?:json)?[\s\S]*?```', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'json```[\s\S]*?```', '', clean, flags=re.IGNORECASE).strip()
        clean = re.sub(r'\[\s*\{[\s\S]*?\}\s*(?:,\s*\{[\s\S]*?\}\s*)*\]', '', clean, flags=re.DOTALL).strip()
        clean = re.sub(r'\{\s*\"?name\"?[\s\S]*?\}', '', clean, flags=re.DOTALL).strip()

        # Remove trailing colons, dashes, etc.
        clean = re.sub(r'[:\s\-]+$', '', clean).strip()

        if not clean:
            count = len(products)
            return f"🛍️ {count} محصول پیدا شد"

        return clean
    
    def _detect_query_type(self, response: str, products: list) -> str:
        """Detect the type of query based on structured output, not keywords."""
        if products:
            return "direct"

        text = (response or "").strip()
        if text.startswith("❓ NEED_CLARIFICATION:"):
            return "unclear"
        if text.startswith("❌ NO_RESULTS:"):
            return "no_results"
        if text.startswith("__AGENT_ERROR__:"):
            return "error"
        if text:
            return "chat"
        return "unknown"

    async def health_check(self) -> dict:
        """Check if agent is healthy."""
        try:
            await self.initialize()
            health = {"status": "ok", "latency_ms": 0}

            # Include agent cache stats
            if self._cache and self._cache.available:
                cache_stats = await self._cache.get_stats()
                health["agent_cache"] = cache_stats
            else:
                health["agent_cache"] = {"available": False}

            return health
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Global service instance
_agent_service: Optional[AgentService] = None


def get_agent_service() -> AgentService:
    """Get or create the agent service instance."""
    global _agent_service
    if _agent_service is None:
        _agent_service = AgentService()
    return _agent_service
