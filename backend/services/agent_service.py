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
from typing import Optional

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.agent import ShoppingAgent
from src.agent_cache import AgentResponseCache
from src.logging_config import get_logger
from src.pipeline_logger import log_latency_summary
from src.mcp_client import InterpretMCPClient, SearchMCPClient
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

            self._interpret_client = InterpretMCPClient(
                settings.mcp_interpret_url,
                timeout=settings.interpret_mcp_timeout,
            )
            self._search_client = SearchMCPClient(
                settings.mcp_search_url,
                timeout=settings.search_mcp_timeout,
            )

    @property
    def agent(self) -> ShoppingAgent:
        """Get the agent instance."""
        if not self._initialized or self._agent is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self._agent

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
        
        # ── Cache miss → full pipeline ─────────────────────────────
        try:
            stage_start = perf_counter()
            route_meta: dict[str, str] = {}
            if settings.deterministic_router_enabled:
                response, products, query_type, route_meta, det_timings = await asyncio.wait_for(
                    self._chat_deterministic(message, session_id),
                    timeout=timeout,
                )
                timings.update(det_timings)
                clean_response = response
                timings["extract_products_ms"] = 0
                timings["clean_response_ms"] = 0
                timings["detect_query_type_ms"] = 0
            else:
                response, session_id = await asyncio.wait_for(
                    self.agent.chat(message, session_id),
                    timeout=timeout,
                )
                # Extract products from response (if any)
                parse_start = perf_counter()
                products = self._extract_products(response)
                timings["extract_products_ms"] = int((perf_counter() - parse_start) * 1000)
                
                # Clean response text (remove product details if extracted)
                clean_start = perf_counter()
                clean_response = self._clean_response_text(response, products)
                timings["clean_response_ms"] = int((perf_counter() - clean_start) * 1000)
                
                # Determine query type
                detect_start = perf_counter()
                query_type = self._detect_query_type(response, products)
                timings["detect_query_type_ms"] = int((perf_counter() - detect_start) * 1000)
            timings["agent_chat_ms"] = int((perf_counter() - stage_start) * 1000)
            took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
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
            
            # ── Store in agent cache (only direct product searches) ──────
            # Skip caching for: suggestions, clarifications, greetings, errors
            is_cacheable = (
                products
                and query_type == "direct"
                and not any(
                    kw in clean_response
                    for kw in ["پیشنهاد", "انتخاب کنید", "کدوم", "کدام", "منظورتون"]
                )
            )
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
                    **route_meta,
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
            log_latency_summary(
                "AGENT",
                "agent_service.chat",
                int((perf_counter() - request_start) * 1000),
                breakdown_ms=timings,
                meta={"cache": "miss", "query_type": "error", "success": False},
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
                    "latency_breakdown_ms": timings,
                },
                "error": str(e),
            }

    async def _chat_deterministic(
        self,
        message: str,
        session_id: str,
    ) -> tuple[str, list[dict], str, dict[str, str], dict[str, int]]:
        """
        Deterministic orchestration:
        - Agent routes: direct/unclear/abstract/follow_up/chat
        - direct + searchable => always call search (no tool-call skip risk)
        - abstract/follow_up/chat => agent conversational mode (no tools)
        """
        timings: dict[str, int] = {}
        meta: dict[str, str] = {"orchestrator": "deterministic_v1"}

        route_start = perf_counter()
        route_info = await self.agent.classify_route(message)
        timings["router_classify_ms"] = int((perf_counter() - route_start) * 1000)
        route = str(route_info.get("route", "unclear")).strip().lower()
        if route not in {"direct", "abstract", "follow_up", "chat", "unclear"}:
            route = "unclear"
        meta["route"] = route

        if route in {"chat", "abstract", "follow_up"}:
            conv_start = perf_counter()
            response = await self.agent.chat_without_tools(
                message,
                route_hint=route,
            )
            timings["router_conversation_ms"] = int((perf_counter() - conv_start) * 1000)
            return response, [], route, meta, timings

        # Route says direct/unclear -> use interpret as retrieval gate (direct/unclear only).
        if self._interpret_client is None or self._search_client is None:
            raise RuntimeError("Deterministic clients are not initialized")

        interpret_start = perf_counter()
        interpret_result = await self._interpret_client.interpret_query(
            query=message,
            session_id=session_id,
            context={"direct_unclear_only": True},
        )
        timings["router_interpret_ms"] = int((perf_counter() - interpret_start) * 1000)

        interpret_type = str(interpret_result.get("query_type", "unclear")).strip().lower()
        searchable = bool(interpret_result.get("searchable"))
        meta["interpret_type"] = interpret_type

        if interpret_type == "direct" and searchable and settings.deterministic_force_search:
            search_params = self._build_search_params_from_interpret(
                interpret_result.get("search_params", {}),
                original_message=message,
            )

            search_start = perf_counter()
            search_result = await self._search_client.search_products(
                search_params=search_params,
                session_id=session_id,
                use_cache=True,
            )
            timings["router_search_ms"] = int((perf_counter() - search_start) * 1000)

            products = self._products_from_search_result(search_result)
            response = self._build_search_response_text(
                products=products,
                search_result=search_result,
            )
            meta["search_executed"] = "true"
            return response, products, "direct", meta, timings

        # Fallback path: unclear or non-direct result from interpret.
        fallback_text = self._build_clarification_from_interpret(interpret_result)
        if not fallback_text:
            conv_start = perf_counter()
            fallback_text = await self.agent.chat_without_tools(
                message,
                route_hint="unclear",
                extra_context=f"interpret_type={interpret_type}",
            )
            timings["router_conversation_ms"] = int((perf_counter() - conv_start) * 1000)

        meta["search_executed"] = "false"
        return fallback_text, [], "unclear", meta, timings

    @staticmethod
    def _as_positive_int(value) -> Optional[int]:
        try:
            if value is None:
                return None
            iv = int(float(value))
            return iv if iv > 0 else None
        except Exception:
            return None

    def _build_search_params_from_interpret(
        self,
        search_params: dict,
        original_message: str,
    ) -> dict:
        product = str(search_params.get("product") or "").strip() or original_message.strip()
        intent = str(search_params.get("intent") or "browse").strip().lower()
        if intent not in {"browse", "find_cheapest", "find_best", "compare"}:
            intent = "browse"

        payload = {
            "product": product,
            "intent": intent,
            "persian_full_query": str(search_params.get("persian_full_query") or original_message).strip(),
        }

        brand = str(search_params.get("brand") or "").strip()
        if brand:
            payload["brand"] = brand

        categories = search_params.get("categories_fa")
        if isinstance(categories, list):
            cleaned = [str(c).strip() for c in categories if str(c).strip()]
            if cleaned:
                payload["categories_fa"] = cleaned[:5]

        price_range = search_params.get("price_range") if isinstance(search_params.get("price_range"), dict) else {}
        min_price = self._as_positive_int(price_range.get("min"))
        max_price = self._as_positive_int(price_range.get("max"))
        if min_price is not None or max_price is not None:
            payload["price_range"] = {}
            if min_price is not None:
                payload["price_range"]["min"] = min_price
            if max_price is not None:
                payload["price_range"]["max"] = max_price

        attributes = search_params.get("attributes")
        if isinstance(attributes, dict) and attributes:
            payload["attributes"] = attributes

        return payload

    def _products_from_search_result(self, search_result: dict) -> list[dict]:
        rows = search_result.get("results", []) if isinstance(search_result, dict) else []
        if not isinstance(rows, list):
            rows = []

        products: list[dict] = []
        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            normalized = self._normalize_product(
                {
                    "name": item.get("product_name", item.get("name", "")),
                    "brand": item.get("brand_name", item.get("brand", "")),
                    "price": item.get("price", 0),
                    "discount_price": item.get("discount_price"),
                    "has_discount": item.get("has_discount", False),
                    "discount_percentage": item.get("discount_percentage", 0),
                    "product_url": item.get("product_url", item.get("url", "")),
                },
                i,
            )
            if normalized:
                normalized["image_url"] = item.get("image_url")
                products.append(normalized)
        return products

    @staticmethod
    def _build_search_response_text(products: list[dict], search_result: dict) -> str:
        if not products:
            return "متأسفانه محصول مورد نظر شما یافت نشد. می‌تونید با کلمات دقیق‌تر دوباره جستجو کنید."
        total_hits = int(search_result.get("total_hits", len(products))) if isinstance(search_result, dict) else len(products)
        if total_hits > len(products):
            return "این محصولات رو پیدا کردم"
        return "این محصولات رو پیدا کردم"

    @staticmethod
    def _build_clarification_from_interpret(interpret_result: dict) -> str:
        clarification = interpret_result.get("clarification")
        if not isinstance(clarification, dict):
            return ""
        question = str(clarification.get("question") or "").strip()
        suggestions = clarification.get("suggestions")
        if not isinstance(suggestions, list) or not suggestions:
            return question

        suggestion_lines: list[str] = []
        for idx, item in enumerate(suggestions[:4], 1):
            if isinstance(item, dict):
                title = str(item.get("product") or item.get("search_query") or "").strip()
            else:
                title = str(item).strip()
            if title:
                suggestion_lines.append(f"{idx}. {title}")

        if suggestion_lines and question:
            return f"{question}\n" + "\n".join(suggestion_lines)
        if suggestion_lines:
            return "برای اینکه بهتر کمک کنم یکی از گزینه‌های زیر رو انتخاب کنید:\n" + "\n".join(suggestion_lines)
        return question

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
        """Detect the type of query based on response."""
        # Check for greeting indicators
        greeting_indicators = ["سلام", "👋", "کمک", "خدمت", "😊", "😄"]
        if any(ind in response for ind in greeting_indicators) and not products:
            return "chat"
        
        # Check for product search indicators
        product_indicators = ["📦", "💰", "تومان", "قیمت", "برند"]
        if any(ind in response for ind in product_indicators):
            return "direct"
        
        # Check for no results
        no_result_indicators = ["یافت نشد", "پیدا نشد", "متأسفانه"]
        if any(ind in response for ind in no_result_indicators):
            return "no_results"
        
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
