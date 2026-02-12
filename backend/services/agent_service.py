"""
Agent Service

Wrapper around the ShoppingAgent for use in FastAPI.
Handles async operations and extracts structured data from responses.
Includes agent response caching (Level 2 cache) to skip LLM on repeat queries.
"""

import asyncio
import hashlib
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

    async def initialize(self) -> None:
        """Initialize the agent and response cache (lazy loading)."""
        if not self._initialized:
            self._agent = ShoppingAgent()
            self._initialized = True

            if settings.ff_interpret_warmup:
                try:
                    await self._agent.warmup_mcp_sessions()
                except Exception as e:
                    logger.warning(f"MCP warmup skipped due to error: {e}")

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
    def _normalize_text(value: str) -> str:
        """Normalize Persian text for stable matching."""
        if not value:
            return ""
        normalized = value.strip().lower()
        normalized = normalized.replace("ي", "ی").replace("ك", "ک").replace("‌", " ")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _is_smalltalk_message(self, message: str) -> bool:
        """Detect pure social/greeting messages that should not hit search."""
        normalized = self._normalize_text(message)
        if not normalized:
            return False

        social_markers = {
            "سلام",
            "سلامم",
            "سلام خوبی",
            "خوبی",
            "چطوری",
            "روز بخیر",
            "شب بخیر",
            "صبح بخیر",
            "ممنون",
            "مرسی",
            "تشکر",
            "ممنونم",
            "خداحافظ",
            "فعلا",
            "خسته نباشید",
            "خودتو معرفی کن",
            "خودتو معرفی میکنی",
            "میشه خودتو معرفی کنی",
        }
        shopping_markers = {
            "میخوام",
            "می‌خوام",
            "میخام",
            "دنبال",
            "بخر",
            "خرید",
            "قیمت",
            "ارزون",
            "ارزان",
            "زیر",
            "تومان",
            "محصول",
            "کالا",
            "مدل",
            "برند",
        }

        if any(marker in normalized for marker in shopping_markers):
            return False
        if normalized in social_markers:
            return True
        return any(
            normalized.startswith(prefix)
            for prefix in ("سلام", "خوبی", "چطوری", "ممنون", "مرسی", "خداحافظ")
        )

    def _build_smalltalk_response(self, message: str) -> str:
        """Generate deterministic social response."""
        normalized = self._normalize_text(message)
        if "معرفی" in normalized:
            return "من دستیار خرید شما هستم و کمک می‌کنم محصول مناسب پیدا کنید."
        if any(token in normalized for token in ("ممنون", "مرسی", "تشکر")):
            return "خواهش می‌کنم، خوشحال می‌شم کمکتون کنم."
        if any(token in normalized for token in ("خداحافظ", "فعلا")):
            return "خدانگهدار، هر زمان نیاز داشتید در خدمتم."
        return "سلام، خوبم ممنون. بفرمایید دنبال چه محصولی هستید؟"

    def _should_force_clarification(
        self,
        message: str,
        query_type: str,
        searchable: bool,
        search_params: dict[str, Any],
    ) -> bool:
        """
        Prevent generic vague requests from being treated as direct product searches.
        """
        if query_type != "direct" or not searchable:
            return False

        message_norm = self._normalize_text(message)
        product_norm = self._normalize_text(str(search_params.get("product") or ""))
        if not message_norm:
            return False

        vague_markers = {
            "یه چیز",
            "یک چیز",
            "چیزی",
            "نمیدونم چی",
            "نمی‌دونم چی",
            "یه مورد",
            "برای هدیه",
            "برای مهمونی",
            "برای پوشیدن",
        }
        has_vague_language = any(marker in message_norm for marker in vague_markers)
        categories = search_params.get("categories_fa") or []
        has_specific_product = bool(product_norm) and ("چیز" not in product_norm)

        if has_vague_language and not has_specific_product:
            return True
        if has_vague_language and product_norm == message_norm and not categories:
            return True
        return False

    def _apply_hard_constraints(
        self,
        rows: list[dict[str, Any]],
        search_params: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Enforce hard filters (price/brand) before returning results to the user.
        """
        if not rows:
            return rows, {"enabled": True, "dropped": 0, "kept": 0}

        price_range = search_params.get("price_range") or {}
        min_price = self._safe_float(price_range.get("min"), default=0.0)
        max_price = self._safe_float(price_range.get("max"), default=0.0)
        brand = self._normalize_text(str(search_params.get("brand") or ""))

        has_min = min_price > 0
        has_max = max_price > 0
        has_brand = bool(brand)
        if not (has_min or has_max or has_brand):
            return rows, {"enabled": False, "dropped": 0, "kept": len(rows)}

        filtered: list[dict[str, Any]] = []
        dropped = 0
        for row in rows:
            row_brand = self._normalize_text(str(row.get("brand_name") or row.get("brand") or ""))
            raw_price = row.get("discount_price")
            if raw_price in (None, "", 0, "0"):
                raw_price = row.get("price")
            price_value = self._safe_float(raw_price, default=0.0)

            brand_ok = (not has_brand) or (brand in row_brand or row_brand in brand)
            min_ok = (not has_min) or (price_value >= min_price)
            max_ok = (not has_max) or (price_value <= max_price)
            if brand_ok and min_ok and max_ok:
                filtered.append(row)
            else:
                dropped += 1

        return filtered, {
            "enabled": True,
            "dropped": dropped,
            "kept": len(filtered),
            "min_price": min_price if has_min else None,
            "max_price": max_price if has_max else None,
            "brand": brand if has_brand else None,
        }

    def _tokenize_text(self, value: str) -> list[str]:
        """Tokenize text with light Persian stopword filtering."""
        stopwords = {
            "یه", "یک", "من", "تو", "با", "از", "به", "برای", "در", "و", "یا",
            "رو", "را", "که", "این", "اون", "میخوام", "می‌خوام", "میخام", "میخواهم",
            "می", "خوام", "دارم", "م", "هستم",
        }
        normalized = self._normalize_text(value)
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        tokens: list[str] = []
        for token in normalized.split():
            t = token.strip()
            if len(t) < 2 or t in stopwords:
                continue
            tokens.append(t)
        return tokens

    def _token_overlap_score(self, query: str, candidate: str) -> float:
        """Soft overlap score to avoid strict keyword-only filtering."""
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return 0.5

        candidate_norm = self._normalize_text(candidate)
        candidate_tokens = set(self._tokenize_text(candidate))

        hits = 0
        for token in query_tokens:
            if token in candidate_tokens:
                hits += 1
                continue
            if len(token) >= 3 and token in candidate_norm:
                hits += 1

        return min(1.0, hits / max(1, len(query_tokens)))

    def _category_alignment_score(self, expected_categories: list[str], rows: list[dict[str, Any]]) -> float:
        """Category alignment as a soft signal (not a hard filter)."""
        normalized_expected = {
            self._normalize_text(str(c)) for c in (expected_categories or []) if str(c).strip()
        }
        if not normalized_expected:
            return 1.0

        compared = 0
        matched = 0
        for row in rows[:3]:
            category_name = self._normalize_text(str(row.get("category_name") or ""))
            if not category_name:
                continue
            compared += 1
            if any(exp in category_name or category_name in exp for exp in normalized_expected):
                matched += 1

        if compared == 0:
            return 0.5
        return matched / compared

    def _evaluate_direct_fastpath(
        self,
        search_params: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Decide whether direct fastpath results are reliable enough to skip final LLM.

        Uses a hybrid confidence score: rerank relevancy + margin + token overlap + category alignment.
        """
        if not rows:
            return {
                "accepted": True,
                "reason": "no_results",
                "confidence": 1.0,
                "top1_relevancy": 0.0,
                "top1_top2_margin": 0.0,
                "avg_top3_relevancy": 0.0,
                "query_overlap": 0.0,
                "category_alignment": 1.0,
            }

        query = str(search_params.get("product") or search_params.get("persian_full_query") or "")
        intent = self._normalize_text(str(search_params.get("intent") or "browse"))
        top_rows = rows[:3]

        relevancies: list[float] = []
        for row in top_rows:
            rel = row.get("relevancy_score")
            if rel is None:
                rel = self._token_overlap_score(query, str(row.get("product_name") or row.get("name") or ""))
            relevancies.append(max(0.0, min(1.0, self._safe_float(rel, default=0.0))))

        top1 = relevancies[0] if relevancies else 0.0
        top2 = relevancies[1] if len(relevancies) > 1 else 0.0
        margin = max(0.0, top1 - top2)
        avg_top3 = sum(relevancies) / len(relevancies) if relevancies else 0.0

        top_name = str(top_rows[0].get("product_name") or top_rows[0].get("name") or "")
        query_overlap = self._token_overlap_score(query, top_name)
        query_token_count = len(self._tokenize_text(query))
        category_alignment = self._category_alignment_score(
            search_params.get("categories_fa") or [],
            top_rows,
        )

        margin_norm = min(1.0, margin / 0.25)
        confidence = (
            0.40 * top1
            + 0.20 * avg_top3
            + 0.15 * margin_norm
            + 0.15 * query_overlap
            + 0.10 * category_alignment
        )

        t1 = self._safe_float(getattr(settings, "router_guard_t1", 0.55), default=0.55)
        t2 = self._safe_float(getattr(settings, "router_guard_t2", 0.08), default=0.08)
        min_conf = self._safe_float(
            getattr(settings, "router_guard_min_confidence", 0.58),
            default=0.58,
        )

        required_margin = t2
        if intent in {"find_cheapest", "find_best_value", "find_high_quality"}:
            required_margin = t2 * 0.5

        has_strong_top1 = top1 >= max(t1 + 0.25, 0.85)
        needs_margin_check = len(relevancies) >= 2 and query_token_count <= 2 and not has_strong_top1
        has_margin = (not needs_margin_check) or margin >= required_margin
        accepted = bool(top1 >= t1 and has_margin and confidence >= min_conf)

        reason = "accepted" if accepted else "low_confidence"
        if not accepted and top1 < t1:
            reason = "low_top1_relevancy"
        elif not accepted and not has_margin:
            reason = "low_top1_top2_margin"

        return {
            "accepted": accepted,
            "reason": reason,
            "confidence": round(confidence, 4),
            "top1_relevancy": round(top1, 4),
            "top1_top2_margin": round(margin, 4),
            "avg_top3_relevancy": round(avg_top3, 4),
            "query_overlap": round(query_overlap, 4),
            "category_alignment": round(category_alignment, 4),
            "query_token_count": query_token_count,
            "thresholds": {
                "t1": round(t1, 4),
                "t2": round(required_margin, 4),
                "min_confidence": round(min_conf, 4),
            },
        }

    @staticmethod
    def _stable_rollout_bucket(seed: str) -> int:
        """Deterministic 0..99 bucket for rollout decisions."""
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100

    def _is_in_rollout(self, feature: str, session_id: str) -> tuple[bool, dict[str, Any]]:
        """Decide if feature is enabled for this session based on rollout percent."""
        feature_key = (feature or "").strip().lower()
        percent_map = {
            "direct_fastpath": getattr(settings, "direct_fastpath_rollout_percent", 100),
            "final_llm": getattr(settings, "final_llm_rollout_percent", 100),
        }
        raw_percent = percent_map.get(feature_key, 100)
        try:
            percent = int(raw_percent)
        except Exception:
            percent = 100
        percent = max(0, min(100, percent))

        bucket = self._stable_rollout_bucket(f"{feature_key}:{session_id}")
        enabled = bucket < percent
        return enabled, {
            "feature": feature_key,
            "rollout_percent": percent,
            "bucket": bucket,
        }

    async def _try_router_fastpath(
        self,
        message: str,
        session_id: str,
        timings: dict[str, int],
    ) -> Optional[dict]:
        """Try routing request to fast deterministic paths. Return None to fallback."""
        if not settings.ff_router_enabled:
            return None

        if self._is_smalltalk_message(message):
            return {
                "success": True,
                "response": self._build_smalltalk_response(message),
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "query_type": "chat",
                    "total_results": 0,
                    "from_agent_cache": False,
                    "from_router_fastpath": True,
                    "router_route": "smalltalk_fastpath",
                    "latency_breakdown_ms": timings,
                },
            }

        stage_start = perf_counter()
        try:
            interpret = await self.agent.interpret_message(message, session_id)
        except Exception as e:
            logger.warning(f"Router interpret failed, fallback to agent.chat: {e}")
            return None
        if not isinstance(interpret, dict):
            return None
        timings["router_interpret_ms"] = int((perf_counter() - stage_start) * 1000)

        query_type = str(interpret.get("query_type") or "unknown")
        searchable = bool(interpret.get("searchable"))
        search_params = interpret.get("search_params") or {}
        clarification = interpret.get("clarification") or {}

        if self._should_force_clarification(message, query_type, searchable, search_params):
            question = "دقیق‌تر بفرمایید چه محصولی مدنظرتونه تا بهتر راهنمایی کنم."
            return {
                "success": True,
                "response": question,
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "query_type": "abstract",
                    "total_results": 0,
                    "from_agent_cache": False,
                    "from_router_fastpath": True,
                    "router_route": "direct_to_clarification_guard",
                    "latency_breakdown_ms": timings,
                },
            }

        # Abstract/unclear fastpath: respond from interpret clarification (no full agent loop).
        if (
            settings.ff_abstract_fastpath
            and query_type in {"abstract", "unclear"}
            and not searchable
        ):
            question = str(clarification.get("question") or "لطفاً دقیق‌تر بگید دنبال چه محصولی هستید؟").strip()
            suggestions = clarification.get("suggestions") or []
            suggestion_names = [
                str(s.get("product") or "").strip()
                for s in suggestions
                if isinstance(s, dict) and str(s.get("product") or "").strip()
            ][:5]

            response = question
            if suggestion_names:
                response = f"{question}\nپیشنهادها: " + " | ".join(
                    f"{i + 1}) {name}" for i, name in enumerate(suggestion_names)
                )

            return {
                "success": True,
                "response": response,
                "session_id": session_id,
                "products": [],
                "metadata": {
                    "query_type": query_type,
                    "total_results": 0,
                    "from_agent_cache": False,
                    "from_router_fastpath": True,
                    "router_route": "abstract_fastpath",
                    "latency_breakdown_ms": timings,
                },
            }

        # Direct fastpath: interpret -> search -> deterministic response.
        if (
            settings.ff_direct_fastpath
            and query_type == "direct"
            and searchable
            and search_params
        ):
            in_rollout, rollout_meta = self._is_in_rollout("direct_fastpath", session_id)
            log_latency_summary(
                "AGENT",
                "agent.router.rollout",
                0,
                meta={**rollout_meta, "enabled": in_rollout},
            )
            if not in_rollout:
                logger.info(
                    "Direct fastpath skipped by rollout percentage; fallback to full agent pipeline",
                    extra=rollout_meta,
                )
                return None

            stage_start = perf_counter()
            try:
                search_result = await self.agent.search_from_params(search_params, session_id)
            except Exception as e:
                logger.warning(f"Router direct search failed, fallback to agent.chat: {e}")
                return None
            timings["router_search_ms"] = int((perf_counter() - stage_start) * 1000)

            rows = search_result.get("results") or []
            rows, constraint_meta = self._apply_hard_constraints(rows, search_params)
            stage_start = perf_counter()
            guard = self._evaluate_direct_fastpath(search_params, rows)
            timings["router_guard_ms"] = int((perf_counter() - stage_start) * 1000)
            log_latency_summary(
                "AGENT",
                "agent.router.direct_guard",
                timings["router_guard_ms"],
                meta={
                    "accepted": guard.get("accepted"),
                    "reason": guard.get("reason"),
                    "confidence": guard.get("confidence"),
                    "top1_relevancy": guard.get("top1_relevancy"),
                    "top1_top2_margin": guard.get("top1_top2_margin"),
                },
            )
            router_route = "direct_fastpath"
            final_llm_meta: dict[str, Any] = {}
            if settings.ff_conditional_final_llm and not bool(guard.get("accepted")):
                in_rollout, final_rollout_meta = self._is_in_rollout("final_llm", session_id)
                log_latency_summary(
                    "AGENT",
                    "agent.router.rollout",
                    0,
                    meta={**final_rollout_meta, "enabled": in_rollout},
                )
                if not in_rollout:
                    logger.info(
                        "Final LLM fallback skipped by rollout percentage; fallback to full agent pipeline",
                        extra=final_rollout_meta,
                    )
                    return None

                timeout_s = max(1, int(getattr(settings, "final_llm_timeout_seconds", 5)))
                top_n = max(3, int(getattr(settings, "final_llm_top_n", 8)))
                stage_start = perf_counter()
                try:
                    final_llm = await asyncio.wait_for(
                        self.agent.final_rerank_direct(
                            user_message=message,
                            search_params=search_params,
                            search_rows=rows,
                            top_n=top_n,
                        ),
                        timeout=timeout_s,
                    )
                    timings["router_final_llm_ms"] = int((perf_counter() - stage_start) * 1000)
                    log_latency_summary(
                        "AGENT",
                        "agent.router.final_llm",
                        timings["router_final_llm_ms"],
                        meta={"success": True, "timeout_s": timeout_s, "top_n": top_n},
                    )
                    rows = final_llm.get("rows") or rows
                    rows, constraint_meta = self._apply_hard_constraints(rows, search_params)
                    final_llm_meta = final_llm.get("meta") or {}
                    final_llm_meta["response"] = str(final_llm.get("response") or "").strip()
                    router_route = "direct_final_llm"
                except asyncio.TimeoutError:
                    timings["router_final_llm_ms"] = int((perf_counter() - stage_start) * 1000)
                    log_latency_summary(
                        "AGENT",
                        "agent.router.final_llm",
                        timings["router_final_llm_ms"],
                        meta={"success": False, "timeout": True, "timeout_s": timeout_s, "top_n": top_n},
                    )
                    logger.info(
                        "Router final LLM timed out; fallback to full agent pipeline",
                        extra={"guard": guard, "timeout_s": timeout_s},
                    )
                    return None
                except Exception as e:
                    timings["router_final_llm_ms"] = int((perf_counter() - stage_start) * 1000)
                    log_latency_summary(
                        "AGENT",
                        "agent.router.final_llm",
                        timings["router_final_llm_ms"],
                        meta={"success": False, "timeout_s": timeout_s, "top_n": top_n},
                    )
                    logger.warning(f"Router final LLM failed, fallback to agent.chat: {e}")
                    return None

            products: list[dict] = []
            for i, item in enumerate(rows[:5]):
                products.append(
                    {
                        "id": str(item.get("id") or item.get("product_id") or i + 1),
                        "name": str(item.get("product_name") or item.get("name") or "").strip(),
                        "brand": str(item.get("brand_name") or item.get("brand") or "").strip(),
                        "price": float(item.get("price") or 0),
                        "discount_price": (
                            float(item.get("discount_price"))
                            if item.get("discount_price") is not None
                            else None
                        ),
                        "has_discount": bool(item.get("has_discount", False)),
                        "discount_percentage": float(item.get("discount_percentage") or 0),
                        "product_url": str(item.get("product_url") or item.get("url") or "").strip(),
                    }
                )

            if products:
                response = str(final_llm_meta.get("response") or "").strip() or "این محصولات رو پیدا کردم:"
                routed_query_type = "direct"
            else:
                response = "متأسفانه محصول مورد نظر شما یافت نشد. میتونید با کلمات دیگه جستجو کنید."
                routed_query_type = "no_results"

            result = {
                "success": True,
                "response": response,
                "session_id": session_id,
                "products": products,
                "metadata": {
                    "query_type": routed_query_type,
                    "total_results": len(products),
                    "from_agent_cache": False,
                    "from_router_fastpath": True,
                    "router_route": router_route,
                    "router_guard": guard,
                    "constraint_filter": constraint_meta,
                    "final_llm": final_llm_meta,
                    "latency_breakdown_ms": timings,
                },
            }

            # Reuse level-2 agent cache for direct fastpath responses.
            if self._cache and self._cache.available and products:
                stage_start = perf_counter()
                await self._cache.set(message, result)
                timings["agent_cache_set_ms"] = int((perf_counter() - stage_start) * 1000)
                result["metadata"]["latency_breakdown_ms"] = timings

            return result

        return None

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

        # ── Router fastpath (after cache miss) ─────────────────────
        stage_start = perf_counter()
        routed = await self._try_router_fastpath(message, session_id, timings)
        timings["router_total_ms"] = int((perf_counter() - stage_start) * 1000)
        if routed is not None:
            took_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            routed["metadata"]["took_ms"] = took_ms
            routed["metadata"]["latency_breakdown_ms"] = timings
            log_latency_summary(
                "AGENT",
                "agent_service.chat",
                int((perf_counter() - request_start) * 1000),
                breakdown_ms=timings,
                meta={
                    "cache": "router_fastpath",
                    "query_type": routed["metadata"].get("query_type"),
                    "success": routed.get("success", True),
                    "products": len(routed.get("products") or []),
                },
            )
            return routed
        
        # ── Cache miss → full pipeline ─────────────────────────────
        try:
            # Call agent with timeout
            stage_start = perf_counter()
            response, session_id = await asyncio.wait_for(
                self.agent.chat(message, session_id),
                timeout=timeout,
            )
            timings["agent_chat_ms"] = int((perf_counter() - stage_start) * 1000)
            
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
        if products:
            return "direct"

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
