"""
Shopping AI Assistant Agent

ReAct-based conversational agent for Persian shopping assistance.
The LLM decides which tools to use based on user queries.
"""

import asyncio
import json
import uuid
from time import perf_counter
from contextvars import ContextVar
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from pydantic import Field
from pydantic_settings import BaseSettings

from src.mcp_client import InterpretMCPClient, SearchMCPClient
from src.pipeline_logger import (
    get_current_trace,
    log_agent,
    log_cache,
    log_error,
    log_latency_summary,
    log_query_summary,
    trace_query,
)

# ============================================================================
# Configuration
# ============================================================================


class Settings(BaseSettings):
    """Agent settings."""

    agent_model_provider: str = Field(default="openrouter", alias="AGENT_MODEL_PROVIDER")
    agent_model: str = Field(default="", alias="AGENT_MODEL")

    openrouter_api_key: str = Field(default="", alias="OPEN_ROUTERS_API_KEY")
    openrouter_model: str = Field(default="meta-llama/llama-3.3-70b-instruct", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_provider_order: str = Field(default="", alias="OPENROUTER_PROVIDER_ORDER")
    openrouter_fallback_to_groq: bool = Field(default=True, alias="OPENROUTER_FALLBACK_TO_GROQ")

    # Groq fallback
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    interpret_url: str = Field(default="http://localhost:5004", alias="MCP_INTERPRET_URL")
    search_url: str = Field(default="http://localhost:5002", alias="MCP_SEARCH_URL")
    interpret_mcp_timeout: float = Field(default=90.0, alias="INTERPRET_MCP_TIMEOUT")
    search_mcp_timeout: float = Field(default=60.0, alias="SEARCH_MCP_TIMEOUT")

    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # Redis (for LLM response cache inside agent)
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    llm_cache_ttl: int = Field(default=86400, alias="LLM_CACHE_TTL")  # 24h
    agent_recursion_limit: int = Field(default=12, alias="AGENT_RECURSION_LIMIT")
    max_search_tool_calls_per_turn: int = Field(default=3, alias="MAX_SEARCH_TOOL_CALLS_PER_TURN")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# ============================================================================
# LLM Response Cache (Level 3)
# ============================================================================
# Caches the LLM's final formatted response keyed by search parameters.
# When search_products is called with the same params, the cached LLM output
# is returned directly — the LLM just passes it through without re-ranking.
# ============================================================================

import hashlib
import redis.asyncio as aioredis

_llm_cache_redis: Optional[aioredis.Redis] = None
_last_search_cache_key: Optional[str] = None  # Tracks current search key for post-LLM storage
_llm_cache_hit: bool = False  # Flag: True when Level 3 cache was hit inside search_products
_llm_cached_response: Optional[str] = None  # Holds the cached LLM response when Level 3 hits
_active_session_id: ContextVar[str] = ContextVar("active_session_id", default="")
_search_tool_call_counts: dict[str, int] = {}


def _get_active_session_id() -> str:
    """Get current request session id from context."""
    current = _active_session_id.get()
    return current or str(uuid.uuid4())


async def _get_llm_cache() -> Optional[aioredis.Redis]:
    """Get or create the Redis connection for LLM response cache."""
    global _llm_cache_redis
    if _llm_cache_redis is not None:
        return _llm_cache_redis
    try:
        _llm_cache_redis = aioredis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
        )
        await _llm_cache_redis.ping()
        log_agent("LLM response cache connected to Redis", {
            "host": settings.redis_host,
            "port": settings.redis_port,
        })
        return _llm_cache_redis
    except Exception as e:
        log_error("AGENT", f"LLM cache Redis connection failed: {e}", e)
        _llm_cache_redis = None
        return None


def _make_search_cache_key(search_params: dict) -> str:
    """
    Build a cache key from search parameters.
    
    Key is based on: product + brand + intent + price_range
    so that identical searches produce the same key.
    """
    parts = [
        str(search_params.get("product", "") or ""),
        str(search_params.get("brand", "") or ""),
        str(search_params.get("intent", "browse") or "browse"),
    ]
    # Include categories in cache key
    categories = search_params.get("categories_fa") or []
    if categories:
        parts.append("|".join(sorted(str(c) for c in categories if c)))
    
    price_range = search_params.get("price_range") or {}
    pr_min = price_range.get("min") if price_range else None
    pr_max = price_range.get("max") if price_range else None
    # Only add price_range to key if there are real values
    if (pr_min and pr_min > 0) or (pr_max and pr_max > 0):
        parts.append(str(pr_min or 0))
        parts.append(str(pr_max or 0))
    
    raw = "|".join(parts).lower().strip()
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"cache:v1:llm_response:{h}"


async def _get_cached_llm_response(key: str) -> Optional[str]:
    """Look up a cached LLM response for a search key."""
    r = await _get_llm_cache()
    if not r:
        return None
    try:
        val = await r.get(key)
        if val:
            log_cache("LLM response cache HIT", {"key": key})
        else:
            log_cache("LLM response cache MISS", {"key": key})
        return val
    except Exception as e:
        log_error("AGENT", f"LLM cache GET failed: {e}", e)
        return None


async def _store_llm_response(key: str, response: str) -> bool:
    """Store the LLM's formatted response in Redis."""
    r = await _get_llm_cache()
    if not r:
        return False
    try:
        await r.set(key, response, ex=settings.llm_cache_ttl)
        log_cache("LLM response cache SET", {"key": key, "ttl": settings.llm_cache_ttl})
        return True
    except Exception as e:
        log_error("AGENT", f"LLM cache SET failed: {e}", e)
        return False


# ============================================================================
# System Prompt (English for better LLM understanding)
# ============================================================================

SYSTEM_PROMPT = """You are a Persian shopping assistant.

Your job has two phases:
1) Decide the correct mode for this user turn.
2) Execute only that mode.

## Decision Modes (exactly one per turn)
- CHAT: greeting, thanks, normal talk
- CLARIFY: vague/abstract/insufficient request
- SEARCH: concrete product request, ready for search
- DETAILS: follow-up about a known product/result item

## Strict Decision Policy
Choose SEARCH only when ALL are true:
1) User explicitly names a purchasable product or product-type.
2) User intent is product finding/buying now.
3) Request is actionable enough for search.
4) Message is not just greeting/chat.
5) Message is not abstract lifestyle advice without clear product noun.

If any condition fails, DO NOT call search tools. Use CLARIFY.

## When to use each mode

### CHAT (no tools)
For: سلام، خوبی، ممنون، خداحافظ, casual conversation without shopping intent.

### CLARIFY (no tools)
For vague or abstract requests:
- "یه چیز گرم میخوام"
- "یه هدیه میخوام"
- "یه چیزی میخوام"
- "اه"
- "چی بگیرم خوبه؟" (without product type)

CLARIFY behavior:
- Ask max 2 short clarifying questions.
- Offer 3-5 concrete product-type suggestions.
- Ask user to pick one.
- Keep it short and practical.

### SEARCH (use search_and_deliver exactly once)
For concrete product requests:
- "شورت مردانه ارزون میخوام"
- "گوشی سامسونگ زیر ۲۰ میلیون"
- "بهترین لپتاپ برای برنامه نویسی"
- "کفش پیاده‌روی زنانه"

Call:
`search_and_deliver` with the user's exact request.

### DETAILS (usually no search)
For follow-up references:
- "اولی رو میخوام"
- "همون رو باز کن"
- "دومی قیمتش چنده؟"

Rules:
- If previous results exist in conversation memory, resolve reference from those results.
- If product id is available and user asks details, call `get_product_details`.
- If no reliable previous result exists, DO NOT guess. Use CLARIFY.

## Tool Output Handling
- If tool returns prefix "🔍 SEARCH_RESULTS:", show content as-is (without prefix).
- If tool returns prefix "✅ CACHED_RESPONSE:", show content as-is (without prefix).
- If tool returns prefix "❓ NEED_CLARIFICATION:", continue with CLARIFY.
- If tool returns prefix "❌ NO_RESULTS:", explain shortly and offer alternatives.

## Hard Safety Rule
If uncertainty is noticeable, prefer CLARIFY over SEARCH.
Never force a tool call just to continue conversation.

## Response Style
- Always Persian.
- Helpful, concise, natural.
- Emojis are allowed but not excessive.

## Output Discipline (very important)
- Never reveal internal mode names (CHAT/CLARIFY/SEARCH/DETAILS).
- Never explain internal reasoning, policies, or tool-decision process.
- Never output analysis text like "the correct mode is ...".
- Output only user-facing assistant text.
"""

# ============================================================================
# MCP Clients
# ============================================================================

_interpret_client: Optional[InterpretMCPClient] = None
_search_client: Optional[SearchMCPClient] = None


def get_interpret_client() -> InterpretMCPClient:
    global _interpret_client
    if _interpret_client is None:
        _interpret_client = InterpretMCPClient(
            settings.interpret_url, timeout=settings.interpret_mcp_timeout
        )
    return _interpret_client


def get_search_client() -> SearchMCPClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchMCPClient(
            settings.search_url, timeout=settings.search_mcp_timeout
        )
    return _search_client


# ============================================================================
# Tool Definitions
# ============================================================================


@tool(return_direct=True)
async def search_and_deliver(query: str) -> str:
    """
    Interpret user query, search for products, and return formatted results.
    
    This is the main tool for product searches. It handles the full pipeline:
    1. Interprets the query (extract product, brand, intent, categories)
    2. If query is unclear/not a product search, returns clarification
    3. If searchable, executes search and returns formatted results
    
    ONLY call this when user clearly wants a specific product or product type.
    Do NOT call for abstract/vague requests, greetings, or follow-ups.
    
    Args:
        query: The EXACT user message about a product they want
    
    Returns:
        Formatted search results ready to show to user, OR clarification request
    """
    global _last_search_cache_key
    global _llm_cache_hit
    global _llm_cached_response

    try:
        tool_start = perf_counter()
        timings: dict[str, int] = {}
        current_trace = get_current_trace()
        trace_id = current_trace.trace_id if current_trace else None
        request_session_id = _get_active_session_id()
        tool_calls = _search_tool_call_counts.get(request_session_id, 0) + 1
        _search_tool_call_counts[request_session_id] = tool_calls

        log_agent("search_and_deliver called", {
            "query": query,
            "session": request_session_id[:8],
            "tool_calls_this_turn": tool_calls,
        })

        if tool_calls > settings.max_search_tool_calls_per_turn:
            log_agent("search_and_deliver loop guard triggered", {
                "tool_calls_this_turn": tool_calls,
                "max_allowed": settings.max_search_tool_calls_per_turn,
            })
            return (
                "❓ NEED_CLARIFICATION:برای اینکه نتیجه دقیق‌تری بگیرم، لطفاً یکی از این گزینه‌ها رو مشخص کنید:\n"
                "🛍️ نوع دقیق محصول\n"
                "💰 بازه بودجه نزدیک‌تر\n"
                "🏷️ برند ترجیحی"
            )

        # ── Step 1: Interpret the query ──
        interpret_client = get_interpret_client()
        interpret_start = perf_counter()
        interpret_result = await interpret_client.interpret_query(
            query=query,
            session_id=request_session_id,
            context={"trace_id": trace_id} if trace_id else {},
        )
        timings["interpret_ms"] = int((perf_counter() - interpret_start) * 1000)

        query_type = interpret_result.get("query_type", "direct")
        searchable = interpret_result.get("searchable", False)

        log_agent("search_and_deliver interpret result", {
            "query_type": query_type,
            "searchable": searchable,
            "product": interpret_result.get("search_params", {}).get("product"),
        })

        # ── Step 1.5: Interpret re-checks unclear ──
        # If interpret says unclear, it means the agent sent a bad query.
        # Return clarification back to the agent so it can chat with user.
        if query_type == "unclear" or not searchable:
            clarification = interpret_result.get("clarification", {})
            question = clarification.get("question", "لطفاً دقیق‌تر بگید دنبال چه محصولی هستید؟")
            suggestions = clarification.get("suggestions", [])
            
            suggestion_text = ""
            if suggestions:
                parts = []
                for s in suggestions:
                    if isinstance(s, dict):
                        emoji = s.get("emoji", "🛒")
                        product = s.get("product", "")
                        parts.append(f"{emoji} {product}")
                    else:
                        parts.append(f"🛒 {s}")
                suggestion_text = "\n".join(parts)
            
            log_agent("search_and_deliver returning clarification", {
                "query_type": query_type,
                "question": question,
            })
            log_latency_summary(
                "AGENT",
                "agent.tool.search_and_deliver",
                int((perf_counter() - tool_start) * 1000),
                breakdown_ms=timings,
                meta={"result": "clarification", "query_type": query_type},
            )
            
            return f"❓ NEED_CLARIFICATION:{question}\n{suggestion_text}"

        # ── Step 2: Build search params ──
        search_params = interpret_result.get("search_params", {})
        product = search_params.get("product", query)
        brand = search_params.get("brand")
        intent = search_params.get("intent", "browse")
        categories_fa = search_params.get("categories_fa", [])
        price_range = search_params.get("price_range", {})

        # Build params for search
        final_search_params = {
            "product": product,
            "intent": intent,
            "persian_full_query": search_params.get("persian_full_query", product),
        }

        if brand and str(brand).strip():
            final_search_params["brand"] = brand

        if categories_fa and len(categories_fa) > 0:
            final_search_params["categories_fa"] = [c for c in categories_fa if c and str(c).strip()]

        if price_range:
            pr = {}
            if price_range.get("min") and price_range["min"] > 0:
                pr["min"] = price_range["min"]
            if price_range.get("max") and price_range["max"] > 0:
                pr["max"] = price_range["max"]
            if pr:
                final_search_params["price_range"] = pr

        if trace_id:
            final_search_params["trace_id"] = trace_id

        # Track search key for LLM response caching
        _last_search_cache_key = _make_search_cache_key(final_search_params)
        log_agent("search_and_deliver cache key", {
            "key": _last_search_cache_key,
            "params_used": {k: v for k, v in final_search_params.items() if k not in ("persian_full_query", "trace_id")},
        })

        # ── Level 3: Check LLM response cache ──
        if not settings.debug_mode:
            cache_lookup_start = perf_counter()
            cached_llm = await _get_cached_llm_response(_last_search_cache_key)
            timings["llm_cache_lookup_ms"] = int((perf_counter() - cache_lookup_start) * 1000)
            if cached_llm:
                log_agent("⚡ LLM response cache HIT in search_and_deliver", {
                    "key": _last_search_cache_key,
                    "product": product,
                })
                _llm_cache_hit = True
                _llm_cached_response = cached_llm
                log_latency_summary(
                    "AGENT",
                    "agent.tool.search_and_deliver",
                    int((perf_counter() - tool_start) * 1000),
                    breakdown_ms=timings,
                    meta={"cache": "llm_hit", "product": product},
                )
                return f"✅ CACHED_RESPONSE:{cached_llm}"
        else:
            timings["llm_cache_lookup_ms"] = 0

        # ── Step 3: Execute search ──
        search_client = get_search_client()
        search_start = perf_counter()
        result = await search_client.search_products(
            search_params=final_search_params,
            session_id=request_session_id,
            use_cache=not settings.debug_mode,
            use_semantic=True,
        )
        timings["search_ms"] = int((perf_counter() - search_start) * 1000)

        # ── Step 4: Format results for user ──
        results = result.get("results", [])
        if not results:
            log_agent("search_and_deliver no results", {"product": product})
            log_latency_summary(
                "AGENT",
                "agent.tool.search_and_deliver",
                int((perf_counter() - tool_start) * 1000),
                breakdown_ms=timings,
                meta={"result": "no_results", "product": product},
            )
            _last_search_cache_key = None
            suggestions: list[str] = []
            if brand:
                suggestions.append("همین محصول بدون محدودیت برند")
            if price_range and price_range.get("max"):
                suggestions.append("افزایش سقف بودجه")
            if price_range and price_range.get("min"):
                suggestions.append("کاهش حداقل بودجه")
            suggestions.extend([
                "عبارت کوتاه‌تر از همین محصول",
                "یک دسته نزدیک به همین محصول",
            ])
            suggestion_lines = "\n".join([f"🛒 {s}" for s in suggestions[:5]])
            return (
                "❓ NEED_CLARIFICATION:متأسفانه با این شرط‌ها نتیجه‌ای پیدا نشد. "
                "یکی از گزینه‌های زیر رو انتخاب کنید تا دوباره دقیق‌تر جستجو کنم:\n"
                f"{suggestion_lines}"
            )

        # Format as ready-to-show response
        formatted_products = []
        for item in results[:10]:
            formatted_products.append({
                "name": item.get("product_name", ""),
                "brand": item.get("brand_name", ""),
                "price": item.get("price", 0),
                "discount_price": item.get("discount_price"),
                "has_discount": item.get("has_discount", False),
                "discount_percentage": item.get("discount_percentage", 0),
                "product_url": item.get("product_url", ""),
            })

        products_json = json.dumps(formatted_products, ensure_ascii=False, indent=2)
        response_text = f"این محصولات رو پیدا کردم:\n```json\n{products_json}\n```"

        log_agent("search_and_deliver success", {
            "product": product,
            "results_count": len(formatted_products),
        })
        log_query_summary(
            query=product,
            query_type="search",
            product=product,
            results_count=result.get("total_hits", 0),
            from_cache=result.get("from_cache", False),
            total_ms=result.get("took_ms", 0),
        )
        log_latency_summary(
            "AGENT",
            "agent.tool.search_and_deliver",
            int((perf_counter() - tool_start) * 1000),
            breakdown_ms=timings,
            meta={
                "result": "success",
                "product": product,
                "results": len(formatted_products),
                "search_cache": result.get("from_cache", False),
            },
        )

        return f"🔍 SEARCH_RESULTS:{response_text}"

    except Exception as e:
        _last_search_cache_key = None
        log_error("AGENT", f"search_and_deliver failed: {e}", e)
        return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)


@tool
async def get_product_details(product_id: str) -> str:
    """
    Get detailed information about a specific product.
    
    Use when user asks about a specific product from results.
    
    Args:
        product_id: The product identifier
    
    Returns:
        JSON with product details
    """
    try:
        log_agent("get_product_details called", {"product_id": product_id})
        
        client = get_search_client()
        result = await client.get_product(product_id)
        
        return json.dumps(result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        log_error("AGENT", f"get_product_details failed: {e}", e)
        return json.dumps({"error": str(e), "success": False}, ensure_ascii=False)


# ============================================================================
# Shopping Agent Class
# ============================================================================


class ShoppingAgent:
    """
    ReAct-based shopping assistant agent.
    
    The LLM decides which tools to use based on user queries.
    Supports multi-turn conversation with memory.
    """

    @staticmethod
    def _resolve_model_config() -> tuple[str, str, str, str]:
        """
        Resolve LLM provider/model from environment settings.

        Returns:
            provider, api_key, base_url, model
        """
        provider = (settings.agent_model_provider or "openrouter").strip().lower()

        if provider == "groq":
            api_key = settings.groq_api_key
            base_url = settings.groq_base_url
            model = (settings.agent_model or settings.groq_model).strip()
        elif provider in {"openrouter", "open_router"}:
            api_key = settings.openrouter_api_key
            base_url = settings.openrouter_base_url
            model = (settings.agent_model or settings.openrouter_model).strip()
            provider = "openrouter"
        else:
            raise ValueError(
                "AGENT_MODEL_PROVIDER must be one of: openrouter, groq"
            )

        if not model:
            raise ValueError("No model configured for selected AGENT_MODEL_PROVIDER")
        if not api_key:
            raise ValueError(f"API key is empty for provider '{provider}'")

        return provider, api_key, base_url, model

    @staticmethod
    def _resolve_openrouter_provider_order() -> list[str]:
        """
        Parse OPENROUTER_PROVIDER_ORDER into provider names for OpenRouter routing.
        Example: "deepinfra,groq"
        """
        raw = (settings.openrouter_provider_order or "").strip()
        if not raw:
            return []

        mapping = {
            "deepinfra": "DeepInfra",
            "groq": "Groq",
            "together": "Together",
            "togetherai": "Together",
            "openai": "OpenAI",
            "fireworks": "Fireworks",
            "novita": "Novita",
        }
        providers: list[str] = []
        for part in raw.split(","):
            value = part.strip()
            if not value:
                continue
            providers.append(mapping.get(value.lower(), value))
        return providers

    def __init__(self):
        provider, api_key, base_url, model = self._resolve_model_config()
        log_agent("Initializing ShoppingAgent", {"provider": provider, "model": model})

        llm_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": 0.3,
        }
        if provider == "openrouter":
            provider_order = self._resolve_openrouter_provider_order()
            if provider_order:
                # OpenRouter expects provider routing inside request extra_body.
                # Passing "provider" as a top-level field causes OpenAI client errors.
                llm_kwargs["extra_body"] = {
                    "provider": {"order": provider_order},
                }
                log_agent("OpenRouter provider routing enabled", {"order": provider_order})

        self.llm = ChatOpenAI(
            **llm_kwargs,
        )
        
        self.tools = [search_and_deliver, get_product_details]
        self.memory = MemorySaver()
        self._fallback_agent = None
        self._provider = provider
        self._model_name = model

        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=SYSTEM_PROMPT,
        )

        if (
            provider == "openrouter"
            and settings.openrouter_fallback_to_groq
            and settings.groq_api_key
        ):
            try:
                fallback_llm = ChatOpenAI(
                    api_key=settings.groq_api_key,
                    base_url=settings.groq_base_url,
                    model=settings.groq_model,
                    temperature=0.3,
                )
                self._fallback_agent = create_react_agent(
                    model=fallback_llm,
                    tools=self.tools,
                    checkpointer=self.memory,
                    prompt=SYSTEM_PROMPT,
                )
                log_agent(
                    "OpenRouter fallback agent enabled",
                    {"provider": "groq", "model": settings.groq_model},
                )
            except Exception as e:
                log_error("AGENT", f"Failed to initialize fallback agent: {e}", e)
        
        log_agent("ShoppingAgent initialized", {"tools": [t.name for t in self.tools]})

    @staticmethod
    def _message_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(p for p in parts if p).strip()
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _extract_text_response(result: dict) -> str:
        response = ""
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                response = ShoppingAgent._message_content_to_text(msg.content)
                if response:
                    break
            if isinstance(msg, ToolMessage) and msg.content:
                response = ShoppingAgent._message_content_to_text(msg.content)
                if response:
                    break
        return response

    @staticmethod
    def _is_tool_support_error(err: Exception) -> bool:
        text = str(err or "")
        return (
            "No endpoints found that support tool use" in text
            or ("Error code: 404" in text and "tool use" in text.lower())
        )

    async def persist_external_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        """
        Persist an externally generated turn (e.g. direct bypass results)
        into the agent's session memory so follow-up references can work.
        """
        if not session_id or not user_message or not assistant_message:
            return False

        config = {"configurable": {"thread_id": session_id}}
        values = {
            "messages": [
                HumanMessage(content=user_message),
                AIMessage(content=assistant_message),
            ]
        }

        try:
            await self.agent.aupdate_state(config, values)
            log_agent(
                "Persisted external turn to memory",
                {
                    "session": session_id[:8],
                    "user_len": len(user_message),
                    "assistant_len": len(assistant_message),
                },
            )
            return True
        except Exception as e:
            log_error("AGENT", f"Failed to persist external turn: {e}", e)
            return False

    async def chat(self, message: str, session_id: Optional[str] = None) -> tuple[str, str]:
        """
        Process a user message and return response.
        
        Flow:
        1. LangGraph agent runs: LLM → interpret_query → search_products
        2. Inside search_products: check LLM response cache (Level 3)
           If HIT → return cached LLM response as tool output (LLM passes it through)
           If MISS → return normal ES results, track cache key
        3. After LLM responds → store response in LLM cache keyed by search params

        Args:
            message: User message in Persian
            session_id: Optional session ID for conversation continuity

        Returns:
            Tuple of (response, session_id)
        """
        global _last_search_cache_key
        global _llm_cache_hit
        global _llm_cached_response
        chat_start = perf_counter()
        timings: dict[str, int] = {}
        
        if not session_id:
            session_id = str(uuid.uuid4())

        session_token = _active_session_id.set(session_id)
        try:
            with trace_query(message, session_id):
                config = {
                    "configurable": {"thread_id": session_id},
                    "recursion_limit": settings.agent_recursion_limit,
                }

                # Reset trackers before each message
                _last_search_cache_key = None
                _llm_cache_hit = False
                _llm_cached_response = None
                _search_tool_call_counts[session_id] = 0

                log_agent("Processing message", {"session": session_id[:8], "message": message[:50]})

                # ── Full LLM pipeline (cache is checked inside search_products) ──
                try:
                    ainvoke_start = perf_counter()
                    result = await self.agent.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=config,
                    )
                    timings["react_ainvoke_ms"] = int((perf_counter() - ainvoke_start) * 1000)

                    response = self._extract_text_response(result)

                    if not response:
                        response = "متأسفانه نتونستم پاسخ مناسبی پیدا کنم. لطفاً دوباره سوالتون رو بپرسید."
                    if "Sorry, need more steps to process this request." in response:
                        response = (
                            "برای رسیدن به نتیجه دقیق‌تر، لطفاً یکی از این موارد رو مشخص کنید:\n"
                            "🛍️ نوع دقیق محصول\n"
                            "💰 بازه بودجه\n"
                            "🏷️ برند دلخواه"
                        )

                    # ── Level 3 HIT: use the exact cached response instead of LLM output ──
                    if _llm_cache_hit and _llm_cached_response:
                        log_agent("⚡ Using cached LLM response (ignoring LLM re-generation)", {
                            "key": _last_search_cache_key,
                            "cached_len": len(_llm_cached_response),
                            "llm_len": len(response),
                        })
                        response = _llm_cached_response

                    # Clean up prefixes from search_and_deliver that the LLM might have kept
                    for prefix in ("🔍 SEARCH_RESULTS:", "❓ NEED_CLARIFICATION:", "❌ NO_RESULTS:", "✅ CACHED_RESPONSE:"):
                        if response.startswith(prefix):
                            response = response[len(prefix):]
                            break

                    # ── Store LLM response in cache (keyed by search params) ──
                    # Only store if this was NOT a cache hit (avoid re-storing same data)
                    llm_cache_was_hit = _llm_cache_hit
                    if _last_search_cache_key and not settings.debug_mode and not _llm_cache_hit:
                        log_agent("Storing LLM response in cache", {
                            "key": _last_search_cache_key,
                            "response_len": len(response),
                        })
                        cache_store_start = perf_counter()
                        await _store_llm_response(_last_search_cache_key, response)
                        timings["llm_cache_store_ms"] = int((perf_counter() - cache_store_start) * 1000)
                    elif _llm_cache_hit:
                        log_agent("Skipping cache store (Level 3 was HIT)", {
                            "key": _last_search_cache_key,
                        })
                        timings["llm_cache_store_ms"] = 0
                    else:
                        timings["llm_cache_store_ms"] = 0
                    _last_search_cache_key = None
                    _llm_cache_hit = False
                    _llm_cached_response = None

                    log_agent("Response generated", {"session": session_id[:8], "response_len": len(response)})
                    log_latency_summary(
                        "AGENT",
                        "agent.chat",
                        int((perf_counter() - chat_start) * 1000),
                        breakdown_ms=timings,
                        meta={"success": True, "llm_cache_hit": bool(llm_cache_was_hit)},
                    )
                    return response, session_id

                except Exception as e:
                    if self._is_tool_support_error(e) and self._fallback_agent is not None:
                        log_error(
                            "AGENT",
                            "Primary model lacks tool-use endpoint; switching to fallback model",
                            e,
                        )
                        try:
                            fallback_start = perf_counter()
                            fallback_result = await self._fallback_agent.ainvoke(
                                {"messages": [HumanMessage(content=message)]},
                                config=config,
                            )
                            timings["fallback_ainvoke_ms"] = int((perf_counter() - fallback_start) * 1000)
                            fallback_response = self._extract_text_response(fallback_result)
                            if not fallback_response:
                                fallback_response = "متأسفانه نتونستم پاسخ مناسبی پیدا کنم. لطفاً دوباره سوالتون رو بپرسید."
                            log_latency_summary(
                                "AGENT",
                                "agent.chat",
                                int((perf_counter() - chat_start) * 1000),
                                breakdown_ms=timings,
                                meta={"success": True, "fallback_model_used": True},
                            )
                            return fallback_response, session_id
                        except Exception as fallback_error:
                            log_error("AGENT", f"Fallback model failed: {fallback_error}", fallback_error)

                    error_text = str(e)
                    if "tool_calls that do not have a corresponding ToolMessage" in error_text:
                        new_session_id = str(uuid.uuid4())
                        retry_config = {"configurable": {"thread_id": new_session_id}}
                        log_error(
                            "AGENT",
                            "Detected invalid chat history; retrying with new session",
                            e,
                        )
                        retry_token = _active_session_id.set(new_session_id)
                        try:
                            retry_result = await self.agent.ainvoke(
                                {"messages": [HumanMessage(content=message)]},
                                config=retry_config,
                            )

                            retry_response = ""
                            for msg in reversed(retry_result.get("messages", [])):
                                if isinstance(msg, AIMessage) and msg.content:
                                    retry_response = msg.content
                                    break

                            if not retry_response:
                                retry_response = "متأسفانه نتونستم پاسخ مناسبی پیدا کنم. لطفاً دوباره سوالتون رو بپرسید."

                            log_agent("Response generated after history reset", {
                                "old_session": session_id[:8],
                                "new_session": new_session_id[:8],
                                "response_len": len(retry_response),
                            })
                            log_latency_summary(
                                "AGENT",
                                "agent.chat",
                                int((perf_counter() - chat_start) * 1000),
                                breakdown_ms=timings,
                                meta={"success": True, "retry_with_new_session": True},
                            )
                            return retry_response, new_session_id
                        except Exception as retry_error:
                            log_error("AGENT", f"Retry after history reset failed: {retry_error}", retry_error)
                        finally:
                            _active_session_id.reset(retry_token)

                    log_error("AGENT", f"Chat error: {e}", e)
                    log_latency_summary(
                        "AGENT",
                        "agent.chat",
                        int((perf_counter() - chat_start) * 1000),
                        breakdown_ms=timings,
                        meta={"success": False},
                    )
                    error_payload = {
                        "stage": "agent.chat",
                        "error_type": e.__class__.__name__,
                        "message": str(e),
                    }
                    return f"__AGENT_ERROR__:{json.dumps(error_payload, ensure_ascii=False)}", session_id
                finally:
                    _search_tool_call_counts.pop(session_id, None)
        finally:
            _active_session_id.reset(session_token)

    async def close(self):
        """Cleanup resources."""
        global _interpret_client, _search_client
        
        if _interpret_client:
            await _interpret_client.close()
            _interpret_client = None
            
        if _search_client:
            await _search_client.close()
            _search_client = None
        
        log_agent("Agent connections closed", {})


# ============================================================================
# Factory Function
# ============================================================================


async def create_agent() -> ShoppingAgent:
    """Create and initialize the shopping agent."""
    return ShoppingAgent()


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    async def main():
        """Interactive CLI for testing the agent."""
        print("🚀 Starting Shopping Agent...")
        agent = ShoppingAgent()
        
        print("\n" + "=" * 50)
        print("🛒 دستیار خرید هوشمند")
        print("برای خروج 'exit' یا 'خروج' تایپ کنید")
        print("=" * 50 + "\n")
        
        session_id = None
        
        while True:
            try:
                user_input = input("👤 شما: ").strip()
                
                if user_input.lower() in ["quit", "exit", "خروج", "q"]:
                    print("👋 خداحافظ!")
                    break
                
                if not user_input:
                    continue
                
                print("🤔 در حال پردازش...")
                response, session_id = await agent.chat(user_input, session_id)
                print(f"\n🤖 دستیار:\n{response}\n")
                
            except KeyboardInterrupt:
                print("\n👋 خداحافظ!")
                break
            except Exception as e:
                print(f"❌ خطا: {e}")
        
        await agent.close()

    asyncio.run(main())
