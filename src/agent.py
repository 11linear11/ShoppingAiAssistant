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

from langchain_core.messages import AIMessage, HumanMessage
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
    openrouter_model: str = Field(default="qwen/qwen2.5-vl-32b-instruct", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")
    openrouter_provider_order: str = Field(default="", alias="OPENROUTER_PROVIDER_ORDER")

    # Groq fallback
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")

    interpret_url: str = Field(default="http://localhost:5004", alias="MCP_INTERPRET_URL")
    search_url: str = Field(default="http://localhost:5002", alias="MCP_SEARCH_URL")

    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    # Redis (for LLM response cache inside agent)
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    llm_cache_ttl: int = Field(default=86400, alias="LLM_CACHE_TTL")  # 24h

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

SYSTEM_PROMPT = """You are a friendly Persian shopping assistant. You help users find products.

## CRITICAL RULES:

### For Greetings and Casual Chat ONLY:
- DO NOT use any tools for pure greetings/thanks/goodbyes
- This ONLY applies to: سلام، خوبی، چطوری، ممنون، خداحافظ (with NO product or shopping intent)
- Just respond naturally and warmly in Persian

### For EVERYTHING ELSE (including abstract/vague requests):
You MUST ALWAYS call `interpret_query` FIRST. NEVER skip this step. NEVER answer directly.

Examples that MUST go to interpret_query:
- "خسته ام یه چیز خوب میخوام" -> interpret_query("خسته ام یه چیز خوب میخوام")
- "ارزان ترین شامپو" -> interpret_query("ارزان ترین شامپو")  
- "یه هدیه میخوام" -> interpret_query("یه هدیه میخوام")
- "کفش ورزشی مردانه" -> interpret_query("کفش ورزشی مردانه")
- "یه چیز برای آرامش" -> interpret_query("یه چیز برای آرامش")
- "بهترین لپتاپ" -> interpret_query("بهترین لپتاپ")
- "میخوام آروم بشم" -> interpret_query("میخوام آروم بشم")

## Tool Flow:

### Step 1: interpret_query
ALWAYS call this first for any non-greeting message. Pass the EXACT user message as-is.

### Step 2: Read interpret_query result carefully
The result JSON contains:
- `query_type`: "direct", "abstract", "follow_up", or "unclear"
- `searchable`: true or false
- `search_params`: has `product`, `brand`, `intent`, `price_range`, `categories_fa`
- `clarification`: for abstract/unclear queries, has question and suggestions

### Step 3: Act based on result

**If searchable=true (direct queries):**
Call `search_products` with ALL parameters from interpret result:
- product: from search_params.product
- brand: from search_params.brand (only if not null)
- min_price: from search_params.price_range.min (only if > 0)
- max_price: from search_params.price_range.max (only if > 0)
- intent: from search_params.intent EXACTLY as returned ("find_cheapest", "find_best", "browse", etc.)
- categories_fa: from search_params.categories_fa (only if non-empty list)

CRITICAL: Do NOT change or ignore the intent! If interpret says intent="find_cheapest", pass intent="find_cheapest" to search_products.

**If searchable=false (abstract/unclear queries):**
DO NOT call search_products.
Use the clarification from interpret result.
Present the suggestions in a friendly Persian message and ask user to pick one.

## IMPORTANT for search_products:
- Only pass parameters that have actual non-null values from interpret result
- If brand is null, DO NOT pass brand parameter
- If price_range is empty, DO NOT pass min_price or max_price
- If categories_fa is empty or null, DO NOT pass categories_fa parameter
- ALWAYS pass the intent value from interpret result
- Never pass null values for optional parameters

## CRITICAL: Presenting Search Results

### Re-ranking and Filtering:
After getting search results:
1. **Filter irrelevant products**: Remove products that don't match
2. **Re-rank by relevance**: Most relevant first
3. **Show 5-10 products**

### Formatting Results:
Return response in this EXACT format:
1. First line: Short Persian intro (e.g., "این محصولات رو پیدا کردم:")
2. New line: JSON block in ```json ... ``` with products array

JSON objects must have:
- "name": string
- "brand": string (empty if unknown)
- "price": number (standard digits 0-9, NOT Persian digits)
- "discount_price": number or null (standard digits)
- "has_discount": boolean
- "discount_percentage": number
- "product_url": string

Example:
این محصولات رو پیدا کردم:
```json
[
  {"name": "product name", "brand": "brand", "price": 12500000, "discount_price": 11000000, "has_discount": true, "discount_percentage": 12, "product_url": ""},
  {"name": "product name 2", "brand": "brand", "price": 8500000, "discount_price": null, "has_discount": false, "discount_percentage": 0, "product_url": ""}
]
```

### No Results:
Say in Persian: "متأسفانه محصول مورد نظر شما یافت نشد. میتونید با کلمات دیگه جستجو کنید."

IMPORTANT:
- Do NOT format products with emojis or bullet points
- ALWAYS use ```json code block format
- Use standard digits (0-9) for numbers in JSON

## CACHED RESPONSE:
If search_products returns a response starting with "✅ CACHED_RESPONSE:", that is a COMPLETE ready-to-send response from a previous identical search. You MUST return it EXACTLY as-is (without the "✅ CACHED_RESPONSE:" prefix). Do NOT modify, re-rank, re-format, or add anything. Just remove the prefix and return the rest verbatim.

## Response Language:
- ALWAYS respond in Persian (Farsi)
- Use emojis to make responses friendly
"""

# ============================================================================
# MCP Clients
# ============================================================================

_interpret_client: Optional[InterpretMCPClient] = None
_search_client: Optional[SearchMCPClient] = None


def get_interpret_client() -> InterpretMCPClient:
    global _interpret_client
    if _interpret_client is None:
        _interpret_client = InterpretMCPClient(settings.interpret_url, timeout=90.0)
    return _interpret_client


def get_search_client() -> SearchMCPClient:
    global _search_client
    if _search_client is None:
        _search_client = SearchMCPClient(settings.search_url, timeout=60.0)
    return _search_client


# ============================================================================
# Tool Definitions
# ============================================================================


@tool
async def interpret_query(query: str) -> str:
    """
    Interpret and analyze ANY user query that is not a simple greeting.
    
    MUST be called for ALL non-greeting messages, including:
    - Direct product requests: "کفش ورزشی مردانه"
    - Cheapest/best queries: "ارزان ترین شامپو"
    - Abstract/vague requests: "خسته ام یه چیز خوب میخوام"
    - Any message with shopping intent
    
    Args:
        query: The EXACT user message, passed as-is without modification
    
    Returns:
        JSON with:
        - searchable: true if this is a valid product search
        - search_params: extracted parameters (product, brand, price_range, intent)
    """
    try:
        tool_start = perf_counter()
        current_trace = get_current_trace()
        trace_id = current_trace.trace_id if current_trace else None
        request_session_id = _get_active_session_id()

        log_agent("interpret_query called", {
            "query": query,
            "session": request_session_id[:8],
        })
        
        client = get_interpret_client()
        mcp_call_start = perf_counter()
        result = await client.interpret_query(
            query=query,
            session_id=request_session_id,
            context={"trace_id": trace_id} if trace_id else {},
        )
        mcp_call_ms = int((perf_counter() - mcp_call_start) * 1000)
        
        log_agent("interpret_query result", {
            "searchable": result.get("searchable"),
            "query_type": result.get("query_type"),
        })
        log_latency_summary(
            "AGENT",
            "agent.tool.interpret_query",
            int((perf_counter() - tool_start) * 1000),
            breakdown_ms={"mcp_call_ms": mcp_call_ms},
            meta={
                "searchable": result.get("searchable"),
                "query_type": result.get("query_type"),
            },
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        log_latency_summary(
            "AGENT",
            "agent.tool.interpret_query",
            int((perf_counter() - tool_start) * 1000),
            meta={"success": False},
        )
        log_error("AGENT", f"interpret_query failed: {e}", e)
        return json.dumps({"error": str(e), "searchable": False}, ensure_ascii=False)


@tool
async def search_products(
    product: str,
    brand: Optional[str] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    intent: str = "browse",
    categories_fa: Optional[list[str]] = None,
    limit: int = 10
) -> str:
    """
    Search for products in the database.
    
    MUST only be called after interpret_query returns searchable=true.
    Use the values from interpret_query's search_params to fill these parameters.
    
    Args:
        product: Product name from interpret result's search_params.product (required)
        brand: Brand from interpret result's search_params.brand (optional, skip if null)
        min_price: From interpret result's search_params.price_range.min (optional, skip if null)
        max_price: From interpret result's search_params.price_range.max (optional, skip if null)
        intent: From interpret result's search_params.intent - MUST pass exactly as returned (e.g. "find_cheapest", "find_best", "browse"). This controls result sorting.
        categories_fa: From interpret result's search_params.categories_fa - list of Persian category names (optional, skip if empty)
        limit: Max results (default: 10)
    
    Returns:
        JSON with search results
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

        log_agent("search_products called", {
            "product": product,
            "brand": brand,
            "min_price": min_price,
            "max_price": max_price,
            "intent": intent,
            "categories_fa": categories_fa,
            "session": request_session_id[:8],
        })
        
        client = get_search_client()
        
        # Build search params - only include non-None values
        search_params = {
            "product": product,
            "intent": intent,
            "persian_full_query": product,
        }
        
        if brand and brand.strip():
            search_params["brand"] = brand
        
        if categories_fa and len(categories_fa) > 0:
            search_params["categories_fa"] = [c for c in categories_fa if c and c.strip()]
        
        if min_price is not None or max_price is not None:
            search_params["price_range"] = {}
            if min_price is not None and min_price > 0:
                search_params["price_range"]["min"] = min_price
            if max_price is not None and max_price > 0:
                search_params["price_range"]["max"] = max_price

        if trace_id:
            search_params["trace_id"] = trace_id
        
        # Track search key for post-LLM caching
        _last_search_cache_key = _make_search_cache_key(search_params)
        log_agent("search_products cache key", {
            "key": _last_search_cache_key,
            "params_used": {k: v for k, v in search_params.items() if k != "persian_full_query"},
        })
        
        # ── Level 3: Check if we have a cached LLM response for these search params ──
        if not settings.debug_mode:
            cache_lookup_start = perf_counter()
            cached_llm = await _get_cached_llm_response(_last_search_cache_key)
            timings["llm_cache_lookup_ms"] = int((perf_counter() - cache_lookup_start) * 1000)
            if cached_llm:
                log_agent("⚡ LLM response cache HIT inside search_products — returning cached LLM output", {
                    "key": _last_search_cache_key,
                    "product": product,
                    "cached_len": len(cached_llm),
                })
                # Set flag so chat() won't re-store the same response
                _llm_cache_hit = True
                _llm_cached_response = cached_llm
                log_latency_summary(
                    "AGENT",
                    "agent.tool.search_products",
                    int((perf_counter() - tool_start) * 1000),
                    breakdown_ms=timings,
                    meta={"cache": "llm_hit", "product": product},
                )
                # Return the previous LLM-formatted response directly as tool output.
                # The LLM will see this and pass it through as-is.
                return f"✅ CACHED_RESPONSE:{cached_llm}"
        else:
            timings["llm_cache_lookup_ms"] = 0
        
        # ── Level 3 MISS: proceed with normal search ──
        mcp_search_start = perf_counter()
        result = await client.search_products(
            search_params=search_params,
            session_id=request_session_id,
            use_cache=not settings.debug_mode,
        )
        timings["mcp_search_ms"] = int((perf_counter() - mcp_search_start) * 1000)
        
        # Format results
        formatted_results = []
        for item in result.get("results", [])[:limit]:
            formatted_results.append({
                "id": item.get("id", item.get("product_id", "")),
                "name": item.get("product_name", ""),
                "brand": item.get("brand_name", ""),
                "price": item.get("price", 0),
                "discount_price": item.get("discount_price"),
                "has_discount": item.get("has_discount", False),
                "discount_percentage": item.get("discount_percentage", 0),
                "url": item.get("product_url", ""),
            })
        
        output = {
            "success": result.get("success", True),
            "total_hits": result.get("total_hits", 0),
            "results": formatted_results,
            "took_ms": result.get("took_ms", 0),
        }
        
        log_query_summary(
            query=product,
            query_type="search",
            product=product,
            results_count=output["total_hits"],
            from_cache=result.get("from_cache", False),
            total_ms=output["took_ms"],
        )
        log_latency_summary(
            "AGENT",
            "agent.tool.search_products",
            int((perf_counter() - tool_start) * 1000),
            breakdown_ms=timings,
            meta={
                "cache": "llm_miss",
                "search_cache": result.get("from_cache", False),
                "results": output["total_hits"],
                "product": product,
            },
        )
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        _last_search_cache_key = None
        log_latency_summary(
            "AGENT",
            "agent.tool.search_products",
            int((perf_counter() - tool_start) * 1000),
            meta={"success": False, "product": product},
        )
        log_error("AGENT", f"search_products failed: {e}", e)
        return json.dumps({"error": str(e), "success": False, "results": []}, ensure_ascii=False)


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
                llm_kwargs["model_kwargs"] = {
                    "extra_body": {
                        "provider": {"order": provider_order},
                    },
                }
                log_agent("OpenRouter provider routing enabled", {"order": provider_order})

        self.llm = ChatOpenAI(
            **llm_kwargs,
        )
        
        self.tools = [interpret_query, search_products, get_product_details]
        self.memory = MemorySaver()
        
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            prompt=SYSTEM_PROMPT,
        )
        
        log_agent("ShoppingAgent initialized", {"tools": [t.name for t in self.tools]})

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
                config = {"configurable": {"thread_id": session_id}}

                # Reset trackers before each message
                _last_search_cache_key = None
                _llm_cache_hit = False
                _llm_cached_response = None

                log_agent("Processing message", {"session": session_id[:8], "message": message[:50]})

                # ── Full LLM pipeline (cache is checked inside search_products) ──
                try:
                    ainvoke_start = perf_counter()
                    result = await self.agent.ainvoke(
                        {"messages": [HumanMessage(content=message)]},
                        config=config,
                    )
                    timings["react_ainvoke_ms"] = int((perf_counter() - ainvoke_start) * 1000)

                    response = ""
                    for msg in reversed(result.get("messages", [])):
                        if isinstance(msg, AIMessage) and msg.content:
                            response = msg.content
                            break

                    if not response:
                        response = "متأسفانه نتونستم پاسخ مناسبی پیدا کنم. لطفاً دوباره سوالتون رو بپرسید."

                    # ── Level 3 HIT: use the exact cached response instead of LLM output ──
                    if _llm_cache_hit and _llm_cached_response:
                        log_agent("⚡ Using cached LLM response (ignoring LLM re-generation)", {
                            "key": _last_search_cache_key,
                            "cached_len": len(_llm_cached_response),
                            "llm_len": len(response),
                        })
                        response = _llm_cached_response

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
                    return f"متأسفانه مشکلی پیش اومد: {str(e)}", session_id
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
