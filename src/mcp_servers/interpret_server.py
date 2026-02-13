"""
Interpret MCP Server (MCP Protocol Version)
Port: 5004

LLM-based query interpretation for the Shopping AI Assistant.
Uses LangChain with GitHub Models API (Llama-3.3-70B-Instruct) for classification and extraction.

This is the MCP protocol version using FastMCP SDK.
"""

import asyncio
import json
import re
import sys
from time import perf_counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

import hashlib
import redis.asyncio as aioredis

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.pipeline_logger import (
    TraceContext,
    log_interpret,
    log_error,
    log_latency_summary,
    reset_current_trace,
    set_current_trace,
    trace_stage,
)

# Load .env from project root
from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


# ============================================================================
# Configuration
# ============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # GitHub API settings for Interpret Server (separate from Agent's Groq)
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_base_url: str = Field(
        default="https://models.inference.ai.azure.com",
        alias="GITHUB_BASE_URL",
    )
    github_model: str = Field(
        default="Llama-3.3-70B-Instruct",
        alias="GITHUB_MODEL",
    )

    # Embedding model for category matching (local)
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base",
        alias="EMBEDDING_MODEL",
    )

    # Redis settings for embedding cache
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    cache_embedding_ttl: int = Field(default=604800, alias="CACHE_EMBEDDING_TTL")  # 7 days

    # Debug mode - disables caching
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    model_config = {"extra": "ignore"}


settings = Settings()


# Shared singleton across HTTP sessions to avoid re-loading heavy models.
_shared_interpreter: Optional["QueryInterpreter"] = None
_shared_interpreter_lock = asyncio.Lock()


# ============================================================================
# Models
# ============================================================================


class QueryType(str, Enum):
    """Types of user queries."""

    DIRECT = "direct"
    ABSTRACT = "abstract"
    FOLLOW_UP = "follow_up"
    UNCLEAR = "unclear"


# ============================================================================
# LangChain LLM Client (using ChatOpenAI with Groq endpoint)
# ============================================================================


class LangChainLLM:
    """LangChain-based LLM client using ChatOpenAI with GitHub Models endpoint."""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=settings.github_token,
            base_url=settings.github_base_url,
            model=settings.github_model,
            temperature=0.1,
            max_tokens=1500,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> str:
        """Generate text using LangChain with Groq API."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            # Use ainvoke for async call
            response = await self.llm.ainvoke(messages)
            return response.content

        except Exception as e:
            log_error("INTERPRET", f"LangChain LLM error: {e}", e)
            raise

    async def close(self):
        """Cleanup (no-op for LangChain)."""
        pass


# ============================================================================
# Query Interpreter
# ============================================================================


class QueryInterpreter:
    """
    LLM-based query interpreter.

    Uses LangChain for classification AND extraction in one call.
    Uses local SentenceTransformer for category matching.
    """

    def __init__(self, llm: LangChainLLM):
        self.llm = llm
        self._category_embeddings: dict[str, list[float]] = {}
        self._category_names: list[str] = []
        self._embedding_cache_redis: Optional[aioredis.Redis] = None
        
        # Local embedding model for category matching
        log_interpret("Loading embedding model", {"model": settings.embedding_model})
        self._embedding_model = SentenceTransformer(
            settings.embedding_model, 
            device="cpu"
        )
        log_interpret("Embedding model loaded", {"model": settings.embedding_model})

    async def _init_embedding_cache(self):
        """Initialize Redis connection for embedding cache."""
        if self._embedding_cache_redis is not None:
            return
        try:
            self._embedding_cache_redis = aioredis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password or None,
                db=settings.redis_db,
                decode_responses=True,
            )
            await self._embedding_cache_redis.ping()
            log_interpret("Embedding cache connected to Redis", {
                "host": settings.redis_host,
                "port": settings.redis_port,
            })
        except Exception as e:
            log_error("INTERPRET", f"Embedding cache Redis connection failed: {e}", e)
            self._embedding_cache_redis = None

    def _make_embedding_cache_key(self, text: str) -> str:
        """Build a Redis key for embedding cache."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"cache:v1:embedding:{text_hash}"

    async def _get_cached_embedding(self, text: str) -> Optional[list[float]]:
        """Look up a cached embedding."""
        if not self._embedding_cache_redis or settings.debug_mode:
            return None
        try:
            key = self._make_embedding_cache_key(text)
            val = await self._embedding_cache_redis.get(key)
            if val:
                return json.loads(val)
        except Exception as e:
            log_error("INTERPRET", f"Embedding cache GET failed: {e}", e)
        return None

    async def _store_embedding(self, text: str, embedding: list[float]) -> None:
        """Store embedding in Redis cache."""
        if not self._embedding_cache_redis or settings.debug_mode:
            return
        try:
            key = self._make_embedding_cache_key(text)
            await self._embedding_cache_redis.set(
                key,
                json.dumps(embedding),
                ex=settings.cache_embedding_ttl,
            )
        except Exception as e:
            log_error("INTERPRET", f"Embedding cache SET failed: {e}", e)

    async def load_category_embeddings(
        self, filepath: str = "full_category_embeddings.json"
    ):
        """Load category embeddings for semantic matching."""
        try:
            project_root = Path(__file__).parent.parent.parent
            file_path = project_root / filepath

            with open(file_path, "r", encoding="utf-8") as f:
                self._category_embeddings = json.load(f)

            self._category_names = list(self._category_embeddings.keys())
            log_interpret(
                "Category embeddings loaded from file",
                {"count": len(self._category_names), "file": str(file_path)},
            )
            log_interpret(
                "Category embeddings loaded", {"count": len(self._category_names)}
            )
        except FileNotFoundError:
            log_error("INTERPRET", f"Category embeddings file not found: {filepath}")
        except Exception as e:
            log_error("INTERPRET", f"Error loading category embeddings: {e}", e)

    async def interpret(
        self,
        query: str,
        session_id: str,
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Main interpretation pipeline.

        1. Normalize input
        2. Call LLM for classification + extraction
        3. Process based on type
        4. Match categories (for direct queries)
        """
        context = context or {}
        total_start = perf_counter()
        timings: dict[str, int] = {}

        log_interpret("📥 Received query", {"query": query, "session": session_id})

        # Normalize input
        normalize_start = perf_counter()
        with trace_stage("INTERPRET", "Normalize Persian text"):
            normalized = self._normalize_persian(query)
            log_interpret(
                "Normalized", {"original": query, "normalized": normalized}
            )
        timings["normalize_ms"] = int((perf_counter() - normalize_start) * 1000)

        def _log_summary(query_type: str, searchable: bool):
            log_latency_summary(
                "INTERPRET",
                "interpret.pipeline",
                int((perf_counter() - total_start) * 1000),
                breakdown_ms=timings,
                meta={
                    "query_type": query_type,
                    "searchable": searchable,
                },
            )

        # Quick check: is it just a number? (follow-up selection)
        if normalized.strip().isdigit():
            log_interpret("Detected number selection", {"number": normalized})
            response = self._handle_number_selection(normalized.strip(), context)
            _log_summary(
                query_type=response.get("query_type", "follow_up"),
                searchable=bool(response.get("searchable")),
            )
            return response

        # LLM Classification + Extraction
        llm_start = perf_counter()
        with trace_stage("INTERPRET", "LLM Classification"):
            llm_result = await self._classify_and_extract(normalized, context)
            log_interpret(
                "LLM Result",
                {
                    "query_type": llm_result.get("query_type"),
                    "product": llm_result.get("product"),
                    "confidence": llm_result.get("confidence"),
                },
            )
        timings["llm_classification_ms"] = int((perf_counter() - llm_start) * 1000)

        # Process based on type
        query_type_str = llm_result.get("query_type", "direct")
        log_interpret(
            f"Query type: {query_type_str}",
            {"reasoning": llm_result.get("reasoning", "")[:100]},
        )

        if query_type_str == "direct":
            build_start = perf_counter()
            response = await self._build_direct_response(llm_result, normalized)
            timings["build_response_ms"] = int((perf_counter() - build_start) * 1000)
            log_interpret(
                "✅ DIRECT response",
                {
                    "product": response.get("search_params", {}).get("product"),
                    "categories": response.get("search_params", {}).get(
                        "categories_fa", []
                    ),
                },
            )
            _log_summary(
                query_type=response.get("query_type", "direct"),
                searchable=bool(response.get("searchable")),
            )
            return response

        elif query_type_str == "abstract":
            build_start = perf_counter()
            response = self._build_abstract_response(llm_result, normalized)
            timings["build_response_ms"] = int((perf_counter() - build_start) * 1000)
            log_interpret(
                "✅ ABSTRACT response",
                {
                    "suggestions": [
                        s.get("product")
                        for s in response.get("clarification", {}).get(
                            "suggestions", []
                        )
                    ]
                },
            )
            _log_summary(
                query_type=response.get("query_type", "abstract"),
                searchable=bool(response.get("searchable")),
            )
            return response

        elif query_type_str == "follow_up":
            build_start = perf_counter()
            response = self._build_followup_response(llm_result, context)
            timings["build_response_ms"] = int((perf_counter() - build_start) * 1000)
            log_interpret(
                "✅ FOLLOW_UP response",
                {"session_update": response.get("session_update")},
            )
            _log_summary(
                query_type=response.get("query_type", "follow_up"),
                searchable=bool(response.get("searchable")),
            )
            return response

        else:  # UNCLEAR
            build_start = perf_counter()
            response = self._build_unclear_response(llm_result)
            timings["build_response_ms"] = int((perf_counter() - build_start) * 1000)
            log_interpret("✅ UNCLEAR response")
            _log_summary(
                query_type=response.get("query_type", "unclear"),
                searchable=bool(response.get("searchable")),
            )
            return response

    def _normalize_persian(self, text: str) -> str:
        """Normalize Persian text."""
        replacements = {
            "ك": "ک",
            "ي": "ی",
            "٤": "۴",
            "٥": "۵",
            "٦": "۶",
            "ة": "ه",
            "ؤ": "و",
        }
        for ar, fa in replacements.items():
            text = text.replace(ar, fa)
        return re.sub(r"\s+", " ", text).strip()

    def _handle_number_selection(self, number: str, context: dict) -> dict[str, Any]:
        """Handle numeric selection (follow-up)."""
        return {
            "query_type": "follow_up",
            "searchable": True,
            "session_update": {
                "selection": int(number),
                "selection_type": "numeric",
            },
        }

    async def _classify_and_extract(self, query: str, context: dict) -> dict:
        """Use LLM to classify AND extract in one call."""

        context_str = ""
        if context.get("last_results"):
            context_str = f"Previous results: {len(context['last_results'])} products"
        if context.get("last_query"):
            context_str += f"\nPrevious query: {context['last_query']}"

        prompt = f"""You are a Persian shopping query analyzer.

User query: "{query}"
{context_str}

Analyze this query and return a JSON response.

## CLASSIFICATION RULES:

### DIRECT (most common):
Use when user mentions ANY specific product name, even with modifiers like "cheapest", "best", etc.
Examples:
- "یخچال" → DIRECT (product: یخچال)
- "ارزون ترین یخچال" → DIRECT (product: یخچال, intent: find_cheapest)
- "بهترین گوشی سامسونگ" → DIRECT (product: گوشی, brand: سامسونگ, intent: find_best)
- "لپتاپ زیر ۲۰ میلیون" → DIRECT (product: لپتاپ, price_range.max: 20000000)
- "شورت مردانه" → DIRECT (product: شورت مردانه)
- "پاستیل" → DIRECT (product: پاستیل)
- "ارزون ترین پاستیل" → DIRECT (product: پاستیل, intent: find_cheapest)

### ABSTRACT:
Use ONLY when user describes a feeling/need WITHOUT any product name.
Examples:
- "خسته‌ام" → ABSTRACT
- "میخوام آروم بشم" → ABSTRACT  
- "یه چیز خوب میخوام" → ABSTRACT (no product specified)

### FOLLOW_UP:
Use ONLY when user refers to previous results with pronouns or ordinals WITHOUT mentioning a product.
Examples:
- "اولی" → FOLLOW_UP (referring to item 1)
- "همون" → FOLLOW_UP
- "این یکی" → FOLLOW_UP
- "شماره ۳" → FOLLOW_UP
NOT follow_up:
- "ارزون ترین پاستیل" → This is DIRECT (has product name "پاستیل")

### UNCLEAR:
Use when query is too short or meaningless.
Examples:
- "اه" → UNCLEAR
- "چی" → UNCLEAR

## INTENT VALUES:
- browse: just looking (default)
- find_cheapest: wants lowest price (ارزون‌ترین، کمترین قیمت)
- find_best: wants best quality/value (بهترین)
- compare: wants to compare options

Return JSON only:
{{
    "query_type": "direct|abstract|follow_up|unclear",
    "confidence": 0.0-1.0,
    "reasoning": "short explanation in Persian",
    "product": "product name or null",
    "brand": "brand name or null",
    "price_range": {{"min": null, "max": null}},
    "intent": "browse|find_cheapest|find_best|compare",
    "feeling": "user feeling for abstract only",
    "suggested_products": ["suggestion1", "suggestion2", "suggestion3"],
    "reference_type": "ordinal|pronoun for follow_up",
    "reference_value": "reference value"
}}

Return only valid JSON."""

        system_prompt = """You are a smart shopping query analyzer.
Always return only valid JSON.
If a product name is mentioned, it's DIRECT - not ABSTRACT.
"ارزون ترین X" means DIRECT with intent=find_cheapest, not ABSTRACT."""

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.1,
            )

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"query_type": "direct", "product": query}

        except json.JSONDecodeError:
            return {"query_type": "direct", "product": query}
        except Exception as e:
            log_error("INTERPRET", f"LLM classify/extract error: {e}", e)
            return {"query_type": "direct", "product": query}

    async def _build_direct_response(
        self, llm_result: dict, query: str
    ) -> dict[str, Any]:
        """Build response for DIRECT queries."""

        product = llm_result.get("product", query)
        with trace_stage("INTERPRET", "Category matching"):
            categories = await self._match_categories(product)

        return {
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "intent": llm_result.get("intent", "browse"),
                "product": product,
                "brand": llm_result.get("brand"),
                "persian_full_query": query,
                "categories_fa": categories,
                "attributes": llm_result.get("attributes", {}),
                "price_range": llm_result.get("price_range", {}),
            },
            "session_update": {
                "last_query": query,
                "last_product": product,
            },
        }

    def _build_abstract_response(
        self, llm_result: dict, query: str
    ) -> dict[str, Any]:
        """Build response for ABSTRACT queries."""

        suggested = llm_result.get("suggested_products", [])
        feeling = llm_result.get("feeling", "")

        emojis = ["🛒", "👕", "🧥", "👟", "📱", "💻", "🎁", "🔥"]
        suggestions = []
        for i, prod in enumerate(suggested[:5], 1):
            suggestions.append(
                {
                    "id": i,
                    "product": prod,
                    "emoji": emojis[i % len(emojis)],
                    "reason": f"پیشنهاد برای {feeling}" if feeling else "پیشنهاد",
                    "search_query": prod,
                }
            )

        question = "دنبال چه محصولی هستید؟"
        if feeling:
            question = f"برای {feeling}، کدوم محصول رو می‌خواید؟"

        return {
            "query_type": "abstract",
            "searchable": False,
            "clarification": {
                "needed": True,
                "question": question,
                "suggestions": suggestions,
            },
            "session_update": {
                "last_query": query,
                "abstract_pattern": feeling,
                "pending_suggestions": [s["product"] for s in suggestions],
            },
        }

    def _build_followup_response(
        self, llm_result: dict, context: dict
    ) -> dict[str, Any]:
        """Build response for FOLLOW_UP queries."""

        ref_type = llm_result.get("reference_type", "ordinal")
        ref_value = llm_result.get("reference_value", "1")

        return {
            "query_type": "follow_up",
            "searchable": True,
            "session_update": {
                "selection": ref_value,
                "selection_type": ref_type,
            },
        }

    def _build_unclear_response(self, llm_result: dict) -> dict[str, Any]:
        """Build response for UNCLEAR queries."""

        return {
            "query_type": "unclear",
            "searchable": False,
            "clarification": {
                "needed": True,
                "question": "لطفاً دقیق‌تر بگید دنبال چه محصولی هستید؟",
                "suggestions": [],
            },
        }

    async def _match_categories(self, product: str) -> list[str]:
        """Match product to categories using embedding similarity with caching."""
        if not self._category_embeddings:
            log_interpret("No category embeddings loaded", {})
            return []

        try:
            # Check embedding cache first
            prefixed_text = f"query: {product}"
            cached_emb = await self._get_cached_embedding(prefixed_text)
            
            if cached_emb is not None:
                product_vec = np.array(cached_emb)
                log_interpret("Embedding cache HIT for category matching", {"product": product})
            else:
                # Generate embedding using local model (with E5 prefix)
                product_embedding = self._embedding_model.encode(
                    prefixed_text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                product_vec = np.array(product_embedding)
                
                # Store in cache
                await self._store_embedding(prefixed_text, product_vec.tolist())
                log_interpret("Embedding cache MISS, stored", {"product": product})

            similarities = []
            for cat_name, cat_embedding in self._category_embeddings.items():
                cat_vec = np.array(cat_embedding)
                similarity = float(np.dot(product_vec, cat_vec))
                similarities.append((cat_name, similarity))

            similarities.sort(key=lambda x: x[1], reverse=True)

            # Log top matches for debugging
            top_5 = similarities[:5]
            log_interpret("Category matching results", {
                "product": product,
                "top_matches": [(cat, round(sim, 4)) for cat, sim in top_5]
            })

            threshold = 0.75
            matched = [cat for cat, sim in similarities if sim >= threshold]
            
            if not matched:
                log_interpret("No categories above threshold", {
                    "threshold": threshold,
                    "best_match": top_5[0] if top_5 else None
                })
            
            return matched[:5]

        except Exception as e:
            log_interpret("Category matching error", {"error": str(e)})
            log_error("INTERPRET", f"Category matching failed: {e}", e)
            return []

    async def close(self):
        """Cleanup resources."""
        await self.llm.close()


# ============================================================================
# MCP Server Setup
# ============================================================================


@dataclass
class AppContext:
    """Application context with interpreter."""

    interpreter: QueryInterpreter


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle."""
    global _shared_interpreter

    if _shared_interpreter is None:
        async with _shared_interpreter_lock:
            if _shared_interpreter is None:
                llm_client = LangChainLLM()
                interpreter = QueryInterpreter(llm_client)
                await interpreter.load_category_embeddings()
                await interpreter._init_embedding_cache()
                _shared_interpreter = interpreter
                log_interpret("Interpret server started", {"model": settings.github_model})
                if settings.debug_mode:
                    log_interpret("DEBUG mode active")
    else:
        log_interpret("Interpret server warm instance reused", {"model": settings.github_model})

    try:
        yield AppContext(interpreter=_shared_interpreter)
    finally:
        # Keep warm singleton alive across sessions/process lifespan.
        pass


# Create MCP server
mcp = FastMCP(
    "Interpret Server",
    lifespan=app_lifespan,
)

# Allow Docker internal hostnames for transport security
mcp.settings.transport_security.allowed_hosts.extend([
    "interpret:*",
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
    "http://interpret:*",
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
async def interpret_query(
    query: str,
    session_id: str,
    context: Optional[dict] = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Interpret a user query.

    Uses LLM to:
    1. Classify query type (direct/abstract/follow_up/unclear)
    2. Extract structured data (product, brand, attributes, etc.)

    Args:
        query: User's search query (Persian/English)
        session_id: Session identifier for context
        context: Optional session context (last_results, last_query, etc.)

    Returns:
        Dict with query_type, searchable flag, search_params or clarification
    """
    if not query or len(query) > 1000:
        return {"success": False, "error": "Query must be 1-1000 characters"}

    interpreter = ctx.request_context.lifespan_context.interpreter
    incoming_trace_id = (context or {}).get("trace_id")
    trace_token = None

    if incoming_trace_id:
        trace = TraceContext(query=query, session_id=session_id)
        trace.trace_id = str(incoming_trace_id)
        trace_token = set_current_trace(trace)

    try:
        result = await interpreter.interpret(
            query=query,
            session_id=session_id,
            context=context,
        )
        result["success"] = True
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if trace_token is not None:
            reset_current_trace(trace_token)


@mcp.tool()
async def classify_query(
    query: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Quick classification endpoint - just returns query type.

    Args:
        query: User's search query

    Returns:
        Dict with query_type, confidence, and reasoning
    """
    if not query:
        return {"success": False, "error": "Query is required"}

    interpreter = ctx.request_context.lifespan_context.interpreter

    try:
        result = await interpreter._classify_and_extract(query, {})
        return {
            "success": True,
            "query_type": result.get("query_type"),
            "confidence": result.get("confidence"),
            "reasoning": result.get("reasoning"),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def get_interpreter_info(ctx: Context = None) -> dict[str, Any]:
    """
    Get interpreter configuration info.

    Returns:
        Dict with model name, status, and settings
    """
    interpreter = ctx.request_context.lifespan_context.interpreter

    return {
        "success": True,
        "model": settings.github_model,
        "categories_loaded": len(interpreter._category_names),
        "embedding_model": settings.embedding_model,
        "debug_mode": settings.debug_mode,
    }


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("interpret://config")
def interpret_config_resource() -> str:
    """Expose interpreter configuration as a resource."""
    return json.dumps(
        {
            "github_model": settings.github_model,
            "embedding_model": settings.embedding_model,
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
        log_interpret("Running with stdio transport")
    else:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 5004
        log_interpret("Running with HTTP transport", {"url": "http://0.0.0.0:5004/mcp"})

    mcp.run(transport=transport)
