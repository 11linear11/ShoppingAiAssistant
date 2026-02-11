"""
Embedding MCP Server (MCP Protocol Version)
Port: 5003

Provides text embedding generation using multilingual-e5-base model.
Supports caching of embeddings for performance optimization.

This is the MCP protocol version using FastMCP SDK.
"""

import asyncio
import hashlib
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.pipeline_logger import log_embed


# ============================================================================
# Configuration
# ============================================================================


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_name: str = Field(
        default="intfloat/multilingual-e5-base",
        alias="EMBEDDING_MODEL",
    )
    device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    max_seq_length: int = Field(default=512, alias="EMBEDDING_MAX_SEQ_LENGTH")
    batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")

    # Debug mode - disables caching
    debug_mode: bool = Field(default=False, alias="DEBUG_MODE")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()


# Shared singleton across HTTP sessions to avoid re-loading heavy models.
_shared_embedding_manager: Optional["EmbeddingManager"] = None
_shared_embedding_lock = asyncio.Lock()


# ============================================================================
# Embedding Manager
# ============================================================================


class EmbeddingManager:
    """
    Manager for text embeddings using multilingual-e5-base.

    Features:
    - Single and batch embedding generation
    - Embedding normalization
    - Cosine similarity calculation
    - In-memory caching
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize the embedding manager.

        Args:
            model_name: HuggingFace model name
            device: Device to run model on (cpu/cuda)
        """
        log_embed("Loading embedding model", {"model": model_name, "device": device})
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = settings.max_seq_length
        self.dimensions = self.model.get_sentence_embedding_dimension()
        log_embed("Embedding model loaded", {"dimensions": self.dimensions})

        # In-memory cache for embeddings
        self._cache: dict[str, list[float]] = {}

    def generate_embedding(
        self,
        text: str,
        normalize: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[float], bool]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            normalize: Whether to L2 normalize the embedding
            use_cache: Whether to use in-memory cache

        Returns:
            Tuple of (embedding vector, from_cache flag)
        """
        # In DEBUG_MODE, disable caching
        if settings.debug_mode:
            use_cache = False

        # Check cache
        cache_key = self._get_cache_key(text, normalize)
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key], True

        # Generate embedding
        # For E5 models, we need to add prefix for better performance
        prefixed_text = f"query: {text}"
        embedding = self.model.encode(
            prefixed_text,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        # Convert to list and store in cache
        embedding_list = embedding.tolist()
        if use_cache:
            self._cache[cache_key] = embedding_list

        return embedding_list, False

    def generate_embeddings_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[list[float]], int]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            normalize: Whether to L2 normalize embeddings
            use_cache: Whether to use in-memory cache

        Returns:
            Tuple of (list of embeddings, cache hit count)
        """
        # In DEBUG_MODE, disable caching
        if settings.debug_mode:
            use_cache = False

        results = []
        cache_hits = 0
        texts_to_encode = []
        indices_to_encode = []

        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text, normalize)
            if use_cache and cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
                cache_hits += 1
            else:
                texts_to_encode.append(text)
                indices_to_encode.append(i)
                results.append((i, None))

        # Generate embeddings for uncached texts
        if texts_to_encode:
            # Add E5 prefix
            prefixed_texts = [f"query: {t}" for t in texts_to_encode]
            embeddings = self.model.encode(
                prefixed_texts,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                batch_size=settings.batch_size,
            )

            # Store in cache and update results
            for idx, (original_idx, text) in enumerate(
                zip(indices_to_encode, texts_to_encode)
            ):
                embedding_list = embeddings[idx].tolist()
                cache_key = self._get_cache_key(text, normalize)
                if use_cache:
                    self._cache[cache_key] = embedding_list

                # Update result at original index
                for j, (result_idx, _) in enumerate(results):
                    if result_idx == original_idx:
                        results[j] = (original_idx, embedding_list)
                        break

        # Sort by original index and extract embeddings
        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results], cache_hits

    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        normalize: bool = True,
    ) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            normalize: Whether to normalize embeddings

        Returns:
            Cosine similarity score [-1, 1]
        """
        emb1, _ = self.generate_embedding(text1, normalize)
        emb2, _ = self.generate_embedding(text2, normalize)

        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        # If already normalized, dot product gives cosine similarity
        if normalize:
            return float(np.dot(vec1, vec2))

        # Otherwise calculate full cosine similarity
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def clear_cache(self) -> int:
        """Clear in-memory cache."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_cache_size(self) -> int:
        """Get number of cached embeddings."""
        return len(self._cache)

    @staticmethod
    def _get_cache_key(text: str, normalize: bool) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{text_hash}_{normalize}"


# ============================================================================
# MCP Server Setup with Lifespan
# ============================================================================


@dataclass
class AppContext:
    """Application context with embedding manager."""

    embedding_manager: EmbeddingManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Manage application lifecycle.

    Loads the embedding model on startup and provides it to tools.
    """
    global _shared_embedding_manager

    if _shared_embedding_manager is None:
        async with _shared_embedding_lock:
            if _shared_embedding_manager is None:
                log_embed("Starting embedding server", {"model": settings.model_name})
                _shared_embedding_manager = EmbeddingManager(
                    model_name=settings.model_name,
                    device=settings.device,
                )
                log_embed("Embedding server started", {"model": settings.model_name})
                if settings.debug_mode:
                    log_embed("DEBUG mode active: embedding caching disabled")
    else:
        log_embed("Embedding server warm instance reused", {"model": settings.model_name})

    try:
        yield AppContext(embedding_manager=_shared_embedding_manager)
    finally:
        # Keep warm singleton alive across sessions/process lifespan.
        pass


# Create MCP server
mcp = FastMCP(
    "Embedding Server",
    lifespan=app_lifespan,
)

# Allow Docker internal hostnames for transport security
mcp.settings.transport_security.allowed_hosts.extend([
    "embedding:*",
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
    "http://embedding:*",
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
def generate_embedding(
    text: str,
    normalize: bool = True,
    use_cache: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Generate embedding for a single text.

    Uses multilingual-e5-base model for high-quality
    multilingual embeddings supporting Persian and English.

    Args:
        text: Input text to embed (max 8192 chars)
        normalize: Whether to L2 normalize the embedding
        use_cache: Whether to use in-memory cache

    Returns:
        Dict with embedding vector and metadata
    """
    if not text or len(text) > 8192:
        return {"success": False, "error": "Text must be 1-8192 characters"}

    manager = ctx.request_context.lifespan_context.embedding_manager

    embedding, from_cache = manager.generate_embedding(
        text=text,
        normalize=normalize,
        use_cache=use_cache,
    )

    return {
        "success": True,
        "text": text,
        "embedding": embedding,
        "dimensions": len(embedding),
        "normalized": normalize,
        "from_cache": from_cache,
    }


@mcp.tool()
def generate_embeddings_batch(
    texts: list[str],
    normalize: bool = True,
    use_cache: bool = True,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Generate embeddings for multiple texts.

    More efficient than calling generate_embedding multiple times
    due to batch processing.

    Args:
        texts: List of input texts (max 100)
        normalize: Whether to L2 normalize embeddings
        use_cache: Whether to use in-memory cache

    Returns:
        Dict with list of embeddings and metadata
    """
    if not texts or len(texts) > 100:
        return {"success": False, "error": "texts must have 1-100 items"}

    manager = ctx.request_context.lifespan_context.embedding_manager

    embeddings, cache_hits = manager.generate_embeddings_batch(
        texts=texts,
        normalize=normalize,
        use_cache=use_cache,
    )

    return {
        "success": True,
        "embeddings": embeddings,
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "count": len(embeddings),
        "normalized": normalize,
        "cache_hits": cache_hits,
    }


@mcp.tool()
def calculate_similarity(
    text1: str,
    text2: str,
    ctx: Context = None,
) -> dict[str, Any]:
    """
    Calculate cosine similarity between two texts.

    Returns a score between -1 and 1, where:
    - 1 = identical meaning
    - 0 = unrelated
    - -1 = opposite meaning

    Args:
        text1: First text
        text2: Second text

    Returns:
        Dict with similarity score
    """
    if not text1 or not text2:
        return {"success": False, "error": "Both texts are required"}

    manager = ctx.request_context.lifespan_context.embedding_manager

    similarity = manager.calculate_similarity(text1=text1, text2=text2)

    return {
        "success": True,
        "text1": text1,
        "text2": text2,
        "similarity": round(similarity, 6),
    }


@mcp.tool()
def get_embedding_cache_stats(ctx: Context = None) -> dict[str, Any]:
    """
    Get embedding cache statistics.

    Returns:
        Dict with cache size and model info
    """
    manager = ctx.request_context.lifespan_context.embedding_manager

    return {
        "success": True,
        "cache_size": manager.get_cache_size(),
        "model": settings.model_name,
        "dimensions": manager.dimensions,
    }


@mcp.tool()
def clear_embedding_cache(ctx: Context = None) -> dict[str, Any]:
    """
    Clear embedding cache.

    Returns:
        Dict with number of cleared entries
    """
    manager = ctx.request_context.lifespan_context.embedding_manager

    cleared = manager.clear_cache()
    return {
        "success": True,
        "cleared": cleared,
    }


@mcp.tool()
def get_model_info(ctx: Context = None) -> dict[str, Any]:
    """
    Get embedding model information.

    Returns:
        Dict with model name, dimensions, and settings
    """
    manager = ctx.request_context.lifespan_context.embedding_manager

    return {
        "success": True,
        "model_name": settings.model_name,
        "dimensions": manager.dimensions,
        "max_seq_length": settings.max_seq_length,
        "device": settings.device,
        "debug_mode": settings.debug_mode,
    }


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("embedding://model/info")
def model_info_resource(ctx: Context = None) -> str:
    """Expose model information as a resource."""
    import json

    return json.dumps(
        {
            "model_name": settings.model_name,
            "max_seq_length": settings.max_seq_length,
            "device": settings.device,
            "batch_size": settings.batch_size,
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
        log_embed("Running with stdio transport")
    else:
        # Configure host/port via settings
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = 5003
        log_embed("Running with HTTP transport", {"url": "http://0.0.0.0:5003/mcp"})

    mcp.run(transport=transport)
