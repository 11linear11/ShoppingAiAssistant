"""
MCP Server for Embeddings and Category Classification
Port: 5003

This server handles:
- get_embedding: Generate embeddings for text
- classify_categories: Find matching categories for a query
"""

import os
import sys
import json
import logging
import numpy as np
from typing import List, Dict, Any
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
SERVER_NAME = "embedding-server"
SERVER_PORT = 5003
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Path to category embeddings
CATEGORY_EMBEDDINGS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "full_category_embeddings.json"
)

# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(SERVER_NAME)


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))


def load_category_embeddings() -> Dict[str, List[float]]:
    """Load pre-computed category embeddings from JSON file."""
    try:
        with open(CATEGORY_EMBEDDINGS_PATH, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)
        logger.info(f"✅ Loaded {len(embeddings)} category embeddings")
        return embeddings
    except FileNotFoundError:
        logger.error(f"❌ Category embeddings file not found: {CATEGORY_EMBEDDINGS_PATH}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing category embeddings JSON: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
# Embedding Service Class
# ═══════════════════════════════════════════════════════════════
class EmbeddingService:
    """Handles text embeddings and category classification."""
    
    def __init__(self):
        logger.info("🔧 Initializing EmbeddingService...")
        
        # Load SentenceTransformer model
        model_name = 'intfloat/multilingual-e5-base'
        logger.info(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("✅ Embedding model loaded")
        
        # Load category embeddings
        self.category_embeddings = load_category_embeddings()
        logger.info(f"✅ Loaded {len(self.category_embeddings)} category embeddings")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.model.encode([text])[0]
        return embedding.tolist()
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.model.encode(texts)
        return [emb.tolist() for emb in embeddings]
    
    def classify_categories(
        self, 
        query: str, 
        top_k: int = 3, 
        threshold: float = 0.25
    ) -> List[Dict[str, Any]]:
        """
        Find top-k categories most similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of top categories to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of dicts with 'category' and 'similarity' keys
        """
        # Generate query embedding
        query_vec = self.get_embedding(query)
        
        # Calculate similarity with all categories
        scores = []
        for cat, cat_emb in self.category_embeddings.items():
            sim = cosine_similarity(query_vec, cat_emb)
            if sim >= threshold:
                scores.append({"category": cat, "similarity": round(sim, 4)})
        
        # Sort by similarity descending
        scores.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        return scores[:top_k]


# ═══════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════
embedding_service: EmbeddingService = None


# ═══════════════════════════════════════════════════════════════
# MCP Server Setup
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on startup."""
    global embedding_service
    logger.info(f"🚀 Starting {SERVER_NAME} on port {SERVER_PORT}...")
    embedding_service = EmbeddingService()
    logger.info(f"✅ {SERVER_NAME} ready!")
    yield {"embedding_service": embedding_service}
    logger.info(f"👋 Shutting down {SERVER_NAME}...")


# Create MCP server
mcp = FastMCP(
    SERVER_NAME,
    lifespan=lifespan
)


# ═══════════════════════════════════════════════════════════════
# MCP Tools
# ═══════════════════════════════════════════════════════════════
@mcp.tool()
def get_embedding(text: str) -> str:
    """
    Generate embedding vector for a text.
    
    Args:
        text: The text to generate embedding for
        
    Returns:
        JSON string containing the embedding vector
    """
    global embedding_service
    logger.debug(f"📥 get_embedding called with text: '{text[:50]}...'")
    
    try:
        embedding = embedding_service.get_embedding(text)
        result = {
            "success": True,
            "text": text,
            "embedding": embedding,
            "dimension": len(embedding)
        }
        logger.debug(f"✅ Embedding generated (dim={len(embedding)})")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


@mcp.tool()
def classify_categories(query: str, top_k: int = 3) -> str:
    """
    Find the most relevant product categories for a search query.
    
    Args:
        query: The search query to classify
        top_k: Number of top categories to return (default: 3)
        
    Returns:
        JSON string with list of matching categories and their similarity scores
    """
    global embedding_service
    logger.debug(f"📥 classify_categories called with query: '{query}'")
    
    try:
        categories = embedding_service.classify_categories(query, top_k=top_k)
        result = {
            "success": True,
            "query": query,
            "categories": categories,
            "count": len(categories)
        }
        logger.debug(f"✅ Found {len(categories)} categories")
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error classifying categories: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════
# Configure mount path
mcp.settings.streamable_http_path = "/"

# Create ASGI app for uvicorn
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting {SERVER_NAME} MCP Server...")
    logger.info(f"📡 Port: {SERVER_PORT}")
    logger.info(f"🔧 Debug Mode: {DEBUG_MODE}")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
