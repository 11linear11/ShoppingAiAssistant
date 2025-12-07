"""
MCP Server for DSL Processing
Port: 5006

This server handles:
- process_dsl: Transform English DSL to Persian and add semantic search component

This server is the bridge between EQuIP (English DSL) and Elasticsearch (Persian data).
It handles:
1. Token replacement: English words → Persian equivalents
2. Adding semantic search (embedding-based) to the query
3. Fixing category filters with Persian names
4. Ensuring proper query structure for hybrid search
"""

import os
import json
import re
import logging
import copy
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
SERVER_NAME = "dsl-processor-server"
SERVER_PORT = 5006
EMBEDDING_SERVER_URL = "http://localhost:5003"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(SERVER_NAME)


# ═══════════════════════════════════════════════════════════════
# DSL Processor Service Class
# ═══════════════════════════════════════════════════════════════
class DSLProcessorService:
    """Handles DSL transformation from English to Persian with semantic search."""
    
    def __init__(self):
        logger.info("🔧 Initializing DSLProcessorService...")
        logger.info("✅ DSLProcessorService initialized")
    
    async def call_embedding_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call the embedding server's MCP tool."""
        try:
            async with streamablehttp_client(f"{EMBEDDING_SERVER_URL}/") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    if result.content:
                        return json.loads(result.content[0].text)
                    return {"success": False, "error": "No content"}
        except Exception as e:
            logger.error(f"❌ Error calling embedding server: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding from embedding server."""
        try:
            result = await self.call_embedding_tool("get_embedding", {"text": text})
            if result.get("success"):
                return result.get("embedding", [])
            return []
        except Exception as e:
            logger.error(f"❌ Error getting embedding: {e}")
            return []
    
    async def process(
        self,
        dsl: Dict,
        token_mapping: Dict[str, str],
        persian_full_query: str,
        categories_fa: List[str],
        add_semantic: bool = True,
        semantic_boost: float = 2.0,
        bm25_boost: float = 1.5,
        phrase_boost: float = 3.0
    ) -> Dict[str, Any]:
        """
        Transform English DSL to Persian hybrid DSL.
        
        Args:
            dsl: Original Elasticsearch DSL (with English terms)
            token_mapping: Dictionary mapping English words to Persian equivalents
            persian_full_query: Full Persian product description for exact matching
            categories_fa: List of Persian category names
            add_semantic: Whether to add semantic search component
            semantic_boost: Boost factor for semantic search
            bm25_boost: Boost factor for BM25 search
            phrase_boost: Boost factor for phrase matching
            
        Returns:
            Dict containing:
            - success: bool
            - dsl: Transformed DSL with Persian terms and semantic search
            - original_dsl: Original input DSL
        """
        logger.info(f"🔄 Processing DSL with {len(token_mapping)} token mappings")
        logger.debug(f"📝 persian_full_query: '{persian_full_query}'")
        logger.debug(f"📁 categories_fa: {categories_fa}")
        
        try:
            # Deep copy to avoid modifying original
            processed_dsl = copy.deepcopy(dsl)
            
            # Step 1: Replace English tokens with Persian equivalents
            processed_dsl = self._replace_tokens(processed_dsl, token_mapping)
            logger.debug("✅ Token replacement complete")
            
            # Step 2: Ensure proper bool query structure
            processed_dsl = self._ensure_bool_structure(processed_dsl)
            logger.debug("✅ Bool structure ensured")
            
            # Step 3: Add match_phrase for full Persian query (highest priority)
            processed_dsl = self._add_phrase_match(processed_dsl, persian_full_query, phrase_boost)
            logger.debug("✅ Phrase match added")
            
            # Step 4: Add BM25 multi-match for Persian query
            processed_dsl = self._add_bm25_match(processed_dsl, persian_full_query, bm25_boost)
            logger.debug("✅ BM25 match added")
            
            # Step 5: Add semantic search if requested
            if add_semantic:
                embedding = await self.get_embedding(persian_full_query)
                if embedding:
                    processed_dsl = self._add_semantic_search(processed_dsl, embedding, semantic_boost)
                    logger.debug("✅ Semantic search added")
                else:
                    logger.warning("⚠️ Could not get embedding, skipping semantic search")
            
            # Step 6: Fix category filters with Persian names
            if categories_fa:
                processed_dsl = self._fix_category_filters(processed_dsl, categories_fa)
                logger.debug("✅ Category filters fixed")
            
            # Step 7: Ensure minimum_should_match
            if "query" in processed_dsl and "bool" in processed_dsl["query"]:
                processed_dsl["query"]["bool"]["minimum_should_match"] = 1
            
            # Step 8: Ensure reasonable size
            if "size" not in processed_dsl:
                processed_dsl["size"] = 50
            
            logger.info("✅ DSL processing complete")
            
            return {
                "success": True,
                "dsl": processed_dsl,
                "original_dsl": dsl
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing DSL: {e}")
            return {
                "success": False,
                "error": str(e),
                "dsl": dsl,
                "original_dsl": dsl
            }
    
    def _replace_tokens(self, dsl: Dict, token_mapping: Dict[str, str]) -> Dict:
        """Replace English tokens with Persian equivalents in DSL."""
        if not token_mapping:
            return dsl
        
        dsl_str = json.dumps(dsl, ensure_ascii=False)
        
        # Sort by length (longer first) to avoid partial replacements
        sorted_mappings = sorted(token_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for en_token, fa_token in sorted_mappings:
            # Case-insensitive replacement with word boundaries
            pattern = rf'\b{re.escape(en_token)}\b'
            dsl_str = re.sub(pattern, fa_token, dsl_str, flags=re.IGNORECASE)
        
        return json.loads(dsl_str)
    
    def _ensure_bool_structure(self, dsl: Dict) -> Dict:
        """Ensure DSL has proper bool query structure."""
        if "query" not in dsl:
            dsl = {"query": {"bool": {"should": []}}, **dsl}
        
        query_part = dsl.get("query", {})
        
        # If query is not a bool query, wrap it
        if "bool" not in query_part:
            if query_part:
                dsl["query"] = {"bool": {"should": [query_part]}}
            else:
                dsl["query"] = {"bool": {"should": []}}
        
        # Ensure should array exists
        bool_query = dsl["query"]["bool"]
        if "should" not in bool_query:
            bool_query["should"] = []
        
        return dsl
    
    def _add_phrase_match(self, dsl: Dict, persian_query: str, boost: float) -> Dict:
        """Add match_phrase for full Persian query."""
        phrase_clause = {
            "match_phrase": {
                "product_name": {
                    "query": persian_query,
                    "boost": boost
                }
            }
        }
        
        # Insert at the beginning for highest priority
        dsl["query"]["bool"]["should"].insert(0, phrase_clause)
        
        return dsl
    
    def _add_bm25_match(self, dsl: Dict, persian_query: str, boost: float) -> Dict:
        """Add BM25 multi-match for Persian query."""
        bm25_clause = {
            "multi_match": {
                "query": persian_query,
                "fields": ["product_name^2", "brand_name", "category_name"],
                "type": "best_fields",
                "operator": "or",
                "boost": boost
            }
        }
        
        dsl["query"]["bool"]["should"].append(bm25_clause)
        
        return dsl
    
    def _add_semantic_search(self, dsl: Dict, embedding: List[float], boost: float) -> Dict:
        """Add semantic search using script_score."""
        semantic_clause = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                    "params": {"query_vector": embedding}
                },
                "boost": boost
            }
        }
        
        dsl["query"]["bool"]["should"].append(semantic_clause)
        
        return dsl
    
    def _fix_category_filters(self, dsl: Dict, categories_fa: List[str]) -> Dict:
        """Fix category filters with Persian category names."""
        bool_query = dsl["query"]["bool"]
        
        # Initialize filter array if needed
        if "filter" not in bool_query:
            bool_query["filter"] = []
        elif not isinstance(bool_query["filter"], list):
            bool_query["filter"] = [bool_query["filter"]]
        
        # Remove any existing category filters
        bool_query["filter"] = [
            f for f in bool_query["filter"]
            if not self._is_category_filter(f)
        ]
        
        # Add Persian category filter
        if categories_fa:
            bool_query["filter"].append({
                "terms": {
                    "category_name.keyword": categories_fa
                }
            })
        
        return dsl
    
    def _is_category_filter(self, filter_clause: Dict) -> bool:
        """Check if a filter clause is a category filter."""
        filter_str = json.dumps(filter_clause).lower()
        return "category" in filter_str


# ═══════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════
dsl_processor_service: DSLProcessorService = None


# ═══════════════════════════════════════════════════════════════
# MCP Server Setup
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on startup."""
    global dsl_processor_service
    logger.info(f"🚀 Starting {SERVER_NAME} on port {SERVER_PORT}...")
    dsl_processor_service = DSLProcessorService()
    logger.info(f"✅ {SERVER_NAME} ready!")
    yield {"dsl_processor_service": dsl_processor_service}
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
async def process_dsl(
    dsl: str,
    token_mapping: str,
    persian_full_query: str,
    categories_fa: str = "[]",
    add_semantic: bool = True,
    semantic_boost: float = 2.0,
    bm25_boost: float = 1.5,
    phrase_boost: float = 3.0
) -> str:
    """
    Transform English DSL to Persian hybrid DSL with semantic search.
    
    Args:
        dsl: JSON string of original Elasticsearch DSL (with English terms)
        token_mapping: JSON string of dictionary mapping English words to Persian equivalents
        persian_full_query: Full Persian product description for exact matching
        categories_fa: JSON string of list of Persian category names
        add_semantic: Whether to add semantic search component
        semantic_boost: Boost factor for semantic search (default: 2.0)
        bm25_boost: Boost factor for BM25 search (default: 1.5)
        phrase_boost: Boost factor for phrase matching (default: 3.0)
        
    Returns:
        JSON string with:
        - success: Whether processing was successful
        - dsl: Transformed DSL ready for Elasticsearch
        - original_dsl: Original input DSL
        - error: Error message if failed
    """
    global dsl_processor_service
    logger.debug(f"📥 process_dsl called")
    
    try:
        # Parse JSON inputs
        dsl_dict = json.loads(dsl) if isinstance(dsl, str) else dsl
        token_mapping_dict = json.loads(token_mapping) if isinstance(token_mapping, str) else token_mapping
        categories_fa_list = json.loads(categories_fa) if isinstance(categories_fa, str) else categories_fa
        
        result = await dsl_processor_service.process(
            dsl=dsl_dict,
            token_mapping=token_mapping_dict,
            persian_full_query=persian_full_query,
            categories_fa=categories_fa_list,
            add_semantic=add_semantic,
            semantic_boost=semantic_boost,
            bm25_boost=bm25_boost,
            phrase_boost=phrase_boost
        )
        
        return json.dumps(result, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        return json.dumps({
            "success": False,
            "error": f"JSON parsing error: {str(e)}",
            "dsl": {},
            "original_dsl": {}
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error processing DSL: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "dsl": {},
            "original_dsl": {}
        }, ensure_ascii=False)


@mcp.tool()
async def create_hybrid_dsl(
    persian_query: str,
    categories_fa: str = "[]",
    intent: str = "find_best_value",
    price_sensitivity: float = 0.5
) -> str:
    """
    Create a hybrid DSL directly from Persian query (without EQuIP).
    Useful as a fallback or for simple queries.
    
    Args:
        persian_query: Persian product search query
        categories_fa: JSON string of list of Persian category names
        intent: Shopping intent for sorting
        price_sensitivity: How price-conscious (0-1)
        
    Returns:
        JSON string with hybrid DSL (BM25 + Semantic)
    """
    global dsl_processor_service
    logger.debug(f"📥 create_hybrid_dsl called for: '{persian_query}'")
    
    try:
        categories_list = json.loads(categories_fa) if isinstance(categories_fa, str) else categories_fa
        
        # Get embedding
        embedding = await dsl_processor_service.get_embedding(persian_query)
        
        # Build hybrid DSL
        dsl = {
            "query": {
                "bool": {
                    "should": [
                        # Phrase match (highest priority)
                        {
                            "match_phrase": {
                                "product_name": {
                                    "query": persian_query,
                                    "boost": 3.0
                                }
                            }
                        },
                        # BM25 multi-match
                        {
                            "multi_match": {
                                "query": persian_query,
                                "fields": ["product_name^2", "brand_name", "category_name"],
                                "type": "best_fields",
                                "boost": 1.5
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 50
        }
        
        # Add semantic search if embedding available
        if embedding:
            dsl["query"]["bool"]["should"].append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    },
                    "boost": 2.0
                }
            })
        
        # Add category filter
        if categories_list:
            dsl["query"]["bool"]["filter"] = [{
                "terms": {
                    "category_name.keyword": categories_list
                }
            }]
        
        # Add sort based on intent
        if intent == "find_cheapest":
            dsl["sort"] = [{"price": "asc"}, "_score"]
        elif intent == "find_high_quality":
            dsl["sort"] = ["_score"]
        else:
            dsl["sort"] = ["_score", {"discount_percentage": "desc"}]
        
        return json.dumps({
            "success": True,
            "dsl": dsl
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"❌ Error creating hybrid DSL: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "dsl": {}
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
    logger.info(f"🔗 Embedding Server: {EMBEDDING_SERVER_URL}")
    logger.info(f"🔧 Debug Mode: {DEBUG_MODE}")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
