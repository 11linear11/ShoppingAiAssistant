"""
MCP Server for EQuIP DSL Generation
Port: 5005

This server handles:
- generate_dsl: Generate Elasticsearch DSL from natural language using EQuIP 3B model

EQuIP 3B is a fine-tuned model specifically designed for generating Elasticsearch DSL queries.
It runs on Ollama and is accessed via LangChain's Ollama integration.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from mcp.server.fastmcp import FastMCP

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
SERVER_NAME = "equip-server"
SERVER_PORT = 5005
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# EQuIP/Ollama configuration
EQUIP_BASE_URL = os.getenv("EQUIP_BASE_URL", "https://struggle-historical-athletics-developed.trycloudflare.com")
EQUIP_MODEL = os.getenv("EQUIP_MODEL", "EQuIP/EQuIP_3B")

# ═══════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(SERVER_NAME)


# ═══════════════════════════════════════════════════════════════
# Index Schema for EQuIP Context
# ═══════════════════════════════════════════════════════════════
INDEX_SCHEMA = """
Elasticsearch Index: shopping_products

Fields:
- product_name (text): Product name with features (e.g., "pink men shorts xl")
- brand_name (text): Brand name (e.g., "کاله", "سونی")
- category_name (keyword): Category name (e.g., "لبنیات", "پوشاک")
- price (long): Original price in Rials
- discount_price (long): Discounted price in Rials
- has_discount (boolean): Whether product has discount
- discount_percentage (float): Discount percentage (0-100)
- product_embedding (dense_vector, dim=768): Product embedding vector

Available sort fields: price, discount_price, discount_percentage
Available filters: category_name.keyword, brand_name.keyword, has_discount, price range
"""


# ═══════════════════════════════════════════════════════════════
# System Prompt for DSL Generation
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an Elasticsearch DSL generator for a shopping product search.

{schema}

RULES:
1. product_name goes in "must" clause with multi_match on product_name field
2. Categories go in "filter" clause with "terms" (OR logic) NOT "must" (AND logic)
3. Sort field can be: price, discount_price, discount_percentage
4. Always include "size": 50
5. Generate ONLY valid JSON DSL (no explanation, no markdown)

Example output for "product_name: laptop sort: price_asc":
{{"query": {{"bool": {{"must": [{{"multi_match": {{"query": "laptop", "fields": ["product_name^3", "brand_name"], "type": "best_fields"}}}}]}}}}, "sort": [{{"price": "asc"}}], "size": 50}}"""


# ═══════════════════════════════════════════════════════════════
# EQuIP Service Class with LangChain
# ═══════════════════════════════════════════════════════════════
class EQuIPService:
    """Handles DSL generation using EQuIP 3B model via LangChain + Ollama."""
    
    def __init__(self):
        logger.info("🔧 Initializing EQuIPService with LangChain...")
        
        self.base_url = EQUIP_BASE_URL
        self.model_name = EQUIP_MODEL
        
        # Initialize LangChain ChatOllama
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.1,
            num_predict=1024,
            timeout=120,
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{query}\n{sort_instruction}\n\nElasticsearch DSL:")
        ])
        
        # Create chain
        self.chain = self.prompt_template | self.llm
        
        logger.info(f"🤖 EQuIP URL: {self.base_url}")
        logger.info(f"🤖 EQuIP Model: {self.model_name}")
        logger.info("✅ EQuIPService initialized with LangChain")
    
    async def check_status(self) -> Dict[str, Any]:
        """Check if EQuIP/Ollama is online and model is available."""
        try:
            # Try a simple request to check connectivity
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = [m.get("name", "") for m in data.get("models", [])]
                        model_available = any(self.model_name in m for m in models)
                        return {
                            "status": "online",
                            "base_url": self.base_url,
                            "target_model": self.model_name,
                            "available_models": models,
                            "model_available": model_available
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"HTTP {response.status}",
                            "base_url": self.base_url
                        }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "base_url": self.base_url
            }
    
    async def generate_dsl(self, query: str, intent: str = "find_best_value") -> Dict[str, Any]:
        """
        Generate Elasticsearch DSL from natural language query using EQuIP 3B.
        
        Args:
            query: English natural language query (e.g., "product_name: laptop sort: price_asc")
            intent: Shopping intent for context
            
        Returns:
            Dict containing:
            - success: bool
            - dsl: Elasticsearch DSL query object
            - raw_response: Raw model response
        """
        logger.info(f"🔍 Generating DSL for: '{query}'")
        
        # Build sort instruction based on intent
        sort_instruction = self._get_sort_instruction(intent)
        
        try:
            # Invoke chain with LangChain
            response = await self.chain.ainvoke({
                "schema": INDEX_SCHEMA,
                "query": query,
                "sort_instruction": sort_instruction
            })
            
            # Extract content from response
            raw_response = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"📄 EQuIP raw response: {raw_response[:500]}...")
            
            # Extract DSL from response
            dsl = self._extract_dsl(raw_response)
            
            if dsl:
                logger.info("✅ DSL generated successfully")
                return {
                    "success": True,
                    "dsl": dsl,
                    "raw_response": raw_response
                }
            else:
                logger.warning("⚠️ Could not extract DSL, using fallback")
                return {
                    "success": False,
                    "error": "Could not extract DSL from response",
                    "dsl": self._fallback_dsl(query),
                    "raw_response": raw_response
                }
                
        except Exception as e:
            logger.error(f"❌ Error in generate_dsl: {e}")
            return {
                "success": False,
                "error": str(e),
                "dsl": self._fallback_dsl(query)
            }
    
    def _get_sort_instruction(self, intent: str) -> str:
        """Get sort instruction based on intent."""
        instructions = {
            "find_cheapest": "Sort results by price ascending (cheapest first).",
            "find_high_quality": "Sort by relevance and consider brand quality.",
            "find_best_value": "Consider both price and discount percentage.",
            "compare": "Return results sorted by relevance for comparison.",
            "find_by_feature": "Sort by relevance to find best matching features."
        }
        return instructions.get(intent, "")
    
    def _extract_dsl(self, response: str) -> Optional[Dict]:
        """Extract valid JSON DSL from model response."""
        try:
            # Clean response
            response = response.strip()
            
            # Try to parse the entire response as JSON
            try:
                dsl = json.loads(response)
                if self._is_valid_dsl(dsl):
                    return dsl
            except json.JSONDecodeError:
                pass
            
            # Try to find JSON block in response
            json_patterns = [
                r'```json\s*([\s\S]*?)\s*```',  # Markdown JSON block
                r'```\s*([\s\S]*?)\s*```',       # Generic code block
                r'(\{[\s\S]*\})',                 # Any JSON object
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response)
                if match:
                    json_str = match.group(1).strip()
                    try:
                        dsl = json.loads(json_str)
                        if self._is_valid_dsl(dsl):
                            return dsl
                    except json.JSONDecodeError:
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error extracting DSL: {e}")
            return None
    
    def _is_valid_dsl(self, dsl: Dict) -> bool:
        """Check if the DSL looks like a valid Elasticsearch query."""
        return "query" in dsl or "bool" in dsl or "match" in dsl or "multi_match" in dsl
    
    def _fallback_dsl(self, query: str) -> Dict:
        """Generate a simple fallback DSL query."""
        # Extract product name from query
        product_match = re.search(r'product_name:\s*([^\s]+(?:\s+[^\s]+)*?)(?:\s+sort:|$)', query, re.IGNORECASE)
        product_name = product_match.group(1).strip() if product_match else query
        
        # Extract sort from query
        sort_match = re.search(r'sort:\s*(\w+)_?(asc|desc)?', query, re.IGNORECASE)
        sort_field = sort_match.group(1) if sort_match else "price"
        sort_order = sort_match.group(2) if sort_match and sort_match.group(2) else "asc"
        
        return {
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": product_name,
                                "fields": ["product_name^3", "brand_name"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "sort": [{sort_field: sort_order}],
            "size": 50
        }


# ═══════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════
equip_service: EQuIPService = None


# ═══════════════════════════════════════════════════════════════
# MCP Server Setup
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on startup."""
    global equip_service
    logger.info(f"🚀 Starting {SERVER_NAME} on port {SERVER_PORT}...")
    equip_service = EQuIPService()
    logger.info(f"✅ {SERVER_NAME} ready!")
    yield {"equip_service": equip_service}
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
async def generate_dsl(query: str, intent: str = "find_best_value") -> str:
    """
    Generate Elasticsearch DSL from a structured query using EQuIP 3B model.
    
    Args:
        query: Structured query in format "product_name: <product> sort: <field>_<order>"
               Example: "product_name: laptop sort: price_asc"
        intent: Shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        
    Returns:
        JSON string with:
        - success: bool
        - dsl: Elasticsearch DSL query object
        - raw_response: Raw model response (if successful)
        - error: Error message (if failed)
    """
    global equip_service
    logger.debug(f"📥 generate_dsl called with: query='{query}', intent='{intent}'")
    
    try:
        result = await equip_service.generate_dsl(query, intent)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error in generate_dsl tool: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "dsl": equip_service._fallback_dsl(query) if equip_service else {}
        }, ensure_ascii=False)


@mcp.tool()
async def check_equip_status() -> str:
    """
    Check if EQuIP/Ollama server is online and model is available.
    
    Returns:
        JSON string with:
        - status: "online" or "offline" or "error"
        - base_url: Ollama server URL
        - target_model: Expected model name
        - available_models: List of available models
        - model_available: Whether target model is available
    """
    global equip_service
    logger.debug("📥 check_equip_status called")
    
    try:
        result = await equip_service.check_status()
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error checking status: {e}")
        return json.dumps({
            "status": "error",
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
    
    logger.info(f"🚀 Starting {SERVER_NAME} MCP Server with LangChain...")
    logger.info(f"📡 Port: {SERVER_PORT}")
    logger.info(f"🔗 Ollama URL: {EQUIP_BASE_URL}")
    logger.info(f"🤖 Model: {EQUIP_MODEL}")
    logger.info(f"🔧 Debug Mode: {DEBUG_MODE}")
    
    # Run with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
