"""
MCP Server for Query Interpretation
Port: 5004

This server handles:
- interpret_query: Analyze user shopping intent and prepare structured output for EQuIP DSL generation

New Architecture:
- Generates English query for EQuIP 3B model
- Provides token_mapping for English→Persian translation in DSL
- Keeps Persian full query for BM25 and semantic search
"""

import os
import sys
import json
import re
import logging
import asyncio
from typing import Dict, Any, List
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
SERVER_NAME = "interpret-server"
SERVER_PORT = 5004
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
# LLM Service Class
# ═══════════════════════════════════════════════════════════════
class InterpretService:
    """Handles query interpretation using LLM for EQuIP DSL generation."""
    
    def __init__(self):
        logger.info("🔧 Initializing InterpretService...")
        
        # Initialize LLM - Using NVIDIA NIM
        nvidia_models = [
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-8b-instruct",
        ]
        model = os.getenv("NVIDIA_MODEL", nvidia_models[0])
        logger.info(f"🤖 Using NVIDIA model: {model}")
        
        self.llm = ChatNVIDIA(
            model=model,
            api_key=os.getenv("api_key"),
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            max_tokens=2000
        )
        
        logger.info("✅ InterpretService initialized")
    
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
    
    async def classify_categories(self, query: str, top_k: int = 3) -> list:
        """Call embedding server to classify categories."""
        try:
            result = await self.call_embedding_tool("classify_categories", {
                "query": query,
                "top_k": top_k
            })
            if result.get("success"):
                return result.get("categories", [])
            return []
        except Exception as e:
            logger.error(f"❌ Error classifying categories: {e}")
            return []
    
    async def interpret(self, query: str) -> Dict[str, Any]:
        """
        Analyze user shopping intent and prepare structured output for EQuIP.
        
        Args:
            query: User's shopping query in Persian
            
        Returns:
            Dict with:
            - equip_prompt: English natural language query for EQuIP
            - persian_full_query: Full Persian product keywords for search
            - token_mapping: English -> Persian word mapping for DSL translation
            - categories_fa: Persian category names
            - intent: Shopping intent
            - price_sensitivity: 0-1
            - quality_sensitivity: 0-1
        """
        logger.info(f"🧠 Interpreting query: '{query}'")
        
        # Default values
        equip_prompt = ""
        persian_full_query = query
        token_mapping = {}
        intent = "find_best_value"
        price_sens = 0.5
        quality_sens = 0.5
        
        # Build prompt for LLM
        # NOTE: Categories are NOT determined by LLM - they come from classify_categories after this step
        prompt = f"""You are a bilingual shopping query interpreter (Persian → English).

Your task is to analyze the user's Persian shopping query and create:
1. A STRUCTURED English query for EQuIP (Elasticsearch DSL generator)
2. Extract Persian product keywords that should remain in Persian
3. Create a mapping between English words and their Persian equivalents

IMPORTANT RULES:
- The dataset has Persian product names like: "شورت صورتی مردانه xl", "شیر کم چرب کاله"
- Product names contain all features together (color, size, type, brand)
- You must keep the FULL Persian product description for search
- DO NOT include category in equip_prompt - categories are determined separately

Output ONLY valid JSON in this EXACT format:
{{
    "equip_prompt": "product_name: <english product name with features> sort: <sort field> filter: <optional filters>",
    "persian_full_query": "کلمات کلیدی فارسی محصول با تمام ویژگی‌ها",
    "token_mapping": {{
        "english_word": "معادل_فارسی"
    }},
    "intent": "find_cheapest|find_best_value|find_high_quality|find_by_feature|compare",
    "price_sensitivity": 0.0-1.0,
    "quality_sensitivity": 0.0-1.0
}}

### equip_prompt Structure:
- product_name: English product name with all attributes (color, size, brand, type)
- sort: price_asc, price_desc, relevance, quality (based on user intent)
- filter: optional filters like brand, size, color (only if explicitly mentioned)

### Intent Types:
- find_cheapest: user wants the cheapest option ("ارزان", "ارزان‌ترین") → sort: price_asc
- find_best_value: user wants best price/quality ratio ("مقرون‌به‌صرفه") → sort: relevance
- find_high_quality: user prioritizes quality ("کیفیت بالا", "محکم") → sort: quality
- compare: user wants to compare options ("مقایسه") → sort: relevance
- find_by_feature: user mentions specific features (color, size, etc.) → sort: relevance

### Price Sensitivity:
- 1.0: words like "ارزون", "ارزان‌ترین", "مقرون‌به‌صرفه"
- 0.5: indirect or unclear
- 0.0: no price-related mention

### Quality Sensitivity:
- 1.0: words like "کیفیت بالا", "محکم", "مارک‌دار"
- 0.5: unclear
- 0.0: no quality mention

### Examples:

Input: "شورت صورتی مردانه xl میخوام ارزون"
Output: {{
    "equip_prompt": "product_name: pink men shorts xl sort: price_asc",
    "persian_full_query": "شورت صورتی مردانه xl",
    "token_mapping": {{
        "pink": "صورتی",
        "men": "مردانه",
        "shorts": "شورت",
        "xl": "xl"
    }},
    "intent": "find_cheapest",
    "price_sensitivity": 1.0,
    "quality_sensitivity": 0.0
}}

Input: "شیر ارزون میخوام"
Output: {{
    "equip_prompt": "product_name: milk sort: price_asc",
    "persian_full_query": "شیر",
    "token_mapping": {{
        "milk": "شیر"
    }},
    "intent": "find_cheapest",
    "price_sensitivity": 1.0,
    "quality_sensitivity": 0.0
}}

Input: "من سردمه"
Output: {{
    "equip_prompt": "product_name: jacket coat sweater warm clothing sort: relevance",
    "persian_full_query": "کاپشن ژاکت پالتو",
    "token_mapping": {{
        "jacket": "کاپشن",
        "coat": "پالتو",
        "sweater": "ژاکت"
    }},
    "intent": "find_by_feature",
    "price_sensitivity": 0.5,
    "quality_sensitivity": 0.5
}}

Input: "هدفون سونی باکیفیت"
Output: {{
    "equip_prompt": "product_name: Sony headphones sort: quality filter: brand=Sony",
    "persian_full_query": "هدفون سونی",
    "token_mapping": {{
        "headphones": "هدفون",
        "Sony": "سونی"
    }},
    "intent": "find_high_quality",
    "price_sensitivity": 0.0,
    "quality_sensitivity": 1.0
}}

Input: "لپتاپ ایسوس گیمینگ"
Output: {{
    "equip_prompt": "product_name: ASUS gaming laptop sort: relevance filter: brand=ASUS",
    "persian_full_query": "لپتاپ ایسوس گیمینگ",
    "token_mapping": {{
        "laptop": "لپتاپ",
        "ASUS": "ایسوس",
        "gaming": "گیمینگ"
    }},
    "intent": "find_by_feature",
    "price_sensitivity": 0.5,
    "quality_sensitivity": 0.5
}}

-----------------------------------------------
User Query: {query}

Output JSON:"""

        try:
            # Invoke LLM
            logger.debug("💭 Invoking LLM for intent analysis...")
            response = self.llm.invoke(prompt)
            
            # Extract content from response
            response_text = ""
            if hasattr(response, 'content') and response.content:
                response_text = response.content.strip()
            elif hasattr(response, 'text') and response.text:
                response_text = response.text.strip()
            elif isinstance(response, dict):
                response_text = response.get('content', '') or response.get('text', '') or str(response)
            elif isinstance(response, str):
                response_text = response.strip()
            else:
                response_text = str(response).strip()
            
            logger.debug(f"📄 LLM response: '{response_text[:300] if response_text else 'EMPTY'}'")
            
            if response_text:
                # Extract JSON from response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        equip_prompt = parsed.get("equip_prompt", "")
                        persian_full_query = parsed.get("persian_full_query", query)
                        token_mapping = parsed.get("token_mapping", {})
                        intent = parsed.get("intent", intent)
                        price_sens = float(parsed.get("price_sensitivity", price_sens))
                        quality_sens = float(parsed.get("quality_sensitivity", quality_sens))
                        
                        logger.info(f"🎯 Parsed: equip_prompt='{equip_prompt[:50]}...', persian='{persian_full_query}'")
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Failed to parse LLM JSON: {e}")
            
            logger.info(f"🎯 LLM extracted persian_full_query: '{persian_full_query}'")
            
        except Exception as e:
            logger.error(f"❌ Error in LLM call: {str(e)}")
            # Fallback: create simple mapping
            equip_prompt = f"find {query}"
            persian_full_query = query
        
        # Call embedding server for category classification
        logger.debug(f"🏷️ Classifying categories for: '{persian_full_query}'")
        categories = await self.classify_categories(persian_full_query, top_k=3)
        categories_fa = [c.get("category", c) if isinstance(c, dict) else c for c in categories]
        
        # Clamp values
        price_sens = max(0.0, min(1.0, price_sens))
        quality_sens = max(0.0, min(1.0, quality_sens))
        
        # NOTE: Categories are NOT added to equip_prompt!
        # They will be added by dsl_processor_server after EQuIP generates the base DSL.
        # This prevents confusion in the EQuIP model with Persian text.
        
        result = {
            "equip_prompt": equip_prompt,
            "persian_full_query": persian_full_query,
            "token_mapping": token_mapping,
            "categories_fa": categories_fa,
            "intent": intent,
            "price_sensitivity": price_sens,
            "quality_sensitivity": quality_sens,
            "original_query": query
        }
        
        logger.info(f"✅ Interpretation complete: {json.dumps(result, ensure_ascii=False)[:200]}...")
        return result


# ═══════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════
interpret_service: InterpretService = None


# ═══════════════════════════════════════════════════════════════
# MCP Server Setup
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on startup."""
    global interpret_service
    logger.info(f"🚀 Starting {SERVER_NAME} on port {SERVER_PORT}...")
    interpret_service = InterpretService()
    logger.info(f"✅ {SERVER_NAME} ready!")
    yield {"interpret_service": interpret_service}
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
async def interpret_query(query: str) -> str:
    """
    Analyze user shopping intent and prepare structured output for EQuIP DSL generation.
    
    Args:
        query: User's shopping query in natural language (Persian)
        
    Returns:
        JSON string with:
        - equip_prompt: English natural language query for EQuIP model
        - persian_full_query: Full Persian product description for BM25/semantic search
        - token_mapping: Dictionary mapping English words to Persian equivalents
        - categories_fa: List of Persian category names
        - intent: Shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        - price_sensitivity: 0-1 (higher = more price-conscious)
        - quality_sensitivity: 0-1 (higher = more quality-focused)
        - original_query: Original user query
    """
    global interpret_service
    logger.debug(f"📥 interpret_query called with: '{query}'")
    
    try:
        result = await interpret_service.interpret(query)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error interpreting query: {e}")
        return json.dumps({
            "error": str(e),
            "equip_prompt": f"find {query}",
            "persian_full_query": query,
            "token_mapping": {},
            "categories_fa": [],
            "intent": "find_best_value",
            "price_sensitivity": 0.5,
            "quality_sensitivity": 0.5,
            "original_query": query
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
