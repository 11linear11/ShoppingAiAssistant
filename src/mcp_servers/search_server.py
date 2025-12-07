"""
MCP Server for Product Search
Port: 5002

This server handles:
- search_with_interpretation: Complete search pipeline (EQuIP + DSL Processor + ES)
- search_products_semantic: Search products with Elasticsearch and reranking
- execute_dsl: Execute pre-built DSL query directly on Elasticsearch

This server internally calls:
- equip_server (5005): For generating Elasticsearch DSL from equip_prompt
- dsl_processor_server (5006): For converting English DSL to Persian + semantic search
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional
from statistics import median
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
SERVER_NAME = "search-server"
SERVER_PORT = 5002
EMBEDDING_SERVER_URL = os.getenv("MCP_EMBEDDING_URL", "http://localhost:5003")
EQUIP_SERVER_URL = os.getenv("MCP_EQUIP_URL", "http://localhost:5005")
DSL_PROCESSOR_URL = os.getenv("MCP_DSL_PROCESSOR_URL", "http://localhost:5006")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Path to brand scores
BRAND_SCORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "BrandScore.json"
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
def load_brand_scores() -> Dict[str, float]:
    """Load brand scores from JSON file."""
    try:
        with open(BRAND_SCORE_PATH, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        logger.info(f"✅ Loaded {len(scores)} brand scores")
        return scores
    except FileNotFoundError:
        logger.error(f"❌ Brand scores file not found: {BRAND_SCORE_PATH}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing brand scores JSON: {e}")
        return {}


# ═══════════════════════════════════════════════════════════════
# Search Service Class
# ═══════════════════════════════════════════════════════════════
class SearchService:
    """Handles product search with Elasticsearch."""
    
    def __init__(self):
        logger.info("🔧 Initializing SearchService...")
        
        # Elasticsearch configuration
        ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
        ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
        ES_USERNAME = os.getenv("ELASTICSEARCH_USER")
        ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
        scheme = os.getenv("ELASTICSEARCH_SCHEME", "http")
        
        es_url = f"{scheme}://{ES_HOST}:{ES_PORT}"
        logger.info(f"🔌 Connecting to Elasticsearch: {es_url}")
        
        # Create Elasticsearch client
        try:
            if ES_USERNAME and ES_PASSWORD:
                self.es = Elasticsearch(es_url, basic_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False)
            else:
                self.es = Elasticsearch(es_url, verify_certs=False)
        except TypeError:
            # Fallback for older versions
            logger.debug("⚠️ Using fallback Elasticsearch connection method")
            host_dict = {"host": ES_HOST, "port": int(ES_PORT), "scheme": scheme}
            if ES_USERNAME and ES_PASSWORD:
                self.es = Elasticsearch([host_dict], http_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False)
            else:
                self.es = Elasticsearch([host_dict], verify_certs=False)
        
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "shopping_products")
        logger.info(f"📚 Index name: {self.index_name}")
        
        # Load brand scores
        self.brand_scores = load_brand_scores()
        
        logger.info("✅ SearchService initialized")
    
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
    
    async def search(
        self,
        query_text: str,
        top_k: int = 100,
        min_similarity: float = 0.3,
        categories: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Perform hybrid search combining BM25 and semantic similarity.
        """
        logger.info(f"🔍 Starting search for: '{query_text}'")
        logger.debug(f"📊 Parameters: top_k={top_k}, min_similarity={min_similarity}, categories={categories}")
        
        # Get embedding from embedding server
        logger.debug("🧠 Getting query embedding from embedding server...")
        query_embedding = await self.get_embedding(query_text)
        
        if not query_embedding:
            logger.error("❌ Failed to get embedding, falling back to text-only search")
            query_embedding = [0.0] * 768  # Default dimension
        
        logger.debug(f"✅ Embedding received (dim={len(query_embedding)})")
        
        # Build filter for categories
        filter_clause = []
        if categories and len(categories) > 0:
            filter_clause.append({
                "terms": {
                    "category_name.keyword": categories
                }
            })
            logger.debug(f"🏷️ Category filter applied: {categories}")
        
        # Hybrid search query
        search_body = {
            "size": 50,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["product_name^2", "brand_name", "category_name"],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                                    "params": {"query_vector": query_embedding}
                                },
                                "boost": 2.0
                            }
                        }
                    ],
                    "filter": filter_clause if filter_clause else None,
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            logger.debug("📡 Executing Elasticsearch query...")
            response = self.es.search(index=self.index_name, body=search_body)
            logger.debug(f"✅ ES response: {response['hits']['total']['value']} total hits")
            
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                score = hit['_score']
                similarity = min(1.0, score / 3.0)
                
                results.append({
                    'product_id': source.get('product_id', ''),
                    'product_name': source.get('product_name', ''),
                    'brand_name': source.get('brand_name', ''),
                    'price': source.get('price', 0),
                    'discount_price': source.get('discount_price', 0),
                    'category_name': source.get('category_name', ''),
                    'has_discount': source.get('has_discount', False),
                    'discount_percentage': source.get('discount_percentage', 0),
                    'similarity': similarity,
                    'score': score
                })
            
            logger.debug(f"📦 Collected {len(results)} raw results")
            
            # Dynamic filtering
            if results:
                similarities = [r['similarity'] for r in results]
                dyn_thresh = max(0.38, median(similarities) - 0.15)
                filtered = [r for r in results if r['similarity'] >= dyn_thresh]
                logger.info(f"🔎 Dynamic filter: threshold={dyn_thresh:.3f}, kept {len(filtered)}/{len(results)}")
            else:
                filtered = []
            
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch error: {str(e)}")
            return []
    
    def calculate_value_scores(
        self,
        results: List[Dict],
        intent: str,
        price_sensitivity: float,
        quality_sensitivity: float
    ) -> List[Dict]:
        """Calculate value scores for products based on intent."""
        
        for product in results:
            price = product['price']
            discount_percentage = product['discount_percentage']
            final_price = price - (price * discount_percentage / 100)
            
            brand_name = product['brand_name'] if product['brand_name'] else ""
            brand_score = self.brand_scores.get(brand_name, 0.5)
            
            normalized_price = final_price / 1000000
            
            if intent == "find_cheapest":
                value_score = (
                    -normalized_price * 2.0 +
                    product['similarity'] * 0.3 +
                    brand_score * 0.1
                )
            elif intent == "find_high_quality":
                value_score = (
                    brand_score * 1.5 +
                    product['similarity'] * 0.5 +
                    (discount_percentage / 100) * 0.3 -
                    normalized_price * 0.1
                )
            elif intent == "find_by_feature":
                value_score = (
                    product['similarity'] * 1.5 +
                    brand_score * quality_sensitivity * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * price_sensitivity * 0.3
                )
            elif intent == "compare":
                value_score = (
                    brand_score * 0.3 +
                    product['similarity'] * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * 0.3
                )
            else:  # find_best_value or default
                value_score = (
                    brand_score * quality_sensitivity +
                    product['similarity'] * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * price_sensitivity
                )
            
            product['final_price'] = int(final_price)
            product['brand_score'] = round(brand_score, 3)
            product['value_score'] = round(value_score, 3)
        
        return results
    
    async def call_mcp_tool(self, server_url: str, tool_name: str, arguments: Dict) -> Dict:
        """Call an MCP tool on another server."""
        try:
            async with streamablehttp_client(f"{server_url}/") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    if result.content:
                        return json.loads(result.content[0].text)
                    return {"success": False, "error": "No content"}
        except Exception as e:
            logger.error(f"❌ Error calling {server_url}: {e}")
            return {"success": False, "error": str(e)}
    
    async def call_equip_server(self, equip_prompt: str, intent: str = "find_best_value") -> Dict:
        """Call equip_server to generate DSL from equip_prompt."""
        logger.info(f"🤖 Calling EQuIP server with prompt: '{equip_prompt}'")
        result = await self.call_mcp_tool(
            EQUIP_SERVER_URL,
            "generate_dsl",
            {"query": equip_prompt, "intent": intent}
        )
        return result
    
    async def call_dsl_processor(
        self,
        dsl: Dict,
        token_mapping: Dict[str, str],
        persian_full_query: str,
        categories_fa: List[str]
    ) -> Dict:
        """Call dsl_processor_server to transform DSL."""
        logger.info("🔄 Calling DSL Processor server")
        result = await self.call_mcp_tool(
            DSL_PROCESSOR_URL,
            "process_dsl",
            {
                "dsl": json.dumps(dsl) if isinstance(dsl, dict) else dsl,
                "token_mapping": json.dumps(token_mapping) if isinstance(token_mapping, dict) else token_mapping,
                "persian_full_query": persian_full_query,
                "categories_fa": json.dumps(categories_fa) if isinstance(categories_fa, list) else categories_fa
            }
        )
        return result
    
    async def full_search_pipeline(
        self,
        equip_prompt: str,
        token_mapping: Dict[str, str],
        persian_full_query: str,
        categories_fa: List[str],
        intent: str = "find_best_value",
        price_sensitivity: float = 0.5,
        quality_sensitivity: float = 0.5
    ) -> Dict:
        """
        Complete search pipeline:
        1. Call EQuIP to generate DSL from equip_prompt
        2. Call DSL Processor to transform to Persian + add semantic search
        3. Execute on Elasticsearch
        4. Calculate value scores and return ranked results
        """
        logger.info(f"🚀 Starting full search pipeline")
        logger.debug(f"📝 equip_prompt: '{equip_prompt}'")
        logger.debug(f"🔤 token_mapping: {token_mapping}")
        logger.debug(f"🇮🇷 persian_full_query: '{persian_full_query}'")
        logger.debug(f"📁 categories_fa: {categories_fa}")
        
        # Step 1: Generate DSL via EQuIP
        equip_result = await self.call_equip_server(equip_prompt, intent)
        if not equip_result.get("success"):
            logger.error(f"❌ EQuIP failed: {equip_result.get('error')}")
            return {"products": [], "error": f"EQuIP error: {equip_result.get('error')}"}
        
        raw_dsl = equip_result.get("dsl", {})
        logger.info(f"✅ EQuIP generated DSL")
        logger.debug(f"📋 Raw DSL: {json.dumps(raw_dsl)[:200]}...")
        
        # Step 2: Process DSL (English → Persian + Semantic)
        processor_result = await self.call_dsl_processor(
            raw_dsl, token_mapping, persian_full_query, categories_fa
        )
        if not processor_result.get("success"):
            logger.error(f"❌ DSL Processor failed: {processor_result.get('error')}")
            return {"products": [], "error": f"DSL Processor error: {processor_result.get('error')}"}
        
        final_dsl = processor_result.get("dsl", {})
        logger.info(f"✅ DSL processed to Persian with semantic search")
        
        # Step 3: Execute on Elasticsearch
        try:
            logger.debug("📡 Executing Elasticsearch query...")
            response = self.es.search(index=self.index_name, body=final_dsl)
            logger.info(f"✅ ES response: {response['hits']['total']['value']} total hits")
            
            # Process results
            results = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                score = hit.get('_score') or 1.0  # Default to 1.0 if score is None
                similarity = min(1.0, score / 5.0)  # Adjusted for hybrid search
                
                results.append({
                    'product_id': source.get('product_id', ''),
                    'product_name': source.get('product_name', ''),
                    'brand_name': source.get('brand_name', ''),
                    'price': source.get('price', 0),
                    'discount_price': source.get('discount_price', 0),
                    'category_name': source.get('category_name', ''),
                    'has_discount': source.get('has_discount', False),
                    'discount_percentage': source.get('discount_percentage', 0),
                    'similarity': similarity,
                    'score': score
                })
            
            if not results:
                logger.warning("⚠️ No products found from pipeline")
                return {"products": [], "message": "متاسفانه هیچ محصولی یافت نشد."}
            
            logger.debug(f"📦 Got {len(results)} products")
            
            # Step 4: Calculate value scores
            results = self.calculate_value_scores(
                results, intent, price_sensitivity, quality_sensitivity
            )
            
            # Sort by value_score
            results.sort(key=lambda x: x['value_score'], reverse=True)
            
            # Limit results
            MAX_RESULTS = 10
            results = results[:MAX_RESULTS]
            
            return {"products": results, "success": True}
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch error: {str(e)}")
            return {"products": [], "error": f"Elasticsearch error: {str(e)}"}


# ═══════════════════════════════════════════════════════════════
# Global Service Instance
# ═══════════════════════════════════════════════════════════════
search_service: SearchService = None


# ═══════════════════════════════════════════════════════════════
# MCP Server Setup
# ═══════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize resources on startup."""
    global search_service
    logger.info(f"🚀 Starting {SERVER_NAME} on port {SERVER_PORT}...")
    search_service = SearchService()
    logger.info(f"✅ {SERVER_NAME} ready!")
    yield {"search_service": search_service}
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
async def search_with_interpretation(
    equip_prompt: str,
    token_mapping: str,
    persian_full_query: str,
    categories_fa: str,
    intent: str = "find_best_value",
    price_sensitivity: float = 0.5,
    quality_sensitivity: float = 0.5
) -> str:
    """
    Complete search pipeline using interpret_server output.
    
    This is the main search tool that:
    1. Takes interpretation output (from interpret_query tool)
    2. Internally calls EQuIP to generate DSL
    3. Internally calls DSL Processor to transform to Persian + add semantic search
    4. Executes on Elasticsearch
    5. Returns ranked products
    
    Args:
        equip_prompt: Structured prompt for EQuIP (e.g., "product_name: laptop sort: price_asc")
        token_mapping: JSON string of English→Persian word mappings (e.g., {"laptop": "لپتاپ"})
        persian_full_query: Full Persian product description for exact matching
        categories_fa: JSON string of Persian category names list
        intent: Shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        price_sensitivity: How price-conscious (0-1), higher = cheaper preferred
        quality_sensitivity: How quality-focused (0-1), higher = quality preferred
        
    Returns:
        JSON string with ranked products and metadata.
    """
    global search_service
    logger.info(f"🔍 Search with interpretation: '{equip_prompt}'")
    
    try:
        # Parse JSON strings
        token_mapping_dict = json.loads(token_mapping) if isinstance(token_mapping, str) else token_mapping
        categories_list = json.loads(categories_fa) if isinstance(categories_fa, str) else categories_fa
        
        # Run full pipeline
        result = await search_service.full_search_pipeline(
            equip_prompt=equip_prompt,
            token_mapping=token_mapping_dict,
            persian_full_query=persian_full_query,
            categories_fa=categories_list,
            intent=intent,
            price_sensitivity=price_sensitivity,
            quality_sensitivity=quality_sensitivity
        )
        
        if result.get("error"):
            return json.dumps({
                "products": [],
                "error": result.get("error")
            }, ensure_ascii=False)
        
        products = result.get("products", [])
        
        # Format output
        formatted_products = []
        for product in products:
            formatted_products.append({
                "name": product['product_name'],
                "price": int(product['price']),
                "final_price": product.get('final_price', product['price']),
                "brand": product['brand_name'] if product['brand_name'] else "",
                "brand_score": product.get('brand_score', 0.5),
                "discount": int(product['discount_percentage']) if product.get('has_discount') else 0,
                "product_id": str(product['product_id']),
                "similarity": round(product.get('similarity', 0), 3),
                "value_score": product.get('value_score', 0),
                "category": product['category_name'] if product['category_name'] else ""
            })
        
        # Calculate meta info
        if formatted_products:
            prices = [p['final_price'] for p in formatted_products]
            min_price = min(prices)
            max_price = max(prices)
            avg_similarity = sum(p['similarity'] for p in formatted_products) / len(formatted_products)
        else:
            min_price = max_price = 0
            avg_similarity = 0
        
        return json.dumps({
            "products": formatted_products,
            "meta": {
                "total_found": len(formatted_products),
                "avg_similarity": round(avg_similarity, 3),
                "price_range": {
                    "min": min_price,
                    "max": max_price
                },
                "intent": intent,
                "execution_method": "full_pipeline"
            }
        }, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {e}")
        return json.dumps({
            "products": [],
            "error": f"Invalid JSON input: {str(e)}"
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error in search pipeline: {str(e)}")
        return json.dumps({
            "products": [],
            "error": f"خطا در جستجو: {str(e)}"
        }, ensure_ascii=False)


@mcp.tool()
async def search_products_semantic(
    query: str,
    quality_sensitivity: float = 0.5,
    price_sensitivity: float = 0.5,
    categories: Optional[List[str]] = None,
    intent: Optional[str] = None
) -> str:
    """
    Search for products using semantic search with Elasticsearch and intelligent reranking.
    
    Args:
        query: The product search query in natural language
        quality_sensitivity: User's quality preference (0-1), higher means quality matters more
        price_sensitivity: User's price preference (0-1), higher means cheaper is better
        categories: List of product category filters
        intent: User's shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        
    Returns:
        JSON string with reranked products based on value score calculation.
    """
    global search_service
    logger.info(f"🛍️ Product search: '{query}'")
    logger.debug(f"⚙️ Params: quality={quality_sensitivity:.2f}, price={price_sensitivity:.2f}, intent={intent}")
    
    if categories:
        logger.info(f"🏷️ Category filters: {categories}")
    
    # Normalize intent
    if intent:
        intent = intent.lower().strip()
    else:
        intent = "find_best_value"
    
    try:
        # Perform search
        results = await search_service.search(query, top_k=100, min_similarity=0.3, categories=categories)
        
        if not results:
            logger.warning(f"⚠️ No products found for query: '{query}'")
            return json.dumps({
                "products": [],
                "message": f"متاسفانه هیچ محصولی با جستجوی '{query}' پیدا نشد."
            }, ensure_ascii=False)
        
        logger.debug(f"📦 Got {len(results)} products from search")
        
        # Calculate value scores
        results = search_service.calculate_value_scores(
            results, intent, price_sensitivity, quality_sensitivity
        )
        
        # Sort by value_score
        results.sort(key=lambda x: x['value_score'], reverse=True)
        
        # Filter and limit results
        MIN_RESULTS = 1
        MAX_RESULTS = 5
        RELEVANCE_THRESHOLD = 0.4
        
        relevant_results = [r for r in results if r['similarity'] >= RELEVANCE_THRESHOLD]
        
        if not relevant_results:
            logger.warning(f"⚠️ No products above relevance threshold")
            relevant_results = results[:MIN_RESULTS] if results else []
            is_relevant = False
        else:
            is_relevant = True
        
        results = relevant_results[:MAX_RESULTS]
        
        logger.info(f"✅ Returning {len(results)} products")
        
        # Format results
        products = []
        for product in results:
            products.append({
                "name": product['product_name'],
                "price": int(product['price']),
                "final_price": product['final_price'],
                "brand": product['brand_name'] if product['brand_name'] else "",
                "brand_score": product['brand_score'],
                "discount": int(product['discount_percentage']) if product['has_discount'] else 0,
                "product_id": str(product['product_id']),
                "similarity": round(product['similarity'], 3),
                "value_score": product['value_score'],
                "category": product['category_name'] if product['category_name'] else ""
            })
        
        # Calculate meta info
        if products:
            prices = [p['final_price'] for p in products]
            min_price = min(prices)
            max_price = max(prices)
            avg_similarity = sum(p['similarity'] for p in products) / len(products)
        else:
            min_price = max_price = 0
            avg_similarity = 0
        
        return json.dumps({
            "products": products,
            "meta": {
                "query": query,
                "total_found": len(products),
                "is_relevant": is_relevant,
                "avg_similarity": round(avg_similarity, 3),
                "price_range": {
                    "min": min_price,
                    "max": max_price
                },
                "intent": intent
            }
        }, ensure_ascii=False)
        
    except Exception as e:
        logger.error(f"❌ Error in search: {str(e)}")
        return json.dumps({
            "products": [],
            "error": "خطا در جستجوی محصولات. لطفاً دوباره تلاش کنید."
        }, ensure_ascii=False)


@mcp.tool()
async def execute_dsl(
    dsl: str,
    intent: str = "find_best_value",
    price_sensitivity: float = 0.5,
    quality_sensitivity: float = 0.5
) -> str:
    """
    Execute a pre-built Elasticsearch DSL query directly.
    
    This tool is designed to work with the new architecture:
    1. interpret_server generates equip_prompt + token_mapping
    2. equip_server generates DSL from equip_prompt
    3. dsl_processor_server converts DSL to Persian + adds semantic search
    4. This tool executes the final DSL
    
    Args:
        dsl: JSON string of Elasticsearch DSL query (already processed with Persian terms)
        intent: Shopping intent for value score calculation
        price_sensitivity: How price-conscious (0-1)
        quality_sensitivity: How quality-focused (0-1)
        
    Returns:
        JSON string with reranked products based on value score calculation.
    """
    global search_service
    logger.info(f"🔍 Executing DSL query")
    logger.debug(f"⚙️ Params: intent={intent}, price={price_sensitivity:.2f}, quality={quality_sensitivity:.2f}")
    
    try:
        # Parse DSL
        dsl_dict = json.loads(dsl) if isinstance(dsl, str) else dsl
        
        # Execute on Elasticsearch
        logger.debug("📡 Executing Elasticsearch query...")
        response = search_service.es.search(index=search_service.index_name, body=dsl_dict)
        
        logger.debug(f"✅ ES response: {response['hits']['total']['value']} total hits")
        
        # Process results
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            score = hit['_score']
            similarity = min(1.0, score / 3.0)
            
            results.append({
                'product_id': source.get('product_id', ''),
                'product_name': source.get('product_name', ''),
                'brand_name': source.get('brand_name', ''),
                'price': source.get('price', 0),
                'discount_price': source.get('discount_price', 0),
                'category_name': source.get('category_name', ''),
                'has_discount': source.get('has_discount', False),
                'discount_percentage': source.get('discount_percentage', 0),
                'similarity': similarity,
                'score': score
            })
        
        if not results:
            logger.warning("⚠️ No products found from DSL query")
            return json.dumps({
                "products": [],
                "message": "متاسفانه هیچ محصولی یافت نشد."
            }, ensure_ascii=False)
        
        logger.debug(f"📦 Got {len(results)} products from DSL execution")
        
        # Calculate value scores
        results = search_service.calculate_value_scores(
            results, intent, price_sensitivity, quality_sensitivity
        )
        
        # Sort by value_score
        results.sort(key=lambda x: x['value_score'], reverse=True)
        
        # Limit results
        MAX_RESULTS = 10
        results = results[:MAX_RESULTS]
        
        logger.info(f"✅ Returning {len(results)} products")
        
        # Format results
        products = []
        for product in results:
            products.append({
                "name": product['product_name'],
                "price": int(product['price']),
                "final_price": product.get('final_price', product['price']),
                "brand": product['brand_name'] if product['brand_name'] else "",
                "brand_score": product.get('brand_score', 0.5),
                "discount": int(product['discount_percentage']) if product['has_discount'] else 0,
                "product_id": str(product['product_id']),
                "similarity": round(product['similarity'], 3),
                "value_score": product.get('value_score', 0),
                "category": product['category_name'] if product['category_name'] else ""
            })
        
        # Calculate meta info
        if products:
            prices = [p['final_price'] for p in products]
            min_price = min(prices)
            max_price = max(prices)
            avg_similarity = sum(p['similarity'] for p in products) / len(products)
        else:
            min_price = max_price = 0
            avg_similarity = 0
        
        return json.dumps({
            "products": products,
            "meta": {
                "total_found": len(products),
                "avg_similarity": round(avg_similarity, 3),
                "price_range": {
                    "min": min_price,
                    "max": max_price
                },
                "intent": intent,
                "execution_method": "dsl_direct"
            }
        }, ensure_ascii=False)
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid DSL JSON: {e}")
        return json.dumps({
            "products": [],
            "error": f"Invalid DSL format: {str(e)}"
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Error executing DSL: {str(e)}")
        return json.dumps({
            "products": [],
            "error": f"خطا در اجرای جستجو: {str(e)}"
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
