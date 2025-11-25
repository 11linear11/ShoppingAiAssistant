"""
Product Search Tool using Elasticsearch and Multilingual Embeddings
This module provides semantic search functionality for products.
"""

import os
import json
import re
import logging
from typing import List, Dict
from statistics import median
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI


load_dotenv()

# Setup logging
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
logger = logging.getLogger(__name__)


class ProductSearchEngine:
    """Elasticsearch-based product search with semantic embeddings."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProductSearchEngine, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if ProductSearchEngine._initialized:
            return
        
        logger.info("🔧 Initializing ProductSearchEngine...")
        
        # Load model silently
        model_name = 'intfloat/multilingual-e5-base'
        logger.debug(f"📦 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.debug("✅ Embedding model loaded")
        
        # Elasticsearch configuration
        ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
        ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
        ES_USERNAME = os.getenv("ELASTICSEARCH_USER")
        ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
        scheme = os.getenv("ELASTICSEARCH_SCHEME", "http")
        
        es_url = f"{scheme}://{ES_HOST}:{ES_PORT}"
        logger.debug(f"🔌 Connecting to Elasticsearch: {es_url}")
        
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
        logger.debug(f"📚 Index name: {self.index_name}")
        logger.info("✅ ProductSearchEngine initialized successfully")
        
        ProductSearchEngine._initialized = True
    
    def search(self, query_text: str, top_k: int = 5, min_similarity: float = 0.3, category: str = None) -> List[Dict]:
        """
        Perform hybrid search combining BM25 text matching and semantic similarity.
        
        Args:
            query_text: Search query
            top_k: Number of results
            min_similarity: Minimum similarity score (0-1)
            category: Optional category filter (e.g., "لپ تاپ", "گوشی موبایل")
            
        Returns:
            List of product dictionaries sorted by combined relevance score
        """
        logger.info(f"🔍 Starting search for: '{query_text}'")
        logger.debug(f"📊 Parameters: top_k={top_k}, min_similarity={min_similarity}, category={category}")
        
        # Generate embedding for semantic search
        logger.debug("🧠 Generating query embedding...")
        query_embedding = self.model.encode([query_text])[0].tolist()
        logger.debug(f"✅ Embedding generated (dim={len(query_embedding)})")
        
        # Build query with BM25 + embedding hybrid search
        must_conditions = []
        
        # Add category filter if provided
        if category:
            must_conditions.append({"match": {"category_name": category}})
            logger.debug(f"🏷️ Category filter applied: {category}")
        
        # Hybrid search: BM25 (text) + semantic (embedding)
        search_body = {
            "size": 50, 
            "query": {
                "bool": {
                    "must": must_conditions if must_conditions else [{"match_all": {}}],
                    "should": [
                        # BM25 text search on product name and brand
                        {
                            "multi_match": {
                                "query": query_text,
                                "fields": ["product_name^2", "brand_name", "category_name"],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        # Semantic similarity search
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
                    "minimum_should_match": 1
                }
            }
        }
        
        try:
            logger.debug("📡 Executing Elasticsearch query...")
            response = self.es.search(index=self.index_name, body=search_body)
            logger.debug(f"✅ Elasticsearch response received: {response['hits']['total']['value']} total hits")
            
            results = []
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                score = hit['_score']
                
                # Calculate semantic similarity from embedding
                # Note: exact similarity needs recalculation, using score as proxy
                similarity = min(1.0, score / 3.0)  # Normalize combined score
                
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
            
            # Dynamic filtering based on similarity distribution
            if results:
                similarities = [r['similarity'] for r in results]
                dyn_thresh = max(0.38, median(similarities) - 0.15)
                filtered = [r for r in results if r['similarity'] >= dyn_thresh]
                logger.info(f"🔎 Dynamic filter: threshold={dyn_thresh:.3f}, kept {len(filtered)}/{len(results)} results")
            else:
                filtered = []
                logger.warning("⚠️ No results found")
            
            logger.info(f"✅ Search completed: returning {len(filtered)} products")
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch error: {str(e)}")
            return []

# Global LLM instance for interpret_query
_llm_instance = None


def get_llm_instance():
    """Get or create a cached LLM instance for query interpretation."""
    global _llm_instance
    if _llm_instance is None:
        # Try different NVIDIA models - some may be unavailable or return empty responses
        nvidia_models = [
            "meta/llama-3.1-70b-instruct",  # More stable model
            "meta/llama-3.1-8b-instruct",   # Lighter fallback
            "openai/gpt-oss-120b",          # Original model
        ]
        
        model = os.getenv("NVIDIA_MODEL", nvidia_models[0])
        logger.info(f"🤖 Using NVIDIA model: {model}")
        
        _llm_instance = ChatNVIDIA(
            model=model,
            api_key=os.getenv("api_key"),
            base_url="https://integrate.api.nvidia.com/v1",   
            temperature=0.1,
            max_tokens=1000
        )
        # Alternative: OpenAI
        # _llm_instance = ChatOpenAI(
        #     model="openai/gpt-4o",
        #     api_key = os.getenv("OPENAI_API_KEY"),
        #     base_url = "https://models.inference.ai.azure.com",   
        # )
        
    return _llm_instance


@tool
def interpret_query(query: str) -> str:
    """
    Analyze user shopping intent and extract structured information.
    
    This tool helps understand what the user is looking for by analyzing their query
    and returning key insights about their shopping preferences and intent.
    
    Args:
        query: User's shopping query in natural language (Persian, English, or other languages)
        
    Returns:
        JSON string with the following fields:
        - category: Product category (e.g., "لپ تاپ", "گوشی", "هدفون")
        - intent: Shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        - price_sensitivity: 0-1 (higher = more price-conscious)
        - quality_sensitivity: 0-1 (higher = more quality-focused)
        - suggested_query: A specific product keyword to search for
    """
    
    # Available categories in Elasticsearch
    AVAILABLE_CATEGORIES = [
        "لوازم الکترونیکی",
        "لوازم برقی و دیجیتال",
        "مد و پوشاک",
        "طلا",
        "خانه و سبک زندگی",
        "خانه و آشپزخانه",
        "آرایشی و بهداشتی",
        "نوشیدنی",
        "لبنیات",
        "کالاهای اساسی",
        "کودک و نوزاد",
        "خواربار و نان",
        "بهداشت و سلامت",
        "شوینده و مواد ضد عفونی کننده",
        "مواد پروتئینی",
        "میوه و سبزیجات تازه",
        "دستمال و شوینده",
        "محصولات سلولزی",
        "چاشنی و افزودنی",
        "تنقلات",
        "کنسرو و غذای آماده",
        "صبحانه",
        "کنسرو و غذاهای آماده",
        "آجیل و خشکبار",
        "خشکبار، دسر و شیرینی",
        "لوازم تحریر و اداری",
        "دسر و شیرینی پزی",
        "نان و شیرینی"
    ]
    
    # Construct a clear, structured prompt
    prompt =   f"""
You are a purchase-intent interpreter. Your job is to convert each user message into structured data that can be used by the product search engine.

Your output must be exactly one valid JSON object. Produce no additional text.

-----------------------------------------------
📋 Available categories in the system:
{AVAILABLE_CATEGORIES}

-----------------------------------------------
Important Rules:

1. **CATEGORY Detection:**

   a) If the user mentions a *specific product*:
   - "شورت", "تیشرت", "کفش" → "مد و پوشاک"
   - "دوغ", "آبمیوه", "نوشابه" → "نوشیدنی"
   - "ماست", "پنیر", "شیر" → "لبنیات"
   - "لپ تاپ", "گوشی", "هدفون" → "لوازم الکترونیکی"
   - "شامپو", "کرم", "رژلب" → "آرایشی و بهداشتی"
   - "مایع ظرفشویی", "پودر لباسشویی" → "شوینده و مواد ضد عفونی کننده"
   - "چیپس", "شکلات", "پفک" → "تنقلات"

   b) If the user only gives *abstract properties*:
   - "نرم", "لطیف", "راحت" → likely "مد و پوشاک" or "دستمال و شوینده"
   - "ترد", "خوشمزه" → likely "تنقلات"
   - "تند", "فلفلی", "تیز" → likely "تنقلات" or "چاشنی و افزودنی"
   - "خنک", "خوشبو" (for liquids) → likely "نوشیدنی" or "آرایشی و بهداشتی"
   - "خوشبو" (cleaning items) → likely "شوینده و مواد ضد عفونی کننده"
   - "تازه" → likely "میوه و سبزیجات تازه" or "لبنیات" or "نان و شیرینی"

   c) **Important rule:** you must ALWAYS guess a category!  
      Only return null if the message is completely unrelated to buying.
      - If the user mentions a taste (تند, شیرین, ترش) → "تنقلات" or "چاشنی و افزودنی"
      - If the user mentions a physical feel (نرم, سبک) → "مد و پوشاک"

2. **intent** must be one of the following (never return null!):
   - find_cheapest        → user wants the cheapest option
   - find_best_value      → user wants best price/quality ratio
   - find_high_quality    → user prioritizes quality
   - compare              → user wants to compare options
   - find_by_feature      → user mentions a specific feature (size, softness, flavor, etc.)

3. **price_sensitivity:**
   - 1.0 → words like "ارزون", "ارزان‌ترین", "مقرون‌به‌صرفه", "اقتصادی"
   - 0.5 → indirect or unclear mention of cost
   - 0   → no price-related mention

4. **quality_sensitivity:**
   - 1.0 → words like "کیفیت بالا", "محکم", "دوام", "پرفورمنس بالا", "نرم", "لطیف"
   - 0.5 → unclear or partial quality mention
   - 0   → no quality mention

5. **suggested_query (مهم!):**
   این فیلد باید یک کلمه کلیدی مشخص محصول باشد که برای جستجو استفاده بشه.
   
   اگر کاربر محصول مشخص گفت → همون محصول
   اگر کاربر مبهم صحبت کرد → یک محصول مناسب پیشنهاد بده
   
   مثال‌ها:
   - "یچیز میخوام بپوشم سردم نشه" → suggested_query: "کاپشن"
   - "گشنمه" → suggested_query: "بیسکویت"
   - "تشنمه" → suggested_query: "آب معدنی"
   - "میخوام موهامو بشورم" → suggested_query: "شامپو"
   - "پوستم خشکه" → suggested_query: "کرم مرطوب کننده"
   - "یچیز تند میخوام" → suggested_query: "چیپس تند"
   - "یچیز شیرین میخوام" → suggested_query: "شکلات"
   - "میخوام یه چیز گرم بخورم" → suggested_query: "سوپ"
   - "خوابم میاد" → suggested_query: "قهوه"
   - "هدفون میخوام" → suggested_query: "هدفون"

-----------------------------------------------
### Direct Examples (specific product)

User: "دوغ خوشمزه و ارزون"
{{
  "category": "نوشیدنی",
  "intent": "find_best_value",
  "price_sensitivity": 1,
  "quality_sensitivity": 1,
  "suggested_query": "دوغ"
}}

User: "شورت ورزشی نرم می‌خوام"
{{
  "category": "مد و پوشاک",
  "intent": "find_by_feature",
  "price_sensitivity": 0,
  "quality_sensitivity": 1,
  "suggested_query": "شورت ورزشی"
}}

User: "چیپس ساده ارزان"
{{
  "category": "تنقلات",
  "intent": "find_cheapest",
  "price_sensitivity": 1,
  "quality_sensitivity": 0,
  "suggested_query": "چیپس"
}}

-----------------------------------------------
### Implicit Intent Examples (user describes need, not product)

User: "یچیز میخوام بپوشم سردم نشه"
{{
  "category": "مد و پوشاک",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "کاپشن"
}}

User: "گشنمه یچیز بده"
{{
  "category": "تنقلات",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "بیسکویت"
}}

User: "تشنمه"
{{
  "category": "نوشیدنی",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "آب معدنی"
}}

User: "پوستم خشکه"
{{
  "category": "آرایشی و بهداشتی",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "کرم مرطوب کننده"
}}

User: "خوابم میاد باید بیدار بمونم"
{{
  "category": "نوشیدنی",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "قهوه"
}}

-----------------------------------------------
### Ambiguous Examples (only features)

User: "من یه چیز تند میخوام"
{{
  "category": "تنقلات",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "چیپس تند"
}}

User: "یه چیز نرم"
{{
  "category": "مد و پوشاک",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "تیشرت"
}}

User: "یه نوشیدنی خنک"
{{
  "category": "نوشیدنی",
  "intent": "find_by_feature",
  "price_sensitivity": 0.5,
  "quality_sensitivity": 0.5,
  "suggested_query": "آب معدنی"
}}

-----------------------------------------------
### Final Rules
- Think step-by-step internally, but do NOT output your reasoning. 
Never include analysis, thoughts, explanations, or chain-of-thought in the output. 
Output only the final JSON.
- You MUST output ONLY one valid JSON object.
If you output anything else (text, explanation, markdown, analysis), it will be considered an error.
- If multiple interpretations exist, choose the most likely one.
- If any feature is mentioned → intent = find_by_feature
- **Always guess a category!**  
- **Always return an intent! Never return null.**
- **Always provide a suggested_query!** This is the keyword for product search.

-----------------------------------------------
User Query: {query}

Your output must match this exact structure:

{{
  "category": "...",
  "intent": "...",
  "price_sensitivity": 0.0,
  "quality_sensitivity": 0.0,
  "suggested_query": "..."
}}

Do not add or remove any fields.
Do not write any text outside this JSON.


"""

    
    logger.info(f"🧠 Interpreting query: '{query}'")
    
    try:
        # Get LLM instance
        logger.debug("🤖 Getting LLM instance for interpretation...")
        llm = get_llm_instance()
        
        # Invoke LLM
        logger.debug("💭 Invoking LLM for intent analysis...")
        response = llm.invoke(prompt)
        
        # Debug: Log the full response object structure
        logger.debug(f"📄 Raw response type: {type(response)}")
        logger.debug(f"📄 Raw response: {response}")
        
        # Try multiple ways to extract content
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
            # Last resort: convert to string
            response_text = str(response).strip()
        
        logger.debug(f"📄 Extracted response_text: '{response_text[:200] if response_text else 'EMPTY'}'")
        
        # Check if response is empty
        if not response_text:
            logger.warning("⚠️ LLM returned empty response! Using fallback values.")
            fallback = {
                "category": "نامشخص",
                "intent": "find_best_value",
                "price_sensitivity": 0.5,
                "quality_sensitivity": 0.5,
                "suggested_query": query,
            }
            return json.dumps(fallback, ensure_ascii=False)
        
        # Extract JSON from response (in case LLM adds extra text)
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            logger.debug("✅ JSON extracted from response")
        else:
            json_str = response_text
            logger.debug("⚠️ Using full response as JSON")
        
        # Validate JSON structure
        try:
            parsed = json.loads(json_str)
            logger.debug(f"✅ JSON parsed successfully: {parsed}")
            logger.debug(f"🔍 Parsed fields: {list(parsed.keys())}")
            
            # Handle null values from JSON (become None in Python)
            category = parsed.get("category")
            intent = parsed.get("intent")
            price_sens = parsed.get("price_sensitivity")
            quality_sens = parsed.get("quality_sensitivity")
            suggested_query = parsed.get("suggested_query")
            
            # Smart defaults for null/None values
            if category is None:
                category = "نامشخص"
                logger.debug("⚠️ Category was null, using 'نامشخص'")
            
            if intent is None:
                intent = "find_by_feature"  # Default for ambiguous queries
                logger.debug("⚠️ Intent was null, using 'find_by_feature'")
            
            if suggested_query is None:
                # اگر suggested_query نبود، از query اصلی استفاده کن
                suggested_query = query
                logger.debug(f"⚠️ suggested_query was null, using original query: '{query}'")
            
            # Ensure required fields exist with defaults
            result = {
                "category": category,
                "intent": intent,
                "price_sensitivity": float(price_sens) if price_sens is not None else 0.5,
                "quality_sensitivity": float(quality_sens) if quality_sens is not None else 0.5,
                "suggested_query": suggested_query,
            }
            
            # Clamp values between 0 and 1
            for key in ["price_sensitivity", "quality_sensitivity"]:
                result[key] = max(0.0, min(1.0, result[key]))
            
            logger.info(f"✅ Intent analysis complete: category={result['category']}, "
                       f"intent={result['intent']}, "
                       f"suggested_query='{result['suggested_query']}', "
                       f"price_sens={result['price_sensitivity']:.2f}, "
                       f"quality_sens={result['quality_sensitivity']:.2f}")
            
            return json.dumps(result, ensure_ascii=False)
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ JSON decode error: {e}. Using fallback values.")
            # Fallback: return default values
            fallback = {
                "category": "نامشخص",
                "intent": "find_best_value",
                "price_sensitivity": 0.5,
                "quality_sensitivity": 0.5,
                "suggested_query": query,
            }
            return json.dumps(fallback, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"❌ Error in interpret_query: {str(e)}")
        # In case of any error, return safe defaults
        error_response = {
            "category": "خطا",
            "intent": "find_best_value",
            "price_sensitivity": 0.5,
            "quality_sensitivity": 0.5,
            "suggested_query": query,
            "error": str(e)
        }
        return json.dumps(error_response, ensure_ascii=False)








# Global instance
_search_engine = None
_brand_scores = None


def get_search_engine() -> ProductSearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = ProductSearchEngine()
    return _search_engine


def get_brand_scores() -> Dict[str, float]:
    """Load brand scores from JSON file."""
    global _brand_scores
    if _brand_scores is None:
        try:
            brand_score_path = os.path.join(os.path.dirname(__file__), '../../BrandScore.json')
            with open(brand_score_path, 'r', encoding='utf-8') as f:
                _brand_scores = json.load(f)
        except:
            _brand_scores = {}
    return _brand_scores


@tool
def search_products_semantic(query: str, quality_sensitivity: float = 0.5, price_sensitivity: float = 0.5, category: str = None, intent: str = None) -> str:
    """
    Search for products using semantic search with Elasticsearch and intelligent reranking.
    Use this tool when the user wants to find, search, or look for products.
    This tool understands natural language in multiple languages (English, Persian, Arabic, etc.).
    
    Args:
        query: The product search query in natural language (e.g., "لپ تاپ گیمینگ", "cheap smartphone", "هدفون بی سیم")
        quality_sensitivity: User's quality preference (0-1), higher means quality matters more
        price_sensitivity: User's price preference (0-1), higher means cheaper is better
        category: Product category filter (e.g., "مد و پوشاک", "نوشیدنی", "لوازم الکترونیکی")
        intent: User's shopping intent (find_cheapest, find_best_value, find_high_quality, compare, find_by_feature)
        
    Returns:
        JSON string with reranked products based on value score calculation.
    """
    logger.info(f"🛍️ Product search: '{query}'")
    logger.debug(f"⚙️ Sensitivity params: quality={quality_sensitivity:.2f}, price={price_sensitivity:.2f}")
    logger.debug(f"🎯 Intent: {intent}")
    if category:
        logger.info(f"🏷️ Category filter: '{category}'")
    
    # ═══════════════════════════════════════════════════════════════
    # 🎯 تنظیم پارامترها بر اساس intent
    # ═══════════════════════════════════════════════════════════════
    if intent:
        intent = intent.lower().strip()
        
        if intent == "find_cheapest":
            # کاربر ارزان‌ترین رو میخواد - قیمت خیلی مهمه، کیفیت مهم نیست
            price_sensitivity = max(price_sensitivity, 0.9)
            quality_sensitivity = min(quality_sensitivity, 0.2)
            logger.info("💰 Intent 'find_cheapest' → Boosting price_sensitivity to 0.9")
            
        elif intent == "find_high_quality":
            # کاربر کیفیت بالا میخواد - برند و کیفیت مهمه، قیمت مهم نیست
            quality_sensitivity = max(quality_sensitivity, 0.9)
            price_sensitivity = min(price_sensitivity, 0.2)
            logger.info("⭐ Intent 'find_high_quality' → Boosting quality_sensitivity to 0.9")
            
        elif intent == "find_best_value":
            # کاربر بهترین ارزش (نسبت کیفیت به قیمت) رو میخواد
            price_sensitivity = 0.6
            quality_sensitivity = 0.6
            logger.info("⚖️ Intent 'find_best_value' → Balanced sensitivities (0.6, 0.6)")
            
        elif intent == "compare":
            # کاربر میخواد مقایسه کنه - نتایج متنوع‌تر نیاز داره
            price_sensitivity = 0.5
            quality_sensitivity = 0.5
            logger.info("🔄 Intent 'compare' → Neutral sensitivities for diverse results")
            
        elif intent == "find_by_feature":
            # کاربر ویژگی خاصی میخواد - similarity (شباهت به query) مهم‌تره
            logger.info("🔍 Intent 'find_by_feature' → Prioritizing similarity score")
    
    try:
        # Get search engine
        logger.debug("🔧 Getting search engine instance...")
        engine = get_search_engine()
        
        # Perform search with category filter
        logger.debug("🔍 Executing search...")
        results = engine.search(query, top_k=100, min_similarity=0.3, category=category)

        if not results:
            logger.warning(f"⚠️ No products found for query: '{query}'")
            return json.dumps({
                "products": [],
                "message": f"متاسفانه هیچ محصولی با جستجوی '{query}' پیدا نشد."
            }, ensure_ascii=False)
        
        logger.debug(f"📦 Got {len(results)} products from search")
        
        # Load brand scores
        logger.debug("📊 Loading brand scores...")
        brand_scores = get_brand_scores()
        logger.debug(f"✅ Loaded {len(brand_scores)} brand scores")
        
        # ═══════════════════════════════════════════════════════════════
        # 🧮 محاسبه value_score با در نظر گرفتن intent
        # ═══════════════════════════════════════════════════════════════
        logger.debug("🧮 Calculating value scores...")
        for product in results:
            # Calculate final price with discount
            price = product['price']
            discount_percentage = product['discount_percentage']
            final_price = price - (price * discount_percentage / 100)
            
            # Get brand score (default to 0.5 if brand not found)
            brand_name = product['brand_name'] if product['brand_name'] else ""
            brand_score = brand_scores.get(brand_name, 0.5)
            
            # Normalize final_price (divide by 1000000 to scale it appropriately)
            normalized_price = final_price / 1000000
            
            # ═══════════════════════════════════════════════════════════
            # 📐 فرمول‌های مختلف value_score بر اساس intent
            # ═══════════════════════════════════════════════════════════
            if intent == "find_cheapest":
                # فقط قیمت مهمه - هرچی ارزان‌تر بهتر
                value_score = (
                    -normalized_price * 2.0 +          # قیمت پایین‌تر = امتیاز بالاتر
                    product['similarity'] * 0.3 +      # یکم similarity
                    brand_score * 0.1                  # برند کمترین اهمیت
                )
                
            elif intent == "find_high_quality":
                # کیفیت و برند خیلی مهمه
                value_score = (
                    brand_score * 1.5 +                # برند خیلی مهم
                    product['similarity'] * 0.5 +      # شباهت به query
                    (discount_percentage / 100) * 0.3 - # تخفیف یه پلاس
                    normalized_price * 0.1             # قیمت کمترین اهمیت
                )
                
            elif intent == "find_by_feature":
                # similarity خیلی مهمه چون کاربر ویژگی خاصی میخواد
                value_score = (
                    product['similarity'] * 1.5 +      # شباهت به query خیلی مهم
                    brand_score * quality_sensitivity * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * price_sensitivity * 0.3
                )
                
            elif intent == "compare":
                # همه فاکتورها تقریباً برابر - برای تنوع نتایج
                value_score = (
                    brand_score * 0.3 +
                    product['similarity'] * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * 0.3
                )
                
            else:  # find_best_value یا default
                # فرمول اصلی متوازن
                value_score = (
                    brand_score * quality_sensitivity +
                    product['similarity'] * 0.4 +
                    (discount_percentage / 100) * 0.2 -
                    normalized_price * price_sensitivity
                )
            
            # Add to product dict
            product['final_price'] = int(final_price)
            product['brand_score'] = round(brand_score, 3)
            product['value_score'] = round(value_score, 3)
        
        # ═══════════════════════════════════════════════════════════════
        # 📊 تعداد نتایج بر اساس intent
        # ═══════════════════════════════════════════════════════════════
        logger.debug("🔄 Reranking by value_score...")
        results.sort(key=lambda x: x['value_score'], reverse=True)
        
        # حداقل 1 و حداکثر 5 محصول برگردون
        MIN_RESULTS = 1
        MAX_RESULTS = 5
        
        # فیلتر کردن محصولات با similarity خیلی پایین (بی‌ربط)
        RELEVANCE_THRESHOLD = 0.4
        relevant_results = [r for r in results if r['similarity'] >= RELEVANCE_THRESHOLD]
        
        if not relevant_results:
            # اگر هیچ محصول مرتبطی نبود، بهترین‌ها رو برگردون با هشدار
            logger.warning(f"⚠️ No products above relevance threshold {RELEVANCE_THRESHOLD}")
            relevant_results = results[:MIN_RESULTS] if results else []
            is_relevant = False
        else:
            is_relevant = True
        
        # محدود کردن به حداکثر 5 محصول
        results = relevant_results[:MAX_RESULTS]
        
        logger.info(f"✅ Returning {len(results)} products (min={MIN_RESULTS}, max={MAX_RESULTS}, relevant={is_relevant})")
        if results:
            logger.debug(f"🏆 Top product: {results[0]['product_name']} (value_score={results[0]['value_score']:.3f}, similarity={results[0]['similarity']:.3f})")
        
        # Format results as JSON
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
        
        # محاسبه بازه قیمت
        if products:
            prices = [p['final_price'] for p in products]
            min_price = min(prices)
            max_price = max(prices)
            avg_similarity = sum(p['similarity'] for p in products) / len(products)
        else:
            min_price = max_price = 0
            avg_similarity = 0
        
        logger.info(f"📤 Returning {len(products)} products (price range: {min_price}-{max_price})")
        
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
        logger.error(f"❌ Error in search_products_semantic: {str(e)}", exc_info=True)
        return json.dumps({
            "products": [],
            "error": "خطا در جستجوی محصولات. لطفاً دوباره تلاش کنید."
        }, ensure_ascii=False)

