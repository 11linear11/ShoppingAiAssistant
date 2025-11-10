"""
Product Search Tool using Elasticsearch and Multilingual Embeddings
This module provides semantic search functionality for products.
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from langchain_core.tools import tool

load_dotenv()


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
            
        # Load model silently
        model_name = 'intfloat/multilingual-e5-base'
        self.model = SentenceTransformer(model_name)
        
        # Elasticsearch configuration
        ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost")
        ES_PORT = os.getenv("ELASTICSEARCH_PORT", "9200")
        ES_USERNAME = os.getenv("ELASTICSEARCH_USER")
        ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
        scheme = os.getenv("ELASTICSEARCH_SCHEME", "http")
        
        es_url = f"{scheme}://{ES_HOST}:{ES_PORT}"
        
        # Create Elasticsearch client
        try:
            if ES_USERNAME and ES_PASSWORD:
                self.es = Elasticsearch(es_url, basic_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False)
            else:
                self.es = Elasticsearch(es_url, verify_certs=False)
        except TypeError:
            # Fallback for older versions
            host_dict = {"host": ES_HOST, "port": int(ES_PORT), "scheme": scheme}
            if ES_USERNAME and ES_PASSWORD:
                self.es = Elasticsearch([host_dict], http_auth=(ES_USERNAME, ES_PASSWORD), verify_certs=False)
            else:
                self.es = Elasticsearch([host_dict], verify_certs=False)
        
        self.index_name = os.getenv("ELASTICSEARCH_INDEX", "shopping_products")
        
        ProductSearchEngine._initialized = True
    
    def search(self, query_text: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Dict]:
        """
        Perform semantic similarity search.
        
        Args:
            query_text: Search query
            top_k: Number of results
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            List of product dictionaries
        """
        # Generate embedding
        query_embedding = self.model.encode([query_text])[0].tolist()
        
        # Search query
        search_body = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=search_body)
            results = []
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                similarity = hit['_score'] - 1.0  # Convert back to cosine similarity
                
                # Filter by minimum similarity
                if similarity >= min_similarity:
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
                        'score': hit['_score']
                    })
            
            return results
            
        except Exception as e:
            return []


# Global instance
_search_engine = None


def get_search_engine() -> ProductSearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = ProductSearchEngine()
    return _search_engine


@tool
def search_products_semantic(query: str) -> str:
    """
    Search for products using semantic search with Elasticsearch.
    Use this tool when the user wants to find, search, or look for products.
    This tool understands natural language in multiple languages (English, Persian, Arabic, etc.).
    
    Args:
        query: The product search query in natural language (e.g., "لپ تاپ گیمینگ", "cheap smartphone", "هدفون بی سیم")
        
    Returns:
        A formatted string with the search results including product details and relevance scores.
        Results are automatically reranked by the AI agent based on semantic relevance to the query.
        It should show all details of each product including product ID, name, brand, price, discount, category, and similarity score .        
        It should be 5 results.
    """
    try:
        # Get search engine
        engine = get_search_engine()
        
        # Perform search
        results = engine.search(query, top_k=5, min_similarity=0.3)

        if not results:
            return json.dumps({
                "products": [],
                "message": f"متاسفانه هیچ محصولی با جستجوی '{query}' پیدا نشد."
            }, ensure_ascii=False)
        
        # Format results as JSON
        products = []
        for product in results:
            products.append({
                "name": product['product_name'],
                "price": int(product['price']),
                "brand": product['brand_name'] if product['brand_name'] else "",
                "discount": int(product['discount_percentage']) if product['has_discount'] else 0,
                "product_id": str(product['product_id']),
                "similarity": round(product['similarity'], 3),
                "category": product['category_name'] if product['category_name'] else ""
            })
        
        return json.dumps({"products": products}, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "products": [],
            "error": "خطا در جستجوی محصولات. لطفاً دوباره تلاش کنید."
        }, ensure_ascii=False)

