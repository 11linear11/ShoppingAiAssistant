#!/usr/bin/env python3
"""
DSL Testing Script

Interactive tool to:
1. Generate DSL using LLM
2. Test DSL against Elasticsearch
3. View full results

Usage:
    python scripts/test_dsl.py
    python scripts/test_dsl.py --query "شورت"
    python scripts/test_dsl.py --query "ارزونترین یخچال" --intent find_cheapest
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
from elasticsearch import AsyncElasticsearch
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

ES_CONFIG = {
    "host": os.getenv("ELASTICSEARCH_HOST", "176.116.18.165"),
    "port": int(os.getenv("ELASTICSEARCH_PORT", 9201)),
    "scheme": os.getenv("ELASTICSEARCH_SCHEME", "http"),
    "user": os.getenv("ELASTICSEARCH_USER", "elastic"),
    "password": os.getenv("ELASTICSEARCH_PASSWORD", ""),
    "index": os.getenv("ELASTICSEARCH_INDEX", "shopping_products"),
}

GROQ_CONFIG = {
    "api_key": os.getenv("GROQ_API_KEY", ""),
    "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "base_url": "https://api.groq.com/openai/v1",
}

EMBEDDING_URL = os.getenv("MCP_EMBEDDING_URL", "http://localhost:5003")


# ============================================================================
# DSL Generation
# ============================================================================

def get_system_prompt() -> str:
    """System prompt for DSL generation."""
    return """You are an Elasticsearch DSL expert for e-commerce.
Generate valid Elasticsearch JSON queries.
Support Persian text search with standard analyzer.
Always include proper boosting and scoring.
Output ONLY valid JSON, no markdown or explanation."""


def get_user_prompt(product: str, intent: str, categories: list[str], brand: str = None) -> str:
    """Build user prompt for DSL generation."""
    # Build category instruction based on whether we have candidates
    category_instruction = ""
    if categories:
        category_instruction = f"""
CATEGORY SELECTION TASK:
Candidate categories from database: {categories}
Your task: Select ONLY the categories that are truly relevant to "{product}".
CRITICAL: You can ONLY use categories from this list! Do NOT invent new categories.
- If a category from the list is clearly related to the product, include it
- If a category seems unrelated (false positive from embedding similarity), DO NOT include it
- You may include 0, 1, or more categories based on actual relevance
- If NONE of the given categories are relevant, DO NOT add any category filter at all

Examples:
- "شورت" (underwear) with ["مد و پوشاک", "لبنیات"] → only "مد و پوشاک"
- "کفش" (shoes) with ["لبنیات", "نوشیدنی"] → NO filter (none relevant)
- "یخچال" with ["لوازم برقی و دیجیتال", "مد و پوشاک"] → only "لوازم برقی و دیجیتال" """

    return f"""Generate Elasticsearch DSL for Persian e-commerce search.

Index Schema:
- product_name (text, Persian/English)
- brand_name (text)
- category_name.keyword (keyword field - MUST use .keyword suffix for exact match)
- price (long)
- discount_price (long)
- has_discount (boolean)
- discount_percentage (float)
- product_embedding (dense_vector[768])

Search Parameters:
- Intent: {intent}
- Search Query: {product}
- Brand: {brand}
- Candidate Categories: {categories}
- Constraints: {{}}
{category_instruction}

Generate a query with:
1. multi_match on product_name^3, brand_name^2 using the Search Query
2. terms filter for relevant categories using category_name.keyword (only if categories are truly relevant)
3. range filter for price (if constraints include price_range)
4. Sort based on intent:
   - find_cheapest: sort by price asc
   - find_high_quality: sort by _score desc
   - find_best_value: sort by _score desc
5. Leave a "script_score" placeholder: {{"script_score": "PLACEHOLDER"}}

IMPORTANT: 
- For category filtering, ALWAYS use "category_name.keyword" not "category_name"
- Only include categories that make semantic sense for the product
- Example: {{"terms": {{"category_name.keyword": ["مد و پوشاک"]}}}}

Return ONLY valid Elasticsearch JSON, no explanation."""


async def generate_dsl_with_llm(product: str, intent: str, categories: list[str] = None) -> dict:
    """Generate DSL using Groq LLM."""
    categories = categories or []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{GROQ_CONFIG['base_url']}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_CONFIG['api_key']}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": get_user_prompt(product, intent, categories)},
                ],
                "temperature": 0.0,
                "max_tokens": 2000,
            },
        )
        
        response.raise_for_status()
        data = response.json()
        raw_response = data["choices"][0]["message"]["content"]
        
        print("\n" + "=" * 60)
        print("📝 RAW LLM RESPONSE:")
        print("=" * 60)
        print(raw_response)
        print("=" * 60 + "\n")
        
        # Parse JSON
        import re
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        raise ValueError("No valid JSON found in LLM response")


def clean_placeholder(dsl: dict) -> dict:
    """Remove PLACEHOLDER script_score from DSL and fix structure."""
    import copy
    dsl = copy.deepcopy(dsl)
    
    # Remove from root DSL
    if "script_score" in dsl:
        del dsl["script_score"]
    
    # Remove from function_score
    if "query" in dsl and "function_score" in dsl["query"]:
        fs = dsl["query"]["function_score"]
        if "script_score" in fs and fs["script_score"] == "PLACEHOLDER":
            del fs["script_score"]
        # Extract inner query
        inner_query = fs.get("query", {"match_all": {}})
        dsl["query"] = inner_query
    
    # Fix bool query structure - must/should/filter should be arrays
    if "query" in dsl and "bool" in dsl["query"]:
        bool_query = dsl["query"]["bool"]
        for clause in ["should", "must", "filter"]:
            if clause in bool_query:
                # Convert single object to array
                if isinstance(bool_query[clause], dict):
                    bool_query[clause] = [bool_query[clause]]
                # Filter out PLACEHOLDER and empty terms entries
                if isinstance(bool_query[clause], list):
                    cleaned = []
                    for item in bool_query[clause]:
                        if not isinstance(item, dict):
                            cleaned.append(item)
                            continue
                        # Skip PLACEHOLDER entries
                        if item.get("script_score") == "PLACEHOLDER":
                            continue
                        # Skip empty terms filter
                        if "terms" in item:
                            terms_val = item["terms"]
                            if isinstance(terms_val, dict):
                                all_empty = all(
                                    isinstance(v, list) and len(v) == 0 
                                    for v in terms_val.values()
                                )
                                if all_empty:
                                    continue
                        cleaned.append(item)
                    
                    bool_query[clause] = cleaned
                    # Remove empty arrays
                    if not bool_query[clause]:
                        del bool_query[clause]
    
    return dsl


async def add_semantic_scoring(dsl: dict, query: str) -> dict:
    """Add semantic vector scoring to DSL."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{EMBEDDING_URL}/embed",
                json={"text": query, "normalize": True},
            )
            
            if response.status_code != 200:
                print("⚠️ Embedding server not available, skipping semantic scoring")
                return dsl
            
            embedding = response.json()["embedding"]
        
        # Clean any PLACEHOLDER first
        dsl = clean_placeholder(dsl)
        
        # Extract original query
        original_query = dsl.get("query", {"match_all": {}})
        
        # If there's a function_score, extract inner query
        if "function_score" in original_query:
            original_query = original_query["function_score"].get("query", {"match_all": {}})
        
        dsl["query"] = {
            "function_score": {
                "query": original_query,
                "functions": [
                    {
                        "script_score": {
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
                                "params": {"query_vector": embedding},
                            }
                        }
                    }
                ],
                "boost_mode": "sum",
                "score_mode": "sum",
            }
        }
        
        return dsl
        
    except Exception as e:
        print(f"⚠️ Error adding semantic scoring: {e}")
        return clean_placeholder(dsl)


# ============================================================================
# Elasticsearch Execution
# ============================================================================

async def execute_search(dsl: dict) -> dict:
    """Execute DSL against Elasticsearch."""
    es_url = f"{ES_CONFIG['scheme']}://{ES_CONFIG['host']}:{ES_CONFIG['port']}"
    
    es = AsyncElasticsearch(
        [es_url],
        basic_auth=(ES_CONFIG["user"], ES_CONFIG["password"]) if ES_CONFIG["password"] else None,
        verify_certs=False,
        request_timeout=30,
    )
    
    try:
        response = await es.search(
            index=ES_CONFIG["index"],
            body=dsl,
        )
        return response
    finally:
        await es.close()


def format_results(response: dict) -> None:
    """Pretty print search results."""
    hits = response.get("hits", {})
    total = hits.get("total", {}).get("value", 0)
    took = response.get("took", 0)
    
    print(f"\n📊 RESULTS: {total} hits in {took}ms\n")
    print("-" * 80)
    
    for i, hit in enumerate(hits.get("hits", [])[:10], 1):
        source = hit.get("_source", {})
        score = hit.get("_score") or 0
        
        name = source.get("product_name", "N/A")
        brand = source.get("brand_name", "N/A")
        category = source.get("category_name", "N/A")
        price = source.get("price") or 0
        discount_price = source.get("discount_price")
        has_discount = source.get("has_discount", False)
        
        price_str = f"{int(price):,} تومان" if price else "N/A"
        if has_discount and discount_price:
            price_str = f"{int(discount_price):,} تومان (was {int(price):,})"
        
        print(f"{i:2}. [{score:.2f}] {name}")
        print(f"     Brand: {brand} | Category: {category}")
        print(f"     Price: {price_str}")
        print()


# ============================================================================
# Interactive Test
# ============================================================================

async def test_dsl(
    product: str,
    intent: str = "browse",
    categories: list[str] = None,
    use_semantic: bool = True,
):
    """Run full DSL test."""
    categories = categories or []
    
    print("\n" + "=" * 80)
    print(f"🔍 TESTING DSL GENERATION")
    print(f"   Product: {product}")
    print(f"   Intent: {intent}")
    print(f"   Categories: {categories}")
    print(f"   Semantic: {use_semantic}")
    print("=" * 80)
    
    # Step 1: Generate DSL with LLM
    print("\n⏳ Generating DSL with LLM...")
    dsl = await generate_dsl_with_llm(product, intent, categories)
    
    print("\n📋 PARSED DSL (before semantic):")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))
    
    # Step 2: Clean PLACEHOLDER
    dsl = clean_placeholder(dsl)
    print("\n📋 CLEANED DSL (after removing PLACEHOLDER):")
    print(json.dumps(dsl, indent=2, ensure_ascii=False))
    
    # Step 3: Add semantic scoring
    if use_semantic:
        print("\n⏳ Adding semantic scoring...")
        dsl = await add_semantic_scoring(dsl, product)
        print("\n📋 FINAL DSL (with semantic):")
        print(json.dumps(dsl, indent=2, ensure_ascii=False))
    
    # Step 4: Execute search
    print("\n⏳ Executing search on Elasticsearch...")
    try:
        response = await execute_search(dsl)
        format_results(response)
    except Exception as e:
        print(f"\n❌ Elasticsearch Error: {e}")
        print("\nTrying without semantic scoring...")
        dsl = clean_placeholder(await generate_dsl_with_llm(product, intent, categories))
        try:
            response = await execute_search(dsl)
            format_results(response)
        except Exception as e2:
            print(f"❌ Still failing: {e2}")


async def interactive_mode():
    """Interactive testing mode."""
    print("\n" + "=" * 80)
    print("🧪 DSL TESTING TOOL - Interactive Mode")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type a product name to search")
    print("  - Use 'intent:find_cheapest' to change intent")
    print("  - Use 'cat:لوازم برقی' to add category filter")
    print("  - Use 'nosem' to disable semantic scoring")
    print("  - Type 'quit' to exit\n")
    
    intent = "browse"
    categories = []
    use_semantic = True
    
    while True:
        try:
            user_input = input("🔎 Search: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break
            
            # Parse special commands
            if user_input.startswith("intent:"):
                intent = user_input.split(":", 1)[1].strip()
                print(f"✓ Intent set to: {intent}")
                continue
            
            if user_input.startswith("cat:"):
                cat = user_input.split(":", 1)[1].strip()
                if cat:
                    categories.append(cat)
                    print(f"✓ Categories: {categories}")
                continue
            
            if user_input == "clearcats":
                categories = []
                print("✓ Categories cleared")
                continue
            
            if user_input == "nosem":
                use_semantic = not use_semantic
                print(f"✓ Semantic scoring: {use_semantic}")
                continue
            
            # Run test
            await test_dsl(user_input, intent, categories, use_semantic)
            
        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSL Testing Tool")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--intent", "-i", default="browse", help="Search intent")
    parser.add_argument("--category", "-c", action="append", help="Category filter")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic scoring")
    
    args = parser.parse_args()
    
    if args.query:
        await test_dsl(
            args.query,
            args.intent,
            args.category or [],
            not args.no_semantic,
        )
    else:
        await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
