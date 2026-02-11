#!/usr/bin/env python3
"""
Seed Data Script

Seeds Elasticsearch with sample product data for testing
the Shopping AI Assistant.
"""

import asyncio
import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import httpx
from elasticsearch import AsyncElasticsearch

# ============================================================================
# Sample Data
# ============================================================================

SAMPLE_PRODUCTS = [
    # Laptops
    {
        "product_name": "لپتاپ ایسوس VivoBook 15",
        "brand_name": "ایسوس",
        "category_name": "لپتاپ",
        "price": 28000000,
        "discount_price": 25500000,
        "has_discount": True,
        "discount_percentage": 8.9,
    },
    {
        "product_name": "لپتاپ ایسوس ROG Strix گیمینگ",
        "brand_name": "ایسوس",
        "category_name": "لپتاپ",
        "price": 65000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "لپتاپ لنوو IdeaPad 3",
        "brand_name": "لنوو",
        "category_name": "لپتاپ",
        "price": 22000000,
        "discount_price": 19800000,
        "has_discount": True,
        "discount_percentage": 10,
    },
    {
        "product_name": "لپتاپ اچ‌پی Pavilion 15",
        "brand_name": "اچ‌پی",
        "category_name": "لپتاپ",
        "price": 32000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "مک‌بوک ایر M2",
        "brand_name": "اپل",
        "category_name": "لپتاپ",
        "price": 72000000,
        "discount_price": 68000000,
        "has_discount": True,
        "discount_percentage": 5.5,
    },
    # Phones
    {
        "product_name": "آیفون 15 پرو مکس",
        "brand_name": "اپل",
        "category_name": "گوشی موبایل",
        "price": 95000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "سامسونگ گلکسی S24 اولترا",
        "brand_name": "سامسونگ",
        "category_name": "گوشی موبایل",
        "price": 78000000,
        "discount_price": 72000000,
        "has_discount": True,
        "discount_percentage": 7.7,
    },
    {
        "product_name": "شیائومی 14 پرو",
        "brand_name": "شیائومی",
        "category_name": "گوشی موبایل",
        "price": 42000000,
        "discount_price": 38000000,
        "has_discount": True,
        "discount_percentage": 9.5,
    },
    {
        "product_name": "گوشی وان‌پلاس 12",
        "brand_name": "وان‌پلاس",
        "category_name": "گوشی موبایل",
        "price": 38000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    # Jackets
    {
        "product_name": "کاپشن زمستانی مردانه",
        "brand_name": "کت‌برد",
        "category_name": "کاپشن",
        "price": 3500000,
        "discount_price": 2800000,
        "has_discount": True,
        "discount_percentage": 20,
    },
    {
        "product_name": "کاپشن پر زنانه",
        "brand_name": "نایک",
        "category_name": "کاپشن",
        "price": 5200000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "کاپشن اسپرت آدیداس",
        "brand_name": "آدیداس",
        "category_name": "کاپشن",
        "price": 4800000,
        "discount_price": 4200000,
        "has_discount": True,
        "discount_percentage": 12.5,
    },
    # Headphones
    {
        "product_name": "هدفون سونی WH-1000XM5",
        "brand_name": "سونی",
        "category_name": "هدفون",
        "price": 18000000,
        "discount_price": 16500000,
        "has_discount": True,
        "discount_percentage": 8.3,
    },
    {
        "product_name": "ایرپاد پرو 2",
        "brand_name": "اپل",
        "category_name": "هدفون",
        "price": 12000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "هدفون گیمینگ ریزر",
        "brand_name": "ریزر",
        "category_name": "هدفون",
        "price": 8500000,
        "discount_price": 7200000,
        "has_discount": True,
        "discount_percentage": 15.3,
    },
    # Watches
    {
        "product_name": "اپل واچ سری 9",
        "brand_name": "اپل",
        "category_name": "ساعت هوشمند",
        "price": 25000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "گلکسی واچ 6 کلاسیک",
        "brand_name": "سامسونگ",
        "category_name": "ساعت هوشمند",
        "price": 18000000,
        "discount_price": 15500000,
        "has_discount": True,
        "discount_percentage": 13.9,
    },
    # Home Appliances
    {
        "product_name": "یخچال ساید بای ساید سامسونگ",
        "brand_name": "سامسونگ",
        "category_name": "یخچال",
        "price": 85000000,
        "discount_price": 78000000,
        "has_discount": True,
        "discount_percentage": 8.2,
    },
    {
        "product_name": "لباسشویی ال‌جی 9 کیلویی",
        "brand_name": "ال‌جی",
        "category_name": "لباسشویی",
        "price": 42000000,
        "discount_price": None,
        "has_discount": False,
        "discount_percentage": 0,
    },
    {
        "product_name": "جاروبرقی دایسون V15",
        "brand_name": "دایسون",
        "category_name": "جاروبرقی",
        "price": 28000000,
        "discount_price": 25000000,
        "has_discount": True,
        "discount_percentage": 10.7,
    },
]


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding using the embedding server."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:5003/embed",
                json={"text": text, "normalize": True},
            )
            if response.status_code == 200:
                return response.json()["embedding"]
    except Exception as e:
        print(f"⚠️  Could not generate embedding: {e}")

    # Return random embedding if server not available
    return [random.random() * 2 - 1 for _ in range(768)]


async def seed_data():
    """Seed Elasticsearch with sample data."""
    print("🌱 Seeding Elasticsearch with sample data...\n")

    # Connect to Elasticsearch
    es = AsyncElasticsearch(
        ["http://localhost:9200"],
        verify_certs=False,
        request_timeout=30,
    )

    try:
        # Check connection
        info = await es.info()
        print(f"✅ Connected to Elasticsearch {info['version']['number']}")

        # Check if index exists
        index_exists = await es.indices.exists(index="shopping_products")

        if not index_exists:
            print("⚠️  Index 'shopping_products' does not exist.")
            print("Run scripts/setup_elasticsearch.sh first.")
            return

        # Delete existing documents
        print("\n🗑️  Clearing existing data...")
        await es.delete_by_query(
            index="shopping_products",
            body={"query": {"match_all": {}}},
            ignore=[404],
        )

        # Index products
        print(f"\n📦 Indexing {len(SAMPLE_PRODUCTS)} products...")

        for i, product in enumerate(SAMPLE_PRODUCTS, 1):
            # Generate embedding for product
            text_for_embedding = (
                f"{product['product_name']} {product['brand_name']} {product['category_name']}"
            )
            embedding = await generate_embedding(text_for_embedding)
            product["product_embedding"] = embedding

            # Index document
            await es.index(
                index="shopping_products",
                document=product,
            )
            print(f"  ✓ {i}/{len(SAMPLE_PRODUCTS)}: {product['product_name']}")

        # Refresh index
        await es.indices.refresh(index="shopping_products")

        # Get count
        count = await es.count(index="shopping_products")
        print(f"\n✅ Successfully indexed {count['count']} products!")

        # Show sample query
        print("\n📊 Sample search test:")
        result = await es.search(
            index="shopping_products",
            body={
                "query": {
                    "multi_match": {
                        "query": "لپتاپ",
                        "fields": ["product_name", "brand_name"],
                    }
                },
                "size": 3,
            },
        )

        print(f"   Query: 'لپتاپ' -> Found {result['hits']['total']['value']} results")
        for hit in result["hits"]["hits"][:3]:
            print(f"   - {hit['_source']['product_name']}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await es.close()


if __name__ == "__main__":
    asyncio.run(seed_data())
