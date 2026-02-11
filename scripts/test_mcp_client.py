#!/usr/bin/env python3
"""
Test MCP Client - verify MCP communication works.
"""

import asyncio
import sys
sys.path.insert(0, "/home/moz/Projects/pyProjects/AiAssistantV3")

from src.mcp_client import InterpretMCPClient, SearchMCPClient, EmbeddingMCPClient, CacheMCPClient


async def test_interpret():
    """Test Interpret MCP server."""
    print("\n" + "="*50)
    print("Testing Interpret MCP Server (port 5004)")
    print("="*50)
    
    client = InterpretMCPClient("http://localhost:5004")
    
    try:
        # Test classify_query
        print("\n1. Testing classify_query...")
        result = await client.classify_query("بهترین گوشی سامسونگ")
        print(f"   Result: {result}")
        
        # Test interpret_query
        print("\n2. Testing interpret_query...")
        result = await client.interpret_query(
            query="گوشی سامسونگ زیر 10 میلیون",
            session_id="test-session-123"
        )
        print(f"   Query Type: {result.get('query_type', 'N/A')}")
        print(f"   Searchable: {result.get('searchable', 'N/A')}")
        print(f"   Params: {result.get('search_params', 'N/A')}")
        
        print("\n✅ Interpret server tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        await client.close()


async def test_search():
    """Test Search MCP server."""
    print("\n" + "="*50)
    print("Testing Search MCP Server (port 5002)")
    print("="*50)
    
    client = SearchMCPClient("http://localhost:5002")
    
    try:
        # Test search with proper params
        print("\n1. Testing search_products...")
        search_params = {
            "intent": "browse",
            "product": "گوشی سامسونگ",
            "persian_full_query": "گوشی سامسونگ"
        }
        result = await client.search_products(
            search_params=search_params,
            session_id="test-session-123",
            use_cache=False
        )
        
        if result.get("success"):
            print(f"   Found {result.get('total_hits', 0)} products")
            results = result.get("results", [])
            for p in results[:2]:
                name = p.get('name', 'N/A')
                print(f"   - {name[:50]}...")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        print("\n✅ Search server tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        await client.close()


async def test_embedding():
    """Test Embedding MCP server."""
    print("\n" + "="*50)
    print("Testing Embedding MCP Server (port 5003)")
    print("="*50)
    
    client = EmbeddingMCPClient("http://localhost:5003")
    
    try:
        # Test embedding generation
        print("\n1. Testing generate_embedding...")
        result = await client.generate_embedding("گوشی موبایل")
        
        if result.get("success"):
            emb = result.get("embedding", [])
            print(f"   Embedding dimension: {len(emb)}")
            print(f"   First 5 values: {emb[:5]}")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        
        print("\n✅ Embedding server tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        await client.close()


async def test_cache():
    """Test Cache MCP server."""
    print("\n" + "="*50)
    print("Testing Cache MCP Server (port 5007)")
    print("="*50)
    
    client = CacheMCPClient("http://localhost:5007")
    
    try:
        # Test set/get component cache
        print("\n1. Testing set_component...")
        result = await client.set_component(
            key="test:mcp:client",
            value={"test": True, "message": "MCP works!"},
            ttl=60
        )
        print(f"   Set result: {result}")
        
        print("\n2. Testing get_component...")
        result = await client.get_component("test:mcp:client")
        print(f"   Get result: {result}")
        
        print("\n3. Testing get_cache_stats...")
        result = await client.get_cache_stats()
        print(f"   Stats: {result}")
        
        print("\n✅ Cache server tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    finally:
        await client.close()


async def main():
    """Run all MCP tests."""
    print("="*60)
    print("   MCP Client Test Suite")
    print("="*60)
    
    results = {}
    
    # Test each server
    results["cache"] = await test_cache()
    results["embedding"] = await test_embedding()
    results["interpret"] = await test_interpret()
    results["search"] = await test_search()
    
    print("\n" + "="*60)
    print("   Summary")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name.capitalize()}: {status}")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
