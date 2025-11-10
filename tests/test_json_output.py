"""
Test JSON output format for Shopping AI Assistant
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import create_agent
from langchain_core.messages import HumanMessage
import json


def test_product_search():
    """Test product search with JSON output."""
    print("=" * 60)
    print("TEST 1: جستجوی محصول")
    print("=" * 60)
    
    graph = create_agent()
    thread_id = "test_json"
    config = {"configurable": {"thread_id": thread_id}}
    
    query = "دوغ پیدا کن برام"
    state = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config
    )
    
    response = state['messages'][-1].content
    print(f"\nResponse:\n{response}\n")
    
    # Try to parse as JSON
    try:
        data = json.loads(response)
        print("✅ Valid JSON!")
        print(f"Number of products: {len(data.get('products', []))}")
        if data.get('products'):
            print(f"First product: {data['products'][0]['name']}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")


def test_normal_chat():
    """Test normal chat with JSON output."""
    print("\n" + "=" * 60)
    print("TEST 2: چت عادی")
    print("=" * 60)
    
    graph = create_agent()
    thread_id = "test_json"
    config = {"configurable": {"thread_id": thread_id}}
    
    query2 = "سلام"
    state2 = graph.invoke(
        {"messages": [HumanMessage(content=query2)]},
        config=config
    )
    
    response2 = state2['messages'][-1].content
    print(f"\nResponse:\n{response2}\n")
    
    try:
        data2 = json.loads(response2)
        print("✅ Valid JSON!")
        print(f"Message: {data2.get('message', 'N/A')}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")


if __name__ == "__main__":
    test_product_search()
    test_normal_chat()
