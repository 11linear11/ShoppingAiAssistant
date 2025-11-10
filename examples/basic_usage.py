"""
Basic usage example for Shopping AI Assistant
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import create_agent
from langchain_core.messages import HumanMessage


def main():
    """Example of basic agent usage."""
    # Create agent
    graph = create_agent()
    
    # Configuration with thread management
    thread_id = "example_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Example queries
    queries = [
        "سلام!",
        "دوغ آبعلی میخوام",
        "شکلات میخوام"
    ]
    
    print("Shopping AI Assistant - Basic Example")
    print("=" * 60)
    
    for query in queries:
        print(f"\n👤 User: {query}")
        
        state = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        response = state['messages'][-1].content
        print(f"🤖 Assistant: {response[:200]}...")  # Show first 200 chars
        print("-" * 60)


if __name__ == "__main__":
    main()
