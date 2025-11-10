"""
Shopping AI Assistant - Main Entry Point
A conversational agent for product search using LangGraph and Elasticsearch.
"""

from src.agent import create_agent
from langchain_core.messages import HumanMessage


def main():
    """Main function to run the agent."""
    print("Shopping AI Assistant\n" + "=" * 50)
    print("Type 'exit' to quit\n")
    
    # Create agent
    graph = create_agent()
    
    # Example conversation with thread management
    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("خداحافظ! (Goodbye!)")
            break
        
        if not user_input.strip():
            continue
            
        try:
            state = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            print(f"Assistant: {state['messages'][-1].content}")
        except Exception as e:
            print(f"خطا: {str(e)}")


if __name__ == "__main__":
    main()
