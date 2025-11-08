from agent import *

def main():
    """Main function to run the agent."""
    print("Shopping AI Assistant\n" + "=" * 50)
    
    # Create agent
    graph = create_agent()
    
    # Example conversation with thread management
    thread_id = "user_session_1"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        state = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config
        )
        print(f"Assistant: {state['messages'][-1].content}")


if __name__ == "__main__":
    main()