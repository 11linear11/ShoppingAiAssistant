"""
Shopping AI Assistant using LangGraph
A conversational agent that can search for products using Elasticsearch and semantic search.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the semantic search tool
from tools.SearchProducts import search_products_semantic


# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://models.inference.ai.azure.com"


class State(TypedDict):
    """State definition for the conversation graph."""
    messages: Annotated[list, add_messages]


def create_agent():
    """
    Create and configure the LangGraph agent with Elasticsearch-based product search.
    
    Returns:
        Compiled graph with memory
    """
    # Initialize LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )
    
    # Bind tools to LLM
    tools = [search_products_semantic]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define chatbot node
    def chatbot_node(state):
        """Process messages and generate responses."""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    # Define routing function
    def should_continue(state):
        """Determine if we should call tools or end."""
        last_message = state["messages"][-1]
        # If there are tool calls, route to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise end
        return END
    
    # Build graph
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    
    # Define edges
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
    builder.add_edge("tools", "chatbot")
    
    # Compile with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


