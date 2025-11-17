"""
Shopping AI Assistant using LangGraph
A conversational agent that can search for products using Elasticsearch and semantic search.
"""

from dotenv import load_dotenv
load_dotenv()

import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode ,tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import the semantic search tool
from .tools.SearchProducts import search_products_semantic


# Configuration
api_key = os.getenv("api_key")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# System prompt
SYSTEM_PROMPT = """شما یک دستیار خرید هوشمند برای یک فروشگاه آنلاین ایرانی هستید.

قوانین مهم:
1. اگر کاربر سلام کرد یا سوال عادی پرسید، به فارسی جواب بده
3. پاسخ های خودت رو در قالب JSON قرار بده به شکل {"message": "متن پاسخ شما"}
2. هیچ متن اضافی قبل یا بعد از JSON نگذار


"""


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
    llm = ChatNVIDIA(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        base_url=BASE_URL
    )
    
    
    # Bind tools to LLM
    tools = [search_products_semantic]
    llm_with_tools = llm.bind_tools(tools)
    
    # Define chatbot node
    def chatbot_node(state):
        """Process messages and generate responses."""
        # Add system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(tools)
    
    
    # Build graph
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    
    # Define edges
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools",END)
    
    # Compile with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
