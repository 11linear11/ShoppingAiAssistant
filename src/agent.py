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
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import the semantic search tool
from .tools.SearchProducts import search_products_semantic


# Configuration
OPENAI_API_KEY = os.getenv("api_key")
BASE_URL = "https://integrate.api.nvidia.com/v1"

# System prompt
SYSTEM_PROMPT = """شما یک دستیار خرید هوشمند برای یک فروشگاه آنلاین ایرانی هستید.

قوانین مهم:
1. اگر کاربر سلام کرد یا سوال عادی پرسید، به فارسی جواب بده
2. وقتی نتایج جستجوی محصول دریافت کردی (از tool)، دقیقاً همان JSON را بدون هیچ تغییری به کاربر برگردان
3. هیچ متن اضافی قبل یا بعد از JSON نگذار

مثال پاسخ برای چت عادی: {"message": "سلام! چطور می‌تونم کمکتون کنم؟"}
مثال پاسخ برای جستجو: فقط JSON دریافتی از tool را برگردان"""


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
        api_key=OPENAI_API_KEY,
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
    
    # JSON response node - returns tool output directly
    def json_response_node(state):
        """Return the JSON from tool directly without LLM processing."""
        # Find the last ToolMessage
        for msg in reversed(state["messages"]):
            if isinstance(msg, ToolMessage):
                # Create an AI message with the tool's JSON output
                return {"messages": [AIMessage(content=msg.content)]}
        # Fallback: let LLM handle it
        return {"messages": []}
    
    # Define routing function from chatbot
    def should_continue(state):
        """Determine if we should call tools or end."""
        last_message = state["messages"][-1]
        # If there are tool calls, route to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise end
        return END
    
    # Define routing function from tools
    def after_tools(state):
        """After tools, go directly to json_response instead of chatbot."""
        return "json_response"
    
    # Build graph
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    builder.add_node("json_response", json_response_node)
    
    # Define edges
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", should_continue, ["tools", END])
    builder.add_edge("tools", "json_response")
    builder.add_edge("json_response", END)
    
    # Compile with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)
