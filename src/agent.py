"""
Shopping AI Assistant using LangGraph with MCP Clients
A conversational agent that uses MCP servers for search and interpretation.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import logging
import asyncio
from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
INTERPRET_SERVER_URL = os.getenv("MCP_INTERPRET_URL", "http://localhost:5004")
SEARCH_SERVER_URL = os.getenv("MCP_SEARCH_URL", "http://localhost:5002")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)

if logger.handlers:
    logger.handlers.clear()

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if DEBUG_MODE:
    file_handler = logging.FileHandler('agent_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Agent DEBUG_MODE is ON")


async def call_mcp_tool(server_url: str, tool_name: str, arguments: dict) -> dict:
    """Call an MCP server tool and return the result."""
    try:
        async with streamablehttp_client(f"{server_url}/") as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if result.content:
                    return json.loads(result.content[0].text)
                return {"error": "No content returned"}
    except Exception as e:
        logger.error(f"Error calling MCP tool {tool_name}: {e}")
        return {"error": str(e)}


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@tool
def interpret_query(query: str) -> str:
    """
    Analyze user shopping intent and extract structured information.
    Use this FIRST to understand what the user wants.
    
    Args:
        query: User's shopping query in natural language (Persian)
        
    Returns:
        JSON with equip_prompt, token_mapping, persian_full_query, categories_fa, intent, price_sensitivity, quality_sensitivity
    """
    logger.info(f"interpret_query called: '{query}'")
    result = run_async(call_mcp_tool(INTERPRET_SERVER_URL, "interpret_query", {"query": query}))
    logger.info(f"interpret_query result keys: {result.keys() if isinstance(result, dict) else 'not dict'}")
    return json.dumps(result, ensure_ascii=False)


@tool
def search_with_interpretation(
    equip_prompt: str,
    token_mapping: str,
    persian_full_query: str,
    categories_fa: str,
    intent: str = "find_best_value",
    price_sensitivity: float = 0.5,
    quality_sensitivity: float = 0.5
) -> str:
    """
    Search for products using interpretation results.
    Use this AFTER interpret_query to get products.
    
    Args:
        equip_prompt: Structured prompt from interpret_query (e.g., "product_name: laptop sort: price_asc")
        token_mapping: JSON string of English→Persian mappings from interpret_query
        persian_full_query: Persian product description from interpret_query
        categories_fa: JSON string of Persian categories from interpret_query
        intent: Shopping intent from interpret_query
        price_sensitivity: Price sensitivity from interpret_query (0-1)
        quality_sensitivity: Quality sensitivity from interpret_query (0-1)
        
    Returns:
        JSON string with ranked products and metadata
    """
    logger.info(f"search_with_interpretation called: equip_prompt='{equip_prompt}'")
    arguments = {
        "equip_prompt": equip_prompt,
        "token_mapping": token_mapping,
        "persian_full_query": persian_full_query,
        "categories_fa": categories_fa,
        "intent": intent,
        "price_sensitivity": price_sensitivity,
        "quality_sensitivity": quality_sensitivity,
    }
    
    result = run_async(call_mcp_tool(SEARCH_SERVER_URL, "search_with_interpretation", arguments))
    logger.info(f"search_with_interpretation found {len(result.get('products', []))} products")
    return json.dumps(result, ensure_ascii=False)


SYSTEM_PROMPT = """You are an intelligent Persian shopping assistant.

You have two tools that MUST be used in order:

1) interpret_query - Analyzes user query, returns:
   - equip_prompt: structured search prompt
   - token_mapping: word translations
   - persian_full_query: full Persian query
   - categories_fa: relevant categories
   - intent: shopping intent
   - price_sensitivity, quality_sensitivity: user preferences

2) search_with_interpretation - Searches products using ALL outputs from interpret_query

WORKFLOW:
1. Call interpret_query with user's Persian query
2. Call search_with_interpretation with ALL fields from interpret_query result
3. Present results nicely in Persian

IMPORTANT:
- Pass token_mapping and categories_fa as JSON strings exactly as received
- Always use the exact values from interpret_query
- Format final response in Persian with product names, prices, and brands
"""


class State(TypedDict):
    messages: Annotated[list, add_messages]


def create_agent():
    logger.info("Creating Shopping AI Agent with MCP Clients...")
    
    llm = ChatOpenAI(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
    )
    
    tools = [interpret_query, search_with_interpretation]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot_node(state):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm_with_tools.invoke(messages)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"Tool calls: {len(response.tool_calls)}")
        
        return {"messages": [response]}
    
    tool_node = ToolNode(tools)
    
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    logger.info("Agent compiled successfully with MCP clients")
    return graph
