"""
Shopping AI Assistant using LangGraph
A conversational agent that can search for products using Elasticsearch and semantic search.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import logging
from typing import Annotated
from typing_extensions import TypedDict

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# Import the semantic search tools
from .tools.SearchProducts import search_products_semantic, interpret_query


# Configuration
api_key = os.getenv("api_key")
BASE_URL = "https://integrate.api.nvidia.com/v1"
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Setup logging properly
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)

# Remove existing handlers
if logger.handlers:
    logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.WARNING)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler for debug mode
if DEBUG_MODE:
    file_handler = logging.FileHandler('agent_debug.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("🔧 Agent DEBUG_MODE is ON")

# System prompt
SYSTEM_PROMPT = """You are an intelligent shopping assistant. Your task is that when the user intends to find a product, you analyze their phrase and then perform a product search.

You only have two tools:

1) interpret_query  
   Input: {"query": "<full user text>"}  
   Output: includes information such as categories (LIST of top 3 categories), intent, price_sensitivity, quality_sensitivity, AND suggested_query

2) search_products_semantic  
   Input: {"query": "<product keyword>", "quality_sensitivity": 0.5, "price_sensitivity": 0.5, "categories": ["cat1", "cat2", ...], "intent": "<intent>"}  
   You must use ALL outputs from interpret_query including categories (as a LIST), intent, AND suggested_query.

-----------------------------------------------
### Mandatory Rules
- If the user intends to buy or search for a product, you **must** use both tools.
- Always call interpret_query first.
- Then based on its output, call search_products_semantic and **you must pass ALL these fields**:
  * query: USE "suggested_query" from interpret_query output! (This is the most important field)
  * categories: from interpret_query output (THIS IS A LIST like ["کتگوری1", "کتگوری2", "کتگوری3"])
  * intent: from interpret_query output (ALWAYS pass this!)
  * price_sensitivity: from interpret_query output
  * quality_sensitivity: from interpret_query output
  
- Example flow 1 (direct product mention):
  1. User says: "یه هدفون ارزان میخوام"
  2. Call interpret_query({"query": "یه هدفون ارزان میخوام"})
  3. Get result: {"categories": ["لوازم الکترونیکی", "لوازم برقی و دیجیتال", "کودک و نوزاد"], "intent": "find_cheapest", "price_sensitivity": 1.0, "quality_sensitivity": 0.0, "suggested_query": "هدفون"}
  4. Call search_products_semantic({
       "query": "هدفون",  ← از suggested_query
       "categories": ["لوازم الکترونیکی", "لوازم برقی و دیجیتال", "کودک و نوزاد"],  ← لیست کتگوری‌ها!
       "intent": "find_cheapest",
       "price_sensitivity": 1.0,
       "quality_sensitivity": 0.0
     })

- Example flow 2 (implicit intent - user describes need):
  1. User says: "یچیز میخوام بپوشم سردم نشه"
  2. Call interpret_query({"query": "یچیز میخوام بپوشم سردم نشه"})
  3. Get result: {"categories": ["مد و پوشاک", "کودک و نوزاد", "خانه و سبک زندگی"], "intent": "find_by_feature", "price_sensitivity": 0.5, "quality_sensitivity": 0.5, "suggested_query": "کاپشن"}
  4. Call search_products_semantic({
       "query": "کاپشن",  ← از suggested_query (نه متن کاربر!)
       "categories": ["مد و پوشاک", "کودک و نوزاد", "خانه و سبک زندگی"],  ← لیست کتگوری‌ها!
       "intent": "find_by_feature",
       "price_sensitivity": 0.5,
       "quality_sensitivity": 0.5
     })

- The final output must only be the result of search_products_semantic in an organized and structured format.

-----------------------------------------------
### Output Format Rules (VERY IMPORTANT!)

When you receive products from search_products_semantic, you MUST:

1. **Check Relevance First:**
   - Compare the user's original query with the found products
   - If products don't match the user's intent, say so clearly
   - Example: User asked for "هدفون" but got "کابل شارژ" → Tell user no relevant products found

2. **Show Min/Max Products:**
   - Minimum: Show at least 1 product (the best match)
   - Maximum: Show at most 5 products
   - If more than 5, show top 5 by value_score

3. **Format Each Product in Persian:**
```
🛒 [نام محصول]
   💰 قیمت: [final_price] تومان
   🏷️ برند: [brand]
   🔥 تخفیف: [discount]%
```

4. **Add Summary at End:**
```
---
📊 خلاصه: [تعداد] محصول یافت شد | بازه قیمت: [min] - [max] تومان
```

5. **Relevance Check Response:**
   If products are NOT relevant to query:
```
متأسفانه محصولی مرتبط با "[query کاربر]" پیدا نشد.
پیشنهاد: [یک پیشنهاد مرتبط]
```


-----------------------------------------------
### Intent meanings (for your reference):
- find_cheapest: User wants the cheapest option → results sorted by lowest price
- find_high_quality: User wants best quality → results sorted by brand score
- find_best_value: User wants best price/quality ratio → balanced results
- find_by_feature: User mentioned specific feature → results prioritize similarity
- compare: User wants to compare options → more diverse results shown

-----------------------------------------------
### Detecting whether the user intends to search for a product:
If the user's text contains any of the following, the user intends to buy:
- Action words: "پیدا کن", "می‌خوام", "جستجو", "بگرد", "نشون بده", "معرفی کن"
- Or contains the name of a product: دوغ، شورت، گوشی، لپتاپ، کفش، مانیتور، هندزفری etc.
- Or describes a NEED: "گشنمه", "تشنمه", "سردمه", "خوابم میاد", "پوستم خشکه"

In this case:  
⇒ You must call the tools.

If the user asks a general question, greeting, or non-shopping topic:  
⇒ Do not use the tools and only respond with:  
{"message": "<your response>"}

-----------------------------------------------
### Non-shopping examples

User: "چطوری؟"  
Response:  
{"message": "I'm good, how about you?"}

User: "داستان انگیزشی بگو"  
Response:  
{"message": "Sure..."}
-----------------------------------------------

### Important Note
- Do not produce any text outside these two modes.
- In normal messages, only return {"message": "..."}.
- ALWAYS use suggested_query from interpret_query as the query for search_products_semantic!


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
    logger.info("🚀 Creating Shopping AI Agent...")
    logger.debug(f"Debug Mode: {DEBUG_MODE}")
    logger.debug(f"LLM Model: openai/gpt-oss-120b")
    
    # Initialize LLM
    llm = ChatNVIDIA(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        base_url=BASE_URL,
        max_tokens=2048,  # افزایش max_tokens برای جلوگیری از cut off
        temperature=0.3,  # کمتر برای پاسخ‌های دقیق‌تر
    )

    logger.debug("✅ LLM initialized successfully")
    
    # Bind tools to LLM
    tools = [search_products_semantic, interpret_query]
    llm_with_tools = llm.bind_tools(tools)
    logger.debug(f"🔧 Tools bound: {[tool.name for tool in tools]}")
    
    # Define chatbot node
    def chatbot_node(state):
        """Process messages and generate responses."""
        logger.debug("=" * 60)
        logger.debug("📥 CHATBOT NODE - Processing messages")
        
        # Add system message if not already present
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            logger.debug("📝 System prompt added to messages")
        
        logger.debug(f"💬 Message count: {len(messages)}")
        if messages:
            last_msg = messages[-1]
            logger.debug(f"📨 Last message type: {type(last_msg).__name__}")
            if hasattr(last_msg, 'content'):
                content_preview = str(last_msg.content)[:100]
                logger.debug(f"📄 Content preview: {content_preview}...")
        
        logger.debug("🤖 Invoking LLM...")
        response = llm_with_tools.invoke(messages)
        
        logger.debug(f"✅ LLM Response received")
        logger.debug(f"📊 Response type: {type(response).__name__}")
        
        # Check if tools are being called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"🔧 Tool calls requested: {len(response.tool_calls)}")
            for i, tool_call in enumerate(response.tool_calls):
                logger.debug(f"  Tool {i+1}: {tool_call.get('name', 'unknown')}")
                logger.debug(f"  Args: {tool_call.get('args', {})}")
        else:
            logger.debug("💬 Direct response (no tool calls)")
            if hasattr(response, 'content'):
                content = str(response.content)
                logger.info(f"📄 Response content length: {len(content)} chars")
                # لاگ کامل برای دیباگ
                if content:
                    logger.debug(f"📄 Full response:\n{content}")
                else:
                    logger.warning("⚠️ LLM returned EMPTY content!")
        
        return {"messages": [response]}
    
    # Create tool node
    tool_node = ToolNode(tools)
    logger.debug("🛠️ Tool node created")
    
    # Build graph
    logger.debug("🏗️ Building graph structure...")
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot_node)
    builder.add_node("tools", tool_node)
    
    # Define edges
    builder.add_edge(START, "chatbot")
    builder.add_conditional_edges("chatbot", tools_condition)
    builder.add_edge("tools", "chatbot")
    
    logger.debug("🔗 Graph edges configured")
    
    # Compile with memory
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    logger.info("✅ Agent compiled successfully with memory")
    logger.debug("=" * 60)
    
    return graph
