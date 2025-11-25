# Shopping AI Assistant - Complete Documentation

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.2.0-orange.svg)](https://www.elastic.co/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.2-green.svg)](https://github.com/langchain-ai/langgraph)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [System Components](#system-components)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## Overview

**Shopping AI Assistant** is a production-ready conversational AI system designed to help users find products through natural language interactions. It leverages cutting-edge technologies including LangGraph for conversation management, Elasticsearch for high-performance search, and multilingual semantic embeddings for understanding queries in multiple languages (primarily Persian/Farsi and English).

The system intelligently interprets user intent, performs hybrid searches combining keyword matching with semantic understanding, and ranks results based on user preferences for price, quality, and relevance.

### What Makes This Special?

- **🧠 Intent-Aware**: Automatically understands whether users want the cheapest option, best quality, or best value
- **🌍 Multilingual**: Native support for Persian/Farsi with cross-lingual capabilities
- **⚡ Fast**: Sub-second response times using Elasticsearch's vector search
- **🎯 Smart Ranking**: Dynamic scoring algorithm that adapts to user preferences
- **💬 Conversational**: Maintains context across multiple turns using LangGraph's state management

---

## Key Features

### 🧠 Intelligent Intent Analysis

The system automatically detects and categorizes user shopping intent into five distinct modes:

1. **find_cheapest**: User prioritizes lowest price
2. **find_high_quality**: User prioritizes brand reputation and quality
3. **find_best_value**: User wants optimal price-to-quality ratio
4. **compare**: User wants to see diverse options for comparison
5. **find_by_feature**: User has specific requirements (size, color, softness, etc.)

**Example:**
```
User: "یه هدفون ارزان میخوام" (I want cheap headphones)
→ Intent: find_cheapest
→ price_sensitivity: 1.0, quality_sensitivity: 0.0
```

### 🔍 Hybrid Search Technology

Combines two complementary search approaches:

1. **BM25 Keyword Search**: Traditional text matching for precise product names and brands
2. **Semantic Vector Search**: Deep learning-based understanding using multilingual embeddings

This hybrid approach ensures both precision (finding exact products) and recall (understanding conceptual queries).

### 🌍 Multilingual Semantic Understanding

Uses the `intfloat/multilingual-e5-base` model, which supports:
- Persian/Farsi (primary)
- English
- Arabic
- And 100+ other languages

**Example queries handled:**
```
"یه چیز میخوام بپوشم سردم نشه" → Understands: needs a warm jacket
"گشنمه" → Understands: needs a snack
"تشنمه" → Understands: needs a beverage
```

### 📊 Dynamic Smart Ranking

Products are ranked using a sophisticated `value_score` formula that considers:

```python
value_score = (
    brand_score × quality_sensitivity +
    similarity_score × 0.4 +
    discount_percentage × 0.2 -
    normalized_price × price_sensitivity
)
```

The formula adapts based on detected intent:
- **find_cheapest**: Price weight = 2.0, Brand weight = 0.1
- **find_high_quality**: Brand weight = 1.5, Price weight = 0.1
- **find_by_feature**: Similarity weight = 1.5 (prioritizes relevance)

### 🎯 Adaptive Relevance Filtering

- **Dynamic threshold**: Adjusts based on result quality distribution
- **Minimum guarantee**: Always returns at least 1 result
- **Maximum limit**: Caps at 5 most relevant products
- **Relevance cutoff**: Filters out products below 0.4 similarity score

### 💬 Conversational Memory

Powered by LangGraph, the system maintains conversation state:
- Remembers previous queries
- Understands context across multiple turns
- Uses thread-based session management

---

## Architecture

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│            "یه هدفون ارزان با کیفیت خوب میخوام"                │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT                              │
│                  (meta/llama-3.1-70b)                           │
│                                                                 │
│  • Receives user message                                       │
│  • Determines if product search is needed                      │
│  • Routes to appropriate tools                                 │
└────────────┬───────────────────────────────┬────────────────────┘
             │                               │
             ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│   interpret_query Tool   │    │   Other Responses            │
│                          │    │   (greetings, general chat)  │
│  INPUT:                  │    └──────────────────────────────┘
│    query: "full text"    │
│                          │
│  OUTPUT:                 │
│    category             │
│    intent               │
│    price_sensitivity    │
│    quality_sensitivity  │
│    suggested_query      │
└────────────┬─────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│              search_products_semantic Tool                      │
│                                                                 │
│  INPUT:                                                         │
│    query: "هدفون"  (from suggested_query)                      │
│    category: "لوازم الکترونیکی"                                │
│    intent: "find_best_value"                                   │
│    price_sensitivity: 0.6                                      │
│    quality_sensitivity: 0.8                                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ELASTICSEARCH 9.2.0                            │
│                                                                 │
│  ┌────────────────────┬─────────────────────────────────────┐  │
│  │   BM25 Search      │   Vector KNN Search                 │  │
│  │                    │                                     │  │
│  │  Match on:         │   Cosine Similarity:                │  │
│  │  • product_name    │   • query_embedding                 │  │
│  │  • brand_name      │   • product_embedding               │  │
│  │  • category        │   (768-dim multilingual-e5-base)    │  │
│  └────────────────────┴─────────────────────────────────────┘  │
│                                                                 │
│  Returns: Top 50 products with combined relevance scores       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  RERANKING ENGINE                               │
│                                                                 │
│  1. Calculate value_score for each product:                    │
│     • Load brand_score from BrandScore.json                    │
│     • Apply intent-specific formula                            │
│     • Consider price, quality, similarity, discount            │
│                                                                 │
│  2. Filter by relevance threshold (similarity ≥ 0.4)           │
│                                                                 │
│  3. Sort by value_score (descending)                           │
│                                                                 │
│  4. Return top 1-5 products                                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH AGENT                              │
│                                                                 │
│  Formats response in Persian:                                  │
│                                                                 │
│  🛒 هدفون بلوتوثی سونی WH-1000XM4                              │
│     💰 قیمت: 3,500,000 تومان                                   │
│     🏷️ برند: Sony                                             │
│     🔥 تخفیف: 15%                                              │
│                                                                 │
│  [... up to 4 more products ...]                               │
│                                                                 │
│  ---                                                            │
│  📊 خلاصه: 5 محصول یافت شد | بازه قیمت: 2M - 5M تومان         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       USER OUTPUT                               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│    main.py   │─────▶│  agent.py   │◀────▶│  LangGraph   │
└──────────────┘      └─────────────┘      └──────────────┘
                            │
                            │ uses tools
                            ▼
                 ┌──────────────────────┐
                 │  SearchProducts.py   │
                 └──────────┬───────────┘
                            │
                ┌───────────┴──────────────┐
                │                          │
                ▼                          ▼
       ┌─────────────────┐      ┌──────────────────┐
       │  Elasticsearch  │      │ SentenceTransform│
       │   (Vector DB)   │      │  (Embedding API) │
       └─────────────────┘      └──────────────────┘
                │
                │ reads
                ▼
         ┌──────────────┐
         │BrandScore.json│
         │CategoryW.json│
         └──────────────┘
```

---

## Technology Stack

### Core Frameworks

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13+ | Programming language |
| **LangGraph** | 1.0.2 | Conversation orchestration and state management |
| **LangChain** | 1.0.5 | LLM integration framework |
| **Elasticsearch** | 9.2.0 | Vector database and search engine |

### AI/ML Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **sentence-transformers** | 5.1.0 | Multilingual semantic embeddings |
| **transformers** | 4.56.1 | HuggingFace transformer models |
| **torch** | 2.8.0 | Deep learning backend |

### LLM Providers

| Provider | Model | Usage |
|----------|-------|-------|
| **NVIDIA AI Endpoints** | meta/llama-3.1-70b-instruct | Primary LLM for agent |
| **NVIDIA AI Endpoints** | openai/gpt-oss-120b | Alternative model |
| **Azure OpenAI** | gpt-4o | Optional alternative (commented) |

### Embedding Model

- **Model**: `intfloat/multilingual-e5-base`
- **Dimensions**: 768
- **Languages**: 100+
- **Primary Language**: Persian/Farsi
- **Performance**: Optimized for cross-lingual semantic search

---

## Project Structure

```
ShoppingAiAssistant/
│
├── main.py                          # Application entry point
│   └── Creates agent and manages conversation loop
│
├── src/
│   ├── __init__.py
│   ├── agent.py                     # LangGraph agent implementation
│   │   ├── System prompt definition
│   │   ├── Agent graph construction
│   │   ├── Tool binding
│   │   └── Memory management
│   │
│   └── tools/
│       ├── __init__.py
│       └── SearchProducts.py        # Core search functionality
│           ├── ProductSearchEngine class
│           ├── interpret_query() tool
│           └── search_products_semantic() tool
│
├── script/
│   ├── BrandScore.py                # Brand scoring algorithm
│   └── shopping_embedding_colab.ipynb  # Data preparation notebook
│
├── config/
│   └── .env.example                 # Configuration template
│
├── BrandScore.json                  # Brand reputation scores
├── CategoryW.json                   # Category weights
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
└── DOCUMENTATION.md                 # This file
```

### File Descriptions

#### Core Application Files

**`main.py`**
- Entry point for the application
- Manages user input/output loop
- Creates agent instance
- Handles thread-based session management

**`src/agent.py`**
- Defines the LangGraph agent
- Contains comprehensive system prompt
- Configures LLM with tool bindings
- Implements chatbot node logic
- Manages conversation state and memory

**`src/tools/SearchProducts.py`**
- Implements Elasticsearch integration
- Defines two LangChain tools:
  - `interpret_query`: Analyzes user intent
  - `search_products_semantic`: Performs product search
- Contains ProductSearchEngine singleton class
- Implements value scoring algorithm

#### Data Files

**`BrandScore.json`**
- Pre-calculated reputation scores for brands
- Format: `{"brand_name": score_value}`
- Scores range from 0.0 to 1.0
- Generated by `script/BrandScore.py`

**`CategoryW.json`**
- Category importance weights
- Used in brand score calculation
- Higher weight = more important category

#### Configuration

**`config/.env.example`**
- Template for environment variables
- Contains all required configuration options
- Must be copied to `.env` and filled with actual values

---

## System Components

### 1. LangGraph Agent (`src/agent.py`)

The conversational brain of the system.

#### Responsibilities:
- Orchestrates tool calls
- Maintains conversation history
- Routes messages appropriately
- Formats final responses

#### Key Features:
```python
# Graph structure
START → chatbot → tools_condition → tools → chatbot → END
                        ↓ (no tools needed)
                       END
```

#### System Prompt:
The agent uses a detailed 200+ line system prompt that:
- Defines tool usage rules
- Specifies output formatting requirements
- Provides intent detection guidelines
- Includes example interactions
- Enforces quality standards

### 2. Intent Interpreter (`interpret_query` tool)

Analyzes raw user input to extract structured shopping preferences.

#### Input:
```json
{
  "query": "یه هدفون ارزان با کیفیت خوب میخوام"
}
```

#### Output:
```json
{
  "category": "لوازم الکترونیکی",
  "intent": "find_best_value",
  "price_sensitivity": 0.7,
  "quality_sensitivity": 0.8,
  "suggested_query": "هدفون"
}
```

#### Processing Steps:
1. **Category Detection**: Matches query against 28 predefined categories
2. **Intent Classification**: Determines one of 5 shopping intents
3. **Sensitivity Scoring**: Analyzes keywords for price/quality preferences
4. **Query Refinement**: Extracts core product keyword from natural language

#### Special Capabilities:

**Implicit Intent Detection:**
```
"یچیز میخوام بپوشم سردم نشه" → suggested_query: "کاپشن"
"گشنمه" → suggested_query: "بیسکویت"
"تشنمه" → suggested_query: "آب معدنی"
"پوستم خشکه" → suggested_query: "کرم مرطوب کننده"
```

### 3. Product Search Engine (`ProductSearchEngine` class)

Singleton class managing Elasticsearch interactions.

#### Features:
- **Lazy initialization**: Model loaded once on first use
- **Connection pooling**: Reuses Elasticsearch client
- **Error handling**: Graceful degradation on failures
- **Logging**: Detailed debug information when enabled

#### Search Method Signature:
```python
def search(
    query_text: str,
    top_k: int = 5,
    min_similarity: float = 0.3,
    category: str = None
) -> List[Dict]
```

#### Elasticsearch Query Structure:
```python
{
  "size": 50,
  "query": {
    "bool": {
      "must": [category_filter],  # Optional
      "should": [
        {
          "multi_match": {  # BM25 keyword search
            "query": query_text,
            "fields": ["product_name^2", "brand_name", "category_name"],
            "boost": 1.0
          }
        },
        {
          "script_score": {  # Semantic vector search
            "script": {
              "source": "cosineSimilarity(params.query_vector, 'product_embedding') + 1.0",
              "params": {"query_vector": query_embedding},
              "boost": 2.0
            }
          }
        }
      ],
      "minimum_should_match": 1
    }
  }
}
```

### 4. Semantic Search Tool (`search_products_semantic`)

LangChain tool wrapper that orchestrates the full search pipeline.

#### Workflow:

```
1. Receive parameters (query, intent, sensitivities, category)
   ↓
2. Adjust sensitivities based on intent
   ↓
3. Execute hybrid search via ProductSearchEngine
   ↓
4. Load brand scores from BrandScore.json
   ↓
5. Calculate value_score for each product
   ↓
6. Filter by relevance threshold (0.4)
   ↓
7. Sort by value_score
   ↓
8. Limit to 1-5 products
   ↓
9. Format as JSON response
```

#### Value Score Calculation:

**Default (find_best_value):**
```python
value_score = (
    brand_score * quality_sensitivity +
    similarity * 0.4 +
    discount_percentage * 0.2 -
    normalized_price * price_sensitivity
)
```

**Intent-Specific Formulas:**

| Intent | Formula |
|--------|---------|
| find_cheapest | `-price × 2.0 + similarity × 0.3 + brand × 0.1` |
| find_high_quality | `brand × 1.5 + similarity × 0.5 + discount × 0.3 - price × 0.1` |
| find_by_feature | `similarity × 1.5 + brand × quality_sens × 0.4 + discount × 0.2 - price × price_sens × 0.3` |
| compare | `brand × 0.3 + similarity × 0.4 + discount × 0.2 - price × 0.3` |

### 5. Brand Scoring System

Pre-computed brand reputation scores based on:

**Formula:**
```python
BrandScore = (
    0.40 × category_weight_total +
    0.25 × product_count +
    0.15 × (1 / (1 + price_std_dev))
)
```

**Components:**
- **Category Weight**: Sum of weights for categories where brand appears
- **Product Count**: Number of products from this brand
- **Price Consistency**: Inverse of price standard deviation (consistent pricing = higher score)

**Example Scores:**
```json
{
  "Samsung": 0.85,
  "Apple": 0.92,
  "Unknown Brand": 0.35
}
```

---

## Installation Guide

### Prerequisites

- **Python**: 3.13 or higher
- **Elasticsearch**: 9.2.0 (running and accessible)
- **API Keys**: NVIDIA AI Endpoints or Azure OpenAI
- **System**: 4GB+ RAM, 2GB+ disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/11linear11/ShoppingAiAssistant.git
cd ShoppingAiAssistant
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Core: langchain, langgraph, python-dotenv
- Search: elasticsearch
- ML: sentence-transformers, torch, transformers
- LLM: langchain-nvidia-ai-endpoints, langchain-openai

### Step 4: Configure Environment

```bash
# Copy example configuration
cp config/.env.example .env

# Edit configuration
nano .env  # or vim, or your preferred editor
```

Fill in the following required fields:

```bash
# Required: NVIDIA API Key
api_key=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Required: Elasticsearch Connection
ELASTICSEARCH_HOST=your_elasticsearch_host
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_INDEX=shopping_products

# Optional: Debug mode
DEBUG_MODE=false
```

### Step 5: Verify Elasticsearch

Ensure your Elasticsearch instance:
- Is running and accessible
- Has the index `shopping_products` (or your configured name)
- Contains documents with required fields:
  - `product_name` (text)
  - `brand_name` (text)
  - `category_name` (text)
  - `price` (float)
  - `discount_price` (float)
  - `has_discount` (boolean)
  - `discount_percentage` (float)
  - `product_embedding` (dense_vector, 768 dimensions)

### Step 6: Test Installation

```bash
python main.py
```

Expected output:
```
Shopping AI Assistant
==================================================
Type 'exit' to quit

User: _
```

---

## Configuration

### Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEBUG_MODE` | No | `false` | Enable detailed logging |
| `api_key` | Yes | - | NVIDIA AI Endpoints API key |
| `BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | NVIDIA API base URL |
| `MODEL_NAME` | No | `openai/gpt-oss-120b` | LLM model to use |
| `ELASTICSEARCH_HOST` | Yes | - | Elasticsearch host address |
| `ELASTICSEARCH_PORT` | No | `9200` | Elasticsearch port |
| `ELASTICSEARCH_USER` | No | - | Elasticsearch username |
| `ELASTICSEARCH_PASSWORD` | No | - | Elasticsearch password |
| `ELASTICSEARCH_INDEX` | No | `shopping_products` | Index name for products |
| `ELASTICSEARCH_SCHEME` | No | `http` | Connection scheme (http/https) |

### Debug Mode

Enable debug logging for troubleshooting:

```bash
DEBUG_MODE=true
```

Creates `shopping_assistant_debug.log` with detailed information:
- LLM requests and responses
- Elasticsearch queries
- Tool invocations
- Value score calculations
- Reranking decisions

### Alternative LLM Providers

#### Using Azure OpenAI:

Uncomment in `src/agent.py`:
```python
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://models.inference.ai.azure.com"
)
```

Set in `.env`:
```bash
OPENAI_API_KEY=your_azure_openai_key
```

#### Using Different NVIDIA Models:

Available models:
- `meta/llama-3.1-70b-instruct` (recommended for stability)
- `meta/llama-3.1-8b-instruct` (lighter, faster)
- `openai/gpt-oss-120b` (default)

Change in `.env`:
```bash
MODEL_NAME=meta/llama-3.1-70b-instruct
```

---

## Usage

### Basic Usage

1. **Start the application:**
```bash
python main.py
```

2. **Enter queries in Persian or English:**
```
User: یه لپ تاپ گیمینگ میخوام
```

3. **Receive formatted results:**
```
Assistant:
🛒 لپ‌تاپ گیمینگ ASUS ROG Strix G15
   💰 قیمت: 45,000,000 تومان
   🏷️ برند: ASUS
   🔥 تخفیف: 10%

🛒 لپ‌تاپ MSI Katana GF66
   💰 قیمت: 38,500,000 تومان
   🏷️ برند: MSI
   🔥 تخفیف: 5%

---
📊 خلاصه: 2 محصول یافت شد | بازه قیمت: 38.5M - 45M تومان
```

4. **Exit:**
```
User: exit
خداحافظ! (Goodbye!)
```

### Advanced Query Examples

#### Example 1: Price-Focused Query
```
User: ارزان‌ترین گوشی رو نشون بده

Intent: find_cheapest
Results: Sorted by lowest price first
```

#### Example 2: Quality-Focused Query
```
User: بهترین هدفون با کیفیت عالی

Intent: find_high_quality
Results: Sorted by brand score first
```

#### Example 3: Feature-Specific Query
```
User: یه شامپوی نرم کننده میخوام

Intent: find_by_feature
Results: Prioritizes similarity to "نرم کننده"
```

#### Example 4: Implicit Need
```
User: سردمه یچیز میخوام بپوشم

Interpretation:
  category: مد و پوشاک
  suggested_query: کاپشن
  intent: find_by_feature
```

#### Example 5: Comparison Query
```
User: چند تا گوشی با برند‌های مختلف نشون بده

Intent: compare
Results: Diverse brands and price ranges
```

### Conversation Context

The system maintains context across multiple turns:

```
User: یه گوشی میخوام
Assistant: [Shows 5 phones]

User: چیزی ارزون‌تر نداری؟
Assistant: [Adjusts to find_cheapest and shows cheaper options]

User: اون سامسونگی رو بیشتر توضیح بده
Assistant: [Provides details about Samsung phone from previous results]
```

### Session Management

Each conversation has a thread ID:

```python
thread_id = "user_session_1"
config = {"configurable": {"thread_id": thread_id}}

state = graph.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config=config
)
```

For multi-user systems, use unique thread IDs per user.

---

## How It Works

### End-to-End Flow Example

**User Query:** "یه هدفون ارزان با صدای خوب"

#### Step 1: Query Reception
```python
# main.py
user_input = "یه هدفون ارزان با صدای خوب"
state = graph.invoke(
    {"messages": [HumanMessage(content=user_input)]},
    config=config
)
```

#### Step 2: Agent Processing
```python
# src/agent.py - chatbot_node
messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(user_input)]
response = llm_with_tools.invoke(messages)
```

Agent detects shopping intent and decides to call tools.

#### Step 3: Intent Interpretation
```python
# Tool call 1: interpret_query
tool_call = {
    "name": "interpret_query",
    "args": {"query": "یه هدفون ارزان با صدای خوب"}
}
```

LLM analyzes query:
```json
{
  "category": "لوازم الکترونیکی",
  "intent": "find_best_value",
  "price_sensitivity": 0.8,
  "quality_sensitivity": 0.7,
  "suggested_query": "هدفون"
}
```

#### Step 4: Product Search
```python
# Tool call 2: search_products_semantic
tool_call = {
    "name": "search_products_semantic",
    "args": {
        "query": "هدفون",
        "category": "لوازم الکترونیکی",
        "intent": "find_best_value",
        "price_sensitivity": 0.8,
        "quality_sensitivity": 0.7
    }
}
```

#### Step 5: Elasticsearch Query
```python
# Generate embedding
query_embedding = model.encode(["هدفون"])[0]  # 768-dim vector

# Execute hybrid search
es_query = {
    "bool": {
        "must": [{"match": {"category_name": "لوازم الکترونیکی"}}],
        "should": [
            {"multi_match": {"query": "هدفون", "fields": ["product_name^2", "brand_name"]}},
            {"script_score": {"script": "cosineSimilarity(...) + 1.0"}}
        ]
    }
}

results = es.search(index="shopping_products", body=es_query)
```

Returns 50 candidates.

#### Step 6: Value Scoring
```python
for product in results:
    brand_score = brand_scores.get(product['brand_name'], 0.5)
    final_price = product['price'] * (1 - product['discount_percentage']/100)
    
    value_score = (
        brand_score * 0.7 +           # quality_sensitivity
        similarity * 0.4 +
        discount_percentage * 0.2 -
        (final_price/1000000) * 0.8   # price_sensitivity
    )
    
    product['value_score'] = value_score
```

#### Step 7: Ranking and Filtering
```python
# Filter by relevance
relevant = [p for p in products if p['similarity'] >= 0.4]

# Sort by value_score
relevant.sort(key=lambda x: x['value_score'], reverse=True)

# Limit to top 5
top_products = relevant[:5]
```

#### Step 8: Response Formatting
```python
# Agent formats final response
response = """
🛒 هدفون بی‌سیم سونی WH-1000XM4
   💰 قیمت: 3,200,000 تومان
   🏷️ برند: Sony
   🔥 تخفیف: 15%

🛒 هدفون JBL Tune 750BTNC
   💰 قیمت: 1,800,000 تومان
   🏷️ برند: JBL
   🔥 تخفیف: 10%

---
📊 خلاصه: 2 محصول یافت شد | بازه قیمت: 1.8M - 3.2M تومان
"""
```

#### Step 9: Output to User
```
Assistant: [formatted response above]
```

---

## API Reference

### Tools API

#### `interpret_query(query: str) -> str`

Analyzes user shopping intent and extracts structured information.

**Parameters:**
- `query` (str): User's natural language query

**Returns:**
- JSON string with fields:
  - `category` (str): Product category
  - `intent` (str): Shopping intent
  - `price_sensitivity` (float): 0-1
  - `quality_sensitivity` (float): 0-1
  - `suggested_query` (str): Refined product keyword

**Example:**
```python
result = interpret_query("یه چیز تند میخوام")
# Returns: {"category": "تنقلات", "intent": "find_by_feature", 
#           "price_sensitivity": 0.5, "quality_sensitivity": 0.5,
#           "suggested_query": "چیپس تند"}
```

#### `search_products_semantic(query, quality_sensitivity, price_sensitivity, category, intent) -> str`

Searches for products using hybrid semantic search.

**Parameters:**
- `query` (str): Product search query
- `quality_sensitivity` (float): 0-1, quality importance
- `price_sensitivity` (float): 0-1, price importance
- `category` (str, optional): Category filter
- `intent` (str, optional): Shopping intent

**Returns:**
- JSON string with fields:
  - `products` (list): Array of product objects
  - `meta` (dict): Search metadata

**Product Object:**
```json
{
  "name": "هدفون سونی",
  "price": 3500000,
  "final_price": 3000000,
  "brand": "Sony",
  "brand_score": 0.85,
  "discount": 15,
  "product_id": "12345",
  "similarity": 0.87,
  "value_score": 1.42,
  "category": "لوازم الکترونیکی"
}
```

**Meta Object:**
```json
{
  "query": "هدفون",
  "total_found": 5,
  "is_relevant": true,
  "avg_similarity": 0.75,
  "price_range": {"min": 1500000, "max": 4500000},
  "intent": "find_best_value"
}
```

### ProductSearchEngine Class

#### `__init__()`
Initializes Elasticsearch connection and embedding model.

#### `search(query_text, top_k=5, min_similarity=0.3, category=None) -> List[Dict]`

Performs hybrid search on Elasticsearch index.

**Parameters:**
- `query_text` (str): Search query
- `top_k` (int): Number of results (default: 5)
- `min_similarity` (float): Minimum similarity threshold (default: 0.3)
- `category` (str, optional): Category filter

**Returns:**
- List of product dictionaries with fields:
  - `product_id`, `product_name`, `brand_name`
  - `price`, `discount_price`, `discount_percentage`
  - `category_name`, `has_discount`
  - `similarity`, `score`

---

## Advanced Features

### 1. Intent-Based Ranking Adaptation

The system dynamically adjusts ranking formulas based on detected intent:

```python
if intent == "find_cheapest":
    price_sensitivity = max(price_sensitivity, 0.9)
    quality_sensitivity = min(quality_sensitivity, 0.2)
elif intent == "find_high_quality":
    quality_sensitivity = max(quality_sensitivity, 0.9)
    price_sensitivity = min(price_sensitivity, 0.2)
```

This ensures results align with user expectations.

### 2. Dynamic Similarity Thresholds

Instead of fixed cutoffs, the system adapts based on result distribution:

```python
similarities = [r['similarity'] for r in results]
dynamic_threshold = max(0.38, median(similarities) - 0.15)
filtered = [r for r in results if r['similarity'] >= dynamic_threshold]
```

Benefits:
- Prevents empty results when no high-similarity matches exist
- Filters aggressively when many good matches are found
- Balances precision and recall

### 3. Brand Scoring System

Pre-computed scores consider:

**Category Diversity:**
```python
category_weight_total = sum(cw.get(cat, 0) for cat in brand_categories)
```

**Product Portfolio Size:**
```python
product_count = number_of_products_from_brand
```

**Price Consistency:**
```python
price_stability = 1 / (1 + std_dev(brand_prices))
```

**Final Score:**
```python
brand_score = 0.40 * category_weight + 0.25 * product_count + 0.15 * price_stability
```

### 4. Multilingual Embedding

Uses `multilingual-e5-base` which:
- Supports 100+ languages
- Produces 768-dimensional embeddings
- Enables cross-lingual search (query in Persian, match English products)
- Handles code-switching ("یه smartphone میخوام")

**Embedding Process:**
```python
model = SentenceTransformer('intfloat/multilingual-e5-base')
embedding = model.encode([query_text])[0]  # Returns 768-dim vector
```

### 5. Hybrid Search Boosting

Combines BM25 and vector search with configurable boosts:

```python
"should": [
    {"multi_match": {..., "boost": 1.0}},      # BM25
    {"script_score": {..., "boost": 2.0}}       # Semantic
]
```

Semantic search gets 2× weight because it better captures intent.

### 6. Conversation Memory

LangGraph's `MemorySaver` maintains state across turns:

```python
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Each user gets a unique thread
config = {"configurable": {"thread_id": "user_123"}}
```

Enables:
- Follow-up questions
- Context-aware responses
- Session persistence

---

## Performance Optimization

### Query Optimization

**1. Elasticsearch Index Settings:**
```json
{
  "settings": {
    "number_of_shards": 3,
    "number_of_replicas": 1,
    "refresh_interval": "5s"
  },
  "mappings": {
    "properties": {
      "product_embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

**2. Batch Embedding:**
```python
# Instead of one-by-one:
for query in queries:
    embedding = model.encode([query])
    
# Batch process:
embeddings = model.encode(queries)  # Much faster
```

**3. Connection Pooling:**
```python
# Singleton pattern ensures single ES client
class ProductSearchEngine:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### Memory Optimization

**1. Lazy Model Loading:**
```python
class ProductSearchEngine:
    _initialized = False
    def __init__(self):
        if ProductSearchEngine._initialized:
            return  # Skip re-initialization
```

**2. Cached Brand Scores:**
```python
_brand_scores = None
def get_brand_scores():
    global _brand_scores
    if _brand_scores is None:
        _brand_scores = json.load(...)
    return _brand_scores
```

### Response Time Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| Model load (first run) | ~2-3s | One-time cost |
| Embedding generation | ~50-100ms | Per query |
| Elasticsearch search | ~100-200ms | Depends on index size |
| Value scoring | ~10-20ms | For 50 products |
| Total (cold start) | ~3s | First query |
| Total (warm) | ~200-400ms | Subsequent queries |

### Scalability Considerations

**For 10,000 products:**
- Search: ~100ms
- Memory: ~500MB (model) + 100MB (ES client)

**For 1,000,000 products:**
- Search: ~200-300ms (with proper indexing)
- Memory: Same (model size doesn't change)
- Recommendations:
  - Use Elasticsearch cluster (3+ nodes)
  - Enable index sharding
  - Consider caching frequent queries

**For Multi-User Deployment:**
- Deploy as web service (FastAPI/Flask)
- Use Redis for session management
- Load balance across multiple instances
- Consider GPU for faster embeddings

---

## Troubleshooting

### Common Issues

#### 1. "No module named 'src'"

**Cause:** Python can't find the `src` package.

**Solution:**
```bash
# Ensure you're in project root
cd ShoppingAiAssistant

# Run from project root
python main.py

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 2. Elasticsearch Connection Error

**Error:** `ConnectionError: Connection refused`

**Solution:**
```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# Check .env configuration
ELASTICSEARCH_HOST=localhost  # Not 127.0.0.1 if using Docker
ELASTICSEARCH_PORT=9200
```

**For Docker:**
```bash
docker run -d -p 9200:9200 -e "discovery.type=single-node" \
  elasticsearch:9.2.0
```

#### 3. Empty or No Results

**Symptom:** Search returns no products

**Debugging:**
```bash
# Enable debug mode
echo "DEBUG_MODE=true" >> .env

# Check log file
tail -f shopping_assistant_debug.log
```

**Common causes:**
- Index doesn't exist
- Query embedding dimension mismatch
- Category filter too restrictive
- No products in specified category

**Solutions:**
```python
# Verify index
GET /shopping_products/_mapping

# Check product count
GET /shopping_products/_count

# Test simple query
GET /shopping_products/_search
{
  "query": {"match_all": {}},
  "size": 1
}
```

#### 4. LLM Returns Empty Response

**Symptom:** `interpret_query` returns fallback values

**Debugging:**
```python
# In SearchProducts.py, check logs
logger.debug(f"📄 Raw response: {response}")
```

**Solutions:**
- Try different NVIDIA model:
  ```bash
  MODEL_NAME=meta/llama-3.1-70b-instruct
  ```
- Check API key validity
- Verify internet connection
- Review rate limits

#### 5. Slow Performance

**Symptom:** Each query takes >5 seconds

**Diagnostics:**
```bash
# Enable debug logging to see timing
DEBUG_MODE=true python main.py
```

**Solutions:**
- **First run slow:** Model loading (~3s) - normal
- **Always slow:**
  - Check Elasticsearch performance
  - Reduce `top_k` parameter
  - Use faster embedding model
  - Enable query caching

#### 6. Installation Errors

**Error:** `torch` installation fails

**Solution:**
```bash
# Install PyTorch separately with CPU-only version
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install rest
pip install -r requirements.txt
```

**Error:** `sentence-transformers` model download fails

**Solution:**
```bash
# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('intfloat/multilingual-e5-base')"
```

### Debug Mode Features

Enable comprehensive logging:

```bash
DEBUG_MODE=true
```

Logs include:
- Full LLM prompts and responses
- Elasticsearch query bodies
- Similarity scores for all products
- Value score calculations
- Reranking decisions
- Tool invocation details

**Log file:** `shopping_assistant_debug.log`

### Testing Components

**Test Elasticsearch connection:**
```python
from src.tools.SearchProducts import ProductSearchEngine

engine = ProductSearchEngine()
results = engine.search("test", top_k=1)
print(f"Found {len(results)} products")
```

**Test embedding model:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-base')
embedding = model.encode(["تست"])
print(f"Embedding dimension: {len(embedding[0])}")  # Should be 768
```

**Test LLM:**
```python
from src.tools.SearchProducts import interpret_query

result = interpret_query("یه هدفون میخوام")
print(result)
```

---

## Future Enhancements

### Planned Features

#### 1. User Preference Learning
- Track user selections and purchases
- Build personalized preference profiles
- Adjust `quality_sensitivity` and `price_sensitivity` automatically
- Implement collaborative filtering

#### 2. Image Search
- Accept product images as queries
- Use CLIP or similar vision-language models
- Enable visual similarity search

#### 3. Price History Tracking
- Store historical price data
- Show price trends (increasing/decreasing)
- Alert users to good deals
- Predict future price changes

#### 4. Advanced Filters
```python
search_products_semantic(
    query="لپ تاپ",
    price_range=(20000000, 50000000),
    brands=["ASUS", "Dell"],
    min_rating=4.0,
    in_stock=True
)
```

#### 5. Multi-Modal Responses
- Include product images
- Show rating stars
- Display availability status
- Add "Add to Cart" links

#### 6. Voice Interface
- Integrate speech-to-text (Whisper)
- Support voice queries
- Text-to-speech responses

#### 7. Comparison Tables
```
User: مقایسه کن این سه تا رو
User: مقایسه کن این سه تا روUser: مقایسه کن این سه تا رو

Response: [Side-by-side comparison table]
```

#### 8. Recommendation Engine
- "Users who bought X also bought Y"
- Personalized recommendations
- Trending products

#### 9. Analytics Dashboard
- Popular search terms
- Conversion rates
- User engagement metrics
- A/B testing for ranking algorithms

#### 10. Mobile App
- Native iOS/Android apps
- Push notifications for deals
- Barcode scanning
- Location-based offers

### Technical Improvements

#### 1. Caching Layer
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query_hash):
    return search_products_semantic(query)
```

#### 2. Query Expansion
- Use synonyms to broaden search
- Handle typos and misspellings
- Expand abbreviations

#### 3. A/B Testing Framework
- Test different ranking formulas
- Experiment with sensitivity values
- Measure user satisfaction

#### 4. Real-Time Index Updates
- Stream product updates to Elasticsearch
- Invalidate cached results
- Hot-reload brand scores

#### 5. Distributed Tracing
- Implement OpenTelemetry
- Track request flow
- Monitor performance bottlenecks

---

## Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues

1. Check existing issues first
2. Provide detailed description
3. Include:
   - OS and Python version
   - Error messages and stack traces
   - Steps to reproduce
   - Expected vs actual behavior

### Pull Request Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes
4. Add tests if applicable
5. Update documentation
6. Commit: `git commit -m 'Add amazing feature'`
7. Push: `git push origin feature/amazing-feature`
8. Open Pull Request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings
- Keep functions focused and small
- Comment complex logic

### Testing

Run tests before submitting:
```bash
pytest tests/
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Technologies Used

- **LangChain & LangGraph**: Conversation orchestration
- **Elasticsearch**: Vector search engine
- **HuggingFace**: Multilingual embedding models
- **NVIDIA AI Endpoints**: LLM inference
- **Sentence Transformers**: Semantic embeddings

### Inspiration

This project was inspired by the need for intelligent shopping assistants that understand natural language queries in Persian/Farsi and can provide context-aware recommendations.

---

## Contact & Support

### Questions?

- **GitHub Issues**: [Create an issue](https://github.com/11linear11/ShoppingAiAssistant/issues)
- **Email**: [Your contact email]

### Commercial Support

For enterprise deployments, custom features, or integration support, contact us at [business email].

---

## Appendix

### A. Category List

Complete list of 28 supported categories:

```python
CATEGORIES = [
    "لوازم الکترونیکی",          # Electronics
    "لوازم برقی و دیجیتال",      # Electrical & Digital
    "مد و پوشاک",                 # Fashion & Clothing
    "طلا",                        # Gold
    "خانه و سبک زندگی",           # Home & Lifestyle
    "خانه و آشپزخانه",            # Home & Kitchen
    "آرایشی و بهداشتی",          # Cosmetics & Hygiene
    "نوشیدنی",                    # Beverages
    "لبنیات",                     # Dairy
    "کالاهای اساسی",              # Basic Goods
    "کودک و نوزاد",               # Baby & Child
    "خواربار و نان",              # Groceries & Bread
    "بهداشت و سلامت",             # Health & Wellness
    "شوینده و مواد ضد عفونی کننده", # Cleaning & Disinfectants
    "مواد پروتئینی",              # Protein Products
    "میوه و سبزیجات تازه",        # Fresh Fruits & Vegetables
    "دستمال و شوینده",            # Tissues & Cleaners
    "محصولات سلولزی",             # Cellulose Products
    "چاشنی و افزودنی",            # Condiments & Additives
    "تنقلات",                     # Snacks
    "کنسرو و غذای آماده",         # Canned & Ready Food
    "صبحانه",                     # Breakfast
    "کنسرو و غذاهای آماده",       # Canned & Ready Foods
    "آجیل و خشکبار",              # Nuts & Dried Fruits
    "خشکبار، دسر و شیرینی",       # Dried Fruits, Desserts & Sweets
    "لوازم تحریر و اداری",        # Stationery & Office
    "دسر و شیرینی پزی",           # Desserts & Confectionery
    "نان و شیرینی"                # Bread & Pastries
]
```

### B. Intent Definitions

| Intent | Description | Example Query | Ranking Priority |
|--------|-------------|---------------|------------------|
| `find_cheapest` | User wants lowest price | "ارزان‌ترین گوشی" | Price ↓ |
| `find_high_quality` | User prioritizes quality | "بهترین لپ تاپ" | Brand Score ↑ |
| `find_best_value` | Optimal price/quality ratio | "لپ تاپ با ارزش خوب" | Balanced |
| `compare` | Diverse options for comparison | "چند تا گوشی نشون بده" | Diversity |
| `find_by_feature` | Specific requirement | "هدفون نرم" | Similarity ↑ |

### C. Elasticsearch Mapping Example

```json
{
  "mappings": {
    "properties": {
      "product_id": { "type": "keyword" },
      "product_name": { 
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "brand_name": { 
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "category_name": { 
        "type": "text",
        "fields": {
          "keyword": { "type": "keyword" }
        }
      },
      "price": { "type": "float" },
      "discount_price": { "type": "float" },
      "discount_percentage": { "type": "float" },
      "has_discount": { "type": "boolean" },
      "product_embedding": {
        "type": "dense_vector",
        "dims": 768,
        "index": true,
        "similarity": "cosine"
      }
    }
  }
}
```

### D. Environment Variables Quick Reference

```bash
# Required
api_key=your_nvidia_api_key
ELASTICSEARCH_HOST=your_es_host
ELASTICSEARCH_PASSWORD=your_password

# Optional with defaults
DEBUG_MODE=false
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_INDEX=shopping_products
ELASTICSEARCH_SCHEME=http
MODEL_NAME=openai/gpt-oss-120b
```

### E. Performance Tuning Checklist

- [ ] Enable Elasticsearch index caching
- [ ] Use connection pooling
- [ ] Implement query result caching
- [ ] Pre-load embedding model at startup
- [ ] Use batch encoding for multiple queries
- [ ] Configure appropriate `top_k` values
- [ ] Set reasonable similarity thresholds
- [ ] Monitor and optimize slow queries
- [ ] Consider GPU acceleration for embeddings
- [ ] Use async/await for I/O operations

### F. Example Product Document

```json
{
  "product_id": "12345",
  "product_name": "هدفون بی‌سیم سونی WH-1000XM4",
  "brand_name": "Sony",
  "category_name": "لوازم الکترونیکی",
  "price": 3500000,
  "discount_price": 3000000,
  "discount_percentage": 14.29,
  "has_discount": true,
  "product_embedding": [0.023, -0.145, 0.089, ..., 0.234]
}
```

### G. Common Query Patterns

**Price-focused:**
```
ارزان‌ترین [product]
[product] با قیمت پایین
[product] اقتصادی
```

**Quality-focused:**
```
بهترین [product]
[product] با کیفیت
[product] برند معتبر
```

**Feature-focused:**
```
[product] [feature]
یه [product] که [requirement]
میخوام [product] [adjective]
```

**Implicit needs:**
```
گشنمه → بیسکویت
تشنمه → آب معدنی
سردمه → کاپشن
پوستم خشکه → کرم مرطوب کننده
```

---

## Version History

### v1.0.0 (Current)
- Initial release
- LangGraph-based conversation agent
- Hybrid search (BM25 + Semantic)
- Intent-aware ranking
- Multilingual support (Persian/English)
- Dynamic filtering and reranking
- Conversation memory

### Roadmap

**v1.1.0** (Planned)
- Query caching
- Performance optimizations
- Additional LLM providers
- Enhanced error handling

**v2.0.0** (Future)
- User preference learning
- Image search
- Price history tracking
- Recommendation engine

---

## Final Notes

This Shopping AI Assistant represents a modern approach to e-commerce search, combining:
- **Semantic Understanding**: Deep learning embeddings
- **Conversation Intelligence**: LangGraph state management
- **Hybrid Search**: Best of keyword and semantic matching
- **Intent Awareness**: Adaptive ranking based on user needs
- **Multilingual Capability**: Native Persian/Farsi support

The system is designed to be:
- **Production-ready**: Error handling, logging, monitoring
- **Scalable**: Singleton patterns, connection pooling
- **Maintainable**: Clear architecture, documented code
- **Extensible**: Modular design, easy to add features

Whether you're building an e-commerce platform, a shopping chatbot, or a product recommendation system, this project provides a solid foundation with state-of-the-art technologies.

**Happy coding! 🛍️🤖**

---

*Documentation last updated: November 25, 2025*
*Project: Shopping AI Assistant*
*Repository: https://github.com/11linear11/ShoppingAiAssistant*
