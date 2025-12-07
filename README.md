# 🛒 Shopping AI Assistant

An intelligent Persian shopping assistant powered by LangGraph, MCP (Model Context Protocol), and Elasticsearch. Uses EQuIP 3B for generating Elasticsearch DSL queries and multilingual embeddings for semantic search.

## ✨ Features

- **Persian Language Support**: Full support for Persian shopping queries
- **Hybrid Search**: Combines BM25 text matching with semantic similarity
- **Smart Ranking**: Value-based product ranking considering brand scores, prices, and discounts
- **Intent Detection**: Understands shopping intents (cheapest, best quality, best value, etc.)
- **Modular Architecture**: Separate MCP servers for each functionality
- **Observability**: Integrated with Logfire for tracing and logging

## 🏗️ Architecture

The system uses a distributed MCP (Model Context Protocol) architecture:

### MCP Servers

| Server | Port | Description |
|--------|------|-------------|
| **embedding-server** | 5003 | Generates multilingual embeddings using intfloat/multilingual-e5-base |
| **interpret-server** | 5004 | Interprets Persian queries, extracts intent, translates to English |
| **equip-server** | 5005 | Generates Elasticsearch DSL using EQuIP 3B model |
| **dsl-processor-server** | 5006 | Transforms English DSL to Persian + adds semantic search |
| **search-server** | 5002 | Orchestrates search pipeline and executes ES queries |

### Data Flow

1. **User Query** → Agent receives Persian shopping query
2. **Interpret** → Extracts intent, translates keywords, identifies categories
3. **EQuIP DSL** → Generates Elasticsearch DSL from structured prompt
4. **DSL Processing** → Converts English terms to Persian, adds semantic search
5. **Elasticsearch** → Executes hybrid search (BM25 + vector similarity)
6. **Ranking** → Applies value-based scoring with brand scores
7. **Response** → Returns ranked products to user

## 📁 Project Structure

```
ShoppingAiAssistant/
├── main.py                      # Main entry point
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
├── BrandScore.json              # Brand reputation scores
├── CategoryW.json               # Category weights
├── full_category_embeddings.json # Pre-computed category embeddings
├── test_mcp_servers.py          # Server testing script
├── src/
│   ├── agent.py                 # LangGraph agent implementation
│   ├── logging_config.py        # Centralized Logfire configuration
│   └── mcp_servers/
│       ├── run_servers.py       # Server orchestrator
│       ├── embedding_server.py  # Embedding generation
│       ├── interpret_server.py  # Query interpretation
│       ├── equip_server.py      # DSL generation
│       ├── dsl_processor_server.py # DSL transformation
│       └── search_server.py     # Search orchestration
├── script/                      # Utility scripts
├── config/                      # Configuration files
└── logs/                        # Server logs
```

## 📋 Prerequisites

- Python 3.11+
- Elasticsearch 8.x with shopping products index
- Ollama (for running EQuIP 3B model)
- GROQ API key (for LLM agent)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/11linear11/ShoppingAiAssistant.git
   cd ShoppingAiAssistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** - Create a .env file with your settings

## ⚙️ Configuration

Copy the example environment file and configure:

```bash
cp config/.env.example .env
```

### Environment Variables

```env
# ============================================
# 🔧 GENERAL SETTINGS
# ============================================
DEBUG_MODE=false

# ============================================
# 🤖 AI/LLM PROVIDERS
# ============================================

# Groq API (Main Agent LLM)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# NVIDIA API (Interpret Server - Query Understanding)
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.1-nemotron-70b-instruct

# EQuIP 3B (DSL Generation - Cloudflare Tunnel)
# ⚠️ This URL changes with each Colab restart!
EQUIP_BASE_URL=https://your-cloudflare-tunnel.trycloudflare.com
EQUIP_MODEL=EQuIP/EQuIP_3B

# ============================================
# 🔍 ELASTICSEARCH
# ============================================
ELASTICSEARCH_HOST=your_elasticsearch_host
ELASTICSEARCH_PORT=9201
ELASTICSEARCH_SCHEME=http
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=your_elasticsearch_password
ELASTICSEARCH_INDEX=shopping_products

# ============================================
# 🌐 MCP SERVERS (Internal URLs)
# ============================================
MCP_SEARCH_URL=http://localhost:5002
MCP_EMBEDDING_URL=http://localhost:5003
MCP_INTERPRET_URL=http://localhost:5004
MCP_EQUIP_URL=http://localhost:5005
MCP_DSL_PROCESSOR_URL=http://localhost:5006

# ============================================
# 📊 LOGGING & MONITORING
# ============================================
LOGFIRE_TOKEN=your_logfire_token_here
LOGFIRE_SERVICE_NAME=shopping-assistant
SEND_TO_LOGFIRE=if-token-present

# ============================================
# 🔐 OPTIONAL API KEYS
# ============================================
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
GITHUB_TOKEN=your_github_token_here
```

## 🔄 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER QUERY                                      │
│                        "دوغ ارزان میخوام"                                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🤖 AGENT (LangGraph)                                 │
│                     Port: N/A (Main Process)                                │
│  • Uses Groq LLM (llama-3.3-70b-versatile)                                  │
│  • Orchestrates tools: interpret_query → search_with_interpretation         │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
┌─────────────────────────────┐         ┌─────────────────────────────────────┐
│  📝 INTERPRET SERVER        │         │      🔍 SEARCH SERVER               │
│  Port: 5004                 │         │      Port: 5002                     │
│                             │         │                                     │
│  • NVIDIA LLM               │         │  Orchestrates internally:           │
│  • Query understanding      │         │  ┌─────────────────────────────┐   │
│  • Intent detection         │         │  │ 🧠 EQUIP SERVER (5005)     │   │
│  • English translation      │         │  │ • EQuIP 3B via Cloudflare  │   │
│  • Category extraction      │         │  │ • Generates ES DSL         │   │
│                             │         │  └──────────────┬──────────────┘   │
│  Calls embedding server     │         │                 │                   │
│  for category matching      │         │                 ▼                   │
└──────────────┬──────────────┘         │  ┌─────────────────────────────┐   │
               │                         │  │ 🔄 DSL PROCESSOR (5006)    │   │
               │                         │  │ • English → Persian        │   │
               │                         │  │ • Adds semantic search     │   │
               │                         │  │ • Adds hybrid scoring      │   │
               │                         │  └──────────────┬──────────────┘   │
               │                         │                 │                   │
               │                         │                 ▼                   │
               │                         │  ┌─────────────────────────────┐   │
               │                         │  │ 🔎 ELASTICSEARCH           │   │
               │                         │  │ • Executes hybrid query    │   │
               │                         │  │ • BM25 + Vector similarity │   │
               │                         │  └──────────────┬──────────────┘   │
               │                         │                 │                   │
               │                         │                 ▼                   │
               │                         │  ┌─────────────────────────────┐   │
               │                         │  │ 📊 VALUE RANKING           │   │
               │                         │  │ • Brand scores             │   │
               │                         │  │ • Price normalization      │   │
               │                         │  │ • Discount consideration   │   │
               │                         │  └─────────────────────────────┘   │
               │                         │                                     │
               ▼                         └─────────────────┬───────────────────┘
┌─────────────────────────────┐                           │
│  🔢 EMBEDDING SERVER        │                           │
│  Port: 5003                 │◄──────────────────────────┤
│                             │                           │
│  • multilingual-e5-base     │                           │
│  • 768-dim embeddings       │                           │
│  • Category matching        │                           │
└─────────────────────────────┘                           │
                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           📦 RANKED PRODUCTS                                 │
│                                                                              │
│  [                                                                           │
│    {"name": "دوغ عالیس", "price": 15000, "score": 0.92},                    │
│    {"name": "دوغ میهن", "price": 18000, "score": 0.87},                     │
│    ...                                                                       │
│  ]                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Steps

| Step | Server | Action |
|------|--------|--------|
| 1 | Agent | Receives user query |
| 2 | Interpret (5004) | Extracts intent, translates, finds categories |
| 3 | EQuIP (5005) | Generates Elasticsearch DSL query |
| 4 | DSL Processor (5006) | Converts English→Persian, adds embeddings |
| 5 | Embedding (5003) | Generates vector for semantic search |
| 6 | Search (5002) | Executes query, applies value ranking |
| 7 | Agent | Returns formatted results to user |

## 🎯 Usage

### Start all MCP servers

```bash
python src/mcp_servers/run_servers.py
```

### Run the agent

```bash
python main.py
```

### Test servers

```bash
python test_mcp_servers.py           # All servers
python test_mcp_servers.py pipeline  # Full pipeline test
python test_mcp_servers.py interpret # Interpret server only
```

## 🔧 API Reference

### interpret_query

Analyzes user shopping query and returns structured data.

**Input:** Persian shopping query
**Output:**
- equip_prompt: Structured prompt for DSL generation
- token_mapping: English to Persian word mappings
- persian_full_query: Full Persian product description
- categories_fa: Relevant Persian category names
- intent: Shopping intent (find_cheapest, find_best_value, etc.)
- price_sensitivity: 0-1 score
- quality_sensitivity: 0-1 score

### search_with_interpretation

Searches products using interpretation results.

**Input:** All outputs from interpret_query
**Output:** Ranked products with scores

## 📊 Shopping Intents

| Intent | Description |
|--------|-------------|
| find_cheapest | User wants the lowest price |
| find_best_value | Balance between price and quality |
| find_high_quality | User prioritizes quality over price |
| find_by_feature | Searching for specific features |
| compare | Comparing multiple products |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 👤 Author

Created by [11linear11](https://github.com/11linear11)
