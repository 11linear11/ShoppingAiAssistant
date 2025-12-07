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

Create a .env file:

```env
# Debug mode
DEBUG_MODE=true

# Elasticsearch
ELASTICSEARCH_HOST=your_elasticsearch_host
ELASTICSEARCH_PORT=9201
ELASTICSEARCH_INDEX=shopping_products

# EQuIP Model (Ollama via Cloudflare tunnel or local)
EQUIP_BASE_URL=https://your-tunnel.trycloudflare.com
EQUIP_MODEL=EQuIP/EQuIP_3B

# LLM for Agent
GROQ_API_KEY=your_groq_api_key

# Logfire (optional)
LOGFIRE_TOKEN=your_logfire_token
```

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
