# Shopping AI Assistant 🛍️

A conversational AI shopping assistant that uses **LangGraph**, **Elasticsearch**, and **multilingual semantic search** to help users find products naturally.

## Features ✨

- **Semantic Search**: Uses `intfloat/multilingual-e5-base` model for understanding queries in multiple languages (including Persian/Farsi)
- **Elasticsearch Integration**: Fast and scalable vector search capabilities
- **Conversational AI**: LangGraph-based agent with conversation memory
- **Tool Integration**: Automatically decides when to search for products based on user intent
- **JSON Output**: Returns product results in structured JSON format
- **Multilingual Support**: Works with English, Persian, and other languages

## Architecture 🏗️

```
User Query → LangGraph Agent → LLM → Tool Selection → Elasticsearch Search
                ↓                                              ↓
         Conversation Memory ← JSON Response ← Semantic Embedding
```

## Project Structure 📁

```
ShoppingAiAssistant/
├── src/                    # Source code
│   ├── agent.py           # LangGraph agent implementation
│   ├── tools/             # Tools package
│   │   ├── SearchProducts.py  # Elasticsearch search tool
│   │   └── __init__.py
│   └── __init__.py
├── tests/                  # Test files
│   └── test_json_output.py
├── examples/               # Usage examples
│   ├── basic_usage.py
│   └── README.md
├── config/                 # Configuration files
│   └── .env.example       # Environment variables template
├── script/                 # Utility scripts
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.example)
├── .gitignore
└── README.md
```

## Installation 🚀

1. **Clone the repository**:
```bash
git clone https://github.com/11linear11/ShoppingAiAssistant.git
cd ShoppingAiAssistant
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**:
```bash
cp config/.env.example .env
# Edit .env and add your API keys
```

## Configuration ⚙️

Edit `.env` file with your credentials:

```env
# NVIDIA AI Endpoints
api_key=your_nvidia_api_key_here
BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=openai/gpt-oss-120b

# Elasticsearch
ELASTICSEARCH_HOST=your_elasticsearch_host
ELASTICSEARCH_PORT=9201
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=your_password
ELASTICSEARCH_INDEX=shopping_products
ELASTICSEARCH_SCHEME=http
```

## Usage 💻

### Basic Usage

Run the interactive CLI:

```bash
python main.py
```

Example conversation:
```
User: سلام
Assistant: {"message": "سلام! چطور می‌تونم کمکتون کنم؟"}

User: دوغ آبعلی میخوام
Assistant: {"products": [...]}
```

### Python API

```python
from src.agent import create_agent
from langchain_core.messages import HumanMessage

# Create agent
graph = create_agent()
config = {"configurable": {"thread_id": "session_1"}}

# Send message
state = graph.invoke(
    {"messages": [HumanMessage(content="دوغ پیدا کن برام")]},
    config=config
)

# Get response
print(state['messages'][-1].content)
```

### Run Tests

```bash
# Test JSON output
python tests/test_json_output.py

# Basic usage example
python examples/basic_usage.py
```

## JSON Response Format 📋

### Product Search Response
```json
{
  "products": [
    {
      "name": "دوغ گازدار آبعلی ۲۶۰ میلی لیتری",
      "price": 27500,
      "brand": "آبعلی",
      "discount": 15,
      "product_id": "3546253",
      "similarity": 0.872,
      "category": "لبنیات"
    }
  ]
}
```

### Chat Response
```json
{
  "message": "سلام! چطور می‌تونم کمکتون کنم؟"
}
```

## Tech Stack 🛠️

- **LangChain & LangGraph**: Agent orchestration and conversation flow
- **NVIDIA AI Endpoints**: LLM inference (gpt-oss-120b)
- **Elasticsearch 9.2.0**: Vector search and product indexing
- **Sentence Transformers**: Multilingual embeddings (intfloat/multilingual-e5-base)
- **Python 3.13**: Runtime environment

## Key Components 🔑

### Agent (src/agent.py)
- LangGraph-based conversational agent
- Automatic tool calling
- JSON response node for direct output
- Memory persistence with MemorySaver

### Search Tool (src/tools/SearchProducts.py)
- Elasticsearch semantic search
- Cosine similarity scoring
- Multilingual support
- JSON formatted output

## Development 🔧

### Adding New Tools

1. Create tool in `src/tools/`
2. Decorate with `@tool`
3. Import in `src/agent.py`
4. Add to tools list

### Cleaning Up

```bash
# Remove cache files
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Remove backup files
rm -f *_old.py *.backup
```

## Troubleshooting 🐛

### Import Errors
Make sure you're running from the project root:
```bash
cd ShoppingAiAssistant
python main.py
```

### Elasticsearch Connection
Check your Elasticsearch credentials in `.env` file.

### Token Limit Issues
The agent uses a `json_response` node to bypass LLM token limits for product results.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📄

This project is licensed under the MIT License.

## Author ✍️

11linear11

## Acknowledgments 🙏

- LangChain team for the amazing framework
- Elasticsearch for powerful search capabilities
- HuggingFace for multilingual embeddings
