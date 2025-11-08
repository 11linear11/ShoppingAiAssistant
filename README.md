# Shopping AI Assistant 🛍️

A conversational AI shopping assistant that uses **LangGraph**, **Elasticsearch**, and **multilingual semantic search** to help users find products naturally.

## Features ✨

- **Semantic Search**: Uses `intfloat/multilingual-e5-base` model for understanding queries in multiple languages (including Persian/Farsi)
- **Elasticsearch Integration**: Fast and scalable vector search capabilities
- **Conversational AI**: LangGraph-based agent with conversation memory
- **Tool Integration**: Automatically decides when to search for products based on user intent
- **Multilingual Support**: Works with English, Persian, and other languages

## Architecture 🏗️

```
User Query → LangGraph Agent → LLM (GPT-4o) → Tool Selection → Elasticsearch Search
                ↓                                                         ↓
         Conversation Memory ← Response Generation ← Semantic Embedding
```

## Prerequisites 📋

1. **Python 3.8+**
2. **Elasticsearch** running on `localhost:9200` (or configure your own)
3. **API Keys** configured in `.env` file

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

4. **Setup Elasticsearch**:
```bash
# Using Docker (easiest way)
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  elasticsearch:8.11.0
```

5. **Configure environment variables**:
Create a `.env` file with:
```env
# LLM API Configuration
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o

# Elasticsearch Configuration
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
ELASTICSEARCH_USER=elastic
ELASTICSEARCH_PASSWORD=
ELASTICSEARCH_INDEX=products
```

## Usage 💻

### 1. Index Sample Products

First, run the search module to index sample products:

```bash
python SearchProducts.py
```

This will:
- Create the Elasticsearch index with proper mappings
- Generate embeddings using `intfloat/multilingual-e5-base`
- Index sample products
- Run test searches

### 2. Run the Agent

```bash
python agent.py
```

### 3. Use in Your Code

```python
from agent import create_agent
from langchain_core.messages import HumanMessage

# Create agent
graph = create_agent()

# Configure session
config = {"configurable": {"thread_id": "user_123"}}

# Chat with agent
state = graph.invoke(
    {"messages": [HumanMessage(content="Find me a gaming laptop")]},
    config=config
)

print(state['messages'][-1].content)
```

## Example Queries 🗣️

The agent understands natural language in multiple languages:

**English:**
- "Show me gaming laptops"
- "I need a cheap phone"
- "Find wireless headphones"

**Persian/Farsi:**
- "لپ تاپ گیمینگ نشون بده"
- "یک گوشی ارزون میخوام"
- "هدفون بی سیم پیدا کن"

## Project Structure 📁

```
ShoppingAiAssistant/
├── agent.py                 # Main LangGraph agent
├── SearchProducts.py        # Elasticsearch + Embedding search tool
├── main.py                  # Entry point (if needed)
├── .env                     # Environment configuration
├── README.md               # This file
├── dataset/
│   └── export-product-list.csv  # Product data
└── script/
    └── shopping_embedding_colab.ipynb  # Jupyter notebook for embeddings
```

## How It Works 🔍

1. **User Query**: User asks a question in natural language
2. **Intent Detection**: LLM (GPT-4o) decides if a product search is needed
3. **Tool Invocation**: If needed, calls `search_products_semantic` tool
4. **Query Embedding**: Converts query to 768-dim vector using multilingual model
5. **Vector Search**: Elasticsearch performs KNN search on product embeddings
6. **Results Ranking**: Returns top-k most relevant products with scores
7. **Response Generation**: LLM formats results into natural conversation

## Customization ⚙️

### Add Your Own Products

Edit `SearchProducts.py` and modify the `index_sample_products()` function:

```python
products = [
    {
        "product_id": "YOUR_ID",
        "name": "Product Name",
        "description": "Detailed description",
        "price": 99.99,
        "category": "Category"
    },
    # Add more products...
]
```

### Change Embedding Model

In `SearchProducts.py`, modify the model name:

```python
model_name = "your-preferred-model"  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
```

### Adjust Search Parameters

```python
results = engine.search(
    query="your query",
    top_k=10,        # Number of results
    min_score=0.3    # Minimum similarity threshold (0-1)
)
```

## Troubleshooting 🔧

### Elasticsearch Connection Error
```bash
# Check if Elasticsearch is running
curl http://localhost:9200

# If not, start it:
docker start elasticsearch
```

### Model Download Issues
The first run will download the embedding model (~500MB). Ensure you have:
- Stable internet connection
- At least 2GB free disk space

### Memory Issues
If you encounter memory errors:
- Use a smaller embedding model
- Reduce batch size in embeddings
- Increase Docker/system memory allocation

## Dependencies 📦

Main packages:
- `langchain` - LLM framework
- `langgraph` - Graph-based agent orchestration
- `elasticsearch` - Search engine client
- `sentence-transformers` - Embedding models
- `python-dotenv` - Environment management

## Contributing 🤝

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

MIT License - feel free to use this project for any purpose.

## Contact 📧

- GitHub: [@11linear11](https://github.com/11linear11)
- Project: [ShoppingAiAssistant](https://github.com/11linear11/ShoppingAiAssistant)

---

Made with ❤️ using LangGraph, Elasticsearch, and Multilingual AI