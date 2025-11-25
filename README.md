# Shopping AI Assistant 🛍️# Shopping AI Assistant 🛍️# Shopping AI Assistant 🛍️



An intelligent shopping assistant powered by LangGraph, Elasticsearch, and multilingual semantic search to help users find products naturally.



## Featuresیک دستیار هوشمند خرید که با استفاده از LangGraph، Elasticsearch و جستجوی معنایی چندزبانه به کاربران کمک می‌کند محصولات را به صورت طبیعی پیدا کنند.A conversational AI shopping assistant that uses **LangGraph**, **Elasticsearch**, and **multilingual semantic search** to help users find products naturally.



- Semantic search using `intfloat/multilingual-e5-base` model

- Hybrid search (BM25 + Embedding)

- Smart ranking based on quality, price, and relevance## ویژگی‌ها## Features ✨

- Intelligent user intent analysis

- Persian/Farsi language support



## Project Structure- جستجوی معنایی با مدل `intfloat/multilingual-e5-base`- **Semantic Search**: Uses `intfloat/multilingual-e5-base` model for understanding queries in multiple languages (including Persian/Farsi)



```- جستجوی ترکیبی (BM25 + Embedding)- **Hybrid Search**: Combines BM25 text matching with semantic embedding search

ShoppingAiAssistant/

├── src/- رتبه‌بندی هوشمند بر اساس کیفیت، قیمت و ارتباط- **Intelligent Reranking**: Products ranked by value_score (quality + price + relevance)

│   ├── agent.py              # LangGraph Agent implementation

│   └── tools/- تحلیل هوشمند نیت کاربر- **Query Intent Analysis**: Automatic understanding of user shopping preferences

│       └── SearchProducts.py # Elasticsearch search tool

├── config/- پشتیبانی از زبان فارسی- **Dynamic Filtering**: Adaptive similarity thresholds based on result quality

│   └── .env.example          # Environment variables template

├── main.py                   # Entry point- **Elasticsearch Integration**: Fast and scalable vector search capabilities

├── requirements.txt          # Dependencies

├── BrandScore.json           # Brand scores## ساختار پروژه- **Conversational AI**: LangGraph-based agent with conversation memory

└── CategoryW.json            # Category weights

```- **Tool Integration**: Automatically decides when to search for products based on user intent



## Installation```- **JSON Output**: Returns product results in structured JSON format



1. Clone the repository:ShoppingAiAssistant/- **Multilingual Support**: Works with English, Persian, and other languages

```bash

git clone https://github.com/11linear11/ShoppingAiAssistant.git├── src/

cd ShoppingAiAssistant

```│   ├── agent.py              # پیاده‌سازی Agent با LangGraph## Architecture 🏗️



2. Create virtual environment:│   └── tools/

```bash

python -m venv venv│       └── SearchProducts.py # ابزار جستجو در Elasticsearch```

source venv/bin/activate

```├── config/User Query → LangGraph Agent → interpret_query (analyze intent)



3. Install dependencies:│   └── .env.example          # نمونه تنظیمات                ↓                         ↓

```bash

pip install -r requirements.txt├── main.py                   # نقطه ورود برنامه         System Prompt          extract preferences

```

├── requirements.txt          # وابستگی‌ها                ↓                         ↓

4. Configure environment variables:

```bash├── BrandScore.json           # امتیاز برندها    Tool Selection    →  search_products_semantic

cp config/.env.example .env

# Edit .env file with your credentials└── CategoryW.json            # وزن دسته‌بندی‌ها                              (BM25 + Embedding)

```

```                                    ↓

## Configuration

                         Hybrid Search Results

Fill in your `.env` file:

## نصب و راه‌اندازی                                    ↓

```env

DEBUG_MODE=false                         Dynamic Filtering



# NVIDIA API1. کلون کردن پروژه:                                    ↓

api_key=your_nvidia_api_key

```bash                    Value Score Reranking

# Elasticsearch

ELASTICSEARCH_HOST=your_hostgit clone https://github.com/11linear11/ShoppingAiAssistant.git              (brand_score × quality + similarity - price)

ELASTICSEARCH_PORT=9200

ELASTICSEARCH_USER=elasticcd ShoppingAiAssistant                                    ↓

ELASTICSEARCH_PASSWORD=your_password

ELASTICSEARCH_INDEX=shopping_products```                         JSON Response → User

```

```

## Usage

2. ساخت محیط مجازی:

Run the application:

```bash```bash### Search Flow:

python main.py

```python -m venv venv1. **Intent Analysis**: `interpret_query` extracts category, intent, price_sensitivity, quality_sensitivity



Example:source venv/bin/activate2. **Hybrid Search**: BM25 (keyword matching) + Semantic (embedding similarity)

```

User: I want cheap headphones```3. **Dynamic Filter**: Median-based threshold removes irrelevant results

Assistant: 

🛒 Bluetooth Headphone XYZ4. **Value Ranking**: Products scored by: `brand_score × quality + 0.4 × similarity - price_sensitivity × final_price`

   💰 Price: 45,000 Toman

   🏷️ Brand: Sony3. نصب وابستگی‌ها:

   🔥 Discount: 15%

``````bash## Project Structure 📁



## APIpip install -r requirements.txt



```python``````

from src.agent import create_agent

from langchain_core.messages import HumanMessageShoppingAiAssistant/



graph = create_agent()4. تنظیم متغیرهای محیطی:├── src/                    # Source code

config = {"configurable": {"thread_id": "session_1"}}

```bash│   ├── agent.py           # LangGraph agent implementation

state = graph.invoke(

    {"messages": [HumanMessage(content="cheap headphones")]},cp config/.env.example .env│   ├── tools/             # Tools package

    config=config

)# فایل .env را ویرایش کنید│   │   ├── SearchProducts.py  # Elasticsearch search tool

print(state['messages'][-1].content)

``````│   │   └── __init__.py



## Tech Stack│   └── __init__.py



- LangChain & LangGraph## تنظیمات├── tests/                  # Test files

- NVIDIA AI Endpoints

- Elasticsearch│   └── test_json_output.py

- Sentence Transformers

فایل `.env` را با اطلاعات خود پر کنید:├── examples/               # Usage examples

## Author

│   ├── basic_usage.py

11linear11

```env│   └── README.md

## License

DEBUG_MODE=false├── config/                 # Configuration files

MIT License

│   └── .env.example       # Environment variables template

# NVIDIA API├── script/                 # Utility scripts

api_key=your_nvidia_api_key├── main.py                # Main entry point

├── requirements.txt       # Python dependencies

# Elasticsearch├── .env                   # Environment variables (create from .env.example)

ELASTICSEARCH_HOST=your_host├── .gitignore

ELASTICSEARCH_PORT=9200└── README.md

ELASTICSEARCH_USER=elastic```

ELASTICSEARCH_PASSWORD=your_password

ELASTICSEARCH_INDEX=shopping_products## Installation 🚀

```

1. **Clone the repository**:

## استفاده```bash

git clone https://github.com/11linear11/ShoppingAiAssistant.git

اجرای برنامه:cd ShoppingAiAssistant

```bash```

python main.py

```2. **Create virtual environment**:

```bash

مثال:python -m venv venv

```source venv/bin/activate  # On Windows: venv\Scripts\activate

User: دوغ آبعلی میخوام```

Assistant: 

🛒 دوغ گازدار آبعلی ۲۶۰ میلی لیتری3. **Install dependencies**:

   💰 قیمت: 23,375 تومان```bash

   🏷️ برند: آبعلیpip install -r requirements.txt

   🔥 تخفیف: 15%```

```

4. **Configure environment variables**:

## API```bash

cp config/.env.example .env

```python# Edit .env and add your API keys

from src.agent import create_agent```

from langchain_core.messages import HumanMessage

## Configuration ⚙️

graph = create_agent()

config = {"configurable": {"thread_id": "session_1"}}Edit `.env` file with your credentials:



state = graph.invoke(```env

    {"messages": [HumanMessage(content="هدفون ارزان")]},# Debug Mode (optional - for detailed logging)

    config=configDEBUG_MODE=false  # Set to true for debugging

)

print(state['messages'][-1].content)# NVIDIA AI Endpoints

```api_key=your_nvidia_api_key_here



## تکنولوژی‌ها# Elasticsearch

ELASTICSEARCH_HOST=your_elasticsearch_host

- LangChain & LangGraphELASTICSEARCH_PORT=9200

- NVIDIA AI EndpointsELASTICSEARCH_USER=elastic

- ElasticsearchELASTICSEARCH_PASSWORD=your_password

- Sentence TransformersELASTICSEARCH_INDEX=shopping_products

ELASTICSEARCH_SCHEME=http

## نویسنده```



11linear11## Usage 💻



## لایسنس### Basic Usage



MIT LicenseRun the interactive CLI:


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

### Debug Mode 🐛

For detailed logging and debugging:

```bash
# Enable debug mode in .env
DEBUG_MODE=true

# Run the app
python main.py

# Check debug log file
tail -f shopping_assistant_debug.log
```

See [DEBUG_GUIDE.md](DEBUG_GUIDE.md) for complete debugging documentation.

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
      "final_price": 23375,
      "brand": "آبعلی",
      "brand_score": 0.762,
      "discount": 15,
      "product_id": "3546253",
      "similarity": 0.872,
      "value_score": 5.234,
      "category": "لبنیات"
    }
  ]
}
```

**New Fields:**
- `final_price`: Price after discount calculation
- `brand_score`: Quality score of the brand (from BrandScore.json)
- `value_score`: Overall value ranking (higher = better deal)

### Query Intent Response
```json
{
  "category": "لپ تاپ",
  "intent": "find_cheapest",
  "price_sensitivity": 0.9,
  "quality_sensitivity": 0.3
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
- Automatic tool calling (interpret_query → search_products_semantic)
- JSON response node for direct output
- Memory persistence with MemorySaver

### Search Tool (src/tools/SearchProducts.py)
- **Hybrid Search**: BM25 + Semantic embedding
- **Query Interpretation**: LLM-based intent analysis
- **Dynamic Filtering**: Median-based similarity threshold
- **Value Reranking**: `brand_score × quality + 0.4 × similarity - price_sensitivity × final_price`
- **Brand Scoring**: Loads quality scores from BrandScore.json
- **Discount Calculation**: Automatic final_price = price - (price × discount / 100)
- Multilingual support (Persian, English, Arabic, etc.)
- JSON formatted output

### Tools Available:
1. **`interpret_query(query)`**: Analyzes user intent and preferences
   - Extracts: category, intent, price_sensitivity, quality_sensitivity
   
2. **`search_products_semantic(query, quality_sensitivity, price_sensitivity)`**: Searches and ranks products
   - Hybrid BM25 + embedding search
   - Dynamic filtering
   - Value-based reranking

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
