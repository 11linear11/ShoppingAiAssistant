# Shopping AI Assistant 🛍️🤖

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.2.0-orange.svg)](https://www.elastic.co/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.2-green.svg)](https://github.com/langchain-ai/langgraph)

An intelligent, production-ready conversational shopping assistant built with **LangGraph**, **Elasticsearch**, and **multilingual semantic search**. It understands Persian/Farsi and English, detects user intent, searches hybridly (BM25 + embeddings), and returns structured, ranked product results.

---

## 🌟 Features

- 🧠 Intent-aware shopping (cheapest, high-quality, best value, compare, by feature)
- 🔍 Hybrid search: BM25 keyword + semantic vector search (multilingual-e5-base)
- 🌍 Multilingual: Native Persian/Farsi support
- 📊 Smart reranking using brand score, price, similarity, discount
- 🎯 Adaptive relevance filtering with dynamic thresholds
- 💬 Conversation memory via LangGraph threads
- ⚡ Fast responses powered by Elasticsearch vector search
- 🖨️ Clean Persian output formatting with summaries

---
## 🏗️ Architecture

## 📁 Project Structure
User ➜ LangGraph Agent
            ├─ interpret_query (LLM)
            └─ search_products_semantic (Elasticsearch + embeddings)
                                       └─ ProductSearchEngine (singleton)
                                             ├─ SentenceTransformer (multilingual-e5-base)
                                             ├─ Elasticsearch (BM25 + script_score cosine)
                                             └─ BrandScore.json (brand reputation)
```

---

## � Project Structure

```
ShoppingAiAssistant/
├── main.py                 # CLI entry point
├── src/
│   ├── agent.py           # LangGraph agent, tools binding, memory
│   └── tools/
│       └── SearchProducts.py  # Search engine + tools (interpret_query, search_products_semantic)
├── config/
│   └── .env.example       # Environment template
├── BrandScore.json        # Precomputed brand scores
├── CategoryW.json         # Category weights (for brand scoring script)
├── requirements.txt       # Dependencies
└── DOCUMENTATION.md       # Full technical documentation
```

---

## ⚙️ Setup

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Configure environment
```bash
cp config/.env.example .env
# Edit values:
# api_key=your_nvidia_api_key
# ELASTICSEARCH_HOST=your_host
# Shopping AI Assistant 🛍️🤖

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-9.2.0-orange.svg)](https://www.elastic.co/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0.2-green.svg)](https://github.com/langchain-ai/langgraph)

An intelligent, production-ready conversational shopping assistant built with **LangGraph**, **Elasticsearch**, and **multilingual semantic search**. It understands Persian/Farsi and English, detects user intent, performs hybrid search (BM25 + embeddings), re-ranks results with smart scoring, and returns clean, structured product answers.

---

## � Overview

This project provides a conversational agent that:
- Interprets user intent (cheapest, high-quality, best value, compare, feature)
- Suggests a refined query keyword (`suggested_query`)
- Searches products via Elasticsearch using BM25 + vector cosine similarity
- Applies dynamic relevance filtering and smart reranking
- Formats answers in Persian with summary lines

---

## 🌟 Features

- 🧠 Intent-aware shopping workflow (via `interpret_query` tool)
- 🔍 Hybrid search (BM25 + semantic embeddings `multilingual-e5-base`)
- 🌍 Multilingual support (native Persian/Farsi, English)
- 📊 Smart reranking with brand score, similarity, discount, price
- 🎯 Dynamic relevance thresholds (median-based) and top-5 cap
- 💬 Conversation memory using LangGraph `MemorySaver` and thread IDs
- ⚡ Fast responses through Elasticsearch vector search

---

## 🏗️ Architecture

```
User ➜ LangGraph Agent
            ├─ interpret_query (LLM → structured intent)
            └─ search_products_semantic (Elasticsearch + embeddings)
                                             └─ ProductSearchEngine (singleton)
                                                   ├─ SentenceTransformer (multilingual-e5-base)
                                                   ├─ Elasticsearch (BM25 + script_score cosine)
                                                   └─ BrandScore.json (brand reputation)
```

---

## 📁 Project Structure

```
ShoppingAiAssistant/
├── main.py                 # CLI entry point
├── src/
│   ├── agent.py           # LangGraph agent, tools binding, memory
│   └── tools/
│       └── SearchProducts.py  # Search engine + tools (interpret_query, search_products_semantic)
├── config/
│   └── .env.example       # Environment template
├── BrandScore.json        # Precomputed brand scores
├── CategoryW.json         # Category weights (script reference)
├── requirements.txt       # Dependencies
└── DOCUMENTATION.md       # Full technical documentation
```

---

## ⚙️ Setup & Run

### 1) Install
```bash
pip install -r requirements.txt
```

### 2) Configure environment
```bash
cp config/.env.example .env
# Edit values:
# DEBUG_MODE=false
# api_key=your_nvidia_api_key
# ELASTICSEARCH_HOST=your_host
# ELASTICSEARCH_PORT=9200
# ELASTICSEARCH_USER=elastic
# ELASTICSEARCH_PASSWORD=your_password
# ELASTICSEARCH_INDEX=shopping_products
# ELASTICSEARCH_SCHEME=http
```

### 3) Run
```bash
python main.py
```

CLI shows:
```
Shopping AI Assistant
==================================================
Type 'exit' to quit
```

---

## 🔧 Configuration requirements

Elasticsearch index must include fields:
- `product_name` (text), `brand_name` (text), `category_name` (text)
- `price` (float), `discount_price` (float), `has_discount` (boolean), `discount_percentage` (float)
- `product_embedding` (dense_vector, dims=768, similarity=cosine)

---

## ▶️ How it works (end-to-end)

1. User enters a message (Persian or English)
2. Agent calls `interpret_query` to extract:
    - `category`, `intent`, `price_sensitivity`, `quality_sensitivity`, `suggested_query`
3. Agent calls `search_products_semantic` with these parameters
4. `ProductSearchEngine` runs a hybrid ES search (BM25 + cosine on embeddings)
5. Products are filtered by dynamic relevance threshold and re-ranked by `value_score`
6. Agent returns 1–5 top products with Persian formatting and a summary line

---

## 🧩 Tools API

### interpret_query(query: str) → JSON string
Returns structured intent, e.g.:
```json
{
   "category": "لوازم الکترونیکی",
   "intent": "find_best_value",
   "price_sensitivity": 0.6,
   "quality_sensitivity": 0.7,
   "suggested_query": "هدفون"
}
```

### search_products_semantic(query, quality_sensitivity, price_sensitivity, category, intent) → JSON string
Returns products and metadata, e.g.:
```json
{
   "products": [{
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
   }],
   "meta": {
      "query": "هدفون",
      "total_found": 5,
      "is_relevant": true,
      "avg_similarity": 0.75,
      "price_range": {"min": 1500000, "max": 4500000},
      "intent": "find_best_value"
   }
}
```

---

## 🔍 Elasticsearch Mapping (example)
```json
{
   "mappings": {
      "properties": {
         "product_id": { "type": "keyword" },
         "product_name": { "type": "text" },
         "brand_name": { "type": "text" },
         "category_name": { "type": "text" },
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

---

## 🧪 Usage examples

- Price-focused: «ارزان‌ترین گوشی رو نشون بده» → `intent = find_cheapest`
- Quality-focused: «بهترین هدفون با کیفیت عالی» → `intent = find_high_quality`
- Feature-focused: «یه شامپوی نرم‌کننده میخوام» → `intent = find_by_feature`
- Implicit need: «سردمه» → `suggested_query = کاپشن`
- Compare: «چند تا گوشی از برندهای مختلف نشون بده» → `intent = compare`

---

## 🛠️ Troubleshooting

- No results: verify index exists and fields; relax category filter; set `DEBUG_MODE=true` to inspect logs
- ES connection errors: check `.env` host/port/user/password; test connectivity
- LLM empty response: try `MODEL_NAME=meta/llama-3.1-70b-instruct`; verify `api_key`
- Slow first run: model loading is one-time; subsequent queries should be fast (< 400ms)

---

## ⚡ Performance Notes

- Singleton ES client and embedding model for warm performance
- Dynamic similarity thresholds (median-based) balance precision/recall
- Reranking leverages `BrandScore.json` for brand reputation

---

## 📝 License

MIT. See `LICENSE`.

---

## 📎 More

For deep technical details (architecture diagrams, formulas, benchmarks, advanced config), see **`DOCUMENTATION.md`**.

Last updated: Nov 26, 2025
2. Verify index exists: `GET /shopping_products/_count`
3. Enable debug mode: `DEBUG_MODE=true`

### Slow Performance?
1. First run is slow (model loading ~3s) - normal
2. Subsequent queries should be <400ms
3. Check Elasticsearch performance

### LLM Errors?
1. Verify API key is valid
2. Try different model: `MODEL_NAME=meta/llama-3.1-70b-instruct`
3. Check internet connection

---

## 📊 Tech Stack

- **LangGraph 1.0.2**: Conversation orchestration
- **Elasticsearch 9.2.0**: Vector search
- **multilingual-e5-base**: Semantic embeddings (768-dim)
- **NVIDIA LLama 3.1 70B**: LLM

---

## 📁 Project Structure

```
ShoppingAiAssistant/
├── main.py                    # Entry point
├── src/
│   ├── agent.py              # LangGraph agent
│   └── tools/
│       └── SearchProducts.py # Search engine
├── config/
│   └── .env.example          # Config template
├── BrandScore.json           # Brand scores
└── requirements.txt          # Dependencies
```

---

## 🎯 How It Works (Simplified)

1. **User sends query**: "یه هدفون ارزان میخوام"

2. **Agent calls interpret_query**:
   ```json
   {
     "category": "لوازم الکترونیکی",
     "intent": "find_cheapest",
     "suggested_query": "هدفون"
   }
   ```

3. **Agent calls search_products_semantic**:
   - Generates embedding for "هدفون"
   - Searches Elasticsearch (hybrid: BM25 + Vector)
   - Gets 50 candidates

4. **Reranking**:
   - Calculates value_score for each
   - Filters by relevance (similarity ≥ 0.4)
   - Sorts by value_score
   - Returns top 5

5. **Agent formats response**:
   ```
   🛒 هدفون JBL Tune 500BT
      💰 قیمت: 850,000 تومان
      🏷️ برند: JBL
      🔥 تخفیف: 10%
   ```

---

## 🔍 Value Score Formula

**Default:**
```python
value_score = (
    brand_score × quality_sensitivity +
    similarity × 0.4 +
    discount × 0.2 -
    normalized_price × price_sensitivity
)
```

**Adjusts based on intent:**
- `find_cheapest`: Price weight = 2.0
- `find_high_quality`: Brand weight = 1.5
- `find_by_feature`: Similarity weight = 1.5

---

## 📖 Full Documentation

For comprehensive details, see:
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete technical documentation (1800+ lines)

Covers:
- Detailed architecture diagrams
- API reference
- Performance optimization
- Advanced features
- Contributing guidelines
- Troubleshooting guide
- Future enhancements

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/name`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/name`
5. Open Pull Request

---

## 📞 Support

- **GitHub Issues**: [Create issue](https://github.com/11linear11/ShoppingAiAssistant/issues)
- **Full Docs**: [DOCUMENTATION.md](DOCUMENTATION.md)

---

**Version**: 1.0.0  
**Last Updated**: November 25, 2025  
**License**: MIT
