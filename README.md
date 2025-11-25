# Shopping AI Assistant - Quick Start Guide

> **For complete documentation, see [DOCUMENTATION.md](DOCUMENTATION.md)**

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp config/.env.example .env
# Edit .env with your credentials:
# - api_key (NVIDIA)
# - ELASTICSEARCH_HOST
# - ELASTICSEARCH_PASSWORD
```

### 3. Run
```bash
python main.py
```

### 4. Try It
```
User: یه هدفون ارزان میخوام
```

---

## 📚 Key Concepts

### System Architecture
```
User Query → LangGraph Agent → interpret_query → search_products_semantic → Elasticsearch → Results
```

### Two Main Tools

1. **interpret_query**: Understands user intent
   - Input: `"یه هدفون ارزان میخوام"`
   - Output: `{category, intent, sensitivities, suggested_query}`

2. **search_products_semantic**: Finds products
   - Uses hybrid search (BM25 + Vector)
   - Ranks by value_score
   - Returns top 1-5 products

### Five Shopping Intents

| Intent | User Wants | Example |
|--------|------------|---------|
| `find_cheapest` | Lowest price | "ارزان‌ترین گوشی" |
| `find_high_quality` | Best quality | "بهترین لپ تاپ" |
| `find_best_value` | Best price/quality | "گوشی با ارزش خوب" |
| `compare` | Multiple options | "چند تا گوشی نشون بده" |
| `find_by_feature` | Specific feature | "هدفون نرم" |

---

## 🔧 Configuration

### Required Environment Variables
```bash
api_key=nvapi-xxxxx              # NVIDIA AI API key
ELASTICSEARCH_HOST=your_host     # ES host address
ELASTICSEARCH_PASSWORD=your_pass # ES password
```

### Optional Variables
```bash
DEBUG_MODE=false                 # Enable detailed logging
ELASTICSEARCH_PORT=9200          # ES port
ELASTICSEARCH_INDEX=shopping_products
MODEL_NAME=openai/gpt-oss-120b  # LLM model
```

---

## 💡 Example Queries

### Price-Focused
```
User: ارزان‌ترین گوشی رو نشون بده
→ Results sorted by lowest price
```

### Quality-Focused
```
User: بهترین هدفون با کیفیت عالی
→ Results sorted by brand score
```

### Feature-Specific
```
User: یه شامپوی نرم کننده میخوام
→ Results match "نرم کننده" feature
```

### Implicit Need
```
User: سردمه
→ Suggests: کاپشن (jacket)
```

```
User: گشنمه
→ Suggests: بیسکویت (biscuit)
```

---

## 🐛 Troubleshooting

### No Results?
1. Check Elasticsearch connection
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
