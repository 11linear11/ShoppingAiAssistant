# API و قراردادهای MCP (فارسی)

## ۱) API بک‌اند
Base: `http://<host>:8080`

### ۱.۱ `POST /api/chat`
نمونه درخواست:
```json
{
  "message": "دوغ میخوام",
  "session_id": "optional-uuid"
}
```

ساختار پاسخ:
```json
{
  "success": true,
  "response": "...",
  "session_id": "...",
  "products": [
    {
      "id": "1",
      "name": "...",
      "brand": "...",
      "price": 0,
      "discount_price": null,
      "has_discount": false,
      "discount_percentage": 0,
      "image_url": null,
      "product_url": ""
    }
  ],
  "metadata": {
    "took_ms": 0,
    "query_type": "direct",
    "total_results": 1,
    "from_agent_cache": false,
    "original_took_ms": null,
    "cached_at": null
  }
}
```

### ۱.۲ `GET /api/health`
وضعیت سلامت سرویس‌ها را برمی‌گرداند.

## ۲) ابزارهای MCP
### ۲.۱ Interpret (`:5004`)
- `interpret_query(query, session_id, context)`
- `classify_query(query)`
- `get_interpreter_info()`

### ۲.۲ Search (`:5002`)
- `search_products(search_params, session_id, use_cache, use_semantic)`
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()`

### ۲.۳ Embedding (`:5003`)
- `generate_embedding(text, normalize, use_cache)`
- `generate_embeddings_batch(texts, normalize, use_cache)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()`
- `clear_embedding_cache()`
- `get_model_info()`

## ۳) قرارداد ابزارهای ایجنت
ایجنت انتظار دارد خروجی interpret شامل موارد زیر باشد:
- `query_type`
- `searchable`
- `search_params.intent`
- `search_params.product`

اگر `searchable=false` باشد، ایجنت نباید search اجرا کند و باید سوال تکمیلی بدهد.

## ۴) الگوی خطا
- در خطای runtime، API پیام امن فارسی برمی‌گرداند.
- در خطای ابزار MCP، پاسخ به شکل `{"success": false, "error": "..."}` است.
- خطاهای transport و session در MCP client retry می‌شوند.
