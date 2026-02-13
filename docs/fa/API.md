# API و قراردادها (فارسی)

## ۱) API بک‌اند
آدرس پایه پیش‌فرض: `http://<host>:8080`

### ۱.۱ `POST /api/chat`
بدنه درخواست (`ChatRequest`):
```json
{
  "message": "گوشی سامسونگ زیر 20 میلیون میخوام",
  "session_id": "optional-uuid"
}
```

بدنه پاسخ (`ChatResponse`):
```json
{
  "success": true,
  "response": "این محصولات رو پیدا کردم",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "products": [
    {
      "id": "1",
      "name": "...",
      "brand": "...",
      "price": 12000000,
      "discount_price": 10900000,
      "has_discount": true,
      "discount_percentage": 9.1,
      "image_url": null,
      "product_url": ""
    }
  ],
  "metadata": {
    "took_ms": 1530,
    "query_type": "direct",
    "total_results": 10,
    "from_agent_cache": false,
    "original_took_ms": null,
    "cached_at": null,
    "latency_breakdown_ms": {
      "initialize_ms": 0,
      "agent_cache_lookup_ms": 2,
      "agent_chat_ms": 1490,
      "extract_products_ms": 4,
      "clean_response_ms": 1,
      "detect_query_type_ms": 0,
      "agent_cache_set_ms": 1
    },
    "error_stage": null,
    "error_type": null
  }
}
```

#### توضیح `metadata`
- `query_type`: در `AgentService` از شکل خروجی تشخیص داده می‌شود (`direct`, `unclear`, `chat`, `no_results`, `error`, ...)
- `from_agent_cache`: نشانگر hit کش سطح ۲
- `latency_breakdown_ms`: تایم هر مرحله

### ۱.۲ `GET /api/health`
سلامت کلی برای:
- `agent`
- `interpret_server`
- `search_server`

نکته:
- سرویس‌های MCP معمولا `/health` ندارند و endpoint اصلی‌شان `/mcp` است؛ در backend پاسخ 404 قابل‌دسترسی به‌عنوان reachable در نظر گرفته می‌شود.

### ۱.۳ `GET /api/`
endpoint ساده برای وضعیت API.

## ۲) سطح پروتکل MCP
همه سرویس‌های MCP از endpoint زیر استفاده می‌کنند:
- `POST /mcp` (JSON-RPC با `initialize` و `tools/call`)

پیاده‌سازی کلاینت: `src/mcp_client.py`.

## ۳) قرارداد Interpret MCP (`:5004`)

### ابزار `interpret_query(query, session_id, context)`
قرارداد اصلی:
```json
{
  "success": true,
  "query_type": "direct|unclear",
  "searchable": true,
  "search_params": {
    "intent": "browse|find_cheapest|find_best|compare",
    "product": "...",
    "brand": "...",
    "persian_full_query": "...",
    "categories_fa": ["..."],
    "price_range": {"min": null, "max": null}
  }
}
```

اگر `query_type=unclear`:
```json
{
  "success": true,
  "query_type": "unclear",
  "searchable": false,
  "clarification": {
    "needed": true,
    "question": "...",
    "suggestions": [{"id": 1, "product": "...", "emoji": "🛒"}]
  }
}
```

### ابزارهای تکمیلی
- `classify_query(query)`
- `get_interpreter_info()`

## ۴) قرارداد Search MCP (`:5002`)

### ابزار `search_products(search_params, session_id, use_cache, use_semantic)`
نمونه پاسخ:
```json
{
  "success": true,
  "query": "...",
  "total_hits": 50,
  "results": [
    {
      "id": "...",
      "product_name": "...",
      "brand_name": "...",
      "category_name": "...",
      "price": 0,
      "discount_price": 0,
      "has_discount": false,
      "discount_percentage": 0,
      "image_url": null,
      "product_url": "",
      "score": 0,
      "relevancy_score": 0,
      "value_score": 0
    }
  ],
  "took_ms": 0,
  "from_cache": false,
  "latency_breakdown_ms": {}
}
```

### ابزارهای تکمیلی
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()`

## ۵) قرارداد Embedding MCP (`:5003`)
- `generate_embedding(text, normalize=true, use_cache=true)`
- `generate_embeddings_batch(texts, normalize=true, use_cache=true)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()`
- `clear_embedding_cache()`
- `get_model_info()`

## ۶) قرارداد ابزار داخلی ایجنت
ابزارهای `ShoppingAgent`:
- `search_and_deliver(query)` -> خروجی متن با prefixهای:
  - `🔍 SEARCH_RESULTS:`
  - `✅ CACHED_RESPONSE:`
  - `❓ NEED_CLARIFICATION:`
  - `❌ NO_RESULTS:`
- `get_product_details(product_id)`

در `AgentService` این prefixها حذف و پاسخ API نرمال‌سازی می‌شود.

## ۷) الگوی خطا
- خطای runtime در backend:
  - `success=false`
  - پیام فارسی امن
  - `metadata.error_stage`, `metadata.error_type`
- خطای ابزار MCP:
  - `{"success": false, "error": "..."}`
- خطاهای transport/session در MCP:
  - با retry هندل می‌شوند
