# API و قراردادها (فارسی)

## ۱) API بک‌اند
آدرس پایه پیش‌فرض: `http://<host>:8080`

### ۱.۱ `POST /api/chat`

درخواست (`ChatRequest`):
```json
{
  "message": "گوشی سامسونگ زیر 20 میلیون میخوام",
  "session_id": "optional-uuid"
}
```

اعتبارسنجی:
- `message`: اجباری، طول `1..1000`
- `session_id`: اختیاری

پاسخ (`ChatResponse`):
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
      "agent_cache_lookup_ms": 1,
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

مقادیر قابل مشاهده `query_type`:
- `direct`
- `unclear`
- `chat`
- `no_results`
- `timeout`
- `error`
- `unknown`

### ۱.۲ `GET /api/health`
وضعیت سلامت این بخش‌ها:
- `agent`
- `interpret_server`
- `search_server`

نکته:
- چون endpoint اصلی سرویس MCP برابر `/mcp` است، پاسخ `404` برای `/health` به‌عنوان reachable در نظر گرفته می‌شود.

### ۱.۳ `GET /api/`
endpoint ساده اطلاعات سرویس.

## ۲) قراردادهای داخلی ابزار ایجنت

### ۲.۱ ابزار `search_and_deliver(query)`
خروجی‌های prefixed:
- `🔍 SEARCH_RESULTS:<text-with-json-products>`
- `✅ CACHED_RESPONSE:<formatted-text>`
- `❓ NEED_CLARIFICATION:<question+suggestions>`
- `❌ NO_RESULTS:<message>`

### ۲.۲ ابزار `get_product_details(product_id)`
متن JSON جزئیات محصول از Search MCP برمی‌گرداند.

## ۳) قرارداد ترنسپورت MCP
همه سرویس‌های MCP از JSON-RPC روی این endpoint استفاده می‌کنند:
- `POST /mcp`

کلاینت این موارد را هندل می‌کند:
- `initialize`
- `tools/call`
- session stateful/stateless
- parsing پاسخ JSON و SSE

پیاده‌سازی:
- `src/mcp_client.py`

## ۴) قرارداد Interpret MCP (`:5004`)

### ۴.۱ `interpret_query(query, session_id, context)`
نمونه حالت direct:
```json
{
  "success": true,
  "query_type": "direct",
  "searchable": true,
  "search_params": {
    "intent": "browse",
    "product": "شورت مردانه",
    "brand": null,
    "persian_full_query": "شورت مردانه میخوام",
    "categories_fa": ["مد و پوشاک"],
    "price_range": {"min": null, "max": null}
  },
  "session_update": {
    "last_query": "شورت مردانه میخوام",
    "last_product": "شورت مردانه"
  }
}
```

نمونه حالت unclear:
```json
{
  "success": true,
  "query_type": "unclear",
  "searchable": false,
  "clarification": {
    "needed": true,
    "question": "لطفاً دقیق‌تر بگید دنبال چه محصولی هستید؟",
    "suggestions": [
      {"id": 1, "product": "گوشی موبایل", "emoji": "🛒"}
    ]
  }
}
```

### ۴.۲ ابزارهای دیگر
- `classify_query(query)`
- `get_interpreter_info()`

## ۵) قرارداد Search MCP (`:5002`)

### ۵.۱ `search_products(search_params, session_id, use_cache, use_semantic)`
نمونه:
```json
{
  "success": true,
  "query": "شورت مردانه",
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

### ۵.۲ ابزارهای دیگر
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()`

## ۶) قرارداد Embedding MCP (`:5003`)
- `generate_embedding(text, normalize=true, use_cache=true)`
- `generate_embeddings_batch(texts, normalize=true, use_cache=true)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()`
- `clear_embedding_cache()`
- `get_model_info()`

## ۷) قرارداد خطا

خطای سطح backend:
```json
{
  "success": false,
  "response": "متأسفانه مشکلی پیش اومد. لطفاً دوباره تلاش کنید.",
  "products": [],
  "metadata": {
    "query_type": "error",
    "error_stage": "agent.chat",
    "error_type": "RuntimeError"
  }
}
```

خطای سطح MCP:
```json
{
  "success": false,
  "error": "..."
}
```

## ۸) منبع حقیقت قراردادها
- Schemaها: `backend/api/schemas.py`
- رفتار endpoint: `backend/api/routes.py`
- رفتار سرویس: `backend/services/agent_service.py`
