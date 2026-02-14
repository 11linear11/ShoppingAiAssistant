# API and Contracts (English)

## 1. Backend HTTP API
Default base URL: `http://<host>:8080`

### 1.1 `POST /api/chat`

Request body (`ChatRequest`):
```json
{
  "message": "گوشی سامسونگ زیر 20 میلیون میخوام",
  "session_id": "optional-uuid"
}
```

Validation:
- `message`: required, length `1..1000`
- `session_id`: optional

Response body (`ChatResponse`):
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

`query_type` values observed in service layer:
- `direct`
- `unclear`
- `chat`
- `no_results`
- `timeout`
- `error`
- `unknown`

### 1.2 `GET /api/health`
Returns aggregated health for:
- `agent`
- `interpret_server`
- `search_server`

Behavior note:
- MCP services are tested with `GET <service>/health`.
- `404` is treated as reachable/healthy because MCP endpoint is `/mcp`.

### 1.3 `GET /api/`
Simple service info endpoint.

## 2. Internal Agent Contracts

### 2.1 Tool: `search_and_deliver(query)`
Possible prefixed outputs:
- `🔍 SEARCH_RESULTS:<text-with-json-products>`
- `✅ CACHED_RESPONSE:<formatted-text>`
- `❓ NEED_CLARIFICATION:<question+suggestions>`
- `❌ NO_RESULTS:<message>`

### 2.2 Tool: `get_product_details(product_id)`
Returns product details JSON text from Search MCP.

## 3. MCP Transport Contract
All MCP services are called using JSON-RPC over:
- `POST /mcp`

Client handles:
- `initialize`
- `tools/call`
- stateful/stateless sessions
- SSE and JSON response parsing

Implementation:
- `src/mcp_client.py`

## 4. Interpret MCP Contracts (`:5004`)

### 4.1 `interpret_query(query, session_id, context)`
Direct example:
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

Unclear example:
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

### 4.2 Other tools
- `classify_query(query)`
- `get_interpreter_info()`

## 5. Search MCP Contracts (`:5002`)

### 5.1 `search_products(search_params, session_id, use_cache, use_semantic)`
Example:
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

### 5.2 Other tools
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()`

## 6. Embedding MCP Contracts (`:5003`)
- `generate_embedding(text, normalize=true, use_cache=true)`
- `generate_embeddings_batch(texts, normalize=true, use_cache=true)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()`
- `clear_embedding_cache()`
- `get_model_info()`

## 7. Error Contract Conventions

Backend-level hard failures:
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

MCP-level failures:
```json
{
  "success": false,
  "error": "..."
}
```

## 8. Source of Truth
- API schemas: `backend/api/schemas.py`
- endpoint behavior: `backend/api/routes.py`
- service behavior: `backend/services/agent_service.py`
