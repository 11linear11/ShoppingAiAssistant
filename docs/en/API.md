# API and Contracts (English)

## 1. HTTP API (Backend)
Base URL (default): `http://<host>:8080`

### 1.1 POST `/api/chat`
Request body (`ChatRequest`):
```json
{
  "message": "گوشی سامسونگ زیر 20 میلیون میخوام",
  "session_id": "optional-uuid"
}
```

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

#### Metadata Notes
- `query_type`: derived in `AgentService` from output shape (`direct`, `unclear`, `chat`, `no_results`, `error`, ...)
- `from_agent_cache`: L2 cache hit marker
- `latency_breakdown_ms`: per-stage timing map

### 1.2 GET `/api/health`
Returns aggregated health for:
- `agent`
- `interpret_server`
- `search_server`

Notes:
- MCP services often return `404` for `/health` because main endpoint is `/mcp`; backend treats reachable 404 as alive.

### 1.3 GET `/api/`
Simple service info endpoint.

## 2. MCP Protocol Surface
All MCP services expose streamable HTTP endpoint:
- `POST /mcp` (JSON-RPC `initialize` + `tools/call`)

Client implementation: `src/mcp_client.py`.

## 3. Interpret MCP Contract (`:5004`)

### Tool: `interpret_query(query, session_id, context)`
Primary output contract:
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

If `query_type=unclear`:
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

### Additional Tools
- `classify_query(query)` (diagnostic/quick classification)
- `get_interpreter_info()` (diagnostic)

## 4. Search MCP Contract (`:5002`)

### Tool: `search_products(search_params, session_id, use_cache, use_semantic)`
Returns:
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

### Additional Tools
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()` (diagnostic)

## 5. Embedding MCP Contract (`:5003`)
- `generate_embedding(text, normalize=true, use_cache=true)`
- `generate_embeddings_batch(texts, normalize=true, use_cache=true)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()` (diagnostic)
- `clear_embedding_cache()` (diagnostic)
- `get_model_info()` (diagnostic)

## 6. Agent Tool Contract (Internal)
`ShoppingAgent` tools:
- `search_and_deliver(query)` -> return-direct text with prefixes:
  - `🔍 SEARCH_RESULTS:`
  - `✅ CACHED_RESPONSE:`
  - `❓ NEED_CLARIFICATION:`
  - `❌ NO_RESULTS:`
- `get_product_details(product_id)`

`AgentService` strips prefixes and returns normalized API payload.

## 7. Error Conventions
- Backend runtime errors:
  - `success=false`
  - safe Persian fallback `response`
  - `metadata.error_stage`, `metadata.error_type`
- MCP tool-level failures:
  - `{"success": false, "error": "..."}`
- MCP transport/session failures:
  - retried by `MCPClient`
