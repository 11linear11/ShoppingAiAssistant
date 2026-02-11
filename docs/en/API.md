# API and MCP Contracts (English)

## 1. HTTP API (Backend)
Base: `http://<host>:8080`

### 1.1 POST `/api/chat`
Request:
```json
{
  "message": "دوغ میخوام",
  "session_id": "optional-uuid"
}
```

Response shape:
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

### 1.2 GET `/api/health`
Returns backend + dependency health summary.

## 2. MCP Tools

### 2.1 Interpret Server (`:5004`)
- `interpret_query(query, session_id, context)`
- `classify_query(query)`
- `get_interpreter_info()`

### 2.2 Search Server (`:5002`)
- `search_products(search_params, session_id, use_cache, use_semantic)`
- `generate_dsl(search_params)`
- `get_product(product_id)`
- `rerank_results(results, preferences, intent)`
- `get_search_info()`

### 2.3 Embedding Server (`:5003`)
- `generate_embedding(text, normalize, use_cache)`
- `generate_embeddings_batch(texts, normalize, use_cache)`
- `calculate_similarity(text1, text2)`
- `get_embedding_cache_stats()`
- `clear_embedding_cache()`
- `get_model_info()`

## 3. Agent Tool Contracts
Agent expects interpret output with:
- `query_type`
- `searchable`
- `search_params.intent`
- `search_params.product`

If `searchable=false`, agent should avoid `search_products` and ask clarification.

## 4. Error Conventions
- API returns safe Persian fallback responses on runtime exceptions.
- MCP layer returns `{"success": false, "error": "..."}` on tool-level failures.
- Transport failures and session expiration are retried in MCP client.
