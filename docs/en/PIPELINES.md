# Pipelines (English)

## 1. End-to-End Chat Pipeline

```mermaid
sequenceDiagram
  participant U as User
  participant FE as Frontend
  participant BE as Backend /api/chat
  participant AS as AgentService
  participant AG as ShoppingAgent
  participant IC as MCP Client
  participant IN as Interpret Server
  participant SE as Search Server
  participant EL as Elasticsearch
  participant RD as Redis

  U->>FE: send message
  FE->>BE: POST /api/chat
  BE->>AS: chat(message, session)
  AS->>RD: L2 agent cache lookup
  alt L2 hit
    AS-->>BE: cached response
  else L2 miss
    AS->>AG: agent.chat
    AG->>IC: interpret_query
    IC->>IN: MCP tools/call
    IN-->>IC: query_type + search_params
    IC-->>AG: interpret result

    alt searchable=true
      AG->>IC: search_products(params)
      IC->>SE: MCP tools/call
      SE->>RD: L1 search cache lookup
      alt L1 miss
        SE->>EL: DSL query
        EL-->>SE: raw hits
        SE->>SE: rerank
        SE->>RD: store search cache
      end
      SE-->>IC: search results
      IC-->>AG: tool result
    end

    AG->>RD: L3 llm-response cache lookup/store
    AG-->>AS: final response text
    AS->>AS: extract products + clean text
    AS->>RD: optional L2 store
    AS-->>BE: structured response
  end
  BE-->>FE: JSON response
```

## 2. Interpret Pipeline
1. Normalize Persian input.
2. Classify and extract via LLM.
3. Build response contract:
   - `searchable`
   - `query_type`
   - `search_params` (product, brand, intent, price_range, categories)
4. For non-searchable requests, return clarification suggestions.

## 3. Search Pipeline
1. Receive structured params from agent.
2. Build cache keys.
3. Check negative cache.
4. Check search cache.
5. If miss:
   - generate DSL
   - execute Elasticsearch query
   - rerank with intent-aware logic
   - set caches
6. Return canonical result payload.

## 4. Embedding Pipeline
1. Receive text or list of texts.
2. Normalize/prepend E5 query prefix when needed.
3. Compute vectors with sentence-transformers.
4. Optional in-memory cache hit path.
5. Return embedding + metadata.

## 5. Logging Pipeline
Debug mode (`DEBUG_LOG=true`):
- full per-stage logs
- start/end stage timings
- query summaries

Non-debug mode (`DEBUG_LOG=false`):
- only `USER_REQUEST` events
- all errors

```mermaid
flowchart LR
  A[log_pipeline call] --> B{DEBUG_LOG?}
  B -->|true| C[write stage log]
  B -->|false| D{Error or USER_REQUEST?}
  D -->|yes| E[write minimal log]
  D -->|no| F[drop event]
```

## 6. Frontend Rendering Pipeline
1. Receive backend response (`response`, `products`).
2. Normalize products.
3. If `products` empty, parse JSON-like payload from `response`.
4. Strip JSON code block from displayed message.
5. Render:
   - single product card-like row
   - or multi-row product table

## 7. Performance Controls
- Rotating pipeline log files (`PIPELINE_LOG_MAX_BYTES`, `PIPELINE_LOG_BACKUP_COUNT`)
- Docker log rotation in compose
- Debug/off modes for cache bypass and log granularity
