# Pipelines (English)

This document describes execution pipelines exactly as implemented in code.

## 1. API Chat Pipeline

```mermaid
sequenceDiagram
  participant FE as Frontend
  participant API as /api/chat
  participant AS as AgentService
  participant AG as ShoppingAgent

  FE->>API: ChatRequest(message, session_id?)
  API->>AS: chat(...)
  AS->>AS: initialize + optional L2 cache lookup
  alt L2 hit
    AS-->>API: cached payload
  else L2 miss
    AS->>AG: chat(message, session_id)
    AG-->>AS: response text
    AS->>AS: extract_products + clean_response + detect_query_type
    AS->>AS: optional L2 cache set
    AS-->>API: ChatResponse payload
  end
  API-->>FE: JSON
```

Main implementation:
- `backend/api/routes.py`
- `backend/services/agent_service.py`

## 2. Agent Turn Pipeline (`ShoppingAgent.chat`)

```mermaid
flowchart TD
  A[Input message + session_id] --> B[trace_query + LangGraph config]
  B --> C[Reset per-turn trackers]
  C --> D[agent.ainvoke]
  D --> E[Extract last AI/Tool message text]
  E --> F{L3 cache hit?}
  F -->|yes| G[Use cached final text]
  F -->|no| H[Use generated text]
  G --> I[Strip tool prefixes]
  H --> I
  I --> J{Need L3 store?}
  J -->|yes| K[Store final text in Redis]
  J -->|no| L[Skip]
  K --> M[Return text + session]
  L --> M
```

Error/fallback branches:
- Tool-use endpoint unavailable (`404` tool-use) -> fallback agent on Groq if configured.
- Invalid tool call history -> retry with new `session_id`.
- Other failures -> return `__AGENT_ERROR__:{...}` envelope.

Main implementation:
- `src/agent.py`

## 3. Tool Pipeline: `search_and_deliver`

```mermaid
flowchart TD
  T0[search_and_deliver(query)] --> T1[Loop guard]
  T1 --> T2[interpret_query via MCP]
  T2 --> T3{direct and searchable?}
  T3 -->|no| T4[Build NEED_CLARIFICATION response]
  T3 -->|yes| T5[Build final_search_params]
  T5 --> T6[L3 cache lookup by search params]
  T6 -->|hit| T7[Return CACHED_RESPONSE prefix]
  T6 -->|miss| T8[search_products via MCP]
  T8 --> T9{results empty?}
  T9 -->|yes| T10[Return NEED_CLARIFICATION with alternatives]
  T9 -->|no| T11[Format product JSON block]
  T11 --> T12[Return SEARCH_RESULTS prefix]
```

Notes:
- This tool is declared `@tool(return_direct=True)`.
- Prefixes are later cleaned by `AgentService`.

Main implementation:
- `src/agent.py`

## 4. Interpret Pipeline

```mermaid
flowchart TD
  I0[query] --> I1[Persian normalization]
  I1 --> I2[LLM classify+extract]
  I2 --> I3[repair/validation]
  I3 --> I4{query_type == direct?}
  I4 -->|yes| I5[category embedding match]
  I5 --> I6[return direct + search_params]
  I4 -->|no| I7[return unclear + clarification]
```

Current hard behavior:
- Any non-direct signal is coerced to `unclear`.
- Classification contract keys:
  - `query_type`, `product`, `brand`, `price_range`, `intent`, `confidence`

Main implementation:
- `src/mcp_servers/interpret_server.py`

## 5. Search Pipeline

```mermaid
flowchart TD
  S0[search_params] --> S1[sanitize categories]
  S1 --> S2[negative cache check]
  S2 --> S3[search cache lookup + lock]
  S3 -->|hit| S4[return cached results]
  S3 -->|miss| S5[generate DSL]
  S5 --> S6[execute ES query]
  S6 --> S7{zero hits and categories?}
  S7 -->|yes| S8[retry once without categories]
  S7 -->|no| S9[continue]
  S8 --> S9
  S9 --> S10{still zero hits?}
  S10 -->|yes| S11[negative cache set + return empty]
  S10 -->|no| S12[rerank]
  S12 --> S13[search cache set]
  S13 --> S14[return top results]
```

DSL generation strategy:
1. Mixtral via OpenRouter (`MIXTRAL_MODEL`)
2. Rule-based fallback DSL
3. Optional KNN append/injection using embedding MCP

Main implementation:
- `src/mcp_servers/search_server.py`

## 6. Latency Instrumentation Pipeline

Every major stage writes `LATENCY_SUMMARY` for analysis.

```mermaid
flowchart LR
  A1[agent_service.chat] --> L[Pipeline logs]
  A2[agent.chat] --> L
  A3[agent.tool.search_and_deliver] --> L
  A4[mcp_client.initialize/call_tool] --> L
  A5[interpret.pipeline] --> L
  A6[search.pipeline] --> L
```

Tools:
- `grep -h "LATENCY_SUMMARY" logs/pipeline-*.log`
- `python3 scripts/analyze_latency_logs.py --log-dir logs --top 30`

## 7. Common Bottleneck Positions
- `mcp_client.initialize`: first call per service/session mode.
- `interpret.pipeline.llm_classification_ms`: interpret LLM latency.
- `mcp_client.call_tool.http_request_ms`: transport/server overhead.
- `search.pipeline.es_search_ms`: Elasticsearch latency.
- `search.pipeline.rerank_ms`: heavy result lists + scoring.

## 8. Accuracy-Sensitive Pipeline Points
- Agent mode selection in system prompt.
- Interpret direct/unclear classification quality.
- Category match quality in interpret (`_match_categories`).
- DSL category filter quality and pruning in search.
- Rerank scoring weights and intent sort behavior.
