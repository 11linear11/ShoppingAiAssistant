# پایپلاین‌ها (فارسی)

این سند مسیرهای اجرایی واقعی سیستم را بر اساس کد فعلی توضیح می‌دهد.

## ۱) پایپلاین API چت

```mermaid
sequenceDiagram
  participant FE as فرانت
  participant API as /api/chat
  participant AS as AgentService
  participant AG as ShoppingAgent

  FE->>API: ChatRequest
  API->>AS: chat(...)
  AS->>AS: initialize + بررسی کش L2
  alt L2 hit
    AS-->>API: payload کش‌شده
  else L2 miss
    AS->>AG: agent.chat()
    AG-->>AS: پاسخ متنی
    AS->>AS: extract_products + clean_response + detect_query_type
    AS->>AS: ذخیره احتمالی در L2
    AS-->>API: ChatResponse
  end
  API-->>FE: JSON
```

فایل‌های اصلی:
- `backend/api/routes.py`
- `backend/services/agent_service.py`

## ۲) پایپلاین نوبت ایجنت (`ShoppingAgent.chat`)

```mermaid
flowchart TD
  A[message + session_id] --> B[trace_query + config]
  B --> C[reset trackers]
  C --> D[agent.ainvoke]
  D --> E[extract final text]
  E --> F{L3 hit?}
  F -->|yes| G[use cached final text]
  F -->|no| H[use generated text]
  G --> I[strip tool prefixes]
  H --> I
  I --> J{store in L3?}
  J -->|yes| K[SET in Redis]
  J -->|no| L[skip]
  K --> M[return]
  L --> M
```

شاخه‌های خطا:
- خطای نبود endpoint tool-use در مدل اصلی -> fallback به Groq (در صورت فعال بودن)
- خطای mismatch history ابزار -> retry با session جدید
- سایر خطاها -> `__AGENT_ERROR__:{...}`

فایل اصلی:
- `src/agent.py`

## ۳) پایپلاین ابزار `search_and_deliver`

```mermaid
flowchart TD
  T0[search_and_deliver(query)] --> T1[loop guard]
  T1 --> T2[interpret_query]
  T2 --> T3{direct + searchable?}
  T3 -->|خیر| T4[بازگشت NEED_CLARIFICATION]
  T3 -->|بله| T5[ساخت final_search_params]
  T5 --> T6[بررسی کش L3 بر اساس search params]
  T6 -->|hit| T7[بازگشت CACHED_RESPONSE]
  T6 -->|miss| T8[search_products]
  T8 --> T9{نتیجه خالی؟}
  T9 -->|بله| T10[clarification + پیشنهاد جایگزین]
  T9 -->|خیر| T11[فرمت JSON محصولات]
  T11 --> T12[بازگشت SEARCH_RESULTS]
```

نکته:
- این ابزار `return_direct=True` است.
- prefixها بعداً در `AgentService` پاک‌سازی می‌شوند.

فایل اصلی:
- `src/agent.py`

## ۴) پایپلاین Interpret

```mermaid
flowchart TD
  I0[query] --> I1[normalize Persian]
  I1 --> I2[LLM classify+extract]
  I2 --> I3[repair/validation]
  I3 --> I4{query_type == direct?}
  I4 -->|yes| I5[category embedding match]
  I5 --> I6[return direct + search_params]
  I4 -->|no| I7[return unclear + clarification]
```

رفتار سخت‌گیر فعلی:
- هر خروجی غیر مستقیم به `unclear` coercion می‌شود.
- کلیدهای قرارداد: `query_type`, `product`, `brand`, `price_range`, `intent`, `confidence`

فایل اصلی:
- `src/mcp_servers/interpret_server.py`

## ۵) پایپلاین Search

```mermaid
flowchart TD
  S0[search_params] --> S1[sanitize categories]
  S1 --> S2[negative cache check]
  S2 --> S3[search cache lookup + lock]
  S3 -->|hit| S4[return cached]
  S3 -->|miss| S5[generate DSL]
  S5 --> S6[execute ES]
  S6 --> S7{zero hit + categories?}
  S7 -->|yes| S8[retry without categories]
  S7 -->|no| S9[continue]
  S8 --> S9
  S9 --> S10{still zero?}
  S10 -->|yes| S11[negative cache set + empty]
  S10 -->|no| S12[rerank]
  S12 --> S13[cache set]
  S13 --> S14[return top results]
```

استراتژی DSL:
1. تولید با Mixtral روی OpenRouter
2. fallback rule-based
3. در صورت semantic فعال، افزودن یا تزریق KNN embedding

فایل اصلی:
- `src/mcp_servers/search_server.py`

## ۶) پایپلاین ثبت تاخیر

```mermaid
flowchart LR
  A1[agent_service.chat] --> L[Pipeline logs]
  A2[agent.chat] --> L
  A3[agent.tool.search_and_deliver] --> L
  A4[mcp_client.initialize/call_tool] --> L
  A5[interpret.pipeline] --> L
  A6[search.pipeline] --> L
```

دستورات تحلیل:
- `grep -h "LATENCY_SUMMARY" logs/pipeline-*.log`
- `python3 scripts/analyze_latency_logs.py --log-dir logs --top 30`

## ۷) نقاط رایج bottleneck
- `mcp_client.initialize` (به‌خصوص درخواست اول)
- `interpret.pipeline.llm_classification_ms`
- `mcp_client.call_tool.http_request_ms`
- `search.pipeline.es_search_ms`
- `search.pipeline.rerank_ms`

## ۸) نقاط حساس دقت
- تصمیم mode در پرامپت ایجنت
- کیفیت direct/unclear در interpret
- کیفیت category matching در interpret
- کیفیت DSL و category pruning در search
- وزن‌دهی rerank و sort وابسته به intent
