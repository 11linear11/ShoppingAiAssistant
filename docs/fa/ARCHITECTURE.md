# معماری سیستم (فارسی)

## ۱) دامنه و سبک معماری
این پروژه یک معماری **Agent-First با Tool-Calling** دارد.

ویژگی‌های اصلی:
- یک درگاه HTTP واحد با FastAPI
- تصمیم‌گیری هر نوبت مکالمه توسط مدل ایجنت
- اجرای ابزارها توسط سرویس‌های MCP روی `/mcp`
- جستجوی محصول روی Elasticsearch با چند لایه کش
- ثبت لاگ ساختاریافته برای تحلیل تاخیر

## ۲) توپولوژی کلان
```mermaid
flowchart LR
  U[کاربر] --> FE[فرانت‌اند]
  FE -->|POST /api/chat| BE[بک‌اند FastAPI]
  BE --> AS[AgentService]
  AS --> AG[ShoppingAgent ReAct]

  AG -->|search_and_deliver| INTP[Interpret MCP :5004]
  AG -->|search_products/get_product| SRCH[Search MCP :5002]
  SRCH --> EMB[Embedding MCP :5003]

  AS <-->|کش L2| REDIS[(Redis)]
  AG <-->|کش L3| REDIS
  INTP <-->|کش embedding| REDIS
  SRCH <-->|کش جستجو/منفی/DSL| REDIS
  SRCH --> ES[(Elasticsearch)]
```

## ۳) مسئولیت اجزا

| جزء | فایل‌های اصلی | مسئولیت |
|---|---|---|
| API Gateway | `backend/main.py`, `backend/api/routes.py` | endpointها، lifecycle، health، CORS |
| Service Layer | `backend/services/agent_service.py` | init ایجنت، timeout، کش L2، نرمال‌سازی پاسخ |
| Agent | `src/agent.py` | تصمیم‌گیری prompt-driven، tool-calling، حافظه، کش L3 |
| MCP Client | `src/mcp_client.py` | initialize/retry/session + parse JSON/SSE |
| Interpret MCP | `src/mcp_servers/interpret_server.py` | تشخیص `direct/unclear` + استخراج پارامتر |
| Search MCP | `src/mcp_servers/search_server.py` | تولید DSL، جستجو ES، rerank، گارد category، کش |
| Embedding MCP | `src/mcp_servers/embedding_server.py` | embedding/similarity + کش درون‌پردازه |
| Telemetry | `src/pipeline_logger.py` | trace id، لاگ مرحله‌ای، `LATENCY_SUMMARY` |

## ۴) جریان کنترل End-to-End
```mermaid
sequenceDiagram
  participant U as کاربر
  participant FE as فرانت
  participant BE as بک‌اند
  participant AS as AgentService
  participant AG as ShoppingAgent
  participant IN as Interpret MCP
  participant SE as Search MCP
  participant RD as Redis
  participant ES as Elasticsearch

  U->>FE: پیام
  FE->>BE: POST /api/chat
  BE->>AS: chat(message, session_id)
  AS->>RD: بررسی کش L2

  alt L2 hit
    AS-->>BE: پاسخ کش‌شده
    BE-->>FE: ChatResponse
  else L2 miss
    AS->>AG: agent.chat()
    AG->>AG: تصمیم مدل برای mode
    alt SEARCH
      AG->>IN: interpret_query
      alt direct
        AG->>SE: search_products
        SE->>RD: بررسی کش‌ها
        alt miss
          SE->>ES: اجرای DSL
          ES-->>SE: hits
          SE->>SE: rerank
          SE->>RD: cache set
        end
        SE-->>AG: نتایج رتبه‌بندی‌شده
      else unclear
        IN-->>AG: پاسخ clarification
      end
    else CHAT/CLARIFY/DETAILS
      AG-->>AS: پاسخ متنی
    end
    AS->>AS: extract products + clean text + detect type
    AS->>RD: ذخیره احتمالی L2
    AS-->>BE: پاسخ ساختاریافته
    BE-->>FE: JSON
  end
```

## ۵) مدل تصمیم ایجنت
ایجنت دو ابزار دارد:
- `search_and_deliver(query)`
- `get_product_details(product_id)`

modeهای پرامپت:
- `CHAT`
- `CLARIFY`
- `SEARCH`
- `DETAILS`

اما خروجی نهایی Interpret فعلاً فقط:
- `direct`
- `unclear`

نتیجه:
- حتی اگر ایجنت مسیر SEARCH را انتخاب کند، Interpret می‌تواند آن را `unclear` برگرداند.

## ۶) معماری کش
```mermaid
flowchart TD
  Q[پیام ورودی] --> L2{کش L2 Agent?}
  L2 -->|hit| R1[بازگشت پاسخ کامل]
  L2 -->|miss| AG[Agent + tools]

  AG --> L3{کش L3 متن نهایی?}
  L3 -->|hit| R2[بازگشت متن کش‌شده]
  L3 -->|miss| S[Search pipeline]

  S --> L1{کش L1 جستجو?}
  L1 -->|hit| R3[بازگشت نتایج کش‌شده]
  L1 -->|miss| ES[ES + rerank]
```

namespaceهای مهم:
- `cache:v1:agent:*`
- `cache:v1:llm_response:*`
- `cache:v2:search:*`
- `cache:v2:negative:*`
- `cache:v1:dsl:*`
- `cache:v1:embedding:*`

## ۷) کنترل کیفیت نتایج در Search
در `search_server`:
- نرمال‌سازی categoryها با لیست معتبر
- حذف فیلتر category نامعتبر از DSL در کد
- یک retry بدون category در حالت صفر نتیجه
- rerank با ترکیب score/relevancy/price/brand/discount

## ۸) سشن و حافظه
- حافظه مکالمه با `MemorySaver` در LangGraph
- `session_id` API به `thread_id` نگاشت می‌شود
- اگر history ابزار خراب شود، ایجنت با سشن جدید retry می‌کند

## ۹) پایداری و fallback
- `MCPClient` برای init/transport/session retry دارد
- در OpenRouter، fallback به Groq قابل فعال‌سازی است
- Backend خطا را به پاسخ امن با `success=false` تبدیل می‌کند

## ۱۰) مدل لاگ و مانیتورینگ
`src/pipeline_logger.py` برای هر trace لاگ ساختاریافته می‌نویسد.

کامپوننت‌های کلیدی latency:
- `agent_service.chat`
- `agent.chat`
- `agent.tool.search_and_deliver`
- `mcp_client.initialize`
- `mcp_client.call_tool`
- `interpret.pipeline`
- `search.pipeline`

## ۱۱) محدودیت‌های شناخته‌شده (نسخه فعلی)
- پاسخ نهایی هنوز بعد از tool output از مسیر LLM عبور می‌کند.
- Interpret فقط `direct/unclear` را پشتیبانی می‌کند.
- نام‌گذاری intent در بخش‌هایی از Search هنوز با خروجی `find_best` کاملاً یکسان نیست (برخی شاخه‌ها legacy هستند).
