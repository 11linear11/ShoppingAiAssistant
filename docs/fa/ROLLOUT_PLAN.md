# برنامه استقرار تغییرات (Deterministic Router v1)

## برنچ اجرا
- `feature/deterministic-router-v1`

## هدف
- حذف وابستگی مسیر `direct` به تصمیم آزاد tool-calling مدل
- انتقال حالت‌های `abstract` و `follow_up` به مکالمه مستقیم Agent
- نگه‌داشتن `interpret` به عنوان دروازه retrieval برای `direct/unclear`
- اضافه‌کردن کش برای category matching در سرویس `interpret`

## تغییرات پیاده‌سازی‌شده

### 1) Orchestrator قطعی در Backend
فایل: `backend/services/agent_service.py`

- مسیر جدید با Feature Flag:
  - `DETERMINISTIC_ROUTER_ENABLED=true`
- فلو جدید:
  1. Agent ابتدا Route را تعیین می‌کند: `direct|abstract|follow_up|chat|unclear`
  2. اگر Route = `abstract` یا `follow_up` یا `chat`:
     - Agent در حالت `chat_without_tools` پاسخ مکالمه‌ای می‌دهد.
  3. اگر Route = `direct` یا `unclear`:
     - `interpret_query` با context `direct_unclear_only=true` صدا زده می‌شود.
     - اگر خروجی `interpret` برابر `direct + searchable=true` بود:
       - `search_products` به صورت **اجباری** اجرا می‌شود.
     - در غیر این صورت، مسیر fallback به clarification برمی‌گردد.

- نتیجه:
  - سناریوی «`interpret` صدا زده شد ولی `search` نه» برای مسیر `direct` عملا حذف می‌شود.

### 2) قابلیت‌های جدید Agent برای Router/Conversation
فایل: `src/agent.py`

- متد جدید `classify_route(message)`:
  - خروجی استاندارد Route برای orchestrator
  - شامل fast-check اولیه برای greeting / follow-up عددی
- متد جدید `chat_without_tools(...)`:
  - مکالمه Agent بدون tool-calling برای `abstract/follow_up/chat`

### 3) Coerce Mode در Interpret
فایل: `src/mcp_servers/interpret_server.py`

- وقتی context شامل `direct_unclear_only=true` باشد:
  - اگر `abstract` یا `follow_up` تشخیص داده شود، به `unclear` تبدیل می‌شود.
- این رفتار fallback را برای misroute تضمین می‌کند.

### 4) کش Category Matching
فایل: `src/mcp_servers/interpret_server.py`

- کش in-memory با کلید versioned:
  - شامل `embedding_model` + hash فایل embeddings + threshold + product normalized
- پشتیبانی از:
  - TTL
  - حداکثر ظرفیت
  - eviction ساده oldest-first
- هدف:
  - حذف embedding/category matching تکراری برای productهای پرتکرار

## تنظیمات جدید `.env`

- `DETERMINISTIC_ROUTER_ENABLED=false`
- `DETERMINISTIC_FORCE_SEARCH=true`
- `CATEGORY_MATCH_THRESHOLD=0.75`
- `CATEGORY_MATCH_CACHE_ENABLED=true`
- `CATEGORY_MATCH_CACHE_TTL=86400`
- `CATEGORY_MATCH_CACHE_MAX_ENTRIES=5000`

## فایل‌های تغییر یافته
- `src/agent.py`
- `backend/services/agent_service.py`
- `src/mcp_servers/interpret_server.py`
- `backend/core/config.py`
- `.env.example`

## روش فعال‌سازی روی سرور
1. checkout برنچ:
   - `git fetch origin`
   - `git checkout feature/deterministic-router-v1`
   - `git pull origin feature/deterministic-router-v1`
2. اعمال env:
   - `DETERMINISTIC_ROUTER_ENABLED=true`
   - `DETERMINISTIC_FORCE_SEARCH=true`
   - `CATEGORY_MATCH_CACHE_ENABLED=true`
3. ری‌استارت سرویس‌ها:
   - `docker compose build backend interpret`
   - `docker compose up -d --force-recreate --no-deps backend interpret`

## معیارهای ارزیابی بعد از استقرار
- نرخ skip برای direct:
  - `searchable=true` و `search_products called=false` باید نزدیک صفر شود.
- p95 latency:
  - مسیر direct عمدتا تابع latency `interpret + search + render` باشد.
- صحت رفتاری:
  - `abstract/follow_up` به جای search مستقیم، مکالمه شفاف‌سازی بدهند.

## Rollback
- خاموش کردن feature:
  - `DETERMINISTIC_ROUTER_ENABLED=false`
- یا برگشت برنچ سرویس به `main`.
