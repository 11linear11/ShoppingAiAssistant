# برنامه استقرار تغییرات (Agent-first In-Model Routing)

## برنچ اجرا
- `feature/deterministic-router-v1`

## هدف
- حذف روتینگ دترمینیستیک چندمرحله‌ای از مسیر اصلی
- تبدیل Agent به تصمیم‌گیر اصلی در همان کال مدل
- جلوگیری از search روی کوئری‌های مبهم/انتزاعی
- نگه‌داشتن fallback ایمن در `interpret` برای خطای route
- حفظ کیفیت خروجی نهایی Agent روی نتایج search

## معماری فعال فعلی
1. پیام کاربر وارد `agent_service.chat` می‌شود.
2. Agent (LangGraph/ReAct) با `SYSTEM_PROMPT` تصمیم می‌گیرد:
   - `chat/abstract/follow_up/unclear`: پاسخ مکالمه‌ای و یک سوال شفاف‌ساز، بدون ابزار
   - `direct`: ابتدا `interpret_query` و در صورت `searchable=true` سپس `search_products`
3. خروجی نهایی همیشه توسط Agent به فارسی ساخته می‌شود.

## Guard ایمنی در Interpret
فایل: `src/mcp_servers/interpret_server.py`

- در حالت `direct_unclear_only=true` اگر درخواست واقعا مبهم باشد:
  - خروجی `direct` به `unclear` coercion می‌شود.
- هدف: اگر Agent اشتباهی کوئری مبهم را direct فرض کرد، search اشتباه اجرا نشود.

## کش Category Matching
فایل: `src/mcp_servers/interpret_server.py`

- کش in-memory برای category embedding matching اضافه شده است.
- کلید کش versioned است و شامل:
  - embedding model
  - hash منبع embeddings
  - threshold
  - normalized product

## تنظیمات کلیدی `.env`
- `DETERMINISTIC_ROUTER_ENABLED=false`
- `CATEGORY_MATCH_CACHE_ENABLED=true`
- `CATEGORY_MATCH_CACHE_TTL=86400`
- `CATEGORY_MATCH_CACHE_MAX_ENTRIES=5000`
- `CATEGORY_MATCH_THRESHOLD=0.75`

## لاگ و مشاهده‌پذیری
- در `agent_service.chat` متادیتای زیر ثبت می‌شود:
  - `orchestrator=agent_react_v2`
  - `routing=in_model`
- رویدادهای `LATENCY_SUMMARY` برای بخش‌های `agent.chat`, `agent.tool.interpret_query`, `agent.tool.search_products`, `interpret.pipeline`, `search.pipeline` فعال هستند.

## وضعیت پیاده‌سازی
- مسیر اصلی روی Agent-first فعال است.
- مسیر deterministic قبلی در کد باقی مانده اما در flow فعال `chat()` استفاده نمی‌شود.

## معیار ارزیابی
- دقت:
  - کوئری‌های مبهم نباید مستقیم search شوند.
  - کوئری‌های direct باید به interpret/search برسند.
- latency:
  - p95 مسیر direct تحت تاثیر `interpret + search + generation` اندازه‌گیری شود.

## Rollback
- checkout به `main` و recreate سرویس‌های پروژه
- یا غیرفعال کردن featureهای جدید از env بر اساس نیاز
