# عملیات و پیکربندی (فارسی)

## ۱) تنظیم محیط
فایل اصلی: `.env`
نمونه: `.env.example`

متغیرهای کلیدی:
- انتخاب مدل: `AGENT_MODEL_PROVIDER`, `AGENT_MODEL`, `AGENT_SECOND_MODEL`
- کلید/مدل ارائه‌دهنده: `OPEN_ROUTERS_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_SECOND_MODEL`, `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_SECOND_MODEL`
- URL سرویس‌ها: `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL`
- دیتاستورها: `REDIS_*`, `ELASTICSEARCH_*`
- لاگ/مانیتورینگ: `DEBUG_LOG`, `PIPELINE_*`, `USE_LOGFIRE`

## ۲) حالت‌های اجرا
### ۲.۱ اجرای لوکال
```bash
python -m src.mcp_servers.run_servers
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### ۲.۲ اجرای Docker
```bash
docker-compose up --build
```

حالت debug:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## ۳) کنترل حجم لاگ و حافظه
برای جلوگیری از پر شدن سرور:
- rotation لاگ پایپلاین با:
  - `PIPELINE_LOG_MAX_BYTES`
  - `PIPELINE_LOG_BACKUP_COUNT`
- rotation لاگ docker در compose
- در production مقدار `DEBUG_LOG=false`

پیشنهاد production:
- `DEBUG_LOG=false`
- `PIPELINE_LOG_MAX_BYTES=5000000`
- `PIPELINE_LOG_BACKUP_COUNT=3`

## ۴) بازگردانی (Rollback)
از تغییرات پاکسازی و نوسازی docs در `_backup/` نسخه پشتیبان گرفته شده است.
برای بازگردانی فایل‌های حذف‌شده:
```bash
bash scripts/restore_cleanup_backup.sh
```

## ۵) وضعیت تست
تست‌های قدیمی حذف و با سوئیت جدید سازگار با معماری جایگزین شده‌اند.
اجرای کامل نهایی (وقتی خواستی):
```bash
pytest -q
```

## ۶) عیب‌یابی
### ۶.۱ لاگ پایپلاین تولید نمی‌شود
بررسی کن:
- `PIPELINE_LOG_TO_FILE=true`
- mount پوشه `logs`
- `PIPELINE_SERVICE_NAME` برای هر سرویس

### ۶.۲ فقط backend لاگ می‌نویسد
برای هر سرویس MCP:
- `DEBUG_LOG` تنظیم شده باشد
- `PIPELINE_LOG_TO_FILE=true`
- `PIPELINE_SERVICE_NAME` یکتا باشد

### ۶.۳ فرانت JSON خام نشان می‌دهد
الان fallback استخراج JSON در `frontend/src/App.jsx` پیاده‌سازی شده است.
اگر بازگشت خطا رخ داد، format خروجی response را بررسی کن.

## ۷) تحلیل تاخیر روی سرور
برای اینکه bottleneck را در production ببینی، لازم نیست `DEBUG_LOG=true` باشد.
رویدادهای `LATENCY_SUMMARY` در حالت non-debug هم ثبت می‌شوند.

### ۷.۱ دیدن لاگ زنده
```bash
docker compose logs -f backend interpret search
```

### ۷.۲ دیدن فقط latency summary
```bash
grep -h "LATENCY_SUMMARY" logs/pipeline-*.log
```

### ۷.۳ گزارش bottleneck
```bash
python scripts/analyze_latency_logs.py --log-dir logs --top 30
```

برای تمرکز روی یک بخش:
```bash
python scripts/analyze_latency_logs.py --log-dir logs --component search.pipeline
python scripts/analyze_latency_logs.py --log-dir logs --component interpret.pipeline
python scripts/analyze_latency_logs.py --log-dir logs --component agent.chat
```
