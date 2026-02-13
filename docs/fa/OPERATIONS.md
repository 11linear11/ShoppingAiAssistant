# عملیات، پیکربندی و استقرار (فارسی)

## ۱) پیکربندی محیط
فایل مرجع متغیرها: `.env.example`

## ۱.۱ گروه‌های کلیدی متغیرها

### اپلیکیشن
- `APP_NAME`, `APP_VERSION`, `HOST`, `PORT`
- `DEBUG`, `DEBUG_MODE`, `DEBUG_LOG`

### مدل ایجنت
- `AGENT_MODEL_PROVIDER=openrouter|groq`
- `AGENT_MODEL` (اختیاری)
- تنظیمات OpenRouter:
  - `OPEN_ROUTERS_API_KEY`
  - `OPENROUTER_MODEL`
  - `OPENROUTER_PROVIDER_ORDER`
  - `OPENROUTER_FALLBACK_TO_GROQ`
- تنظیمات Groq:
  - `GROQ_API_KEY`
  - `GROQ_MODEL`

### مدل Interpret/Search
- `GITHUB_TOKEN`, `GITHUB_BASE_URL`, `GITHUB_MODEL`
- در compose، interpret می‌تواند با OpenRouter override شود:
  - `INTERPRET_OPENROUTER_BASE_URL`
  - `INTERPRET_OPENROUTER_MODEL`

### دیتا و کش
- Elasticsearch: `ELASTICSEARCH_*`
- Redis: `REDIS_*`
- TTLها: `AGENT_CACHE_TTL`, `LLM_CACHE_TTL`, `CACHE_SEARCH_TTL`, `CACHE_DSL_TTL`, `CACHE_EMBEDDING_TTL`

### MCP و timeout
- `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL`
- `INTERPRET_MCP_TIMEOUT`, `SEARCH_MCP_TIMEOUT`, `EMBEDDING_MCP_TIMEOUT`

### لاگ پایپلاین
- `PIPELINE_SERVICE_NAME`
- `PIPELINE_LOG_TO_FILE`
- `PIPELINE_LOG_DIR`
- `PIPELINE_LOG_MAX_BYTES`
- `PIPELINE_LOG_BACKUP_COUNT`

## ۲) حالت‌های اجرا

### ۲.۱ اجرای لوکال
```bash
python3 -m src.mcp_servers.run_servers
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### ۲.۲ اجرای Docker (Production)
```bash
docker compose up --build
```

### ۲.۳ اجرای Docker (Debug)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## ۳) فرایند دیپلوی روی سرور
```bash
# 1) بروزرسانی کد
git pull --ff-only origin <branch>

# 2) در تغییرات backend فقط backend را rebuild/recreate کن
docker compose up -d --build backend

# 3) بررسی وضعیت
docker compose ps backend
docker logs --tail 100 assistant-backend
```

برای rebuild کامل:
```bash
docker compose up -d --build
```

## ۴) Rollback
### ۴.۱ rollback کد
```bash
git revert <commit_sha>
git push origin <branch>
```
سپس backend/container را دوباره deploy کن.

### ۴.۲ بازگردانی snapshotهای cleanup
```bash
bash scripts/restore_cleanup_backup.sh
```

## ۵) عملیات کش
با احتیاط در production انجام شود.

### ۵.۱ مشاهده کلیدها
```bash
redis-cli -h <redis-host> -p <redis-port> KEYS 'cache:*'
```

### ۵.۲ پاک‌سازی فقط namespaceهای همین پروژه
```bash
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:agent:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:llm_response:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:search:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:negative:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:dsl:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:embedding:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
```

## ۶) تحلیل تاخیر و bottleneck

### ۶.۱ لاگ زنده
```bash
docker compose logs -f backend interpret search embedding
```

### ۶.۲ خلاصه latency
```bash
grep -h "LATENCY_SUMMARY" logs/pipeline-*.log
```

### ۶.۳ گزارش تحلیلی
```bash
python3 scripts/analyze_latency_logs.py --log-dir logs --top 30
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component interpret.pipeline
python3 scripts/analyze_latency_logs.py --log-dir logs --component search.pipeline
```

## ۷) عیب‌یابی سریع

### ۷.۱ مدل بدون tool پاسخ می‌دهد
- لاگ startup را چک کن تا provider/model درست باشد
- در لاگ backend وجود `search_and_deliver called` را بررسی کن
- مطمئن شو کد داخل کانتینر با برنچ جاری یکسان است

### ۷.۲ خطای OpenRouter tool-use (404)
- ترکیب provider/model را به مدل tool-capable تغییر بده
- fallback به Groq را فعال نگه دار

### ۷.۳ تاخیر زیاد در درخواست اول
- `mcp_client.initialize` را در latency summary بررسی کن
- warm-up request بزن تا sessionها initialize شوند

### ۷.۴ نتایج نامرتبط سرچ
- لاگ category guard در search را بررسی کن
- DSL تولیدی را با `generate_dsl` بررسی کن
- کیفیت `categories_fa` خروجی interpret را چک کن

## ۸) تست
```bash
pytest -q
```

تست‌های سریع هدفمند:
```bash
pytest -q tests/test_agent_service.py tests/test_agent_cache.py tests/test_pipeline_logger.py
```
