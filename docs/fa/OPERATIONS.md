# عملیات، پیکربندی و استقرار (فارسی)

## ۱) مدل پیکربندی محیط
فایل الگو: `.env.example`

### ۱.۱ متغیرهای کلیدی

| گروه | متغیرها |
|---|---|
| اپلیکیشن | `APP_NAME`, `APP_VERSION`, `HOST`, `PORT`, `DEBUG`, `DEBUG_MODE`, `DEBUG_LOG` |
| مدل/روتینگ ایجنت | `AGENT_MODEL_PROVIDER`, `AGENT_MODEL`, `OPENROUTER_MODEL`, `GROQ_MODEL` |
| OpenRouter | `OPEN_ROUTERS_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_PROVIDER_ORDER`, `OPENROUTER_FALLBACK_TO_GROQ` |
| Groq | `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL` |
| LLM برای Interpret/Search | `GITHUB_TOKEN`, `GITHUB_BASE_URL`, `GITHUB_MODEL` |
| Override Interpret روی OpenRouter | `INTERPRET_OPENROUTER_BASE_URL`, `INTERPRET_OPENROUTER_MODEL` |
| آدرس MCP | `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL` |
| Timeoutها | `AGENT_TIMEOUT`, `INTERPRET_MCP_TIMEOUT`, `SEARCH_MCP_TIMEOUT`, `EMBEDDING_MCP_TIMEOUT`, `MIXTRAL_DSL_TIMEOUT` |
| دیتا | `ELASTICSEARCH_*`, `REDIS_*` |
| TTL کش | `AGENT_CACHE_TTL`, `LLM_CACHE_TTL`, `CACHE_SEARCH_TTL`, `CACHE_DSL_TTL`, `CACHE_EMBEDDING_TTL` |
| لاگ | `PIPELINE_SERVICE_NAME`, `PIPELINE_LOG_TO_FILE`, `PIPELINE_LOG_DIR`, `PIPELINE_LOG_MAX_BYTES`, `PIPELINE_LOG_BACKUP_COUNT` |

## ۲) توپولوژی Docker
سرویس‌های `docker-compose.yml`:
- `frontend`
- `backend`
- `interpret`
- `search`
- `embedding`
- `redis`

نکات wiring:
- Backend با aliasهای docker (`interpret`, `search`) به MCP وصل می‌شود.
- Search با alias `embedding` به Embedding MCP وصل می‌شود.
- Redis در compose فعلی با `host.docker.internal` + `REDIS_HOST_PORT` ست شده است.

## ۳) حالت‌های اجرا

### ۳.۱ Docker production-like
```bash
docker compose up --build
```

### ۳.۲ Docker debug (overlay)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### ۳.۳ اجرای local process
```bash
python3 -m src.mcp_servers.run_servers
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

## ۴) روند پیشنهادی دیپلوی
```bash
# 1) بروزرسانی کد
git fetch origin
git checkout <branch>
git pull --ff-only origin <branch>

# 2) دیپلوی سرویس‌های تغییر کرده
docker compose up -d --build backend interpret search embedding frontend

# 3) بررسی
docker compose ps
docker compose logs --tail=100 backend
```

مثال دیپلوی سرویس خاص:
```bash
# فقط interpret
docker compose up -d --force-recreate interpret

# فقط backend
docker compose up -d --build backend
```

## ۵) Rollback
```bash
git revert <commit_sha>
git push origin <branch>
docker compose up -d --build
```

در صورت نیاز به اسکریپت بازگردانی:
```bash
bash scripts/restore_cleanup_backup.sh
```

## ۶) عملیات کش
فقط namespaceهای همین پروژه را پاک کن:

```bash
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:agent:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:llm_response:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:search:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:negative:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:dsl:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:embedding:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
```

## ۷) مانیتورینگ و تحلیل تاخیر

### ۷.۱ لاگ زنده
```bash
docker compose logs -f backend interpret search embedding
```

### ۷.۲ خلاصه latency
```bash
grep -h "LATENCY_SUMMARY" logs/pipeline-*.log
```

### ۷.۳ گزارش تجمیعی
```bash
python3 scripts/analyze_latency_logs.py --log-dir logs --top 30
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent_service.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component interpret.pipeline
python3 scripts/analyze_latency_logs.py --log-dir logs --component search.pipeline
```

## ۸) Playbook عیب‌یابی

### ۸.۱ خطای OpenRouter tool-use 404
نشانه:
- `No endpoints found that support tool use`

بررسی:
1. مدل `OPENROUTER_MODEL` حتماً tool-capable باشد.
2. `OPENROUTER_FALLBACK_TO_GROQ=true` و `GROQ_API_KEY` معتبر باشد.
3. env واقعی داخل کانتینر را چک کن.

### ۸.۲ خطای اتصال Redis در Interpret
نشانه:
- `Embedding cache Redis connection failed`

بررسی:
1. `REDIS_HOST/REDIS_PORT` داخل env کانتینر interpret.
2. دسترسی کانتینر interpret به Redis.
3. وجود fallback در `interpret_server._init_embedding_cache`.

### ۸.۳ ایجنت ابزار را صدا نمی‌زند
بررسی:
1. در لاگ backend ببین `search_and_deliver called` ثبت می‌شود یا نه.
2. پرامپت runtime در `src/agent.py` با نسخه دیپلوی شده یکی باشد.
3. provider/model استارتاپ backend را چک کن.

### ۸.۴ نتایج search بی‌ربط
بررسی:
1. categoryهای خروجی interpret.
2. DSL تولیدی (`generate_dsl`).
3. امتیازهای rerank (`relevancy_score`, `value_score`).
4. لاگ prune شدن category filter.

## ۹) Smoke Test
```bash
curl -sS http://127.0.0.1:8080/api/health
curl -sS -H 'Content-Type: application/json' \
  -d '{"message":"سلام یه شورت مردانه ارزون میخوام","session_id":"sess_smoke"}' \
  http://127.0.0.1:8080/api/chat
```

## ۱۰) تست‌ها
```bash
pytest -q
```

تست هدفمند:
```bash
pytest -q tests/test_agent_service.py tests/test_mcp_client.py tests/test_agent_cache.py tests/test_pipeline_logger.py
```
