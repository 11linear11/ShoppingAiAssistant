# Operations and Configuration (English)

## 1. Environment Configuration
Primary env template: `.env.example`

## 1.1 Critical Variable Groups

### Application
- `APP_NAME`, `APP_VERSION`, `HOST`, `PORT`
- `DEBUG`, `DEBUG_MODE`, `DEBUG_LOG`

### Agent Model Routing
- `AGENT_MODEL_PROVIDER=openrouter|groq`
- `AGENT_MODEL` (optional override)
- OpenRouter options:
  - `OPEN_ROUTERS_API_KEY`
  - `OPENROUTER_MODEL`
  - `OPENROUTER_PROVIDER_ORDER`
  - `OPENROUTER_FALLBACK_TO_GROQ`
- Groq options:
  - `GROQ_API_KEY`
  - `GROQ_MODEL`

### Interpret/Search Models
- `GITHUB_TOKEN`, `GITHUB_BASE_URL`, `GITHUB_MODEL`
- Interpret container can override to OpenRouter via compose:
  - `INTERPRET_OPENROUTER_BASE_URL`
  - `INTERPRET_OPENROUTER_MODEL`

### Data and Caches
- Elasticsearch: `ELASTICSEARCH_*`
- Redis: `REDIS_*`
- TTLs: `AGENT_CACHE_TTL`, `LLM_CACHE_TTL`, `CACHE_SEARCH_TTL`, `CACHE_DSL_TTL`, `CACHE_EMBEDDING_TTL`

### MCP URLs and Timeouts
- `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL`
- `INTERPRET_MCP_TIMEOUT`, `SEARCH_MCP_TIMEOUT`, `EMBEDDING_MCP_TIMEOUT`

### Pipeline Logging
- `PIPELINE_SERVICE_NAME`
- `PIPELINE_LOG_TO_FILE`
- `PIPELINE_LOG_DIR`
- `PIPELINE_LOG_MAX_BYTES`
- `PIPELINE_LOG_BACKUP_COUNT`

## 2. Run Modes

### 2.1 Local Process Mode
```bash
python3 -m src.mcp_servers.run_servers
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### 2.2 Docker Production Mode
```bash
docker compose up --build
```

### 2.3 Docker Debug Mode
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## 3. Deployment Procedure (Server)
```bash
# 1) Update code
git pull --ff-only origin <branch>

# 2) Rebuild/restart only backend when backend-only changes
docker compose up -d --build backend

# 3) Verify

docker compose ps backend
docker logs --tail 100 assistant-backend
```

For full stack rebuild:
```bash
docker compose up -d --build
```

## 4. Rollback Procedure
### 4.1 Code rollback
```bash
git revert <commit_sha>
git push origin <branch>
```
Then redeploy backend/container.

### 4.2 Restore cleanup backups
```bash
bash scripts/restore_cleanup_backup.sh
```

## 5. Cache Operations
Use with care in production.

### 5.1 Inspect cache keys
```bash
redis-cli -h <redis-host> -p <redis-port> KEYS 'cache:*'
```

### 5.2 Clear only this project cache namespaces
```bash
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:agent:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:llm_response:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:search:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:negative:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:dsl:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:embedding:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
```

## 6. Latency and Bottleneck Analysis

### 6.1 Live logs
```bash
docker compose logs -f backend interpret search embedding
```

### 6.2 Latency summaries
```bash
grep -h "LATENCY_SUMMARY" logs/pipeline-*.log
```

### 6.3 Aggregated analysis
```bash
python3 scripts/analyze_latency_logs.py --log-dir logs --top 30
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component interpret.pipeline
python3 scripts/analyze_latency_logs.py --log-dir logs --component search.pipeline
```

## 7. Troubleshooting

### 7.1 Agent responds without tool usage
- Verify startup logs show expected provider/model.
- Inspect backend pipeline log for `search_and_deliver called` entries.
- Confirm prompt/tool code in running container matches current branch.

### 7.2 Tool-use endpoint errors (OpenRouter 404)
- Use a tool-capable model/provider combination.
- Keep Groq fallback configured when using OpenRouter.

### 7.3 High first-request latency
- Check `mcp_client.initialize` in latency summaries.
- Warm up services by sending a startup request.

### 7.4 Irrelevant products in results
- Inspect category guard logs in search server.
- Inspect generated DSL (`generate_dsl`) and rerank behavior.
- Validate `categories_fa` quality from interpret output.

## 8. Test Commands
```bash
pytest -q
```

Focused checks:
```bash
pytest -q tests/test_agent_service.py tests/test_agent_cache.py tests/test_pipeline_logger.py
```
