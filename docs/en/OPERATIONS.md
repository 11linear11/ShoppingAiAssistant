# Operations and Configuration (English)

## 1. Environment Model
Main template: `.env.example`

### 1.1 Critical Variables

| Group | Variables |
|---|---|
| App | `APP_NAME`, `APP_VERSION`, `HOST`, `PORT`, `DEBUG`, `DEBUG_MODE`, `DEBUG_LOG` |
| Agent routing/model | `AGENT_MODEL_PROVIDER`, `AGENT_MODEL`, `OPENROUTER_MODEL`, `GROQ_MODEL` |
| OpenRouter | `OPEN_ROUTERS_API_KEY`, `OPENROUTER_BASE_URL`, `OPENROUTER_PROVIDER_ORDER`, `OPENROUTER_FALLBACK_TO_GROQ` |
| Groq | `GROQ_API_KEY`, `GROQ_BASE_URL`, `GROQ_MODEL` |
| Interpret/Search LLM | `GITHUB_TOKEN`, `GITHUB_BASE_URL`, `GITHUB_MODEL` |
| Interpret OpenRouter override | `INTERPRET_OPENROUTER_BASE_URL`, `INTERPRET_OPENROUTER_MODEL` |
| MCP endpoints | `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL` |
| Timeouts | `AGENT_TIMEOUT`, `INTERPRET_MCP_TIMEOUT`, `SEARCH_MCP_TIMEOUT`, `EMBEDDING_MCP_TIMEOUT`, `MIXTRAL_DSL_TIMEOUT` |
| Data | `ELASTICSEARCH_*`, `REDIS_*` |
| Cache TTLs | `AGENT_CACHE_TTL`, `LLM_CACHE_TTL`, `CACHE_SEARCH_TTL`, `CACHE_DSL_TTL`, `CACHE_EMBEDDING_TTL` |
| Logging | `PIPELINE_SERVICE_NAME`, `PIPELINE_LOG_TO_FILE`, `PIPELINE_LOG_DIR`, `PIPELINE_LOG_MAX_BYTES`, `PIPELINE_LOG_BACKUP_COUNT` |

## 2. Docker Deployment Topology
`docker-compose.yml` services:
- `frontend`
- `backend`
- `interpret`
- `search`
- `embedding`
- `redis`

Important wiring details:
- Backend talks to MCP via docker aliases (`interpret`, `search`).
- Search talks to embedding via `embedding` alias.
- Redis host in compose services is currently `host.docker.internal` + `REDIS_HOST_PORT`.

## 3. Run Modes

### 3.1 Production-like Docker
```bash
docker compose up --build
```

### 3.2 Debug Docker (overlay)
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### 3.3 Local Process Mode
```bash
python3 -m src.mcp_servers.run_servers
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

## 4. Recommended Deploy Procedure
```bash
# 1) update code
git fetch origin
git checkout <branch>
git pull --ff-only origin <branch>

# 2) deploy only changed services when possible
docker compose up -d --build backend interpret search embedding frontend

# 3) verify
docker compose ps
docker compose logs --tail=100 backend
```

Service-specific deploy examples:
```bash
# interpret-only change
docker compose up -d --force-recreate interpret

# backend-only change
docker compose up -d --build backend
```

## 5. Rollback Procedure
```bash
git revert <commit_sha>
git push origin <branch>
docker compose up -d --build
```

If cleanup rollback script is required:
```bash
bash scripts/restore_cleanup_backup.sh
```

## 6. Cache Operations
Use targeted namespace deletion only.

```bash
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:agent:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:llm_response:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:search:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v2:negative:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:dsl:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
redis-cli -h <redis-host> -p <redis-port> --scan --pattern 'cache:v1:embedding:*' | xargs -r redis-cli -h <redis-host> -p <redis-port> del
```

## 7. Observability and Latency Analysis

### 7.1 Live logs
```bash
docker compose logs -f backend interpret search embedding
```

### 7.2 Latency summaries
```bash
grep -h "LATENCY_SUMMARY" logs/pipeline-*.log
```

### 7.3 Aggregated latency report
```bash
python3 scripts/analyze_latency_logs.py --log-dir logs --top 30
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component agent_service.chat
python3 scripts/analyze_latency_logs.py --log-dir logs --component interpret.pipeline
python3 scripts/analyze_latency_logs.py --log-dir logs --component search.pipeline
```

## 8. Troubleshooting Playbook

### 8.1 OpenRouter tool-use 404
Symptoms:
- `No endpoints found that support tool use`

Checks:
1. Verify `OPENROUTER_MODEL` is tool-capable.
2. Keep `OPENROUTER_FALLBACK_TO_GROQ=true` and valid `GROQ_API_KEY`.
3. Confirm runtime env inside container.

### 8.2 Interpret Redis embedding cache connection errors
Symptoms:
- `Embedding cache Redis connection failed`

Checks:
1. Verify `REDIS_HOST/REDIS_PORT` in interpret container env.
2. Verify Redis service reachable from interpret container.
3. Confirm recent fallback logic in `interpret_server._init_embedding_cache`.

### 8.3 Agent not calling tools
Checks:
1. Inspect `pipeline-...-backend.log` for `search_and_deliver called`.
2. Verify deployed prompt in `src/agent.py`.
3. Verify selected model/provider at backend startup logs.

### 8.4 Search returns irrelevant products
Checks:
1. Inspect interpret output categories.
2. Inspect search DSL (`generate_dsl`).
3. Inspect rerank outputs (`relevancy_score`, `value_score`).
4. Check whether category guard pruned filters.

## 9. Smoke Test Commands
```bash
curl -sS http://127.0.0.1:8080/api/health
curl -sS -H 'Content-Type: application/json' \
  -d '{"message":"سلام یه شورت مردانه ارزون میخوام","session_id":"sess_smoke"}' \
  http://127.0.0.1:8080/api/chat
```

## 10. Test Suite
```bash
pytest -q
```

Targeted:
```bash
pytest -q tests/test_agent_service.py tests/test_mcp_client.py tests/test_agent_cache.py tests/test_pipeline_logger.py
```
