# Shopping AI Assistant V3

Persian shopping assistant with FastAPI, LangGraph, MCP servers, Redis caching, and Elasticsearch search.

## Quick Links
- Documentation index: `docs/README.md`
- English docs: `docs/en/`
- Persian docs: `docs/fa/`
- Backend API entry: `backend/main.py`
- Agent core: `src/agent.py`
- MCP servers: `src/mcp_servers/`

## Services
- `frontend`: `${FRONTEND_HOST_PORT:-3000}`
- `backend` (FastAPI gateway): `${BACKEND_HOST_PORT:-8080}`
- `search` MCP server: `${MCP_SEARCH_HOST_PORT:-5002}`
- `embedding` MCP server: `${MCP_EMBEDDING_HOST_PORT:-5003}`
- `interpret` MCP server: `${MCP_INTERPRET_HOST_PORT:-5004}`
- `redis`: `${REDIS_HOST_PORT:-6379}`

## Local Run
```bash
cp .env.example .env
python -m pip install -r requirements.txt
python -m src.mcp_servers.run_servers
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

## Docker Run
```bash
docker-compose up --build
```

If your server already uses common ports (e.g. `8080`, `6379`), override only host ports in `.env`:
```env
BACKEND_HOST_PORT=8081
REDIS_HOST_PORT=6380
FRONTEND_HOST_PORT=3001
MCP_SEARCH_HOST_PORT=5005
MCP_EMBEDDING_HOST_PORT=5006
MCP_INTERPRET_HOST_PORT=5007
```
Internal service ports and project structure stay unchanged.

Debug stack:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## Logging
- Structured service logs: `logs/shopping-assistant-backend.log` and `.error.log`
- Pipeline logs (per service):
  - `logs/pipeline-shopping-assistant-backend.log`
  - `logs/pipeline-shopping-assistant-interpret.log`
  - `logs/pipeline-shopping-assistant-search.log`
  - `logs/pipeline-shopping-assistant-embedding.log`

## Safety / Rollback
Cleanup backups are stored under `_backup/`.
If a removed file should be restored, use:
```bash
bash scripts/restore_cleanup_backup.sh
```

## Status Note
Test suite has been replaced with a new architecture-aligned suite in `tests/`. Full final validation can be run later with:
```bash
pytest -q
```

Current quick validation (file-by-file) is green for:
- `tests/test_agent_cache.py`
- `tests/test_agent_service.py`
- `tests/test_mcp_client.py`
- `tests/test_pipeline_logger.py`
- `tests/test_backend_api.py`
