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
- `backend` (FastAPI gateway): `:8080`
- `search` MCP server: `:5002`
- `embedding` MCP server: `:5003`
- `interpret` MCP server: `:5004`
- `redis`: `:6379`

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
