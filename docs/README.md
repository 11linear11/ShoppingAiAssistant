# Documentation Index

This folder contains the authoritative project documentation for both English and Persian readers.

## Recommended Reading Order
1. Architecture
2. Pipelines
3. API Contracts
4. Operations

## English
- Architecture: `docs/en/ARCHITECTURE.md`
- Pipelines: `docs/en/PIPELINES.md`
- API: `docs/en/API.md`
- Operations: `docs/en/OPERATIONS.md`

## فارسی
- معماری: `docs/fa/ARCHITECTURE.md`
- پایپلاین‌ها: `docs/fa/PIPELINES.md`
- API و قراردادها: `docs/fa/API.md`
- عملیات و استقرار: `docs/fa/OPERATIONS.md`

## Scope Covered by These Docs
- Backend gateway: `backend/`
- Agent orchestration: `src/agent.py`
- Agent response shaping/cache: `backend/services/agent_service.py`, `src/agent_cache.py`
- MCP transport/client: `src/mcp_client.py`
- MCP servers:
  - `src/mcp_servers/interpret_server.py`
  - `src/mcp_servers/search_server.py`
  - `src/mcp_servers/embedding_server.py`
- Pipeline telemetry: `src/pipeline_logger.py`
- Deployment: `docker-compose.yml`, `docker-compose.dev.yml`

## Fast Navigation by Task
- Need to understand full request flow? -> `ARCHITECTURE.md` + `PIPELINES.md`
- Need request/response schemas and tool contracts? -> `API.md`
- Need env/deploy/rollback/log analysis commands? -> `OPERATIONS.md`
