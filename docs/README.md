# Documentation Index

This folder is the authoritative technical documentation for the current runtime architecture.

## Reading Order
1. `ARCHITECTURE.md`
2. `PIPELINES.md`
3. `API.md`
4. `OPERATIONS.md`

## English
- `docs/en/ARCHITECTURE.md`
- `docs/en/PIPELINES.md`
- `docs/en/API.md`
- `docs/en/OPERATIONS.md`

## فارسی
- `docs/fa/ARCHITECTURE.md`
- `docs/fa/PIPELINES.md`
- `docs/fa/API.md`
- `docs/fa/OPERATIONS.md`

## Coverage
- API gateway and schemas: `backend/main.py`, `backend/api/*`, `backend/services/agent_service.py`
- Agent orchestration: `src/agent.py`
- MCP transport: `src/mcp_client.py`
- MCP services: `src/mcp_servers/*`
- Caching: `src/agent_cache.py`, Redis paths in search/interpret/agent
- Logging and telemetry: `src/pipeline_logger.py`, `src/logging_config.py`
- Deployment and runtime wiring: `docker-compose.yml`, `docker-compose.dev.yml`, `.env.example`

## Notes
- The docs describe the **actual code paths in this repository**.
- If behavior changes in code, update docs in the same PR.
