# Operations and Configuration (English)

## 1. Environment Setup
Primary file: `.env`
Template: `.env.example`

Key variables:
- Model selection: `AGENT_MODEL_PROVIDER`, `AGENT_MODEL`
- Provider keys/models: `OPEN_ROUTERS_API_KEY`, `OPENROUTER_MODEL`, `GROQ_API_KEY`, `GROQ_MODEL`
- MCP URLs: `MCP_INTERPRET_URL`, `MCP_SEARCH_URL`, `MCP_EMBEDDING_URL`
- Data stores: `REDIS_*`, `ELASTICSEARCH_*`
- Logging: `DEBUG_LOG`, `PIPELINE_*`, `USE_LOGFIRE`

## 2. Startup Modes
### 2.1 Local processes
```bash
python -m src.mcp_servers.run_servers
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload
```

### 2.2 Docker
```bash
docker-compose up --build
```

Debug overlay:
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

## 3. Logging and Storage Limits
To avoid filling server disk:
- Pipeline logs use rotating files
  - `PIPELINE_LOG_MAX_BYTES`
  - `PIPELINE_LOG_BACKUP_COUNT`
- Docker logs are also rotated via compose `logging.options`
- In production, keep `DEBUG_LOG=false`

Recommended production defaults:
- `DEBUG_LOG=false`
- `PIPELINE_LOG_MAX_BYTES=5000000`
- `PIPELINE_LOG_BACKUP_COUNT=3`

## 4. Restore / Rollback
Cleanup and docs refresh backups are in `_backup/`.
Restore removed files from latest cleanup snapshot:
```bash
bash scripts/restore_cleanup_backup.sh
```

## 5. Testing Status
The old tests were removed and replaced with architecture-aligned tests in `tests/`.
If environment dependencies are available, run:
```bash
pytest -q
```

## 6. Troubleshooting
### 6.1 No pipeline logs
Check:
- `PIPELINE_LOG_TO_FILE=true`
- mounted `./logs:/app/logs` in docker
- service-level `PIPELINE_SERVICE_NAME`

### 6.2 Only backend logs appear
Verify each MCP service has:
- `DEBUG_LOG` set
- `PIPELINE_LOG_TO_FILE=true`
- `PIPELINE_SERVICE_NAME` unique per service

### 6.3 Frontend shows raw JSON
Current frontend includes fallback JSON extraction and stripping in `frontend/src/App.jsx`.
If regression appears, inspect message payload and JSON block format.
