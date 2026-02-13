"""
Shopping AI Assistant - FastAPI Backend

Main entry point for the backend API server.
Provides REST endpoints for the frontend to interact with the AI agent.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.config import settings
from backend.api.routes import router
from backend.services.agent_service import get_agent_service
from src.logging_config import setup_logging, get_logger

DEBUG_LOG = os.environ.get("DEBUG_LOG", os.environ.get("DEBUG_MODE", "false")).lower() in (
    "true",
    "1",
    "yes",
    "on",
)
setup_logging(
    service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "shopping-assistant-backend"),
    level="DEBUG" if DEBUG_LOG else "ERROR",
)
logger = get_logger(__name__)
AGENT_PROVIDER = os.environ.get("AGENT_MODEL_PROVIDER", "openrouter").strip().lower()
if AGENT_PROVIDER == "groq":
    AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("GROQ_MODEL", "")
    AGENT_SECOND_MODEL = (
        os.environ.get("AGENT_SECOND_MODEL")
        or os.environ.get("GROQ_SECOND_MODEL")
        or AGENT_MODEL
    )
else:
    AGENT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("OPENROUTER_MODEL", "")
    AGENT_SECOND_MODEL = (
        os.environ.get("AGENT_SECOND_MODEL")
        or os.environ.get("OPENROUTER_SECOND_MODEL")
        or AGENT_MODEL
    )


# =============================================================================
# Logfire Instrumentation (optional)
# =============================================================================

USE_LOGFIRE = os.environ.get("USE_LOGFIRE", "false").lower() in ("true", "1", "yes")
_logfire_ready = False

if USE_LOGFIRE:
    try:
        import logfire

        _logfire_token = os.environ.get("LOGFIRE_TOKEN", "")
        if not _logfire_token:
            raise ValueError("LOGFIRE_TOKEN environment variable is not set")

        logfire.configure(
            token=_logfire_token,
            service_name=os.environ.get("LOGFIRE_SERVICE_NAME", "shopping-assistant-backend"),
            service_version="3.0.0",
            environment=os.environ.get("LOGFIRE_ENVIRONMENT", "production"),
        )
        _logfire_ready = True
    except Exception as e:
        logger.warning(f"Logfire init failed: {e}")


# =============================================================================
# Lifespan Manager
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    - Startup: Initialize agent
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting Shopping AI Assistant Backend")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Agent provider: {AGENT_PROVIDER}")
    logger.info(f"Agent model: {AGENT_MODEL or 'not-set'}")
    logger.info(f"Agent second model: {AGENT_SECOND_MODEL or 'not-set'}")
    
    # Initialize agent service
    agent_service = get_agent_service()
    try:
        await agent_service.initialize()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.warning(f"Agent initialization failed: {e}")
        logger.info("Agent will be initialized on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend")


# =============================================================================
# FastAPI App
# =============================================================================


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## 🛒 Shopping AI Assistant API

A Persian shopping assistant powered by AI. 

### Features:
- 💬 Natural language chat in Persian
- 🔍 Product search with smart filtering
- 💰 Price-based recommendations
- 🏷️ Brand filtering

### Endpoints:
- `POST /api/chat` - Send a message and get response
- `GET /api/health` - Check service health
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# =============================================================================
# CORS Middleware
# =============================================================================


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Include Routers
# =============================================================================


app.include_router(router, prefix="/api", tags=["API"])


# =============================================================================
# Logfire FastAPI + HTTPX Instrumentation
# =============================================================================

if USE_LOGFIRE and _logfire_ready:
    try:
        logfire.instrument_fastapi(app)
        logfire.instrument_httpx()
        logger.info("Logfire: FastAPI + HTTPX instrumented")
    except Exception as e:
        logger.warning(f"Logfire instrumentation failed: {e}")


# =============================================================================
# Root Redirect
# =============================================================================


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to API docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
