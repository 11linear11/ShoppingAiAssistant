"""
API Routes

Defines all HTTP endpoints for the Shopping AI Assistant.
"""

import httpx
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatMetadata,
    ErrorResponse,
    HealthResponse,
    ServiceHealth,
    ProductInfo,
)
from backend.core.config import settings
from backend.services.agent_service import AgentService, get_agent_service


# =============================================================================
# Router Setup
# =============================================================================

router = APIRouter()


# =============================================================================
# Chat Endpoint
# =============================================================================


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "Successful response"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Send a chat message",
    description="Send a message to the shopping assistant and receive a response with optional product recommendations.",
)
async def chat(
    request: ChatRequest,
    agent: Annotated[AgentService, Depends(get_agent_service)],
) -> ChatResponse:
    """
    Process a user chat message.
    
    The assistant will:
    - Respond to greetings and casual chat naturally
    - Search for products when user asks for something
    - Provide product recommendations with prices and details
    """
    result = await agent.chat(
        message=request.message,
        session_id=request.session_id,
        timeout=settings.agent_timeout,
    )
    
    # Convert to response model
    products = [ProductInfo(**p) for p in result.get("products", [])]
    metadata = ChatMetadata(**result["metadata"])
    
    return ChatResponse(
        success=result["success"],
        response=result["response"],
        session_id=result["session_id"],
        products=products,
        metadata=metadata,
    )


# =============================================================================
# Health Check Endpoint
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all services.",
)
async def health_check(
    agent: Annotated[AgentService, Depends(get_agent_service)],
) -> HealthResponse:
    """
    Check health of all services.
    
    Returns status of:
    - Agent (LLM connection)
    - Interpret Server (MCP)
    - Search Server (MCP)
    - Cache Server (MCP)
    """
    services = {}
    overall_healthy = True
    
    # Check Agent
    agent_health = await agent.health_check()
    services["agent"] = ServiceHealth(**agent_health)
    if agent_health["status"] != "ok":
        overall_healthy = False
    
    # Check MCP Servers
    mcp_servers = {
        "interpret_server": settings.mcp_interpret_url,
        "search_server": settings.mcp_search_url,
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in mcp_servers.items():
            start = datetime.now()
            try:
                # Try to reach the MCP server
                response = await client.get(f"{url}/health")
                latency = int((datetime.now() - start).total_seconds() * 1000)
                
                if response.status_code == 200:
                    services[name] = ServiceHealth(status="ok", latency_ms=latency)
                elif response.status_code == 404:
                    # MCP servers expose /mcp (not /health) in streamable-http mode.
                    # A 404 here still proves service reachability.
                    services[name] = ServiceHealth(status="ok", latency_ms=latency)
                else:
                    services[name] = ServiceHealth(
                        status="error",
                        latency_ms=latency,
                        error=f"HTTP {response.status_code}",
                    )
                    overall_healthy = False
            except httpx.TimeoutException:
                services[name] = ServiceHealth(status="timeout", error="Request timed out")
                overall_healthy = False
            except Exception as e:
                services[name] = ServiceHealth(status="error", error=str(e)[:100])
                overall_healthy = False
    
    return HealthResponse(
        status="healthy" if overall_healthy else "unhealthy",
        services=services,
        timestamp=datetime.now(),
    )


# =============================================================================
# Root Endpoint (for testing)
# =============================================================================


@router.get(
    "/",
    summary="API Root",
    description="Basic endpoint to verify API is running.",
)
async def root():
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
    }
