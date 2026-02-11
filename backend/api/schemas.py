"""
Pydantic Schemas for API Request/Response

Defines the data models used in API endpoints.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Request Schemas
# =============================================================================


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message in Persian",
        examples=["یخچال میخوام", "سلام", "ارزون ترین شامپو"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )


# =============================================================================
# Response Schemas
# =============================================================================


class ProductInfo(BaseModel):
    """Product information in search results."""

    id: str = Field(..., description="Product ID")
    name: str = Field(..., description="Product name in Persian")
    brand: Optional[str] = Field(None, description="Brand name")
    price: float = Field(..., description="Original price in Tomans")
    discount_price: Optional[float] = Field(None, description="Discounted price")
    has_discount: bool = Field(False, description="Whether product has discount")
    discount_percentage: Optional[float] = Field(None, description="Discount %")
    image_url: Optional[str] = Field(None, description="Product image URL")
    product_url: Optional[str] = Field(None, description="Link to product page")


class ChatMetadata(BaseModel):
    """Metadata about the chat response."""

    took_ms: int = Field(..., description="Processing time in milliseconds")
    query_type: Optional[str] = Field(
        None, description="Type of query (direct/abstract/chat)"
    )
    total_results: Optional[int] = Field(
        None, description="Total number of products found"
    )
    from_agent_cache: Optional[bool] = Field(
        False, description="Whether this response was served from agent cache"
    )
    original_took_ms: Optional[int] = Field(
        None, description="Original processing time before caching (ms)"
    )
    cached_at: Optional[str] = Field(
        None, description="ISO timestamp of when this response was cached"
    )


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    response: str = Field(..., description="Agent's response text in Persian")
    session_id: str = Field(..., description="Session ID for continuity")
    products: list[ProductInfo] = Field(
        default_factory=list, description="List of products (if search was performed)"
    )
    metadata: ChatMetadata = Field(..., description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "response": "اینم یخچال‌های پیدا شده:\n\n📦 یخچال فریزر جی‌پلاس...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "products": [
                    {
                        "id": "abc123",
                        "name": "یخچال فریزر جی‌پلاس",
                        "brand": "جی پلاس",
                        "price": 26500000,
                        "discount_price": 24000000,
                        "has_discount": True,
                        "discount_percentage": 9.4,
                        "image_url": "https://example.com/image.jpg",
                        "product_url": "https://example.com/product",
                    }
                ],
                "metadata": {
                    "took_ms": 1500,
                    "query_type": "direct",
                    "total_results": 42,
                },
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error info")


# =============================================================================
# Health Check Schemas
# =============================================================================


class ServiceHealth(BaseModel):
    """Health status of a single service."""

    status: str = Field(..., description="ok, error, or timeout")
    latency_ms: Optional[int] = Field(None, description="Response latency")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""

    status: str = Field(..., description="Overall status: healthy or unhealthy")
    services: dict[str, ServiceHealth] = Field(
        ..., description="Status of each service"
    )
    timestamp: datetime = Field(..., description="Check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "services": {
                    "agent": {"status": "ok", "latency_ms": 50},
                    "interpret_server": {"status": "ok", "latency_ms": 100},
                    "search_server": {"status": "ok", "latency_ms": 80},
                },
                "timestamp": "2026-02-02T08:30:00Z",
            }
        }
