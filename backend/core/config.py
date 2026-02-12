"""
Backend Configuration Module

Centralized settings management using Pydantic.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App Info
    app_name: str = Field(default="Shopping AI Assistant", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8080, alias="PORT")

    # CORS (for frontend)
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS",
    )

    # Groq API (for Agent)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_base_url: str = Field(
        default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL"
    )

    # MCP Server URLs
    mcp_interpret_url: str = Field(
        default="http://localhost:5004", alias="MCP_INTERPRET_URL"
    )
    mcp_search_url: str = Field(
        default="http://localhost:5002", alias="MCP_SEARCH_URL"
    )

    # Redis (for agent response cache)
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # Agent Response Cache
    agent_cache_enabled: bool = Field(default=True, alias="AGENT_CACHE_ENABLED")
    agent_cache_ttl: int = Field(default=86400, alias="AGENT_CACHE_TTL")  # 24 hours

    # Timeouts
    agent_timeout: int = Field(default=120, alias="AGENT_TIMEOUT")

    # Feature flags
    ff_interpret_warmup: bool = Field(default=True, alias="FF_INTERPRET_WARMUP")
    ff_router_enabled: bool = Field(default=True, alias="FF_ROUTER_ENABLED")
    ff_abstract_fastpath: bool = Field(default=True, alias="FF_ABSTRACT_FASTPATH")
    ff_direct_fastpath: bool = Field(default=False, alias="FF_DIRECT_FASTPATH")
    ff_conditional_final_llm: bool = Field(default=True, alias="FF_CONDITIONAL_FINAL_LLM")

    # Router guard thresholds (for direct fastpath confidence)
    router_guard_t1: float = Field(default=0.55, alias="ROUTER_GUARD_T1")
    router_guard_t2: float = Field(default=0.08, alias="ROUTER_GUARD_T2")
    router_guard_min_confidence: float = Field(
        default=0.58,
        alias="ROUTER_GUARD_MIN_CONFIDENCE",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
