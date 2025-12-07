"""
Centralized Logging Configuration using Logfire
All MCP servers and the agent use this configuration.

Features:
- Structured logging with spans
- MCP server instrumentation
- Console and cloud logging
- Request/response tracking
- Performance metrics
"""

import os
import sys
import time
import functools
from typing import Any, Callable, Optional
from contextlib import contextmanager
from logging import basicConfig, getLogger, Logger, DEBUG, INFO, WARNING

import logfire
from logfire import LogfireSpan

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
SERVICE_NAME = os.getenv("LOGFIRE_SERVICE_NAME", "shopping-assistant")
SEND_TO_LOGFIRE = os.getenv("SEND_TO_LOGFIRE", "if-token-present")
LOG_LEVEL = DEBUG if DEBUG_MODE else INFO

# Emojis for log levels
EMOJI = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "tool": "🔧",
    "search": "🔎",
    "embed": "🧠",
    "llm": "🤖",
    "db": "💾",
    "http": "🌐",
    "time": "⏱️",
}


# ═══════════════════════════════════════════════════════════════
# Logfire Configuration
# ═══════════════════════════════════════════════════════════════
_configured = False


def configure_logging(service_name: str = None) -> "logfire":
    """
    Configure Logfire for the application.
    
    Args:
        service_name: Optional service name override (e.g., "embedding-server")
        
    Returns:
        Configured logfire instance
    """
    global _configured
    
    if _configured:
        return logfire
        
    final_service_name = service_name or SERVICE_NAME
    
    # Configure Logfire with detailed options
    logfire.configure(
        service_name=final_service_name,
        send_to_logfire=SEND_TO_LOGFIRE,
        console=logfire.ConsoleOptions(
            colors='auto',
            span_style='show-parents' if DEBUG_MODE else 'simple',
            include_timestamps=True,
            verbose=DEBUG_MODE,
            min_log_level='debug' if DEBUG_MODE else 'info',
        ),
    )
    
    # Integrate with standard library logging
    basicConfig(
        level=LOG_LEVEL,
        handlers=[logfire.LogfireLoggingHandler()],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Instrument MCP if available
    try:
        logfire.instrument_mcp()
    except Exception:
        pass
    
    _configured = True
    logfire.info(f"{EMOJI['start']} Logging configured for {final_service_name}")
    
    return logfire


# ═══════════════════════════════════════════════════════════════
# Logger Factory
# ═══════════════════════════════════════════════════════════════
class MCPLogger:
    """Enhanced logger for MCP servers with structured logging."""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = getLogger(name)
        self._logger.setLevel(LOG_LEVEL)
        
    def info(self, message: str, **kwargs):
        """Log info with optional structured data."""
        logfire.info(f"{EMOJI['info']} [{self.name}] {message}", **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug with optional structured data."""
        if DEBUG_MODE:
            logfire.debug(f"{EMOJI['debug']} [{self.name}] {message}", **kwargs)
            
    def warning(self, message: str, **kwargs):
        """Log warning with optional structured data."""
        logfire.warn(f"{EMOJI['warning']} [{self.name}] {message}", **kwargs)
        
    def error(self, message: str, error: Exception = None, **kwargs):
        """Log error with optional exception details."""
        if error:
            kwargs['error_type'] = type(error).__name__
            kwargs['error_message'] = str(error)
        logfire.error(f"{EMOJI['error']} [{self.name}] {message}", **kwargs)
        
    def success(self, message: str, **kwargs):
        """Log success message."""
        logfire.info(f"{EMOJI['success']} [{self.name}] {message}", **kwargs)
        
    @contextmanager
    def span(self, operation: str, **attributes):
        """Create a span for tracking an operation."""
        with logfire.span(f"{self.name}.{operation}", **attributes) as span:
            yield span
            
    @contextmanager
    def tool_span(self, tool_name: str, **attributes):
        """Create a span for MCP tool execution."""
        start_time = time.time()
        logfire.info(f"{EMOJI['tool']} [{self.name}] Tool '{tool_name}' started", **attributes)
        
        try:
            with logfire.span(f"tool.{tool_name}", **attributes) as span:
                yield span
                duration = time.time() - start_time
                logfire.info(
                    f"{EMOJI['success']} [{self.name}] Tool '{tool_name}' completed",
                    duration_ms=round(duration * 1000, 2),
                    **attributes
                )
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(
                f"{EMOJI['error']} [{self.name}] Tool '{tool_name}' failed: {e}",
                duration_ms=round(duration * 1000, 2),
                error_type=type(e).__name__,
                **attributes
            )
            raise
            
    @contextmanager  
    def search_span(self, query: str, **attributes):
        """Create a span for search operations."""
        start_time = time.time()
        logfire.info(f"{EMOJI['search']} [{self.name}] Search: '{query}'", **attributes)
        
        try:
            with logfire.span(f"search.{self.name}", query=query, **attributes) as span:
                yield span
                duration = time.time() - start_time
                logfire.info(
                    f"{EMOJI['success']} [{self.name}] Search completed",
                    duration_ms=round(duration * 1000, 2),
                    query=query
                )
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(
                f"{EMOJI['error']} [{self.name}] Search failed: {e}",
                duration_ms=round(duration * 1000, 2),
                query=query,
                error_type=type(e).__name__
            )
            raise
            
    @contextmanager
    def llm_span(self, model: str, prompt_preview: str = None, **attributes):
        """Create a span for LLM operations."""
        start_time = time.time()
        preview = prompt_preview[:100] + "..." if prompt_preview and len(prompt_preview) > 100 else prompt_preview
        logfire.info(f"{EMOJI['llm']} [{self.name}] LLM call to '{model}'", prompt_preview=preview, **attributes)
        
        try:
            with logfire.span(f"llm.{model}", **attributes) as span:
                yield span
                duration = time.time() - start_time
                logfire.info(
                    f"{EMOJI['success']} [{self.name}] LLM response received",
                    model=model,
                    duration_ms=round(duration * 1000, 2)
                )
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(
                f"{EMOJI['error']} [{self.name}] LLM call failed: {e}",
                model=model,
                duration_ms=round(duration * 1000, 2),
                error_type=type(e).__name__
            )
            raise
            
    @contextmanager
    def db_span(self, operation: str, **attributes):
        """Create a span for database operations."""
        start_time = time.time()
        logfire.debug(f"{EMOJI['db']} [{self.name}] DB: {operation}", **attributes)
        
        try:
            with logfire.span(f"db.{operation}", **attributes) as span:
                yield span
                duration = time.time() - start_time
                logfire.debug(
                    f"{EMOJI['success']} [{self.name}] DB operation completed",
                    operation=operation,
                    duration_ms=round(duration * 1000, 2)
                )
        except Exception as e:
            duration = time.time() - start_time
            logfire.error(
                f"{EMOJI['error']} [{self.name}] DB operation failed: {e}",
                operation=operation,
                duration_ms=round(duration * 1000, 2),
                error_type=type(e).__name__
            )
            raise

    def log_request(self, tool_name: str, arguments: dict):
        """Log incoming tool request."""
        # Truncate large values for logging
        truncated_args = {}
        for k, v in arguments.items():
            if isinstance(v, str) and len(v) > 200:
                truncated_args[k] = v[:200] + "..."
            elif isinstance(v, list) and len(v) > 10:
                truncated_args[k] = f"[{len(v)} items]"
            else:
                truncated_args[k] = v
                
        logfire.info(
            f"{EMOJI['http']} [{self.name}] Request: {tool_name}",
            tool=tool_name,
            arguments=truncated_args
        )
        
    def log_response(self, tool_name: str, result: Any, duration_ms: float = None):
        """Log tool response."""
        result_preview = str(result)[:200] if result else "None"
        logfire.info(
            f"{EMOJI['success']} [{self.name}] Response: {tool_name}",
            tool=tool_name,
            result_preview=result_preview,
            duration_ms=duration_ms
        )


def get_logger(name: str) -> MCPLogger:
    """Get an MCPLogger instance for the given name."""
    return MCPLogger(name)


# ═══════════════════════════════════════════════════════════════
# Decorators
# ═══════════════════════════════════════════════════════════════
def log_tool(logger: MCPLogger):
    """Decorator to automatically log MCP tool execution."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            tool_name = func.__name__
            start_time = time.time()
            logger.log_request(tool_name, kwargs)
            
            try:
                with logger.tool_span(tool_name, **{k: str(v)[:100] for k, v in kwargs.items()}):
                    result = await func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    logger.log_response(tool_name, result, duration_ms=round(duration, 2))
                    return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"Tool {tool_name} failed", error=e, duration_ms=round(duration, 2))
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            tool_name = func.__name__
            start_time = time.time()
            logger.log_request(tool_name, kwargs)
            
            try:
                with logger.tool_span(tool_name, **{k: str(v)[:100] for k, v in kwargs.items()}):
                    result = func(*args, **kwargs)
                    duration = (time.time() - start_time) * 1000
                    logger.log_response(tool_name, result, duration_ms=round(duration, 2))
                    return result
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                logger.error(f"Tool {tool_name} failed", error=e, duration_ms=round(duration, 2))
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# Convenience Exports
# ═══════════════════════════════════════════════════════════════
info = logfire.info
debug = logfire.debug
warning = logfire.warn
error = logfire.error
span = logfire.span
