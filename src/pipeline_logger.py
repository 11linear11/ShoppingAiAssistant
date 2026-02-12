"""
Pipeline Logger

Comprehensive logging system for tracking user queries
through the entire MCP pipeline.

Each query gets a unique trace_id that follows it through:
1. Agent → 2. Interpret → 3. Search → 4. Cache → 5. Response

Optionally integrates with Logfire for cloud-based observability
when USE_LOGFIRE=true is set in environment.
"""

import json
import logging
import os
import re
import sys
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Optional

# ============================================================================
# Logfire Integration (optional, toggle via USE_LOGFIRE env)
# ============================================================================

def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    return cleaned or "shopping-assistant"


PIPELINE_SERVICE_NAME = _slugify(
    os.environ.get("PIPELINE_SERVICE_NAME")
    or os.environ.get("LOGFIRE_SERVICE_NAME", "shopping-assistant")
)
DEBUG_LOG = _env_bool("DEBUG_LOG", _env_bool("DEBUG_MODE", False))
PIPELINE_LOG_LEVEL = os.environ.get(
    "PIPELINE_LOG_LEVEL",
    "DEBUG" if DEBUG_LOG else "INFO",
).upper()
PIPELINE_LOG_TO_FILE = _env_bool("PIPELINE_LOG_TO_FILE", True)
PIPELINE_LOG_MAX_BYTES = _env_int("PIPELINE_LOG_MAX_BYTES", 5_000_000)
PIPELINE_LOG_BACKUP_COUNT = _env_int("PIPELINE_LOG_BACKUP_COUNT", 3)
PIPELINE_LOG_DIR = Path(
    os.environ.get("PIPELINE_LOG_DIR", str(Path(__file__).parent.parent / "logs"))
)
PIPELINE_LOG_FILE_NAME = os.environ.get(
    "PIPELINE_LOG_FILE_NAME",
    f"pipeline-{PIPELINE_SERVICE_NAME}.log",
)
PIPELINE_LOG_FILE = PIPELINE_LOG_DIR / PIPELINE_LOG_FILE_NAME
USE_LOGFIRE = _env_bool("USE_LOGFIRE", False)
_logfire_configured = False

if USE_LOGFIRE:
    try:
        import logfire

        _logfire_token = os.environ.get("LOGFIRE_TOKEN", "")
        if not _logfire_token:
            raise ValueError("LOGFIRE_TOKEN environment variable is not set")

        logfire.configure(
            token=_logfire_token,
            service_name=PIPELINE_SERVICE_NAME,
            service_version="3.0.0",
            environment=os.environ.get("LOGFIRE_ENVIRONMENT", "production"),
            console=False,  # We have our own console handler
        )
        _logfire_configured = True
    except Exception as e:
        USE_LOGFIRE = False

# ============================================================================
# Pipeline Logger Setup
# ============================================================================

class PipelineFormatter(logging.Formatter):
    """Custom formatter for pipeline logs with colors and structure."""
    
    COLORS = {
        'AGENT': '\033[94m',      # Blue
        'INTERPRET': '\033[95m',  # Magenta
        'SEARCH': '\033[92m',     # Green
        'CACHE': '\033[93m',      # Yellow
        'EMBED': '\033[96m',      # Cyan
        'ERROR': '\033[91m',      # Red
        'RESET': '\033[0m',
    }
    
    ICONS = {
        'AGENT': '🤖',
        'INTERPRET': '🎯',
        'SEARCH': '🔍',
        'CACHE': '💾',
        'EMBED': '🧠',
        'START': '▶️',
        'END': '✅',
        'ERROR': '❌',
        'DATA': '📊',
    }
    
    def format(self, record):
        # Add icons based on stage
        stage = getattr(record, 'stage', 'AGENT')
        icon = self.ICONS.get(stage, '📋')
        
        # Format the message
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]
        trace_id = getattr(record, 'trace_id', '--------')[:8]
        
        # Build structured message
        msg = f"{timestamp} │ {trace_id} │ {icon} {stage:10} │ {record.getMessage()}"
        
        return msg


def setup_pipeline_logger() -> logging.Logger:
    """Setup dedicated pipeline logger."""
    logger = logging.getLogger("pipeline")
    if logger.handlers:
        return logger

    level = getattr(logging, PIPELINE_LOG_LEVEL, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(PipelineFormatter())
    logger.addHandler(console_handler)
    
    # Optional file handler (avoid shared-file writes in multi-process deployments)
    if PIPELINE_LOG_TO_FILE:
        PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            PIPELINE_LOG_FILE,
            maxBytes=PIPELINE_LOG_MAX_BYTES,
            backupCount=PIPELINE_LOG_BACKUP_COUNT,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(PipelineFormatter())
        logger.addHandler(file_handler)
    
    # Logfire handler (sends logs to Logfire cloud when enabled)
    if USE_LOGFIRE and _logfire_configured:
        try:
            logfire_handler = logfire.LogfireLoggingHandler()
            logfire_handler.setLevel(logging.INFO)
            logger.addHandler(logfire_handler)
        except Exception:
            pass
    
    return logger


# Global pipeline logger
pipeline_logger = setup_pipeline_logger()


# ============================================================================
# Trace Context
# ============================================================================

class TraceContext:
    """Context for tracking a single query through the pipeline."""
    
    def __init__(self, query: str, session_id: str = ""):
        self.trace_id = str(uuid.uuid4())
        self.query = query
        self.session_id = session_id
        self.start_time = time.time()
        self.stages: list[dict] = []
        self.current_stage: Optional[str] = None
        
    def elapsed_ms(self) -> int:
        return int((time.time() - self.start_time) * 1000)
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "session_id": self.session_id,
            "total_ms": self.elapsed_ms(),
            "stages": self.stages,
        }


# Context-local storage for trace context (safe for async concurrency)
_current_trace: ContextVar[Optional[TraceContext]] = ContextVar(
    "pipeline_current_trace",
    default=None,
)


def get_current_trace() -> Optional[TraceContext]:
    """Get the current trace context."""
    return _current_trace.get()


def set_current_trace(trace: Optional[TraceContext]):
    """Set the current trace context."""
    return _current_trace.set(trace)


def reset_current_trace(token):
    """Reset trace context to previous value."""
    _current_trace.reset(token)


# ============================================================================
# Logging Functions
# ============================================================================

def log_pipeline(
    stage: str,
    message: str,
    data: Optional[dict] = None,
    level: int = logging.INFO,
    trace_id: Optional[str] = None,
    exc_info: Any = None,
):
    """
    Log a pipeline event.
    
    Args:
        stage: Pipeline stage (AGENT, INTERPRET, SEARCH, CACHE, EMBED)
        message: Log message
        data: Optional structured data
        level: Log level
        trace_id: Optional trace ID (uses current trace if not provided)
    """
    trace = get_current_trace()
    tid = trace_id or (trace.trace_id if trace else "no-trace")

    # In non-debug mode keep only user requests, latency summaries, and errors.
    if not DEBUG_LOG:
        is_user_request = stage == "AGENT" and message.startswith("USER_REQUEST")
        is_latency_summary = message.startswith("LATENCY_SUMMARY")
        if level < logging.ERROR and not is_user_request and not is_latency_summary:
            return
    
    extra = {
        'stage': stage,
        'trace_id': tid,
    }
    
    # Format message with data if provided
    if data:
        # Truncate long values for readability
        truncated_data = _truncate_data(data)
        message = f"{message} | {json.dumps(truncated_data, ensure_ascii=False, default=str)}"
    
    pipeline_logger.log(level, message, extra=extra, exc_info=exc_info)


def log_user_request(query: str, session_id: str, source: str = "agent"):
    """Persist user requests for offline training dataset collection."""
    log_pipeline(
        "AGENT",
        "USER_REQUEST",
        {"query": query, "session": session_id, "source": source},
        level=logging.INFO,
    )


def _truncate_data(data: dict, max_len: int = 100) -> dict:
    """Truncate long string values in data dict."""
    redacted_keys = {"token", "api_key", "authorization", "password", "secret"}
    result = {}
    for k, v in data.items():
        if str(k).lower() in redacted_keys:
            result[k] = "***REDACTED***"
        elif isinstance(v, str) and len(v) > max_len:
            result[k] = v[:max_len] + "..."
        elif isinstance(v, list) and len(v) > 5:
            result[k] = f"[{len(v)} items]"
        elif isinstance(v, dict):
            result[k] = _truncate_data(v, max_len)
        else:
            result[k] = v
    return result


# ============================================================================
# Stage Decorators
# ============================================================================

@contextmanager
def trace_query(query: str, session_id: str = ""):
    """
    Context manager to trace a query through the entire pipeline.
    
    Usage:
        with trace_query("جوراب مردانه", "session123") as trace:
            # ... process query ...
    """
    trace = TraceContext(query, session_id)
    token = set_current_trace(trace)

    log_user_request(query=query, session_id=session_id, source="agent")
    if DEBUG_LOG:
        log_pipeline(
            "AGENT",
            f"═══ NEW QUERY ═══",
            {"query": query, "session": session_id},
        )
    
    try:
        yield trace
    finally:
        if DEBUG_LOG:
            log_pipeline(
                "AGENT",
                f"═══ QUERY COMPLETE ({trace.elapsed_ms()}ms) ═══",
                {"total_stages": len(trace.stages)},
            )
        reset_current_trace(token)


@contextmanager  
def trace_stage(stage: str, description: str = ""):
    """
    Context manager to trace a pipeline stage.
    
    Usage:
        with trace_stage("INTERPRET", "Classifying query"):
            # ... interpret query ...
    """
    trace = get_current_trace()
    start_time = time.time()
    
    log_pipeline(stage, f"▶ START: {description}")
    
    stage_data = {
        "stage": stage,
        "description": description,
        "start_time": datetime.now().isoformat(),
    }
    
    try:
        yield
        elapsed = int((time.time() - start_time) * 1000)
        stage_data["elapsed_ms"] = elapsed
        stage_data["success"] = True
        log_pipeline(stage, f"✓ END: {description} ({elapsed}ms)")
        
    except Exception as e:
        elapsed = int((time.time() - start_time) * 1000)
        stage_data["elapsed_ms"] = elapsed
        stage_data["success"] = False
        stage_data["error"] = str(e)
        log_pipeline(stage, f"✗ FAILED: {description} - {e}", level=logging.ERROR, exc_info=e)
        raise
        
    finally:
        if trace:
            trace.stages.append(stage_data)


def log_stage(stage: str):
    """
    Decorator to log a function as a pipeline stage.
    
    Usage:
        @log_stage("INTERPRET")
        async def interpret_query(query: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_stage(stage, func.__name__):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_stage(stage, func.__name__):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ============================================================================
# Convenience Functions for Each Stage
# ============================================================================

def log_agent(message: str, data: Optional[dict] = None):
    """Log agent-level event."""
    log_pipeline("AGENT", message, data)


def log_interpret(message: str, data: Optional[dict] = None):
    """Log interpret stage event."""
    log_pipeline("INTERPRET", message, data)


def log_search(message: str, data: Optional[dict] = None):
    """Log search stage event."""
    log_pipeline("SEARCH", message, data)


def log_cache(message: str, data: Optional[dict] = None):
    """Log cache stage event."""
    log_pipeline("CACHE", message, data)


def log_embed(message: str, data: Optional[dict] = None):
    """Log embedding stage event."""
    log_pipeline("EMBED", message, data)


def log_error(stage: str, message: str, error: Optional[Exception] = None):
    """Log an error."""
    data = (
        {"error": str(error), "error_type": error.__class__.__name__}
        if error
        else None
    )
    log_pipeline(stage, f"❌ {message}", data, level=logging.ERROR, exc_info=error)


def log_latency_summary(
    stage: str,
    component: str,
    total_ms: int,
    breakdown_ms: Optional[dict[str, int]] = None,
    meta: Optional[dict[str, Any]] = None,
):
    """Log one compact latency summary event for downstream analysis."""
    payload: dict[str, Any] = {
        "component": component,
        "total_ms": int(total_ms),
    }
    if breakdown_ms:
        payload["breakdown_ms"] = {
            key: int(value) for key, value in breakdown_ms.items()
        }
    if meta:
        payload["meta"] = meta

    log_pipeline(
        stage,
        "LATENCY_SUMMARY",
        payload,
        level=logging.INFO,
    )


# ============================================================================
# Query Summary
# ============================================================================

def log_query_summary(
    query: str,
    query_type: str,
    product: Optional[str],
    results_count: int,
    from_cache: bool,
    total_ms: int,
):
    """Log a summary of the query processing."""
    log_pipeline(
        "AGENT",
        "📊 QUERY SUMMARY",
        {
            "query": query,
            "type": query_type,
            "product": product,
            "results": results_count,
            "cached": from_cache,
            "time_ms": total_ms,
        },
    )
