"""
Logging Configuration

Centralized logging setup with Logfire integration
for the Shopping AI Assistant.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings

# Try to import logfire, but make it optional
try:
    import logfire

    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    logfire_token: str = ""
    logfire_service_name: str = "shopping-assistant"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    log_max_bytes: int = 10_000_000  # 10MB
    log_backup_count: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_prefix = ""
        extra = "ignore"


settings = LoggingSettings()


def setup_logging(
    service_name: Optional[str] = None,
    level: Optional[str] = None,
    force: bool = True,
) -> logging.Logger:
    """
    Setup logging with optional Logfire integration.

    Args:
        service_name: Name of the service (for Logfire)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        force: Force reconfigure root handlers (recommended for app entrypoints)

    Returns:
        Configured logger instance
    """
    service_name = service_name or settings.logfire_service_name
    level = level or settings.log_level

    # Configure log format
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
    ]

    # Add file handler if enabled
    if settings.log_to_file:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Main log file
        log_file = log_dir / f"{service_name}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
        
        # Error log file (only ERROR and above)
        error_log_file = log_dir / f"{service_name}.error.log"
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        error_handler.setFormatter(logging.Formatter(log_format))
        error_handler.setLevel(logging.ERROR)
        handlers.append(error_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        force=force,
    )

    logger = logging.getLogger(service_name)

    # Setup Logfire if available and configured
    if LOGFIRE_AVAILABLE and settings.logfire_token:
        try:
            logfire.configure(
                token=settings.logfire_token,
                service_name=service_name,
            )
            logger.info("✅ Logfire configured successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to configure Logfire: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"{settings.logfire_service_name}.{name}")
