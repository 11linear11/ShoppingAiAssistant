"""Pytest bootstrap helpers for local/test environments."""

import sys
import types


try:
    import pydantic_settings  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pydantic import BaseModel

    shim = types.ModuleType("pydantic_settings")

    class BaseSettings(BaseModel):
        """Minimal fallback used only in tests when pydantic-settings is absent."""

        class Config:
            extra = "ignore"

    shim.BaseSettings = BaseSettings
    sys.modules["pydantic_settings"] = shim
