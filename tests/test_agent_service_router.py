import asyncio
from unittest.mock import AsyncMock

from backend.services import agent_service as agent_service_module
from backend.services.agent_service import AgentService


def _run(coro):
    return asyncio.run(coro)


class FakeRouterAgent:
    def __init__(self, interpret_result: dict, search_result: dict | None = None):
        self._interpret_result = interpret_result
        self._search_result = search_result or {}
        self.chat_called = False

    async def interpret_message(self, message: str, session_id: str, context=None):
        return self._interpret_result

    async def search_from_params(self, search_params: dict, session_id: str, use_cache=None):
        return self._search_result

    async def chat(self, message: str, session_id: str):
        self.chat_called = True
        return "fallback", session_id


def test_router_abstract_fastpath_skips_agent_chat(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "abstract",
            "searchable": False,
            "clarification": {
                "question": "چه نوع هدیه‌ای می‌خواید؟",
                "suggestions": [{"product": "ساعت"}, {"product": "عطر"}],
            },
        }
    )
    service._cache = AsyncMock()
    service._cache.available = False

    monkeypatch.setattr(agent_service_module.settings, "ff_router_enabled", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_abstract_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_direct_fastpath", False, raising=False)

    result = _run(service.chat("یه چیزی می‌خوام هدیه بدم", session_id="sess-a"))
    assert result["success"] is True
    assert result["metadata"]["query_type"] == "abstract"
    assert result["products"] == []
    assert "پیشنهادها" in result["response"]
    assert service._agent.chat_called is False


def test_router_direct_fastpath_uses_search_and_sets_cache(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "شلوار مردانه",
                "intent": "browse",
                "persian_full_query": "شلوار مردانه",
            },
        },
        search_result={
            "results": [
                {
                    "id": "p1",
                    "product_name": "شلوار مردانه نخی",
                    "brand_name": "X",
                    "price": 1000000,
                    "discount_price": 900000,
                    "has_discount": True,
                    "discount_percentage": 10,
                    "product_url": "https://example.com/p1",
                }
            ]
        },
    )
    service._cache = AsyncMock()
    service._cache.available = True
    service._cache.get = AsyncMock(return_value=None)
    service._cache.set = AsyncMock(return_value=True)

    monkeypatch.setattr(agent_service_module.settings, "ff_router_enabled", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_abstract_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_direct_fastpath", True, raising=False)

    result = _run(service.chat("شلوار مردانه", session_id="sess-d"))
    assert result["success"] is True
    assert result["metadata"]["query_type"] == "direct"
    assert len(result["products"]) == 1
    assert service._cache.set.await_count == 1
    assert service._agent.chat_called is False
