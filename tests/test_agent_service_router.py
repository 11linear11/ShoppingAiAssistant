import asyncio
from unittest.mock import AsyncMock

from backend.services import agent_service as agent_service_module
from backend.services.agent_service import AgentService


def _run(coro):
    return asyncio.run(coro)


class FakeRouterAgent:
    def __init__(
        self,
        interpret_result: dict,
        search_result: dict | None = None,
        final_result: dict | None = None,
    ):
        self._interpret_result = interpret_result
        self._search_result = search_result or {}
        self._final_result = final_result
        self.chat_called = False
        self.final_called = False

    async def interpret_message(self, message: str, session_id: str, context=None):
        return self._interpret_result

    async def search_from_params(self, search_params: dict, session_id: str, use_cache=None):
        return self._search_result

    async def final_rerank_direct(self, user_message: str, search_params: dict, search_rows: list[dict], top_n: int = 8):
        self.final_called = True
        if self._final_result is None:
            raise RuntimeError("final llm unavailable")
        return self._final_result

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
                    "category_name": "پوشاک مردانه",
                    "price": 1000000,
                    "discount_price": 900000,
                    "has_discount": True,
                    "discount_percentage": 10,
                    "product_url": "https://example.com/p1",
                    "relevancy_score": 0.92,
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
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 100, raising=False)

    result = _run(service.chat("شلوار مردانه", session_id="sess-d"))
    assert result["success"] is True
    assert result["metadata"]["query_type"] == "direct"
    assert len(result["products"]) == 1
    assert service._cache.set.await_count == 1
    assert service._agent.chat_called is False


def test_router_direct_fastpath_low_confidence_falls_back_to_agent(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "گوشی",
                "intent": "browse",
                "persian_full_query": "گوشی",
            },
        },
        search_result={
            "results": [
                {
                    "id": "a1",
                    "product_name": "کیف لپ تاپ",
                    "brand_name": "Y",
                    "category_name": "کیف و کاور",
                    "price": 800000,
                    "relevancy_score": 0.22,
                },
                {
                    "id": "a2",
                    "product_name": "هدست بی سیم",
                    "brand_name": "Z",
                    "category_name": "صوتی",
                    "price": 1200000,
                    "relevancy_score": 0.21,
                },
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
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t1", 0.55, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t2", 0.08, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_min_confidence", 0.58, raising=False)

    result = _run(service.chat("گوشی", session_id="sess-fallback"))
    assert result["success"] is True
    assert result["response"] == "fallback"
    assert service._agent.chat_called is True
    assert service._cache.set.await_count == 0


def test_router_direct_fastpath_can_skip_fallback_when_flag_disabled(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "گوشی",
                "intent": "browse",
                "persian_full_query": "گوشی",
            },
        },
        search_result={
            "results": [
                {
                    "id": "a1",
                    "product_name": "کیف لپ تاپ",
                    "brand_name": "Y",
                    "category_name": "کیف و کاور",
                    "price": 800000,
                    "relevancy_score": 0.22,
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
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", False, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t1", 0.55, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t2", 0.08, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_min_confidence", 0.58, raising=False)

    result = _run(service.chat("گوشی", session_id="sess-no-fallback"))
    assert result["success"] is True
    assert result["metadata"]["query_type"] in {"direct", "no_results"}
    assert service._agent.chat_called is False


def test_router_direct_fastpath_low_confidence_uses_final_llm(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "گوشی",
                "intent": "browse",
                "persian_full_query": "گوشی",
            },
        },
        search_result={
            "results": [
                {
                    "id": "p1",
                    "product_name": "گوشی موبایل الف",
                    "brand_name": "A",
                    "category_name": "موبایل",
                    "price": 10000000,
                    "relevancy_score": 0.62,
                },
                {
                    "id": "p2",
                    "product_name": "نگهدارنده گوشی",
                    "brand_name": "B",
                    "category_name": "اکسسوری",
                    "price": 300000,
                    "relevancy_score": 0.61,
                },
            ]
        },
        final_result={
            "response": "چند گزینه مرتبط پیدا کردم:",
            "rows": [
                {
                    "id": "p1",
                    "product_name": "گوشی موبایل الف",
                    "brand_name": "A",
                    "category_name": "موبایل",
                    "price": 10000000,
                    "discount_price": 9500000,
                    "has_discount": True,
                    "discount_percentage": 5,
                    "product_url": "https://example.com/p1",
                }
            ],
            "meta": {"mode": "llm_selected", "selected_count": 1},
        },
    )
    service._cache = AsyncMock()
    service._cache.available = True
    service._cache.get = AsyncMock(return_value=None)
    service._cache.set = AsyncMock(return_value=True)

    monkeypatch.setattr(agent_service_module.settings, "ff_router_enabled", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_abstract_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_direct_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t1", 0.55, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t2", 0.08, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_min_confidence", 0.58, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_timeout_seconds", 5, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_top_n", 8, raising=False)

    result = _run(service.chat("گوشی", session_id="sess-final-llm"))
    assert result["success"] is True
    assert result["metadata"]["router_route"] == "direct_final_llm"
    assert result["response"] == "چند گزینه مرتبط پیدا کردم:"
    assert len(result["products"]) == 1
    assert service._agent.final_called is True
    assert service._agent.chat_called is False


def test_router_direct_fastpath_rollout_zero_forces_fallback(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "شلوار",
                "intent": "browse",
                "persian_full_query": "شلوار",
            },
        },
        search_result={
            "results": [
                {"id": "p1", "product_name": "شلوار مردانه", "price": 1000000, "relevancy_score": 0.9}
            ]
        },
        final_result={"response": "ok", "rows": [], "meta": {}},
    )
    service._cache = AsyncMock()
    service._cache.available = True
    service._cache.get = AsyncMock(return_value=None)
    service._cache.set = AsyncMock(return_value=True)

    monkeypatch.setattr(agent_service_module.settings, "ff_router_enabled", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_abstract_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_direct_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 0, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 100, raising=False)

    result = _run(service.chat("شلوار", session_id="sess-rollout-direct"))
    assert result["success"] is True
    assert result["response"] == "fallback"
    assert service._agent.chat_called is True
    assert service._agent.final_called is False


def test_router_final_llm_rollout_zero_forces_legacy_fallback(monkeypatch):
    service = AgentService()
    service._initialized = True
    service._agent = FakeRouterAgent(
        interpret_result={
            "query_type": "direct",
            "searchable": True,
            "search_params": {
                "product": "گوشی",
                "intent": "browse",
                "persian_full_query": "گوشی",
            },
        },
        search_result={
            "results": [
                {"id": "p1", "product_name": "گوشی موبایل", "price": 9000000, "relevancy_score": 0.62},
                {"id": "p2", "product_name": "کیف لپ تاپ", "price": 900000, "relevancy_score": 0.61},
            ]
        },
        final_result={
            "response": "باید استفاده نشود",
            "rows": [{"id": "p1", "product_name": "گوشی موبایل", "price": 9000000}],
            "meta": {},
        },
    )
    service._cache = AsyncMock()
    service._cache.available = True
    service._cache.get = AsyncMock(return_value=None)
    service._cache.set = AsyncMock(return_value=True)

    monkeypatch.setattr(agent_service_module.settings, "ff_router_enabled", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_abstract_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_direct_fastpath", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "ff_conditional_final_llm", True, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "direct_fastpath_rollout_percent", 100, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "final_llm_rollout_percent", 0, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t1", 0.55, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_t2", 0.08, raising=False)
    monkeypatch.setattr(agent_service_module.settings, "router_guard_min_confidence", 0.58, raising=False)

    result = _run(service.chat("گوشی", session_id="sess-rollout-final"))
    assert result["success"] is True
    assert result["response"] == "fallback"
    assert service._agent.chat_called is True
    assert service._agent.final_called is False
