import asyncio
from unittest.mock import AsyncMock

from backend.services.agent_service import AgentService


def _run(coro):
    return asyncio.run(coro)


def test_extract_products_and_clean_response_from_json_block():
    service = AgentService()
    response = (
        "این محصولات رو پیدا کردم:\n"
        "```json\n"
        "[{\"name\":\"شامپو\",\"brand\":\"X\",\"price\":\"1,200\",\"has_discount\":true,\"discount_percentage\":10}]\n"
        "```"
    )

    products = service._extract_products(response)
    assert len(products) == 1
    assert products[0]["name"] == "شامپو"

    clean = service._clean_response_text(response, products)
    assert "```json" not in clean
    assert "این محصولات" in clean


def test_extract_products_from_raw_array_and_detect_types():
    service = AgentService()
    response = (
        "نتیجه:\n"
        "[{\"name\":\"لپتاپ\",\"brand\":\"A\",\"price\":25000000,\"has_discount\":false,\"discount_percentage\":0}]"
    )

    products = service._extract_products(response)
    assert len(products) == 1

    query_type = service._detect_query_type(response, products)
    # Detection is now data-driven: if products exist, treat as direct.
    assert query_type == "direct"


def test_chat_returns_cached_result_when_agent_cache_hits():
    service = AgentService()

    service._initialized = True
    service._agent = AsyncMock()

    cache_payload = {
        "success": True,
        "response": "cached",
        "session_id": "will-be-replaced",
        "products": [],
        "metadata": {"took_ms": 10, "from_agent_cache": True},
    }

    fake_cache = AsyncMock()
    fake_cache.available = True
    fake_cache.get = AsyncMock(return_value=cache_payload)
    service._cache = fake_cache

    result = _run(service.chat("سلام"))
    assert result["success"] is True
    assert result["response"] == "cached"
    assert result["session_id"]
    assert result["metadata"]["from_agent_cache"] is True


def test_chat_timeout_path():
    service = AgentService()
    service._initialized = True

    async def slow_chat(message, session_id):
        await asyncio.sleep(0.05)
        return "ok", session_id

    service._agent = AsyncMock()
    service._agent.chat = slow_chat

    service._cache = AsyncMock()
    service._cache.available = False

    result = _run(service.chat("slow", timeout=0.001))
    assert result["success"] is False
    assert result["metadata"]["query_type"] == "timeout"


def test_clean_response_removes_tool_prefixes():
    service = AgentService()
    clean = service._clean_response_text(
        "❓ NEED_CLARIFICATION:لطفا دقیق‌تر بگید دنبال چه محصولی هستید",
        [],
    )
    assert clean == "لطفا دقیق‌تر بگید دنبال چه محصولی هستید"


def test_chat_agent_error_payload_contains_stage_and_type():
    service = AgentService()
    service._initialized = True

    async def failing_chat(message, session_id):
        return (
            '__AGENT_ERROR__:{"stage":"agent.chat","error_type":"RuntimeError","message":"boom"}',
            session_id,
        )

    service._agent = AsyncMock()
    service._agent.chat = failing_chat
    service._cache = AsyncMock()
    service._cache.available = False

    result = _run(service.chat("test"))
    assert result["success"] is False
    assert result["metadata"]["query_type"] == "error"
    assert result["metadata"]["error_stage"] == "agent.chat"
    assert result["metadata"]["error_type"] == "RuntimeError"


def test_chat_success_path_extracts_products_and_metadata():
    service = AgentService()
    service._initialized = True

    async def fast_chat(message, session_id):
        return (
            "این رو پیدا کردم:\n```json\n[{\"name\":\"کفش\",\"brand\":\"Y\",\"price\":500000}]\n```",
            session_id,
        )

    service._agent = AsyncMock()
    service._agent.chat = fast_chat

    service._cache = AsyncMock()
    service._cache.available = False

    result = _run(service.chat("کفش"))
    assert result["success"] is True
    assert len(result["products"]) == 1
    assert "```json" not in result["response"]
    assert result["metadata"]["query_type"] in {"direct", "unknown"}
