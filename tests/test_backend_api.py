import asyncio

from backend.api.routes import chat
from backend.api.schemas import ChatRequest


class FakeAgentService:
    async def chat(self, message: str, session_id: str | None, timeout: int):
        return {
            "success": True,
            "response": "پاسخ تست",
            "session_id": session_id or "generated-session",
            "products": [
                {
                    "id": "1",
                    "name": "شامپو",
                    "brand": "X",
                    "price": 1000,
                    "discount_price": None,
                    "has_discount": False,
                    "discount_percentage": 0,
                    "product_url": "",
                }
            ],
            "metadata": {
                "took_ms": 12,
                "query_type": "direct",
                "total_results": 1,
                "from_agent_cache": False,
            },
        }


async def _call_chat_route_direct():
    req = ChatRequest(message="سلام", session_id="sess-1")
    res = await chat(request=req, agent=FakeAgentService())
    return res


def test_chat_route_direct_call_returns_typed_response():
    response = asyncio.run(_call_chat_route_direct())
    assert response.success is True
    assert response.products[0].name == "شامپو"
    assert response.metadata.query_type == "direct"


def test_chat_route_generates_session_when_missing():
    req = ChatRequest(message="سلام", session_id=None)
    res = asyncio.run(chat(request=req, agent=FakeAgentService()))
    assert res.success is True
    assert res.session_id == "generated-session"
    assert len(res.products) == 1
