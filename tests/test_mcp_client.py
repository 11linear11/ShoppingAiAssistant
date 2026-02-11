import asyncio
import json
from unittest.mock import AsyncMock

from src.mcp_client import MCPClient


def _run(coro):
    return asyncio.run(coro)


def test_extract_tool_result_parses_json_text_payload():
    client = MCPClient("http://localhost:5004")
    payload = {
        "content": [
            {"type": "text", "text": json.dumps({"ok": True, "value": 42})}
        ]
    }
    result = client._extract_tool_result(payload)
    assert result == {"ok": True, "value": 42}


def test_extract_tool_result_falls_back_to_raw_text():
    client = MCPClient("http://localhost:5004")
    payload = {
        "content": [
            {"type": "text", "text": "plain text"}
        ]
    }
    result = client._extract_tool_result(payload)
    assert result == {"text": "plain text"}


def test_parse_sse_response_reads_data_frame():
    client = MCPClient("http://localhost:5004")
    frame = 'event: message\ndata: {"result": {"content": [{"type": "text", "text": "{\\"a\\": 1}"}]}}\n\n'
    result = client._parse_sse_response(frame)
    assert result == {"a": 1}


def test_call_tool_retries_after_404_session_expired():
    client = MCPClient("http://localhost:5004")

    client._ensure_initialized = AsyncMock(return_value=None)

    resp_404 = AsyncMock()
    resp_404.status_code = 404
    resp_404.headers = {"content-type": "application/json"}
    resp_404.text = "not found"

    resp_200 = AsyncMock()
    resp_200.status_code = 200
    resp_200.headers = {"content-type": "application/json"}
    resp_200.text = '{"result": {"content": [{"type":"text","text":"{\\"ok\\": true}"}]}}'

    fake_http_client = AsyncMock()
    fake_http_client.post = AsyncMock(side_effect=[resp_404, resp_200])

    client._get_client = AsyncMock(return_value=fake_http_client)

    result = _run(client.call_tool("interpret_query", {"query": "x"}))
    assert result["ok"] is True
    assert fake_http_client.post.call_count == 2
