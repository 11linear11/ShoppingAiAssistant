"""
MCP Client for Shopping AI Assistant

Handles communication with MCP servers using the MCP protocol.
Supports session management required by MCP Streamable HTTP transport.
"""

import asyncio
import json
import logging
from typing import Any, Optional
import httpx


logger = logging.getLogger(__name__)


class MCPClient:
    """
    Client for communicating with MCP servers.
    
    MCP servers expect POST requests to /mcp endpoint with JSON-RPC format.
    Handles session initialization and management automatically.
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize MCP client.
        
        Args:
            base_url: Base URL of the MCP server (e.g., http://localhost:5004)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.mcp_url = f"{self.base_url}/mcp"
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._request_id = 0
        self._session_id: Optional[str] = None
        self._initialized = False
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._session_id = None
        self._initialized = False
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _ensure_initialized(self):
        """Ensure MCP session is initialized. Retries if server isn't ready."""
        if self._initialized and self._session_id:
            return
        
        # Reset state for fresh initialization
        self._session_id = None
        self._initialized = False
        
        client = await self._get_client()
        
        # Initialize MCP session
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "shopping-assistant",
                    "version": "1.0"
                }
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        # Retry up to 5 times with backoff (servers may still be loading)
        max_retries = 5
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    self.mcp_url,
                    json=init_request,
                    headers=headers
                )
                
                # Extract session ID from response header
                self._session_id = response.headers.get("mcp-session-id")
                
                if not self._session_id:
                    # Try to extract from SSE response
                    for line in response.text.split("\n"):
                        if "session" in line.lower():
                            logger.debug(f"Session line: {line}")
                
                if response.status_code == 200:
                    self._initialized = True
                    logger.info(f"MCP session initialized: {self._session_id}")
                    return
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(f"MCP init attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_error = str(e)
                logger.warning(f"MCP init attempt {attempt + 1}/{max_retries}: server not ready - {e}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"MCP init attempt {attempt + 1}/{max_retries} error: {e}")
            
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                logger.info(f"Retrying MCP init in {wait}s...")
                await asyncio.sleep(wait)
        
        logger.error(f"Failed to initialize MCP session after {max_retries} attempts: {last_error}")
        raise Exception(f"MCP initialization failed after {max_retries} attempts: {last_error}")
    
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Call an MCP tool.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            
        Returns:
            Tool result as dict
        """
        max_retries = 3
        for attempt in range(max_retries):
            # Ensure session is initialized
            await self._ensure_initialized()
            
            client = await self._get_client()
            
            # MCP JSON-RPC request format
            request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Add session ID if available
            if self._session_id:
                headers["mcp-session-id"] = self._session_id
            
            try:
                response = await client.post(
                    self.mcp_url,
                    json=request,
                    headers=headers,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteError, httpx.RemoteProtocolError) as e:
                logger.warning(
                    f"MCP call transport error on {tool_name} (attempt {attempt + 1}/{max_retries}): {e}"
                )
                self._session_id = None
                self._initialized = False
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                raise Exception(f"MCP transport error while calling {tool_name}: {e}") from e
            
            # Handle session expiration (404) - reset and retry
            if response.status_code == 404 and attempt < max_retries - 1:
                logger.warning(f"MCP session expired (404), re-initializing... (attempt {attempt + 1})")
                self._session_id = None
                self._initialized = False
                continue
            
            # Handle SSE response format (MCP returns SSE)
            if "text/event-stream" in response.headers.get("content-type", ""):
                return self._parse_sse_response(response.text)
            
            if response.status_code == 200:
                return self._parse_json_response(response.text)
            
            raise Exception(f"MCP request failed: HTTP {response.status_code} - {response.text}")
        
        raise Exception("MCP request failed after retries")
    
    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Parse JSON response."""
        result = json.loads(text)
        if "result" in result:
            return self._extract_tool_result(result["result"])
        elif "error" in result:
            raise Exception(f"MCP Error: {result['error']}")
        return result
    
    def _parse_sse_response(self, text: str) -> dict[str, Any]:
        """Parse SSE (Server-Sent Events) response."""
        for line in text.split("\n"):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        parsed = json.loads(data)
                        if "result" in parsed:
                            return self._extract_tool_result(parsed["result"])
                        elif "error" in parsed:
                            raise Exception(f"MCP Error: {parsed['error']}")
                    except json.JSONDecodeError:
                        continue
        return {}
    
    def _extract_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Extract actual result from MCP tool response."""
        content = result.get("content", [])
        if content and isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "{}")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"text": text}
        return result


class InterpretMCPClient(MCPClient):
    """Client for Interpret MCP Server."""
    
    async def interpret_query(
        self,
        query: str,
        session_id: str,
        context: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Interpret a user query.
        
        Args:
            query: User's query text
            session_id: Session identifier
            context: Optional session context
            
        Returns:
            Interpretation result
        """
        return await self.call_tool("interpret_query", {
            "query": query,
            "session_id": session_id,
            "context": context or {}
        })
    
    async def classify_query(self, query: str) -> dict[str, Any]:
        """Classify a query type."""
        return await self.call_tool("classify_query", {"query": query})


class SearchMCPClient(MCPClient):
    """Client for Search MCP Server."""
    
    async def search_products(
        self,
        search_params: dict[str, Any],
        session_id: str,
        use_cache: bool = True,
        use_semantic: bool = False
    ) -> dict[str, Any]:
        """
        Execute full search pipeline.
        
        Args:
            search_params: Search parameters from interpretation
                - intent: browse/find_cheapest/find_best_value
                - product: Product name
                - brand: Brand filter
                - persian_full_query: Full query text
                - categories_fa: Category filters
                - constraints: Price range, etc.
            session_id: Session identifier
            use_cache: Whether to use cache
            use_semantic: Whether to use semantic search
            
        Returns:
            Search results with total_hits, results, took_ms, from_cache
        """
        return await self.call_tool("search_products", {
            "search_params": search_params,
            "session_id": session_id,
            "use_cache": use_cache,
            "use_semantic": use_semantic
        })
    
    async def get_product(self, product_id: str) -> dict[str, Any]:
        """Get product details by ID."""
        return await self.call_tool("get_product", {"product_id": product_id})
    
    async def generate_dsl(
        self,
        search_params: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate Elasticsearch DSL only (no execution)."""
        return await self.call_tool("generate_dsl", {
            "search_params": search_params
        })
    
    async def rerank_results(
        self,
        results: list[dict],
        query: str,
        intent: str = "browse"
    ) -> dict[str, Any]:
        """Rerank search results."""
        return await self.call_tool("rerank_results", {
            "results": results,
            "query": query,
            "intent": intent
        })


class EmbeddingMCPClient(MCPClient):
    """Client for Embedding MCP Server."""
    
    async def generate_embedding(self, text: str) -> dict[str, Any]:
        """Generate embedding for text."""
        return await self.call_tool("generate_embedding", {"text": text})
    
    async def generate_embeddings_batch(self, texts: list[str]) -> dict[str, Any]:
        """Generate embeddings for multiple texts."""
        return await self.call_tool("generate_embeddings_batch", {"texts": texts})
