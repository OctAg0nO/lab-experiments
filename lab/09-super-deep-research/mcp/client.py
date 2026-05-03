"""
MCP client — connects to servers (stdio + SSE), discovers tools, bridges async→sync.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client


@dataclass
class _ServerCtx:
    session: ClientSession
    close_coro: Any


class MCPClient:
    """Connects to MCP servers on a background event loop thread."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers: dict[str, _ServerCtx] = {}

    def _run(self, coro) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def connect_all(self) -> list[dict]:
        all_tools: list[dict] = []
        for name, cfg in self.config.get("mcpServers", {}).items():
            transport = cfg.get("type", "stdio")
            try:
                if transport == "sse":
                    tools = self._run(self._connect_sse(name, cfg["url"]))
                else:
                    params = StdioServerParameters(
                        command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"),
                    )
                    tools = self._run(self._connect_stdio(name, params))
                all_tools.extend(tools)
            except Exception as e:
                print(f"  [!] {name}: {e}")
        return all_tools

    async def _connect_stdio(self, name: str, params: StdioServerParameters) -> list[dict]:
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)
        self._servers[name] = _ServerCtx(session=session, close_coro=_close())
        return await self._list_tools(name, session)

    async def _connect_sse(self, name: str, url: str) -> list[dict]:
        ctx = sse_client(url)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()
        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)
        self._servers[name] = _ServerCtx(session=session, close_coro=_close())
        return await self._list_tools(name, session)

    @staticmethod
    async def _list_tools(name: str, session: ClientSession) -> list[dict]:
        result = await session.list_tools()
        return [
            {"server": name, "name": t.name, "description": t.description, "inputSchema": t.inputSchema}
            for t in result.tools
        ]

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> str:
        session = self._servers[server].session
        result = self._run(session.call_tool(tool_name, arguments=arguments))
        parts = []
        for c in result.content:
            if hasattr(c, "text") and c.text:
                parts.append(c.text)
            elif hasattr(c, "resource") and c.resource:
                parts.append(str(c.resource))
            else:
                parts.append(str(c))
        return "\n".join(parts)

    def close(self):
        async def _cleanup():
            for ctx in self._servers.values():
                await ctx.close_coro
        self._run(_cleanup())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def find_tool(self, tool_defs: list[dict], server: str, name: str) -> dict | None:
        return next((t for t in tool_defs if t["server"] == server and t["name"] == name), None)

    def build_tool_fns(self, tool_defs: list[dict]) -> list:
        """Wrap MCP tool defs into callables for DSPy RLM."""
        fns = []
        for td in tool_defs:
            srv, tn, desc = td["server"], td["name"], td.get("description", "")
            def make(srv=srv, tn=tn, desc=desc):
                def fn(**kwargs: Any) -> str:
                    return self.call_tool(srv, tn, kwargs)
                fn.__name__ = tn
                fn.__doc__ = desc
                return fn
            fns.append(make())
        return fns
