"""
MCP client — connects to servers (stdio + SSE), discovers tools/resources/prompts,
bridges async→sync via background event-loop thread.

Consolidated from labs 09 (proper cleanup, _ServerCtx, build_tool_fns, find_tool),
10/11/12 (simplified session management), with protocol extensions for
Resources, Prompts, and Sampling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server context — holds session + cleanup coroutine
# ---------------------------------------------------------------------------

@dataclass
class _ServerCtx:
    """Holds an MCP session and its cleanup coroutine."""
    session: ClientSession
    close_coro: Any
    server_name: str = ""


# ---------------------------------------------------------------------------
# Config model — typed view into mcpServers JSON
# ---------------------------------------------------------------------------

@dataclass
class MCPServerConfig:
    """Parsed config for a single MCP server entry."""
    name: str
    description: str = ""
    enabled: bool = True
    transport: str = "stdio"  # "stdio" | "sse"
    command: str = ""
    args: list[str] | None = None
    url: str = ""
    env: dict[str, str] | None = None
    timeout: int = 60  # tool call timeout in seconds
    max_retries: int = 0

    @classmethod
    def from_dict(cls, name: str, cfg: dict) -> MCPServerConfig:
        return cls(
            name=name,
            description=cfg.get("description", ""),
            enabled=cfg.get("enabled", True),
            transport=cfg.get("type", "stdio"),
            command=cfg.get("command", ""),
            args=cfg.get("args", []),
            url=cfg.get("url", ""),
            env=cfg.get("env"),
            timeout=cfg.get("timeout", 60),
            max_retries=cfg.get("max_retries", 0),
        )


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------

class MCPClient:
    """Async-to-sync MCP transport bridge.

    Connects to MCP servers (stdio/sse) from a background event-loop thread
    and exposes a synchronous interface.  The background thread is a daemon
    — its subprocess children are reaped by the OS on process exit.

    Usage::

        client = MCPClient("config/mcp_servers.json")
        tool_defs = client.connect_all()
        result = client.call_tool("crawl4ai", "md", {"url": "https://dspy.ai"})
        client.close()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers: dict[str, _ServerCtx] = {}

    # ------------------------------------------------------------------
    # Async → sync bridge
    # ------------------------------------------------------------------

    def _run(self, coro, timeout: float | None = None) -> Any:
        """Run a coroutine on the background event loop and wait for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect_all(self) -> list[dict]:
        """Connect to all enabled MCP servers.

        Returns a flat list of tool definitions::

            [{"server": str, "name": str, "description": str, "inputSchema": dict}, ...]
        """
        all_tools: list[dict] = []
        for name, cfg in self.config.get("mcpServers", {}).items():
            server_cfg = MCPServerConfig.from_dict(name, cfg)
            if not server_cfg.enabled:
                logger.info("  [-] %s: disabled", name)
                continue
            try:
                if server_cfg.transport == "sse":
                    tools = self._run(self._connect_sse(server_cfg))
                else:
                    tools = self._run(self._connect_stdio(server_cfg))
                all_tools.extend(tools)
                logger.info("  [+] %s: %d tool(s)", name, len(tools))
            except Exception as e:
                logger.warning("  [!] %s: %s", name, e)
        return all_tools

    def connect_server(self, server_name: str) -> list[dict] | None:
        """Connect to a single named server (lazy-load)."""
        cfg = self.config.get("mcpServers", {}).get(server_name)
        if cfg is None:
            logger.warning("Server '%s' not found in config", server_name)
            return None
        server_cfg = MCPServerConfig.from_dict(server_name, cfg)
        if not server_cfg.enabled:
            logger.info("  [-] %s: disabled", server_name)
            return None
        try:
            if server_cfg.transport == "sse":
                return self._run(self._connect_sse(server_cfg))
            else:
                return self._run(self._connect_stdio(server_cfg))
        except Exception as e:
            logger.warning("  [!] %s: %s", server_name, e)
            return None

    async def _connect_stdio(self, cfg: MCPServerConfig) -> list[dict]:
        params = StdioServerParameters(
            command=cfg.command,
            args=cfg.args or [],
            env=cfg.env,
        )
        ctx = stdio_client(params)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()

        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)

        self._servers[cfg.name] = _ServerCtx(session=session, close_coro=_close(), server_name=cfg.name)
        return await self._list_tools(cfg.name, session)

    async def _connect_sse(self, cfg: MCPServerConfig) -> list[dict]:
        ctx = sse_client(cfg.url)
        read, write = await ctx.__aenter__()
        session = await ClientSession(read, write).__aenter__()
        await session.initialize()

        async def _close():
            await session.__aexit__(None, None, None)
            await ctx.__aexit__(None, None, None)

        self._servers[cfg.name] = _ServerCtx(session=session, close_coro=_close(), server_name=cfg.name)
        return await self._list_tools(cfg.name, session)

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    @staticmethod
    async def _list_tools(server_name: str, session: ClientSession) -> list[dict]:
        result = await session.list_tools()
        return [
            {"server": server_name, "name": t.name,
             "description": t.description, "inputSchema": t.inputSchema}
            for t in result.tools
        ]

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return string content.

        Handles text, resource, and unknown content types.
        """
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected. Call connect_all() first.")
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

    # ------------------------------------------------------------------
    # Resources (MCP protocol extension)
    # ------------------------------------------------------------------

    def list_resources(self, server: str) -> list[dict]:
        """List available resources from a server."""
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        result = self._run(session.list_resources())
        return [
            {"uri": r.uri, "name": r.name, "description": r.description,
             "mimeType": r.mimeType}
            for r in result.resources
        ]

    def read_resource(self, server: str, uri: str) -> str:
        """Read a resource by URI from a server."""
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        result = self._run(session.read_resource(uri))
        parts = []
        for c in result.contents:
            if hasattr(c, "text") and c.text:
                parts.append(c.text)
            else:
                parts.append(str(c))
        return "\n".join(parts)

    def subscribe_resource(self, server: str, uri: str) -> bool:
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        self._run(session.subscribe(uri))
        return True

    # ------------------------------------------------------------------
    # Prompts (MCP protocol extension)
    # ------------------------------------------------------------------

    def list_prompts(self, server: str) -> list[dict]:
        """List available prompt templates from a server."""
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        result = self._run(session.list_prompts())
        return [
            {"name": p.name, "description": p.description,
             "arguments": [{"name": a.name, "description": a.description,
                            "required": a.required} for a in (p.arguments or [])]}
            for p in result.prompts
        ]

    def get_prompt(self, server: str, name: str, arguments: dict | None = None) -> str:
        """Get a prompt template with arguments filled in."""
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        result = self._run(session.get_prompt(name, arguments=arguments or {}))
        parts = []
        for m in result.messages:
            if hasattr(m, "content") and hasattr(m.content, "text"):
                parts.append(m.content.text)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Sampling (MCP protocol extension — server → client requests)
    # ------------------------------------------------------------------

    def create_message(self, server: str, messages: list[dict],
                       max_tokens: int = 1024, **kwargs) -> str:
        """Request the client to create a message (sampling).

        Allows servers to request LLM completions from the client.
        """
        if server not in self._servers:
            raise RuntimeError(f"Server '{server}' not connected.")
        session = self._servers[server].session
        result = self._run(session.create_message(
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        ))
        if hasattr(result, "content") and hasattr(result.content, "text"):
            return result.content.text
        return str(result)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def find_tool(self, tool_defs: list[dict], server: str, name: str) -> dict | None:
        """Look up a tool definition by server and name."""
        return next(
            (t for t in tool_defs if t["server"] == server and t["name"] == name),
            None,
        )

    def build_tool_fns(self, tool_defs: list[dict]) -> list:
        """Wrap MCP tool defs into callables for DSPy RLM / ReAct.

        Each wrapped function delegates to ``self.call_tool()`` with the
        server and tool name baked in via closure.
        """
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

    def get_connected_servers(self) -> list[str]:
        """Return names of all currently connected servers."""
        return list(self._servers.keys())

    def is_connected(self, server: str) -> bool:
        return server in self._servers

    # ------------------------------------------------------------------
    # Authentication — API key injection into env/headers
    # ------------------------------------------------------------------

    AUTH_ENV_MAP: dict[str, str] = {
        "openrouter": "OPENROUTER_API_KEY",
        "z3-solver": "Z3_API_KEY",
        "arxiv": "ARXIV_API_KEY",
        "lean-lsp": "LEAN_API_KEY",
    }

    @classmethod
    def inject_auth(cls, config: dict, env_prefix: str = "") -> dict:
        """Inject API keys from environment into MCP server configs.

        Reads ``{SERVER}_API_KEY`` or ``{ENV_PREFIX}_API_KEY`` from env
        and injects into each server's ``env`` block.  Also supports
        per-server key lookup via ``AUTH_ENV_MAP``.

        Returns the mutated config dict.
        """
        for name, cfg in config.get("mcpServers", {}).items():
            env_key = cls.AUTH_ENV_MAP.get(name) or f"{name.upper()}_API_KEY"
            if prefix := env_prefix:
                env_key = f"{prefix}_{env_key}"
            api_key = os.environ.get(env_key) or os.environ.get(f"{name.upper()}_API_KEY")
            if api_key and cfg.get("type") == "stdio":
                env = cfg.setdefault("env", {})
                env.setdefault(env_key, api_key)
                env.setdefault("API_KEY", api_key)
            elif api_key and cfg.get("type") == "sse":
                headers = cfg.setdefault("headers", {})
                headers.setdefault("Authorization", f"Bearer {api_key}")
        return config

    @classmethod
    def load_config_with_auth(cls, config_path: str, env_prefix: str = "") -> dict:
        """Load ``mcp_servers.json`` and inject auth keys from environment."""
        with open(config_path) as f:
            config = json.load(f)
        return cls.inject_auth(config, env_prefix=env_prefix)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def health_check(self, server: str | None = None) -> dict[str, dict]:
        """Check connectivity of connected MCP servers.

        Returns ``{server_name: {"status": "ok"|"error", "tools": int,
        "error": str, "latency_ms": float}}``.

        If *server* is given, checks only that server.
        """
        targets = [server] if server else list(self._servers.keys())
        results: dict[str, dict] = {}
        for name in targets:
            if name not in self._servers:
                results[name] = {"status": "disconnected", "tools": 0, "error": "Not connected"}
                continue
            session = self._servers[name].session
            start = time.monotonic()
            try:
                result = self._run(session.list_tools())
                elapsed = (time.monotonic() - start) * 1000
                results[name] = {
                    "status": "ok",
                    "tools": len(result.tools),
                    "latency_ms": round(elapsed, 1),
                }
            except Exception as e:
                elapsed = (time.monotonic() - start) * 1000
                results[name] = {
                    "status": "error",
                    "tools": 0,
                    "error": str(e),
                    "latency_ms": round(elapsed, 1),
                }
        return results

    def auto_reconnect(self, max_attempts: int = 2) -> list[str]:
        """Check all connected servers and reconnect any that are unhealthy.

        Returns list of server names that were reconnected.
        """
        reconnected: list[str] = []
        for name, ctx in list(self._servers.items()):
            session = ctx.session
            try:
                self._run(session.list_tools())
            except Exception:
                logger.warning("Reconnecting %s ...", name)
                for attempt in range(max_attempts):
                    try:
                        cfg = self.config.get("mcpServers", {}).get(name, {})
                        from copy import deepcopy
                        server_cfg = MCPServerConfig.from_dict(name, deepcopy(cfg))
                        if server_cfg.transport == "sse":
                            tools = self._run(self._connect_sse(server_cfg))
                        else:
                            tools = self._run(self._connect_stdio(server_cfg))
                        if tools:
                            reconnected.append(name)
                            logger.info("  Reconnected %s (%d tools)", name, len(tools))
                            break
                    except Exception as e:
                        logger.warning("  Attempt %d/%d failed: %s", attempt + 1, max_attempts, e)
        return reconnected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close all server sessions and stop the background event loop.

        MCP sessions are cleaned up via stored close coroutines. The daemon
        thread is joined with a short timeout.
        """
        if self._servers:
            async def _cleanup():
                for ctx in self._servers.values():
                    try:
                        await ctx.close_coro
                    except Exception:
                        pass
            self._run(_cleanup())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2)

    def reconnect_all(self) -> list[dict]:
        """Close all connections and reconnect from scratch."""
        self.close()
        # Re-create loop and thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._servers = {}
        return self.connect_all()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=1)
