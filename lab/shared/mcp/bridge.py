"""
MCP bridge — dual-format tool adapter for DSPy RLMs and dapr-agents DurableAgents.

Wraps ``MCPClient`` to produce tools in two formats:

* ``get_dspy_tools()`` → list of callables for ``dspy.RLM(tools=...)``
* ``get_agent_tools()`` → list of ``AgentTool`` for ``DurableAgent(tools=...)``

Also provides resource and prompt wrappers for the agent ecosystem.
"""

from __future__ import annotations

import logging
from typing import Any

from dapr_agents import AgentTool

from .client import MCPClient

logger = logging.getLogger(__name__)


class MCPBridge:
    """Wraps MCPClient to produce tools for both DSPy RLMs and DurableAgents.

    Usage::

        client = MCPClient("config/mcp_servers.json")
        tool_defs = client.connect_all()
        bridge = MCPBridge(client, tool_defs)

        # For DSPy
        rlm_tools = bridge.get_dspy_tools()
        rlm = dspy.RLM("task: str -> result: str", tools=rlm_tools)

        # For dapr-agents
        agent_tools = bridge.get_agent_tools()
        agent = DurableAgent(tools=agent_tools)
    """

    def __init__(self, client: MCPClient, tool_defs: list[dict]):
        self.client = client
        self.tool_defs = tool_defs

    # ------------------------------------------------------------------
    # DSPy format — plain callables with __name__ and __doc__
    # ------------------------------------------------------------------

    def get_dspy_tools(self) -> list:
        """Return callables for ``dspy.RLM(tools=...)`` / ``dspy.ReAct(tools=...)``.

        Each callable has:

        * ``__name__`` set to the MCP tool name
        * ``__doc__`` set to the MCP tool description
        """
        fns = []
        for td in self.tool_defs:
            srv, tn, desc = td["server"], td["name"], td.get("description", "")

            def make(srv=srv, tn=tn, desc=desc):
                def fn(**kwargs: Any) -> str:
                    return self.client.call_tool(srv, tn, kwargs)
                fn.__name__ = tn
                fn.__doc__ = desc
                return fn
            fns.append(make())
        return fns

    def get_dspy_tools_filtered(self, names: set[str]) -> list:
        """Return DSPy tools for only the named tools (case-sensitive).

        Useful for role-based tool access::

            search_tools = bridge.get_dspy_tools_filtered({"search", "fetch", "chat"})
        """
        filtered = [td for td in self.tool_defs if td["name"] in names]
        return MCPBridge(self.client, filtered).get_dspy_tools()

    # ------------------------------------------------------------------
    # dapr-agents format — AgentTool list
    # ------------------------------------------------------------------

    def get_agent_tools(self) -> list[AgentTool]:
        """Return ``AgentTool`` list for ``DurableAgent(tools=...)``.

        Each ``AgentTool`` has:

        * ``name`` set to the MCP tool name
        * ``description`` set to the MCP tool description
        * ``func`` delegates to ``client.call_tool()``
        """
        tools = []
        for td in self.tool_defs:
            srv, tn, desc = td["server"], td["name"], td.get("description", "")

            def make(srv=srv, tn=tn, desc=desc):
                def fn(**kwargs: Any) -> str:
                    return self.client.call_tool(srv, tn, kwargs)
                return AgentTool(
                    name=tn,
                    description=desc or f"MCP tool {tn} on {srv}",
                    func=fn,
                    args_model=None,
                )
            tools.append(make())
        return tools

    def get_agent_tools_filtered(self, names: set[str]) -> list[AgentTool]:
        """Return ``AgentTool`` list for only the named tools."""
        filtered = [td for td in self.tool_defs if td["name"] in names]
        return MCPBridge(self.client, filtered).get_agent_tools()

    # ------------------------------------------------------------------
    # Resource and Prompt bridges (protocol extensions)
    # ------------------------------------------------------------------

    def get_resource_descriptors(self) -> list[dict]:
        """Aggregate resources across all connected servers."""
        all_resources = []
        for server_name in self.client.get_connected_servers():
            try:
                resources = self.client.list_resources(server_name)
                for r in resources:
                    r["server"] = server_name
                all_resources.extend(resources)
            except Exception as e:
                logger.warning("  [!] resources from %s: %s", server_name, e)
        return all_resources

    def get_prompt_templates(self) -> list[dict]:
        """Aggregate prompt templates across all connected servers."""
        all_prompts = []
        for server_name in self.client.get_connected_servers():
            try:
                prompts = self.client.list_prompts(server_name)
                for p in prompts:
                    p["server"] = server_name
                all_prompts.extend(prompts)
            except Exception as e:
                logger.warning("  [!] prompts from %s: %s", server_name, e)
        return all_prompts

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of connected MCP servers and tools."""
        servers = set(td["server"] for td in self.tool_defs)
        n_tools = len(self.tool_defs)
        parts = [f"MCP: {n_tools} tool(s) across {len(servers)} server(s)"]
        for srv in sorted(servers):
            srv_tools = [td["name"] for td in self.tool_defs if td["server"] == srv]
            parts.append(f"  [{srv}] {', '.join(sorted(srv_tools))}")
        return "\n".join(parts)
