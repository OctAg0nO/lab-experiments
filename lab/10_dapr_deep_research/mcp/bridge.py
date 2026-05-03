"""
MCP bridge — provides tools in both DSPy RLM format and dapr-agents AgentTool format.
"""

from __future__ import annotations

from typing import Any

from dapr_agents import AgentTool

from ..mcp.client import MCPClient


class MCPBridge:
    """Wraps MCPClient to produce tools for both DSPy RLMs and DurableAgents."""

    def __init__(self, client: MCPClient, tool_defs: list[dict]):
        self.client = client
        self.tool_defs = tool_defs

    def get_dspy_tools(self) -> list:
        """Return callables for dspy.RLM(tools=...)."""
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

    def get_agent_tools(self) -> list[AgentTool]:
        """Return AgentTool list for DurableAgent(tools=...)."""
        tools = []
        for td in self.tool_defs:
            srv, tn, desc = td["server"], td["name"], td.get("description", "")
            def make(srv=srv, tn=tn):
                def fn(**kwargs: Any) -> str:
                    return self.client.call_tool(srv, tn, kwargs)
                return AgentTool(name=tn, description=desc or f"MCP tool {tn}", func=fn)
            tools.append(make())
        return tools
