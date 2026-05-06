# Re-export from shared MCP package (single source of truth)
from lab.shared.mcp import MCPClient, MCPBridge
__all__ = ["MCPClient", "MCPBridge"]
