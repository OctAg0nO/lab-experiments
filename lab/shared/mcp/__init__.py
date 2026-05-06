"""
Shared MCP infrastructure — consolidated from labs 09-12.

Single source of truth for MCP client, bridge, and protocol helpers.
All labs should import from here instead of maintaining local copies.
"""

from .client import MCPClient
from .bridge import MCPBridge

__all__ = ["MCPClient", "MCPBridge"]
