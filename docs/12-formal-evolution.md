# 12 — Formal Evolution: MCP Server Configuration Reference

> Source: `lab/12_formal_evolution/` (fork of lab 11)

Lab 12 is a **zero-code-change** extension of the lab 11 meta-agent. The only difference is `config/mcp_servers.json`, which adds 6 MCP servers for formal verification, multi-model consensus, research, filesystem, and Git operations. The meta-agent auto-discovers all tools via `MCPBridge`.

---

## MCP Server Configuration

The meta-agent reads `config/mcp_servers.json` at startup. Each entry defines a server using `MCPClient` (stdio or SSE transport). The client connects to all enabled servers, flattens their tool lists, and passes them to `MCPBridge`, which wraps each tool as a DSPy-compatible callable.

### Config format

```json
{
  "mcpServers": {
    "server-name": {
      "description": "Human-readable purpose",
      "enabled": true,
      "type": "stdio",
      "command": "uvx",
      "args": ["package-name"]
    }
  }
}
```

All relative paths resolve from the project root.

### Server inventory

| Server | Transport | Tools | Config |
|--------|-----------|-------|--------|
| `crawl4ai` | SSE | `md`, `html`, `crawl`, `screenshot` | `url` → `http://localhost:11235/mcp/sse` |
| `fetch` | stdio | `fetch` | `uvx mcp-server-fetch` |
| `openrouter` | stdio | `chat_completion`, `model_list`, `consensus`, `ensemble`, `usage_stats` | `npx @physics91/openrouter-mcp start` |
| `z3-solver` | stdio | `solve_constraint_problem`, `simple_constraint_solver`, `analyze_relationships`, `simple_relationship_analyzer` | `uv run --directory lab/12_formal_evolution/z3_mcp python -m z3_mcp.server.main` |
| `arxiv` | stdio | `search_papers`, `download_paper`, `read_paper`, `list_papers`, `semantic_search`, `citation_graph` | `uvx arxiv-mcp-server` |
| `lean-lsp` | stdio | `lean_goal`, `lean_build`, `lean_diagnostic_messages`, `lean_run_code`, `lean_leansearch`, 20+ more | `uvx lean-lsp-mcp` |
| `filesystem` | stdio | `read_text_file`, `write_file`, `edit_file`, `search_files`, `directory_tree`, `list_directory` | `npx -y @modelcontextprotocol/server-filesystem .` |
| `git` | stdio | `git_status`, `git_diff`, `git_log`, `git_commit`, `git_branch`, `git_checkout` | `uvx mcp-server-git --repository .` |

### Enabled by default

`crawl4ai`, `fetch`, `openrouter`, `z3-solver`, `arxiv`

Toggle any server by setting `"enabled": false`.

---

## Tool Discovery Flow

```
mcp_servers.json
    → MCPClient.connect_all()      # spawns servers, lists tools
    → MCPBridge.get_dspy_tools()   # wraps as DSPy callables
    → AgentGenerator.analyze()     # BestOfN picks right tools for task
    → RLM/ReAct/CodeAct/CoT agent  # calls tools during execution
```

Each tool becomes a Python function with `__name__` and `__doc__` set from the MCP server's tool definition. The meta-agent's `BestOfN` task analysis decides which tools an agent needs based on the user query.

---

## MCPClient API (from lab 11)

| Method | Description |
|--------|-------------|
| `MCPClient(config_path)` | Connect to servers defined in JSON config |
| `.connect_all()` | Start all enabled servers, return `list[dict]` of tool defs |
| `.call_tool(server, tool_name, args)` | Call MCP tool, return string |
| `.close()` | Stop all sessions and background thread |

Tool def format:
```python
{"server": str, "name": str, "description": str, "inputSchema": dict}
```

---

## MCPBridge API (from lab 11)

| Method | Description |
|--------|-------------|
| `MCPBridge(client, tool_defs)` | Wrap MCP client for DSPy/dapr-agents |
| `.get_dspy_tools()` | Return callables for `dspy.RLM(tools=...)` |
| `.get_agent_tools()` | Return `AgentTool` list for `DurableAgent(tools=...)` |

---

## Key difference from lab 11

| Aspect | Lab 11 | Lab 12 |
|--------|--------|--------|
| MCP servers | 3 (1 enabled) | 8 (5 enabled) |
| Tool count | ~6 | ~40+ |
| Local deps | none | `z3_mcp/` cloned in-tree |
| Code changes | — | none — config only |
