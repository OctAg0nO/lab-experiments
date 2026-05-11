# Lab 15b — A2UI/AG-UI Optimization Plan

**Status:** Planning  
**Goal:** Optimize the A2UI integration for production-grade UX, leveraging AG-UI protocol patterns.

---

## Current State Assessment

The current A2UI integration has:

| Component | Status | Gaps |
|-----------|--------|------|
| `A2UIChannel` — v0.10 protocol + surface management | ✅ Built | No client-side renderer exists yet (frontend must implement) |
| `a2ui_tool.py` — MCP tools for UI | ✅ Built | Tools return strings (no streaming), no bi-directional state |
| LiveKit data channel transport | ✅ Works | No fallback if data channel unavailable. No WebSocket/SSE option |
| Default theme | ✅ Built | Static theme, no runtime customization |
| Progress/loading states | ✅ Built | No determinate progress tracking from agent |
| Surface lifecycle | ✅ Built | No surface persistence across conversation turns |
| Reconnection durability | ✅ Built | In-memory only (lost on worker restart) |

### Missing (Not Yet Built)

| Feature | Impact | Required For |
|---------|--------|-------------|
| Frontend renderer (React/Lit component that renders A2UI JSON) | Blocks all UI | First user-facing deployment |
| Bi-directional state sync (agent ↔ frontend state) | Limited UX | Real-time collaboration |
| Frontend tool calls (browser capabilities as agent tools) | Medium | Geolocation, file access, clipboard |
| Human-in-the-loop (user approval mid-workflow) | High | Production safety |
| Streaming tool output (UI updates as research progresses) | High | Perceived responsiveness |
| WebSocket transport fallback | Medium | Non-LiveKit deployments |

---

## AG-UI Integration Opportunities

AG-UI and A2UI are **complementary, not competitive**:

| Protocol | Role | Our Use |
|----------|------|---------|
| **AG-UI** | Agent↔User interaction protocol (events, streaming, state sync, frontend tools) | Transport + state layer |
| **A2UI** | Generative UI specification (declarative JSON components) | Rendering layer |
| **MCP** | Agent↔Tools protocol | Already have this |

AG-UI's building blocks that directly apply:

| AG-UI Feature | Maps To | Optimization |
|---------------|---------|--------------|
| **Generative UI** (declarative) | A2UI `createSurface`/`updateComponents` | Use AG-UI's declarative UI format alongside A2UI |
| **Shared state** (read-write) | A2UI `updateDataModel` | Bi-directional state sync with conflict resolution |
| **Frontend tool calls** | Reverse of agent tools | Browser APIs as agent tools |
| **Streaming chat** | Research iteration streaming | Real-time progress per iteration |
| **Interrupts** (human-in-loop) | Approval gates | User confirms before agent takes actions |
| **Thinking steps** | Research trace visualization | Show agent's reasoning steps |
| **Backend tool rendering** | A2UI `show_results` | Render tool outputs as UI components |

---

## Optimization Plan — Phased

### Phase 0: Frontend Renderer (Prerequisite, ~1 week)

Before any other optimization, a frontend renderer must exist. Without it, all A2UI messages are invisible.

**Option A: AG-UI + CopilotKit (Recommended)**
Use AG-UI's existing CopilotKit integration. CopilotKit provides React components that render AG-UI events as UI. Since AG-UI supports A2UI as a generative UI spec, we get A2UI rendering for free.

```bash
npx copilotkit@latest init
# Select React + AG-UI
```

Then the frontend receives AG-UI events (which carry A2UI payloads) and renders them via CopilotKit's built-in component system.

**Option B: Custom React/Lit renderer**
Build a custom renderer using `@a2ui/react` or `@a2ui/lit`. More control but more work.

Both options receive A2UI JSON over the data channel. Option A adds AG-UI's bi-directional state and frontend tools for free.

**Deliverable:** User sees A2UI components rendered in their browser.

### Phase 1: Shared State Sync (Week 2, ~4 hours)

**Goal:** Bi-directional state between agent and frontend.

**Current:** Agent pushes `updateDataModel` → frontend receives. No reverse path.
**Target:** Frontend state changes sync back to agent.

**Implementation:**
- Add AG-UI `shared_state` event handler in the agent
- Frontend updates to AG-UI state store are delivered as events to the agent
- Agent can react to user state changes mid-research

```python
# Agent receives state update from frontend
@agui.on("shared_state_update")
async def on_state_update(path: str, value: Any):
    # User changed something in the UI
    # Agent can adjust research direction
    self._meta.frontier.adjust(path, value)
```

**Files:** `livekit/agui_handler.py` (new), `mcp/a2ui_tool.py` (extend)

### Phase 2: Frontend Tool Calls (Week 2, ~3 hours)

**Goal:** User's browser capabilities become agent tools.

**Current:** Agent has MCP tools (crawl4ai, Exa, etc.) — all server-side.
**Target:** Agent can call frontend tools (geolocation, file picker, clipboard).

**Implementation:**
- Register frontend tool schemas via AG-UI
- When agent calls `get_user_location`, AG-UI forwards to frontend
- Frontend returns result, agent continues research

```python
# Agent tool definition — executes on frontend
@tool
def get_user_location() -> dict:
    """Get the user's current GPS location. Executes in browser."""
    # AG-UI forwards this to the frontend
    # Frontend calls navigator.geolocation.getCurrentPosition()
    pass  # Result arrives asynchronously via AG-UI event
```

**Use cases:**
- "Show me restaurants near me" → geolocation
- "Analyze this file" → file picker
- "Copy this to clipboard" → clipboard API

**Files:** `livekit/frontend_tools.py` (new)

### Phase 3: Streaming Research Output (Week 3, ~3 hours)

**Goal:** UI updates progressively as research iterations complete.

**Current:** Agent runs all iterations, then sends results.
**Target:** After each iteration, agent pushes partial results to UI.

**Implementation:**

```python
# In OctAg0nOAgent.llm_node — stream per iteration
for i, result in enumerate(partial_results):
    yield str(result.get("prediction", ""))

    # Push progress + partial data to A2UI
    a2ui_update_progress(
        percent=int((i + 1) / max_iterations * 100),
        message=f"Iteration {i + 1}/{max_iterations} complete"
    )
    a2ui_update_data("/research/latest", str(result)[:200])
```

**UX impact:** Dramatic — user sees progress bar moving and partial data appearing instead of waiting for completion.

**Files:** `livekit/llm_adapter.py` (modify `llm_node`)

### Phase 4: Human-in-the-Loop Interrupts (Week 3, ~4 hours)

**Goal:** Agent pauses and asks user for approval before executing actions.

**Current:** Agent executes tools autonomously.
**Target:** Agent asks user "Should I execute this action?" and waits for response.

**Implementation:**

```python
@tool
async def confirm_action(description: str) -> bool:
    """Ask the user to confirm before proceeding.

    The agent pauses, renders a confirmation card via A2UI,
    and waits for the user's response via AG-UI event.
    """
    a2ui_show_card(
        "confirm", "confirmation",
        title="Confirmation Required",
        content=description,
        badge="Awaiting Input",
    )
    # AG-UI interrupt: agent pauses, frontend shows approve/deny buttons
    response = await agui.wait_for_interrupt("confirm_action")
    return response["approved"]
```

**Use cases:**
- "I found a vulnerability. Should I generate a patch?" → approve/deny
- "This research will use 50 API calls. Proceed?" → approve/deny
- "I need access to your location. Allow?" → approve/deny

**Files:** `livekit/interrupts.py` (new), `mcp/a2ui_tool.py` (extend)

### Phase 5: Thinking Step Visualization (Week 4, ~2 hours)

**Goal:** Show the agent's reasoning process in real-time.

**Current:** Agent is a black box — user hears final result.
**Target:** User sees "Researching X...", "Analyzing Y...", "Synthesizing..." as steps appear.

**Implementation:**

```python
# In llm_node — emit thinking steps before research
yield "Let me research that."
a2ui_update_data("/thinking", "Analyzing query...")

# After each agent generates
for entry in agent_stack:
    a2ui_update_data("/thinking", f"Agent {entry.name} working...")
    yield from run_agent(entry)

a2ui_update_data("/thinking", "Synthesizing results...")
```

**UX impact:** Transparency — user understands what the agent is doing.

**Files:** `livekit/llm_adapter.py` (modify)

### Phase 6: Theme Customization & Component Catalog (Week 4, ~2 hours)

**Goal:** Runtime theme customization and custom component catalogs.

**Current:** `DEFAULT_THEME` hardcoded.
**Target:** Frontend sends theme preferences on connect, agent adapts.

```python
# Frontend sends theme on connect
a2ui_channel = A2UIChannel(room, theme={
    "primaryColor": user_theme.primary,
    "backgroundColor": user_theme.background,
})
```

Plus: Allow agents to register custom component types beyond the basic catalog (e.g., `research-chart`, `citation-list`, `confidence-meter`).

**Files:** `livekit/a2ui_channel.py` (extend)

### Phase 7: WebSocket Transport Fallback (Week 5, ~3 hours)

**Goal:** A2UI works without LiveKit (standalone web chat).

**Current:** A2UI requires LiveKit data channel.
**Target:** Abstract transport layer — LiveKit data channel or WebSocket.

```python
class A2UITransport(ABC):
    @abstractmethod
    async def send(self, payload: bytes): ...

class LiveKitDataChannel(A2UITransport):
    def __init__(self, room): ...

class WebSocketTransport(A2UITransport):
    def __init__(self, ws_url: str): ...
```

This enables A2UI rendering in non-voice contexts (web chat, mobile, Slack).

**Files:** `livekit/a2ui_transport.py` (new), `livekit/a2ui_channel.py` (refactor)

---

## Priority Matrix

| Phase | Effort | UX Impact | Risk | Do First? |
|-------|--------|-----------|------|-----------|
| **P0**: Frontend Renderer | ~1 week | 🔴 Blocking | Medium | ✅ **Yes — blocks everything** |
| **P3**: Streaming Output | ~3 hours | 🟢 High | Low | ✅ Yes — quick win |
| **P4**: Human-in-Loop | ~4 hours | 🟢 High | Medium | ✅ Yes — safety critical |
| **P1**: Shared State | ~4 hours | 🟡 Medium | Medium | ⏸️ After P0/P3 |
| **P5**: Thinking Steps | ~2 hours | 🟡 Medium | Low | ⏸️ After P3 |
| **P2**: Frontend Tools | ~3 hours | 🟡 Medium | High | ⏸️ After P1 |
| **P6**: Theme/Catalog | ~2 hours | 🟢 High | Low | ⏸️ Low effort, high polish |
| **P7**: WebSocket Transport | ~3 hours | 🟡 Medium | Low | ⏸️ When voice not needed |

---

## Quick Wins (Can Be Done Today)

These don't require the frontend renderer and improve the backend architecture:

1. **Add `thinking` steps to `llm_node`** — yield intermediate status messages that TTS can speak
2. **Add `research_progress` event** — push per-iteration updates to data channel even without renderer (logs for debugging)
3. **Add `a2ui_update_progress()` calls in `llm_node`** — data model updates accumulate and are delivered when renderer connects
4. **AG-UI event format compliance** — ensure our A2UI JSON matches AG-UI's event spec for future compatibility

---

## Research References

- **AG-UI Protocol**: [github.com/ag-ui-protocol/ag-ui](https://github.com/ag-ui-protocol/ag-ui) — Event-based agent↔user interaction protocol
- **A2UI + AG-UI**: [docs.ag-ui.com/concepts/generative-ui-specs](https://docs.ag-ui.com/concepts/generative-ui-specs) — A2UI as a generative UI spec within AG-UI
- **AG-UI Dojo**: [dojo.ag-ui.com](https://dojo.ag-ui.com/) — Interactive demos of all AG-UI features
- **CopilotKit**: [github.com/CopilotKit/CopilotKit](https://github.com/CopilotKit/CopilotKit) — AG-UI client for React
- **A2UI Spec**: [github.com/google/a2ui](https://github.com/google/a2ui) — Declarative JSON UI protocol
