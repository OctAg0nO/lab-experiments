# Lab 15b — Integration Plan: LiveKit Agents + A2UI

**Status:** Planning  
**Parent:** Lab 15 (Ray + SGLang: Distributed High-Throughput Meta-Agent)  
**Goal:** Add multimodal human-agent interaction — voice, video, and live UI rendering — to the Durable Meta Agent.

---

## Executive Summary

Lab 15b adds two new layers to the existing DSPy + Dapr + Ray + SGLang stack:

| Layer | Technology | Metaphor | Role |
|-------|-----------|----------|------|
| State & Workflows | **Dapr** (existing) | Brain / Memory | Durable execution, checkpointing, pub/sub |
| Reasoning Engine | **DSPy** (existing) | Logic | Modules, signatures, optimizers, BAMLAdapter |
| Distributed Compute | **Ray** (existing) | Body | Task parallelism, resource isolation |
| Fast Inference | **SGLang** (existing) | Nervous System | RadixAttention, continuous batching, 4-bit AWQ |
| **Media & Voice** | **LiveKit Agents** (new) | Senses | STT → LLM → TTS pipeline, WebRTC, data channels |
| **UI Rendering** | **A2UI** (new) | Face / Hands | Declarative JSON UI protocol, trusted component catalog |

### The "Physical AI" Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                          USER (Web/Mobile)                            │
│  ┌─────────────────────┐  ┌────────────────────────────────────┐    │
│  │ A2UI Renderer       │  │ LiveKit WebRTC SDK                 │    │
│  │ (Lit/React/Flutter) │  │ (Audio, Video, Data Channel)       │    │
│  └─────────┬───────────┘  └───────────────┬────────────────────┘    │
│            │                               │                          │
└────────────┼───────────────────────────────┼──────────────────────────┘
             │                               │
             │  A2UI JSON (via Data Channel) │  WebRTC Audio/Video
             ▼                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    LIVEKIT INFRASTRUCTURE                             │
│                                                                       │
│  ┌────────────────────────┐  ┌──────────────────────────────────┐   │
│  │ LiveKit Server         │  │ LiveKit Agent Worker              │   │
│  │ (Cloud or Self-hosted) │  │                                  │   │
│  │ · Room management      │  │  ┌────────────────────────────┐  │   │
│  │ · WebRTC relay         │  │  │ VoicePipelineAgent         │  │   │
│  │ · Recording/Egress     │  │  │ · STT (DeepGram/Whisper)   │  │   │
│  │                        │  │  │ · LLM → MetaAgent adapter  │  │   │
│  │                        │  │  │ · TTS (Cartesia/ElevenLabs)│  │   │
│  │                        │  │  │ · Data Channel (A2UI)      │  │   │
│  │                        │  │  └───────────┬────────────────┘  │   │
│  └────────────────────────┘  └──────────────┼───────────────────┘   │
│                                             │                         │
└─────────────────────────────────────────────┼─────────────────────────┘
                                               │
                    LiveKit calls MetaAgent as  │  the "LLM"
                    A2UI messages forwarded via │  data channel
                                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    OCTAGONO META-AGENT (The Brain)                    │
│                                                                       │
│  DurableMetaAgent (Dapr) → MetaAgent (DSPy) → AgentGenerator          │
│  ↑ Ray tasks for parallel execution                                   │
│  ↑ SGLang for fast inference + RadixAttention                         │
│  ↑ MCPBridge for tools (crawl4ai, Exa, etc.)                         │
│  ↑ A2UITool for UI rendering (new)                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: LiveKit Agents Integration

### 1.1 Architecture Overview

LiveKit Agents follows an **agent worker model**:
1. **LiveKit Server** manages WebRTC rooms, media routing, and signaling
2. **Agent Workers** connect to the server and handle rooms
3. Each worker runs a **VoicePipelineAgent** that processes STT → LLM → TTS
4. Our custom **LLM adapter** delegates to the OctAg0nO MetaAgent

```
User opens browser → joins LiveKit room → WebRTC audio stream
  → LiveKit Server routes audio to Agent Worker
  → Agent Worker: STT (speech→text) → MetaAgent (reasoning) → TTS (text→speech)
  → Audio streamed back to user via WebRTC
  → Parallel: A2UI messages sent via data channel
```

### 1.2 The MetaAgent LLM Adapter

LiveKit's `AgentSession` accepts an LLM that is OpenAI-compatible or any callable. The MetaAgent can be wrapped as an OpenAI-compatible endpoint (it already exposes one via SGLang). But for tighter integration, we create a thin `MetaAgentLLM` adapter:

```python
# lab/15_ray_sglang/livekit/llm_adapter.py

from livekit.agents.llm import LLM, ChatContext, ChatMessage
from lab._15_ray_sglang.meta.meta_agent import MetaAgent, MetaConfig


class MetaAgentLLM(LLM):
    """Wraps the OctAg0nO MetaAgent as a LiveKit-compatible LLM.

    LiveKit calls chat() with conversation history.
    MetaAgent processes the query through its research loop
    and returns structured results.
    """

    def __init__(self, meta_agent: MetaAgent):
        self._meta = meta_agent

    async def chat(
        self,
        chat_ctx: ChatContext,
        fnc_ctx: FunctionContext | None = None,
    ) -> "LLMStream":
        """Convert LiveKit chat context → MetaAgent task → stream result."""

        # Extract the latest user message as the task query
        last_msg = chat_ctx.messages[-1]
        query = last_msg.text if hasattr(last_msg, 'text') else str(last_msg.content)

        # Run MetaAgent research loop (async via Ray tasks)
        # The executor uses RayModuleExecutor if configured
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._meta.run_stack, query, 3  # 3 iterations for voice latency
        )

        # Stream results back to LiveKit as LLMStream
        stream = LLMStream()
        for r in results:
            prediction = r.get("prediction", "")
            stream.push(LLMChunk(content=str(prediction)))
        stream.end()
        return stream
```

### 1.3 LiveKit Worker with OctAg0nO Brain

```python
# lab/15_ray_sglang/livekit/worker.py

from livekit.agents import AgentServer, JobContext, cli, inference
from livekit.agents.voice_pipeline import VoicePipelineAgent
from livekit.plugins import silero, deepgram, cartesia

from lab._15_ray_sglang.meta.meta_agent import MetaAgent, MetaConfig
from lab._15_ray_sglang.ray.executor import RayModuleExecutor

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Entrypoint for each LiveKit voice session.

    Initializes the OctAg0nO MetaAgent as the reasoning brain,
    connects it to LiveKit's STT → LLM → TTS pipeline,
    and opens a data channel for A2UI messages.
    """

    # 1. Initialize OctAg0nO Brain with SGLang for fast inference
    meta = MetaAgent(config=MetaConfig(
        llm=dspy.LM(
            model="openai/meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:30000/v1",  # SGLang
        ),
        executor=RayModuleExecutor(num_gpus=0, num_cpus=1),
        # ... generator, tool_defs from MCP bridge
    ))

    # 2. Set up A2UI data channel
    a2ui_channel = A2UIChannel(ctx.room)

    # 3. Register A2UI rendering as a tool the agent can call
    a2ui_tool = A2UITool(channel=a2ui_channel)
    # ... add to agent's tool list via AgentGenerator

    # 4. Create MetaAgentLLM adapter
    meta_llm = MetaAgentLLM(meta)

    # 5. Start the LiveKit voice pipeline
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-3"),
        llm=meta_llm,  # OctAg0nO is the LLM
        tts=cartesia.TTS(model="sonic-3", voice="..."),
    )

    agent.start(ctx.room)
    await agent.say("OctAg0nO agent connected. How can I help you?")


if __name__ == "__main__":
    cli.run_app(server)
```

### 1.4 LiveKit Data Channel for A2UI Transport

A2UI messages travel over LiveKit's **data channel** — a low-latency, ordered transport within the WebRTC session:

```python
# lab/15_ray_sglang/livekit/a2ui_channel.py

from livekit.rtc import Room
import json


class A2UIChannel:
    """Manages A2UI JSON messages over LiveKit data channel.

    Each message is an A2UI Response payload that the frontend
    A2UI Renderer resolves into native UI components.
    """

    def __init__(self, room: Room, label: str = "a2ui"):
        self._room = room
        self._label = label

    def send(self, a2ui_response: dict):
        """Send an A2UI Response to the frontend via data channel.

        a2ui_response format:
        {
            "version": "1.0",
            "components": [
                {"id": "card-1", "type": "card", "props": {...}},
                {"id": "chart-1", "type": "chart", "props": {...}},
            ]
        }
        """
        payload = json.dumps(a2ui_response).encode("utf-8")
        self._room.local_participant.publish_data(
            payload,
            topic=self._label,
        )

    def update_component(self, component_id: str, props: dict):
        """Incrementally update a single component by ID.

        A2UI supports incremental updates — the agent can patch
        a chart's data without re-rendering the entire UI.
        """
        payload = json.dumps({
            "version": "1.0",
            "type": "update",
            "component_id": component_id,
            "props": props,
        }).encode("utf-8")
        self._room.local_participant.publish_data(payload, topic=self._label)
```

---

## Part 2: A2UI Integration

### 2.1 What is A2UI?

A2UI (Agent-to-User Interface) is an open protocol from Google that lets agents
"speak UI" — sending declarative JSON that the client renders using its own
native component library.

**Key properties:**
- **Declarative JSON format** — the agent describes UI intent, not implementation
- **Trusted component catalog** — the client controls which components are allowed
- **Incremental updates** — agent can update individual components by ID
- **LLM-friendly flat list** — easy for models to generate
- **Transport agnostic** — works over any channel (WebSocket, data channel, A2A)

**A2UI Response format:**
```json
{
  "version": "1.0",
  "components": [
    {
      "id": "results-table",
      "type": "data-table",
      "props": {
        "columns": ["Name", "Value"],
        "rows": [["Item A", 42], ["Item B", 17]],
        "title": "Research Results"
      }
    },
    {
      "id": "status-badge",
      "type": "badge",
      "props": {
        "label": "Completed",
        "variant": "success"
      }
    }
  ]
}
```

### 2.2 A2UITool — MCP-Compatible UI Rendering

The A2UITool is registered as an **MCP tool** in the OctAg0nO tool layer.
The agent can call it like any other tool — the framework handles transport:

```python
# lab/15_ray_sglang/mcp/a2ui_tool.py

import json
from typing import Any


def a2ui_render(components: list[dict]) -> str:
    """Render UI components on the user's screen.

    Sends an A2UI JSON payload to the frontend via the active
    data channel. Components are rendered using the client's
    trusted component catalog.

    Each component must have:
      - id: unique string identifier (for incremental updates)
      - type: component type from the client's catalog
      - props: component-specific properties

    Supported component types (varies by client):
      card, data-table, chart, badge, text-field,
      button, progress-bar, list, form, accordion

    Example:
        components = [
            {"id": "results", "type": "data-table",
             "props": {"columns": ["X", "Y"], "rows": [[1, 2], [3, 4]]}},
            {"id": "note", "type": "card",
             "props": {"title": "Summary", "content": "Analysis complete"}},
        ]
    """
    # A2UIChannel is set on the session context by the LiveKit worker
    channel = _get_active_a2ui_channel()
    if channel is None:
        return "Error: No active A2UI channel. User may not be connected."

    response = {
        "version": "1.0",
        "components": components,
    }
    channel.send(response)
    return f"Rendered {len(components)} component(s) on user screen."
```

### 2.3 Agent Prompt Integration

The A2UI tool is registered alongside other MCP tools via the `AgentGenerator`.
The LLM (MetaAgent) is prompted to use it at appropriate moments:

```
You have an A2UI tool that can render UI components on the user's screen.

Use it when:
- The user asks for data/analysis → render a data-table or chart
- The user asks for a comparison → render cards side by side
- Showing progress → render a progress bar
- The information is visual/structured → always prefer UI over text

Don't use it for:
- Simple acknowledgments ("OK", "Got it")
- Conversational back-and-forth
```

---

## Part 3: Ray Scaling for LiveKit Workers

LiveKit workers process audio (STT/TTS) which is CPU-intensive.
Running them as Ray actors allows horizontal scaling:

```python
# lab/15_ray_sglang/livekit/ray_worker.py

import ray
from livekit.agents import JobContext, WorkerOptions, AgentServer


@ray.remote(num_cpus=2, num_gpus=0.25)
class LiveKitWorker:
    """A single LiveKit worker that can handle multiple rooms.

    Scaled horizontally via Ray. Each worker uses:
    - 2 CPUs for STT/TTS processing
    - 0.25 GPU (fractional) for SGLang inference calls
    """

    def __init__(self):
        self.server = AgentServer()
        self._setup_entrypoint()

    def _setup_entrypoint(self):
        @self.server.rtc_session()
        async def entrypoint(ctx: JobContext):
            # ... same as worker.py above ...
            pass

    def run(self):
        """Start the worker in a non-blocking way."""
        import asyncio
        asyncio.run(self.server.run())
```

---

## Part 4: Integration with Lab 15 Layers

### 4.1 Pipeline: Voice → MetaAgent → A2UI

```
User speaks: "Show me the results of my research on quantum computing"

1. STT (DeepGram) → text: "Show me research results on quantum computing"
2. LiveKit calls MetaAgentLLM.chat(query)
3. MetaAgent.run_stack("Research quantum computing", iterations=3)
   └─ RayModuleExecutor parallelizes agent calls
   └─ SGLang provides fast inference with RadixAttention
   └─ AgentGenerator creates researcher agent with A2UITool
4. Researcher agent calls a2ui_render(components=[...])
   └─ A2UITool sends JSON over LiveKit data channel
   └─ Frontend A2UI Renderer displays table + chart
5. MetaAgent returns text summary
6. TTS (Cartesia) speaks: "I found three key papers. I've displayed a summary table on your screen."
```

### 4.2 Dapr Durability Across Disconnections

If the user's WebRTC connection drops:
- LiveKit session pauses
- Dapr workflow continues executing (crash-resistant)
- On reconnection, agent can say: "While you were away, I completed the analysis"

```python
# In the DurableMetaAgent workflow:
@workflow_entry
def run_research(self, ctx, input: dict):
    # ... standard research loop ...
    # This continues running even if the LiveKit session ends
    # Results are available when the user reconnects
```

### 4.3 SGLang RadixAttention for Voice Sessions

Voice conversations have high prefix reuse: the system prompt and conversation
history are shared across every turn. SGLang's RadixAttention caches this:

```
Turn 1: [system prompt] + "Show me research on quantum..." → cached prefix
Turn 2: [system prompt] + "Show me research on quantum..." + "Now show me it as a chart"
         → radix tree reuses Turn 1's KV cache → ~3x faster
```

---

## Part 5: File Structure

```
lab/15_ray_sglang/
├── ... (existing Lab 15 files)
│
├── livekit/                    # NEW: LiveKit integration
│   ├── __init__.py
│   ├── worker.py               # LiveKit Agent Server entrypoint
│   ├── llm_adapter.py          # MetaAgentLLM — wraps MetaAgent as LiveKit LLM
│   ├── a2ui_channel.py         # A2UIChannel — data channel transport
│   └── ray_worker.py           # LiveKitWorker — Ray-scalable worker
│
├── mcp/
│   └── a2ui_tool.py            # NEW: A2UI rendering MCP tool
│
├── scripts/
│   ├── launch_sglang.sh        # Existing
│   └── start_livekit_agent.sh  # NEW: start LiveKit agent worker
│
└── config/
    ├── ray_cluster.yaml        # Existing
    └── livekit.yaml            # NEW: LiveKit server config
```

---

## Part 6: Implementation Phases

### Phase 1: LiveKit Worker with MetaAgentLLM Adapter (Week 1)
**Effort: ~4 hours**

| Task | Files | Dependencies |
|------|-------|-------------|
| 1.1 Create `MetaAgentLLM(LLM)` adapter | `livekit/llm_adapter.py` | Lab 15 MetaAgent |
| 1.2 Create basic LiveKit worker | `livekit/worker.py` | LiveKit server |
| 1.3 Test: voice → MetaAgent → voice loop | Manual | 1.1, 1.2 |
| 1.4 Add `livekit-agents` dep to pyproject.toml | `pyproject.toml` | None |

### Phase 2: A2UI Integration (Day 2-3, ~3 hours)

| Task | Files | Dependencies |
|------|-------|-------------|
| 2.1 Create `A2UIChannel` for data channel transport | `livekit/a2ui_channel.py` | Phase 1 |
| 2.2 Create `a2ui_render()` MCP tool | `mcp/a2ui_tool.py` | 2.1 |
| 2.3 Register A2UITool in AgentGenerator | `meta/agent_generator.py` | 2.2 |
| 2.4 Add A2UI rendering prompt to agent instructions | `meta/meta_agent.py` | 2.3 |
| 2.5 Test: agent renders a table on voice command | Manual | 2.4 |

### Phase 3: Ray Scaling for LiveKit Workers (Day 4, ~2 hours)

| Task | Files | Dependencies |
|------|-------|-------------|
| 3.1 Create `LiveKitWorker` Ray actor | `livekit/ray_worker.py` | Phase 1 |
| 3.2 Add `--livekit-workers N` to CLI | `cli.py` | 3.1 |
| 3.3 Test: 3 concurrent voice sessions on one node | Manual | 3.2 |

### Phase 4: Dapr Durability for Voice Sessions (Day 5, ~2 hours)

| Task | Files | Dependencies |
|------|-------|-------------|
| 4.1 Wire Disconnect → Continue workflow pattern | `core/durable_meta_agent.py` | Phase 1 |
| 4.2 Test: disconnect and reconnect mid-research | Manual | 4.1 |

---

## Part 7: CLI Commands

```bash
# Start LiveKit server (Docker)
docker run --rm -p 7880:7880 -p 7881:7881 livekit/livekit-server \
  --config lab/15_ray_sglang/config/livekit.yaml

# Start SGLang server
bash lab/15_ray_sglang/scripts/launch_sglang.sh

# Start LiveKit Agent Worker
uv run python -m lab.15_ray_sglang \
  --sglang-endpoint http://localhost:30000/v1 \
  --ray \
  livekit-worker

# Start with N parallel workers via Ray
uv run python -m lab.15_ray_sglang \
  --sglang-endpoint http://localhost:30000/v1 \
  --ray \
  --livekit-workers 4
```

---

## Part 8: Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Voice latency > 500ms** (STT → MetaAgent → TTS) | High | SGLang RadixAttention for prefix reuse. Cap MetaAgent iterations to 3 for voice. Use 4-bit AWQ for faster inference. |
| **A2UI JSON generation quality** (LLM generates invalid UI) | Medium | A2UI schema validation server-side before sending. LLM prompt engineering with examples. Fallback to text-only mode. |
| **LiveKit + Dapr sidecar conflict** | Medium | LiveKit workers run as separate processes or Ray actors. Dapr sidecar attaches to the orchestrator, not the worker. |
| **WebRTC connection drops** mid-research | Medium | Dapr workflow continues. Results cached. Reconnecting user receives completed research. |
| **GPU contention** (SGLang + STT on same GPU) | Medium | LiveKit workers get fractional GPU (0.25). SGLang gets dedicated GPU. Separate via Ray placement groups. |

---

## Part 9: Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Voice round-trip latency | < 500ms | Time from user speaks → STT → MetaAgent → TTS → user hears |
| A2UI render latency | < 200ms | Time from agent calls a2ui_render → UI visible on screen |
| Concurrent voice sessions | 10+ per node | LiveKit workers as Ray actors |
| Disconnect recovery | < 2s | Reconnect → resume workflow → notify user |
| RadixAttention cache hit rate | > 80% | SGLang cache metrics on shared conversation prefixes |

---

## Research References

- **LiveKit Agents**: [github.com/livekit/agents](https://github.com/livekit/agents) — VoicePipelineAgent, AgentServer, MCP support
- **LiveKit Docs**: [docs.livekit.io](https://docs.livekit.io/agents) — Python SDK, STT/TTS plugins, data channels
- **A2UI Protocol**: [github.com/google/a2ui](https://github.com/google/a2ui) — Spec, renderers, quickstart
- **A2UI + AG-UI (CopilotKit)**: [a2ui.org/quickstart](https://a2ui.org/quickstart) — React/Next.js integration
- **LiveKit MCP Server**: [docs.livekit.io/mcp](https://docs.livekit.io/mcp) — For coding agent assistance
