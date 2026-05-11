# Lab 15 — Integration Plan: Ray + SGLang

**Status:** Planning (Revised — Realistic Assessment)  
**Parent:** Lab 14 (Durable Meta Agent: DSPy + Dapr)  
**Goal:** Integrate SGLang for high-throughput inference and Ray for distributed compute into the Durable Meta Agent. BitNet b1.58 deferred to research track.  
**Revised:** 2026-05-10 — Reality check incorporated. Vision separated from what ships today.

---

## Executive Summary

Lab 15 adds two proven technology layers to the existing DSPy + Dapr + MCP stack:

| Layer | Technology | Status | What It Delivers |
|-------|-----------|--------|-----------------|
| State & Workflows | **Dapr** (existing) | Production | Durable execution, checkpointing, pub/sub |
| Reasoning Engine | **DSPy** (existing) | Production | Modules, signatures, optimizers, BAMLAdapter |
| Tool Integration | **MCP** (existing) | Production | Tool discovery, auth, health checks |
| **Fast Inference** | **SGLang** (new) | **Ship Today** | RadixAttention, continuous batching, 4-bit quant, FP8 KV cache |
| **Distributed Compute** | **Ray** (new) | **Ship Today** | Task parallelism, resource isolation |
| Ultra-Efficient Weights | BitNet b1.58 | **Research Track** | Ternary weights — no 70B model or production kernels exist yet |
| KV Cache Compression | TurboQuant | **Use SGLang built-in** | `--kv-cache-dtype fp8_e4m3` works today |

**Minimum Viable Integration (6 hours):** SGLang server + `dspy.LM(base_url=...)` + Ray tasks for parallel agents. This delivers 2-5x latency reduction and N× throughput on N GPUs with zero DSPy code changes.

**North Star (2027-2028):** BitNet b1.58 + PD disaggregation + Ray Serve autoscaling + formal verification. Documented in Part 6 for reference but not in the critical path.

---

## Reality Check — What Actually Works Today

### The Good (Ship It)

| Integration | Evidence | Effort |
|-------------|----------|--------|
| **SGLang via `dspy.LM(base_url=...)`** | SGLang exposes OpenAI-compatible API. DSPy already supports this. RadixAttention gives 2-5x latency reduction on multi-agent workloads with shared prefixes. | ~1 hour |
| **4-bit AWQ/GPTQ via SGLang** | `--quantization awq` or `--quantization gptq` flags. 70B model at 4-bit fits in ~35 GB VRAM. Works today. | ~30 min |
| **FP8 KV cache** | `--kv-cache-dtype fp8_e4m3` single flag in SGLang. 2x memory reduction for long contexts. No custom kernels. | ~5 min |
| **Ray tasks for parallel agents** | `@ray.remote` + `ray.get()`. Standard Ray pattern. Each agent runs on a separate GPU. | ~4 hours |

### The Not Ready (Defer)

| Integration | Why Not Today | What To Do Instead |
|-------------|--------------|-------------------|
| **BitNet b1.58** | Only a 2B parameter model exists (BitNet-b1.58-2B4T). No 70B. Custom CUDA kernels for ternary MatMul are research-grade. On standard GPUs, BitNet runs through FP16 pipelines — you get memory savings but NOT compute savings. The "10x throughput" requires custom hardware (LPUs, ASICs) nobody has. | Use AWQ/GPTQ 4-bit quantization. Proven, production, available for all major models. |
| **PD Disaggregation** | Requires 2+ nodes, custom IB/NVLink config, complex SGLang Router setup. Marginal gain for single-node. | Single-node tensor parallelism (`--tp N`). Add PD disaggregation when scaling to multi-node. |
| **Ray Serve for SGLang** | Adds complexity (autoscaling config, deployment management) with no benefit over a bare SGLang server for single-node. | Bare `python -m sglang.launch_server`. Add Ray Serve when you need multi-replica autoscaling. |
| **Massive parallel LSE** (100+ branches) | Running 100+ LSE branches in Ray is expensive ($500 compute to optimize a prompt that saves $0.05). Overfitting risk — agent finds a prompt that works for one test case but fails in the real world. | Cap at 10 branches. Only run LSE when quality score drops below threshold. Profile before parallelizing. |
| **Lean 4 formal proofs** | LLMs fail 70-80% on non-trivial Lean 4 proofs. Agent enters infinite generate-fail-retry loop. Burns tokens. | Property-based testing (Hypothesis) + Z3 for simple constraint verification. Lean 4 for research only. |

### The Infrastructure Risk (Manage It)

**Dapr + Ray coexistence is the highest-risk integration point.** Two competing failure-recovery systems with no coordination:

```
Dapr workflow calls ray.get(task.remote()) 
  → Ray worker crashes
  → Dapr workflow hangs forever waiting for ray.get()
  → Dapr's retry policy doesn't know about Ray's retry policy
  → Two competing failure-recovery systems with no coordination
```

**Mitigation:** Circuit-breaker pattern. If `ray.get()` times out, fall back to `InProcessExecutor` for that iteration, log the degradation, and continue. Ray is always opt-in (`--ray` flag). `InProcessExecutor` is the default.

---

## Minimum Viable Architecture (What We Build)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Dapr Durability Layer                         │
│   @workflow_entry · checkpointing · state store · pub/sub       │
├─────────────────────────────────────────────────────────────────┤
│                    Ray Compute Layer (opt-in)                    │
│   RayModuleExecutor · parallel tasks · circuit-breaker fallback │
├─────────────────────────────────────────────────────────────────┤
│                    SGLang Inference Layer                        │
│   RadixAttention · Continuous Batching · 4-bit AWQ/GPTQ        │
│   FP8 KV Cache · Tensor Parallelism · OpenAI-compatible API     │
├─────────────────────────────────────────────────────────────────┤
│                    DSPy Reasoning Engine (unchanged)             │
│   RLM · ReAct · CodeAct · CoT · BAMLAdapter · Optimizers       │
├─────────────────────────────────────────────────────────────────┤
│                    MCP Tool Layer (unchanged)                    │
│   MCPBridge · get_dspy_tools() · get_agent_tools()              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Architecture Principles

### 1.1 Orthogonal Layers, Not Replacements

**Critical principle from Oracle review:** Each layer solves an orthogonal problem. No layer replaces another.

- **Dapr** = "What step am I on? What do I know?" (state, recovery)
- **Ray** = "Where should this run? How many copies?" (distribution, scaling)
- **SGLang** = "How fast can I think? Is my output valid?" (inference speed, structure)
- **DSPy** = "What should I reason about? How do I improve?" (logic, optimization)

The direction of integration is always:

```
Dapr workflow → calls Ray task → calls SGLang endpoint → returns to Dapr → checkpoints
```

Never: Ray → Dapr (Ray workers don't have the Dapr sidecar).

### 1.2 Dual-Path Pattern (Preserved from Lab 14)

Every new subsystem follows the same ABC pattern:

| Subsystem | Dev (no infra) | Production |
|-----------|---------------|------------|
| Frontier | `InMemoryFrontier` | `DaprFrontier` |
| Agent Stack | `AgentStack` | `DaprAgentStack` |
| LSE | `LSEOptimizer` | `DaprLSEOptimizer` |
| **Module Execution** | `InProcessExecutor` | **`RayModuleExecutor`** |
| **LLM Backend** | `dspy.LM(base_url=...)` | **`SGLangLM` (constrained)** |
| **Model Serving** | Local SGLang server | **Ray Serve + SGLang Router** |

### 1.3 SGLang Integration Is Configuration, Not Code

**Key insight:** SGLang exposes an OpenAI-compatible API. DSPy already supports `dspy.LM(model=..., base_url=...)`. RadixAttention, continuous batching, and tensor parallelism are all transparent server-side features.

The simplest SGLang integration is:

```python
dspy.configure(
    lm=dspy.LM(
        model="openai/meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:30000/v1",  # SGLang server
        api_key="None",
    )
)
```

Zero DSPy code changes. RadixAttention caches shared agent prefixes automatically. Continuous batching handles concurrent requests from multiple agents transparently.

---

## Part 2: Ray Integration — Detailed Design

### 2.1 Ray Tasks for Parallel Module Execution

**Oracle recommendation: Use Ray tasks, not Ray actors.** DSPy modules are stateless — they generate fresh predictions each call. Actors add lifecycle complexity with no benefit.

```python
# ray/executor.py

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import dspy

class ModuleExecutor(ABC):
    """ABC for executing DSPy modules — mirrors Frontier/Stack pattern."""
    
    @abstractmethod
    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        ...
    
    @abstractmethod
    def execute_batch(self, modules: list[dspy.Module], batch_kwargs: list[dict]) -> list[dspy.Prediction]:
        ...


class InProcessExecutor(ModuleExecutor):
    """Default: execute modules in the current process. Zero infra."""
    
    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        return module(**kwargs)
    
    def execute_batch(self, modules, batch_kwargs):
        return [m(**kw) for m, kw in zip(modules, batch_kwargs)]


class RayModuleExecutor(ModuleExecutor):
    """Production: execute modules as Ray tasks across a cluster.
    
    Each module execution becomes a remote Ray task with configurable
    resource requirements (GPU/CPU). Results are collected via ray.get().
    """
    
    def __init__(self, num_gpus: float = 0, num_cpus: float = 1, timeout: float = 300):
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.timeout = timeout
    
    @ray.remote(num_gpus=0, num_cpus=1)
    @staticmethod
    def _execute_remote(module: dspy.Module, kwargs: dict) -> dict:
        """Execute a DSPy module in a Ray worker.
        
        Returns serialized prediction to avoid Ray object store bloat.
        """
        prediction = module(**kwargs)
        return {"prediction": prediction}
    
    def execute(self, module: dspy.Module, **kwargs) -> dspy.Prediction:
        ref = self._execute_remote.remote(module, kwargs)
        result = ray.get(ref, timeout=self.timeout)
        return result["prediction"]
    
    def execute_batch(self, modules, batch_kwargs) -> list[dspy.Prediction]:
        """Execute multiple modules in parallel across Ray cluster."""
        refs = [
            self._execute_remote.remote(m, kw)
            for m, kw in zip(modules, batch_kwargs)
        ]
        results = ray.get(refs, timeout=self.timeout)
        return [r["prediction"] for r in results]
```

### 2.2 Ray Serve for SGLang Deployment Management

Ray Serve wraps the SGLang HTTP server for production deployment:

```python
# ray/serve_deployment.py

from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app

# SGLang under Ray Serve — uses vLLM-compatible engine kwargs
# SGLang supports the same engine_kwargs interface
llm_config = LLMConfig(
    model_loading_config={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_source": "meta-llama/Llama-3.1-8B-Instruct",
    },
    accelerator_type="L4",
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 1,
            "max_replicas": 4,
            "target_num_ongoing_requests": 32,
        },
        "max_ongoing_requests": 64,
    },
    engine_kwargs={
        "max_model_len": 8192,
        "tensor_parallel_size": 2,  # Multi-GPU sharding
        "trust_remote_code": True,
    },
)

app = build_openai_app({"llm_configs": [llm_config]})
```

### 2.3 Parallel LSE — Distributed Self-Optimization

LSE (Learning to Self-Evolve) runs multiple evaluation branches. Currently sequential. With Ray:

```python
# ray/lse_parallel.py

@ray.remote(num_cpus=1)
def evaluate_lse_branch(
    agent_module: dspy.Module,
    test_cases: list[dspy.Example],
    metric: callable,
) -> float:
    """Evaluate one LSE branch on a Ray worker."""
    scores = []
    for example in test_cases:
        prediction = agent_module(**example.inputs())
        scores.append(metric(example, prediction))
    return sum(scores) / len(scores)


def parallel_lse_evaluate(
    branches: list[dspy.Module],
    test_cases: list[dspy.Example],
    metric: callable,
) -> list[float]:
    """Evaluate all LSE branches in parallel across the Ray cluster."""
    refs = [evaluate_lse_branch.remote(b, test_cases, metric) for b in branches]
    return ray.get(refs)
```

### 2.4 Resource Isolation

```python
# Ray Placement Groups for GPU bundling
pg = ray.util.placement_group(
    bundles=[{"GPU": 2, "CPU": 4}],  # Bundle for model serving
    strategy="STRICT_SPREAD",
)

# Agent resource constraints
@ray.remote(num_gpus=0.25, num_cpus=2)  # Fractional GPU for inference
class AgentWorker:
    ...
```

### 2.5 Ray + Dapr Coexistence

The boundary is clean:

```
┌──────────────────────────────────────┐
│ Dapr Workflow (DurableMetaAgent)     │
│   for iteration in meta.run_stack(): │
│     yield ctx.set_state(...)          │  ← Dapr owns the outer loop
│     result = ray.get(                 │
│       execute_module.remote(...)      │  ← Ray owns the inner compute
│     )                                 │
│     yield ctx.set_state(...)          │  ← Back to Dapr for checkpointing
└──────────────────────────────────────┘
```

**Rules:**
1. Dapr workflow → calls Ray task → awaits result → checkpoints. Always.
2. Ray workers never call Dapr APIs (no sidecar access).
3. Ray handles compute parallelism; Dapr handles state durability.
4. Swarm mode can use Ray for intra-node parallelism and Dapr for inter-node coordination.

---

## Part 3: SGLang Integration — Detailed Design

### 3.1 Architecture: SGLang as Inference Backend

SGLang provides two modes:

| Mode | API | Use Case |
|------|-----|----------|
| **HTTP Server** (`python -m sglang.launch_server`) | OpenAI-compatible REST | Production, multi-client |
| **Engine** (`sgl.Engine(model_path=...)`) | Python `generate()` | Offline batch, embedded |

**Recommendation:** Use HTTP server mode for production (Ray Serve manages it), Engine mode for testing/dev.

### 3.2 SGLang Server Launch Configuration

```bash
# Single-GPU serving with RadixAttention
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --enable-cache-report

# Multi-GPU tensor parallelism
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --tp 4 \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.85

# PD Disaggregation (Prefill/Decode separation)
# Prefill node (fast context processing)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode prefill \
  --tp 4 \
  --host 0.0.0.0 \
  --port 8000

# Decode node (fast token generation)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-70B-Instruct \
  --disaggregation-mode decode \
  --tp 4 \
  --host 0.0.0.0 \
  --port 8232

# Router (load balances prefill/decode)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --policy cache_aware \
  --prefill http://prefill-host:8000 8995 \
  --decode http://decode-host:8232 \
  --port 30000
```

### 3.3 RadixAttention — The Biggest Win

RadixAttention is the key differentiator for agent workloads:

- **Prefix caching** — When the Meta-Agent forks into 3 sub-tasks, all three share the same system prompt + conversation history prefix. SGLang caches this prefix as a radix tree. Subsequent requests reuse cached KV states.
- **Automatic** — No code changes needed. The SGLang server handles prefix matching internally.
- **Measured impact** — 2-5x latency reduction for multi-agent scenarios with shared context.

```python
# This just works — SGLang's server automatically detects shared prefixes
# across concurrent requests from different DSPy modules

# Agent 1: "You are a research assistant. [shared prefix] ... [task 1 specifics]"
# Agent 2: "You are a research assistant. [shared prefix] ... [task 2 specifics]"  
# Agent 3: "You are a research assistant. [shared prefix] ... [task 3 specifics]"
# → SGLang processes the shared prefix ONCE, reuses KV cache for all three
```

### 3.4 SGLangLM Adapter (Constrained Decoding)

**Only build this if BAMLAdapter retry rate exceeds 10%.** For most cases, standard `dspy.LM(base_url=...)` suffices.

```python
# ray/sglang_lm.py — Conditional, only if needed

class SGLangLM(dspy.LM):
    """Extended LM that maps DSPy response_format → SGLang grammar constraints.
    
    Build this ONLY if:
    - BAMLAdapter retry rate > 10% on structured outputs
    - Z3/Lean4 code generation has > 5% syntax errors
    
    Otherwise, standard dspy.LM(base_url=...) with BAMLAdapter is sufficient.
    """
    
    def __init__(self, model: str, base_url: str = "http://localhost:30000/v1", **kwargs):
        super().__init__(model=model, base_url=base_url, api_key="None", **kwargs)
    
    def _map_response_format(self, response_format: dict) -> dict:
        """Translate DSPy response_format to SGLang grammar parameter."""
        if not response_format:
            return {}
        
        json_schema = response_format.get("json_schema", {})
        return {
            "grammar": json_schema,  # SGLang server-side constrained decoding
            "sampling_params": {
                "max_new_tokens": 4096,
                "temperature": 0.0,  # Deterministic for formal verification
            }
        }
    
    def __call__(self, prompt: str, **kwargs) -> list[dict]:
        """Override to inject grammar constraints for structured output."""
        response_format = kwargs.pop("response_format", None)
        if response_format:
            kwargs.update(self._map_response_format(response_format))
        return super().__call__(prompt, **kwargs)
```

### 3.5 SGLang Engine (Offline/Embedded Mode)

For testing without a running server:

```python
# ray/sglang_engine.py

import sglang as sgl

class EmbeddedSGLangEngine:
    """In-process SGLang engine for dev/testing. No HTTP server needed."""
    
    def __init__(self, model_path: str, tp_size: int = 1):
        self.engine = sgl.Engine(model_path=model_path, tp_size=tp_size)
    
    def generate(self, prompts: list[str], **sampling_params) -> list[dict]:
        return self.engine.generate(prompts, sampling_params)
    
    async def async_generate(self, prompts: list[str], **sampling_params) -> list[dict]:
        return await self.engine.async_generate(prompts, sampling_params)
    
    def shutdown(self):
        self.engine.shutdown()
```

---

## Part 4: BitNet b1.58 Integration — Detailed Design

### 4.1 Architecture Impact

BitNet b1.58 replaces floating-point matrix multiplication with ternary integer addition. This creates a cascade of benefits:

| Aspect | FP16 Baseline | 4-bit Quant | BitNet 1.58 |
|--------|-------------|-------------|-------------|
| Weight storage (70B model) | ~140 GB | ~35 GB | **~13 GB** |
| Core operation | FMA (float) | Integer MatMul | **Integer Addition** |
| Energy per token | ~1x | ~0.3x | **~0.05x** |
| Throughput (same hardware) | Baseline | ~2-3x | **~5-10x** |
| Fits in consumer GPU? | No | Barely (70B) | **Yes (70B in ~13 GB)** |

### 4.2 Integration Path

BitNet models are loaded through SGLang's standard model loading interface. The integration is at the SGLang configuration level:

```bash
# Launch SGLang with a BitNet-quantized model
python -m sglang.launch_server \
  --model-path microsoft/BitNet-b1.58-2B4T \
  --host 0.0.0.0 \
  --port 30000
```

From DSPy's perspective, the model is just another OpenAI-compatible endpoint:

```python
dspy.configure(
    lm=dspy.LM(
        model="openai/microsoft/BitNet-b1.58-2B4T",
        base_url="http://localhost:30000/v1",
    )
)
```

### 4.3 The "Radical Scaling" Opportunity

Because BitNet models are 10-20x smaller:

1. **Single-GPU Multi-Agent Swarms** — Run 10+ agent instances on a single GPU. Each agent is a Ray actor with fractional GPU allocation.
2. **Massive UCB Tree Search** — Explore 1000+ reasoning paths in parallel on a single node. The current UCB frontier explores sequentially; BitNet makes parallel exploration economically viable.
3. **Edge Deployment** — Durable agents that run on laptops, phones, edge servers. State transfer is near-instant due to tiny model size.

### 4.4 The Formal Verification Question

**Risk:** BitNet's 1.58-bit weights may introduce "fuzzy logic" that undermines Z3/Lean4 verification quality.

**Mitigation strategy:**
1. **Hybrid architecture** — Use BitNet for exploration (fast, cheap), FP16 for verification (precise, expensive).
2. **Escalation gate** — BitNet agent generates candidate solutions; FP16 agent validates them.
3. **Benchmark threshold** — If BitNet verification success rate drops below 90% of FP16 baseline, keep BitNet for non-critical paths only.

```python
# Hybrid execution pattern
class HybridExecutor:
    """Use BitNet for exploration, FP16 for verification."""
    
    def __init__(self):
        self.exploration_lm = dspy.LM(
            model="openai/BitNet-b1.58-2B4T",
            base_url="http://localhost:30000/v1",  # BitNet SGLang
        )
        self.verification_lm = dspy.LM(
            model="openai/meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://localhost:30001/v1",  # FP16 SGLang
        )
    
    def explore(self, task: str, n: int = 100) -> list[str]:
        """Generate 100 candidate solutions using cheap BitNet inference."""
        # Run as parallel Ray tasks
        refs = [ray_task.remote(self.exploration_lm, task) for _ in range(n)]
        return ray.get(refs)
    
    def verify(self, candidate: str) -> bool:
        """Verify a single candidate using precise FP16 inference."""
        return verify_with_z3(self.verification_lm, candidate)
```

---

## Part 5: TurboQuant Integration — Detailed Design

### 5.1 KV Cache Compression

TurboQuant compresses the KV cache (short-term memory) to 4-bit, reducing VRAM usage for long-context inference:

| Context Length | FP16 KV Cache | TurboQuant 4-bit | Reduction |
|---------------|--------------|-------------------|-----------|
| 4K tokens | ~2 GB | ~0.5 GB | 4x |
| 32K tokens | ~16 GB | ~4 GB | 4x |
| 128K tokens | ~64 GB | ~16 GB | 4x |

### 5.2 SGLang Configuration

```bash
# SGLang with 4-bit KV cache
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype fp8_e4m3 \
  --host 0.0.0.0 \
  --port 30000
```

### 5.3 Impact on Agent Workloads

Durable agents accumulate long conversation histories. TurboQuant:
- Reduces memory pressure during multi-iteration research loops
- Enables longer context windows (more "memory" per agent)
- Complements BitNet: weights are 1.58-bit, KV cache is 4-bit

---

## Part 6: Combined Architecture — "The Full Stack"

### 6.1 Complete System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT / CLI                                  │
│   uv run python -m lab.15_ray_sglang --query "..." run              │
├─────────────────────────────────────────────────────────────────────┤
│                     DAPR DURABILITY LAYER                            │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐   │
│  │ DurableMeta  │  │ SwarmCoord   │  │ State Store (Redis)     │   │
│  │ Agent        │  │              │  │ · frontier state        │   │
│  │              │  │ · dispatch   │  │ · agent stack           │   │
│  │ · checkpoint │  │ · heartbeat  │  │ · LSE runs             │   │
│  │ · resume     │  │ · reassign   │  │ · iteration results    │   │
│  └──────┬───────┘  └──────┬───────┘  └─────────────────────────┘   │
│         │                 │                                         │
│         │ ray.get()       │ ray.get()                               │
│         ▼                 ▼                                         │
├─────────────────────────────────────────────────────────────────────┤
│                      RAY COMPUTE LAYER                              │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  RayModuleExecutor                                           │   │
│  │  · InProcessExecutor (dev) / RayModuleExecutor (production)  │   │
│  │  · execute_batch() → parallel Ray tasks                      │   │
│  │  · Resource: {GPU: fractional, CPU: N}                       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Ray Serve (SGLang Deployment)                               │   │
│  │  · LLMConfig → autoscaling SGLang replicas                   │   │
│  │  · Placement Groups for multi-GPU tensor parallelism         │   │
│  │  · Load balancing via Serve handles                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Parallel LSE                                                │   │
│  │  · evaluate_lse_branch.remote() × N branches                 │   │
│  │  · Massive parallel self-optimization with BitNet            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                            │
│         ▼                                                            │
├─────────────────────────────────────────────────────────────────────┤
│                     SGLang INFERENCE LAYER                           │
│                                                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │
│  │ SGLang Server  │  │ SGLang Router  │  │ PD Disaggregation    │  │
│  │ · RadixAttn    │  │ · cache_aware  │  │ · Prefill node       │  │
│  │ · Cont. Batch  │  │ · load balance │  │ · Decode node        │  │
│  │ · Struct. Out  │  │ · routing      │  │ · KV cache transfer  │  │
│  │ · Tensor Par.  │  │                │  │                      │  │
│  └───────┬────────┘  └────────────────┘  └──────────────────────┘  │
│          │                                                          │
│          ▼                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                   MODEL OPTIMIZATION LAYER                           │
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐    │
│  │ BitNet b1.58         │  │ TurboQuant                       │    │
│  │ · {-1, 0, +1} weights│  │ · 4-bit KV cache                 │    │
│  │ · MatMul-free        │  │ · 4x memory reduction            │    │
│  │ · 10-20x less VRAM   │  │ · Long context support            │    │
│  └──────────────────────┘  └──────────────────────────────────┘    │
│          │                                                          │
│          ▼                                                          │
├─────────────────────────────────────────────────────────────────────┤
│                   DSPy REASONING ENGINE (unchanged)                  │
│  RLM · ReAct · CodeAct · CoT · BestOfN · MultiChainComparison      │
│  Refine · BAMLAdapter · GFL Pipeline · LSE · Trace2Skill           │
├─────────────────────────────────────────────────────────────────────┤
│                   MCP TOOL LAYER (unchanged)                         │
│  MCPBridge · crawl4ai · Exa · sequential-thinking · ...             │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Resource Efficiency Comparison (Realistic)

| Configuration | VRAM (70B model) | Throughput | Status |
|--------------|-------------------|------------|--------|
| FP16 baseline (API) | 140 GB (server) | 1x | Current |
| + SGLang RadixAttention | 140 GB | 2-3x | **Ship today** |
| + 4-bit AWQ quantization | **~35 GB** | 2-3x | **Ship today** |
| + FP8 KV cache (`--kv-cache-dtype fp8_e4m3`) | **~25 GB** | 2-3x | **Ship today** |
| + Ray parallel (4 GPUs) | ~25 GB × 1 GPU | **8-12x** | **Ship today** |
| + BitNet b1.58 (theoretical) | ~13 GB | 5-10x | **Not ready — no 70B model, no kernels** |

**Realistic result:** 70B model on 2× A100-40G with 4-bit + FP8 KV cache + RadixAttention + Ray parallel = 8-12x baseline throughput. Works today.

---

## Part 7: Implementation Plan — Phased Delivery (Revised)

### Phase 1: SGLang as Inference Backend (Day 1, ~1-2 hours)
**This is the single highest-ROI integration.** Configuration, not code.

| Task | Files | Dependencies |
|------|-------|-------------|
| 1.1 Add `--sglang-endpoint` CLI flag | `cli.py` | None |
| 1.2 Configure `dspy.LM(base_url=...)` | `cli.py` | None |
| 1.3 Add `sglang-warmup` CLI command | `cli.py` | SGLang server |
| 1.4 Add SGLang server launch script (with 4-bit + FP8 KV) | `scripts/launch_sglang.sh` | sglang pip package |
| 1.5 Update `pyproject.toml` with sglang dep | `pyproject.toml` | None |

**SGLang launch configuration (production-ready today):**
```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --quantization awq \
  --kv-cache-dtype fp8_e4m3 \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-metrics \
  --enable-cache-report
```

**Deliverable:** `uv run python -m lab.15_ray_sglang --sglang-endpoint http://localhost:30000/v1 run` works.

### Phase 2: Ray Tasks for Module Execution (Day 2, ~4 hours)

| Task | Files | Dependencies |
|------|-------|-------------|
| 2.1 Create `ModuleExecutor` ABC | `ray/executor.py` | None |
| 2.2 Implement `InProcessExecutor` | `ray/executor.py` | None |
| 2.3 Implement `RayModuleExecutor` with circuit-breaker | `ray/executor.py` | `ray` pip package |
| 2.4 Add `--ray` CLI flag | `cli.py` | 2.2, 2.3 |
| 2.5 Wire executor into `MetaAgent.run_stack_iter()` | `meta/meta_agent.py` | 2.1 |
| 2.6 Test: parallel execution of 4 agents on 4 GPUs | `tests/` | 2.3 |

**Circuit-breaker pattern (critical for Dapr + Ray coexistence):**
```python
def execute_with_fallback(self, module, **kwargs):
    try:
        ref = self._execute_remote.remote(module, kwargs)
        return ray.get(ref, timeout=self.timeout)
    except (ray.exceptions.TimeoutError, ray.exceptions.WorkerCrashedError):
        logger.warning("Ray task failed, falling back to InProcessExecutor")
        return InProcessExecutor().execute(module, **kwargs)
```

**Deliverable:** `uv run python -m lab.15_ray_sglang --ray run` distributes agents across Ray cluster with automatic fallback.

### Phase 3: Parallel LSE — Capped at 10 Branches (Day 3, ~2 hours)
**Only build this if profiling shows LSE is a bottleneck.** Cap at `min(available_GPUs, 10)` branches. Only trigger when quality score drops below threshold.

| Task | Files | Dependencies |
|------|-------|-------------|
| 3.1 Create `parallel_lse_evaluate()` | `ray/lse_parallel.py` | Phase 2 |
| 3.2 Wire into LSE loop with quality gate | `evolution/lse.py` | 3.1 |
| 3.3 Benchmark: sequential vs parallel LSE | Manual | 3.2 |

**Deliverable:** LSE runs up to 10 branches in parallel when quality degrades.

### Phase 4: SGLangLM Adapter (Conditional — Only If Needed)
**Only build if BAMLAdapter retry rate > 10% on structured outputs.** Measure first.

| Task | Files | Dependencies |
|------|-------|-------------|
| 4.1 Measure BAMLAdapter retry rate | Profiling | Phase 1 |
| 4.2 If > 10%: implement `SGLangLM(dspy.LM)` | `ray/sglang_lm.py` | Phase 1 |
| 4.3 Map `response_format` → SGLang grammar | `ray/sglang_lm.py` | 4.2 |

**Deliverable:** Constrained decoding (only if BAMLAdapter isn't sufficient).

### Phase 5: End-to-End Integration (Day 4, ~3 hours)

| Task | Files | Dependencies |
|------|-------|-------------|
| 5.1 Test: Dapr + SGLang + Ray end-to-end | Manual | Phases 1-3 |
| 5.2 Add multi-GPU tensor parallelism script | `scripts/launch_sglang_tp.sh` | Phase 1 |
| 5.3 Documentation and examples | `README.md` | All |
| 5.4 Update project-level README | `../README.md` | 5.3 |

**Deliverable:** Full realistic stack running end-to-end.

### Research Track (Deferred — Not in Critical Path)

These are documented for future reference. Do NOT implement until ecosystem catches up.

| Task | Trigger Condition |
|------|-----------------|
| BitNet b1.58 integration | When a 70B BitNet model + production CUDA kernels exist |
| PD Disaggregation | When scaling to multi-node (2+ nodes) |
| Ray Serve for SGLang | When multi-replica autoscaling is needed |
| Ray Placement Groups | When multi-GPU models need bundled resources |
| Massive parallel LSE (100+ branches) | When profiling proves it's the bottleneck AND cost is justified |
| Lean 4 formal proof integration | When a proof-assistant model (AlphaProof-class) is available |
| SGLang Router for multi-model | When serving multiple model variants simultaneously |
| HybridExecutor (BitNet explore + FP16 verify) | When BitNet quality is benchmarked at ≥90% of FP16 |

---

## Part 8: File Structure

New files for Lab 15 (additions to Lab 14's structure):

```
lab/15_ray_sglang/
├── __init__.py
├── __main__.py
├── cli.py                    # Extended with --sglang-endpoint, --ray flags
├── README.md                 # Updated for Ray + SGLang (realistic)
│
├── core/
│   ├── durable_meta_agent.py # Extended: accepts ModuleExecutor
│   └── __init__.py
│
├── ray/                      # NEW: Ray compute layer
│   ├── __init__.py
│   ├── executor.py           # ModuleExecutor ABC, InProcessExecutor, RayModuleExecutor + circuit-breaker
│   ├── lse_parallel.py       # Parallel LSE (capped at 10 branches)
│   └── sglang_lm.py          # SGLangLM(dspy.LM) — conditional, only if BAMLAdapter retry > 10%
│
├── meta/                     # Existing — minimal changes
│   ├── meta_agent.py         # Modified: accepts executor dependency
│   ├── agent_generator.py    # Unchanged
│   ├── agent_stack.py        # Unchanged
│   └── __init__.py
│
├── evolution/                # Existing — unchanged
├── memory/                   # Existing — unchanged
├── dapr/                     # Existing — unchanged
├── swarm/                    # Existing — can use Ray for intra-node parallelism
├── mcp/                      # Existing — unchanged
│
├── config/
│   ├── mcp_servers.json      # Existing
│   └── ray_cluster.yaml      # NEW: Ray cluster configuration
│
├── scripts/                  # NEW: Infrastructure launch scripts
│   ├── launch_sglang.sh      # Start SGLang server (4-bit AWQ + FP8 KV cache)
│   └── launch_sglang_tp.sh   # Start SGLang with tensor parallelism
│
└── tests/                    # NEW: Integration tests
    ├── test_executor.py
    └── test_sglang_integration.py
```

---

## Part 9: Risks and Mitigations (Revised)

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| **Dapr + Ray conflict** (competing failure recovery) | **High** | **High** | Circuit-breaker: `ray.get()` timeout → `InProcessExecutor` fallback. Strict direction: Dapr → Ray, never reverse. `InProcessExecutor` is default. |
| **State bloat** (Redis grows over 100+ iterations) | High | Medium | Continue-as-New pattern (from Lab 14). Periodic frontier compaction. Dirty-flag batching. |
| **SGLang model loading latency** on first request | Medium | High | `sglang-warmup` CLI command. Pre-load model before starting agent loop. |
| **Ray task cold start** (~100ms overhead per task) | Medium | Medium | `InProcessExecutor` default. `RayModuleExecutor` opt-in. Measure overhead before committing. Only use Ray for batch execution (multiple agents). |
| **LSE overfitting** (optimizes for test case, fails in real world) | High | High | Cap at 10 branches. Only trigger on quality degradation. Validate LSE-optimized prompts on held-out data. |
| **Over-abstracting the LM backend** | Medium | Low | Don't build an LLM provider factory. `dspy.LM(base_url=...)` is the default. `SGLangLM` only if needed. Two concrete paths. |
| **DSPy compilation unpredictability** | Medium | Medium | Keep human-in-the-loop for production decisions. Don't auto-deploy compiled programs without validation. |
| **Infrastructure complexity** (Dapr + Ray + Redis + SGLang) | High | High | Each layer is independently usable. Dual-path pattern ensures dev mode works with zero infra (`uv run ... run`). Build layers incrementally. |
| **BitNet quality degradation** | **N/A (deferred)** | N/A | Deferred to research track. Use AWQ/GPTQ 4-bit today. Revisit when 70B BitNet model + kernels exist. |
| **Formal verification failure rate** | High | High | Downgrade to property-based testing (Hypothesis) + Z3 for simple constraints. Lean 4 for research only. |

---

## Part 10: Success Metrics (Revised — Measurable, Realistic)

| Metric | Baseline (Lab 14) | Target (Lab 15) | How Measured | Status |
|--------|-------------------|-----------------|-------------|--------|
| Inference latency (per agent call) | ~2s | **<500ms** | SGLang RadixAttention on shared prefixes | **Ship today** |
| Parallel agent throughput | 1 agent/time | **N agents on N GPUs** | Ray tasks with circuit-breaker | **Ship today** |
| VRAM usage (70B model) | N/A (API) | **~35 GB** (4-bit AWQ) | SGLang `--quantization awq` | **Ship today** |
| KV cache memory (70B, 32K context) | ~16 GB | **~8 GB** | SGLang `--kv-cache-dtype fp8_e4m3` | **Ship today** |
| LSE evaluation time (10 branches) | ~20s sequential | **~3s parallel** | Ray parallel tasks, capped at 10 | After Phase 3 |
| Structured output retry rate | ~15% | **<10%** | Measure BAMLAdapter retries; add SGLangLM if needed | Conditional |
| Dapr → Ray fallback rate | N/A | **<5%** | Log circuit-breaker activations | After Phase 2 |

---

## Part 11: CLI Commands (Revised)

### Pure DSPy mode (existing, unchanged)
```bash
uv run python -m lab.15_ray_sglang --query "Research topic" run
```

### SGLang mode (highest ROI — do this first)
```bash
# Terminal 1: Start SGLang server with 4-bit + FP8 KV cache
bash lab/15_ray_sglang/scripts/launch_sglang.sh

# Terminal 2: Run with SGLang backend
uv run python -m lab.15_ray_sglang \
  --query "Research topic" \
  --sglang-endpoint http://localhost:30000/v1 \
  run
```

### SGLang + Ray mode (production)
```bash
# Start Ray cluster
ray start --head

# Start SGLang server
bash lab/15_ray_sglang/scripts/launch_sglang.sh

# Run with SGLang + Ray parallelism
uv run python -m lab.15_ray_sglang \
  --query "Research topic" \
  --sglang-endpoint http://localhost:30000/v1 \
  --ray \
  run
```

### Dapr + SGLang + Ray (full production stack)
```bash
# Start infrastructure
redis-server &> /dev/null &
ray start --head
bash lab/15_ray_sglang/scripts/launch_sglang.sh

# Run with all layers
dapr run --app-id ray-meta-agent --app-protocol grpc --app-port 8000 \
  --resources-path lab/15_ray_sglang/dapr/resources -- \
  uv run python -m lab.15_ray_sglang \
  --query "Research topic" \
  --sglang-endpoint http://localhost:30000/v1 \
  --ray \
  dapr-orchestrator --tracing --dapr-frontier --dapr-lse
```

---

## Research References

### Production-Ready (Build Today)
- **SGLang**: [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) — RadixAttention, continuous batching, AWQ/GPTQ support, FP8 KV cache
- **Ray**: [docs.ray.io](https://docs.ray.io) — Ray Tasks, `@ray.remote`, `ray.get()`, resource management
- **DSPy**: [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) — v3.2.0, BAMLAdapter, LM base_url
- **Dapr Agents**: [github.com/dapr-agent](https://github.com/dapr-agent) — DurableAgent, AgentRunner

### Research Track (North Star — Defer Until Ready)
- **BitNet**: [arxiv.org/abs/2402.17764](https://arxiv.org/abs/2402.17764) — "The Era of 1-bit LLMs" (no 70B model or production kernels yet)
- **TurboQuant**: [github.com/bytedance/TurboQuant](https://github.com/bytedance/TurboQuant) — Use SGLang's built-in `--kv-cache-dtype fp8_e4m3` instead
- **SGLang PD Disaggregation**: SGLang docs, `--disaggregation-mode prefill/decode` (requires multi-node)
- **Ray Serve LLM**: [docs.ray.io/en/latest/serve/tutorials/serve-llm](https://docs.ray.io/en/latest/serve/tutorials/deployment-serve-llm) — LLMConfig, autoscaling (overkill for single-node)
