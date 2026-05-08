# experiments — Complete API Reference

> Lab / showroom for AI agent experiments with **DSPy 3.2**, **MCP**, and **dapr-agents**.

## Overview

```
lab/
├── 01-basics/          DSPy foundations: signatures, Predict, ChainOfThought, custom Modules
├── 02-react-tools/     ReAct agent loop with function tools and PythonInterpreter
├── 03-rag-pipeline/    Multi-stage RAG with ColBERTv2 + BootstrapFewShot optimization
├── 04-optimizers/      Optimizer comparison: BootstrapFewShot, GEPA, MIPROv2, BetterTogether
├── 05-rlm/             Recursive Language Model (REPL-based code-gen agent)
├── 06-advanced/        MultiChainComparison, Module.batch, Ensemble, Streaming, Async, Adapters
├── 07-generative-feedback-loops/  GFL patterns: trace→feedback→update loop with 5 optimizers
├── 08-rlm-mcp/         End-to-end: MCP scrape → GFL optimize → RLM agent → Trace2Skill
├── 09_super_deep_research/  Multi-agent research: UCB frontier + LSE evolution + KG memory
├── 10_dapr_deep_research/   Durable multi-agent: Dapr workflows + DSPy deltas + distillation
├── 11_meta_agent/           Meta-agent: dynamic agent generation via LSE + Trace2Skill
├── 12_formal_evolution/    Formal evolution: Z3 + Lean4 + OpenRouter MCP consensus
├── 13_autonomous_factory/  Autonomous Software Factory: 23 MCP servers, IaC, verification
├── 14_durable_meta_agent/  Durable Meta-Agent: DSPy + Dapr production framework + swarm
├── 99-sandbox/         Scratch space
└── shared/             Env config, LM helpers, research primitives (ResearchDirection, ResearchFrontier ABC)
```

## API Reference by Module

| Module | File(s) | Key Classes/Functions | DSPy Modules Used |
|--------|---------|----------------------|-------------------|
| [01-basics](./01-basics.md) | `main.py` | `Classify`, `Extract`, `Outline`, `DraftSection`, `ArticleWriter` | `ChainOfThought`, `Predict`, `Module`, `Signature` |
| [02-react-tools](./02-react-tools.md) | `main.py` | `search()`, `calculator()` | `ReAct`, `PythonInterpreter` |
| [03-rag-pipeline](./03-rag-pipeline.md) | `main.py` | `RAG`, `gold_answer_metric()` | `ColBERTv2`, `ChainOfThought`, `BootstrapFewShot`, `Evaluate` |
| [04-optimizers](./04-optimizers.md) | `main.py` | `NumClassify`, `exact_match()` | `ChainOfThought`, `BootstrapFewShot`, `GEPA`, `MIPROv2`, `BetterTogether` |
| [05-rlm](./05-rlm.md) | `main.py` | (example script) | `RLM` |
| [06-advanced](./06-advanced.md) | `main.py` | `CompareAnswers`, `Summarize`, `Classify`, `Person` | `MultiChainComparison`, `Module.batch`, `Ensemble.compile()`, `streamify`, `asyncify`, `JSONAdapter` |
| [07-gfl](./07-generative-feedback-loops.md) | `main.py` | `ClassifyIntent`, `intent_metric()`, `gepa_metric()`, `eval_score()` | `ChainOfThought`, `BootstrapFewShot`, `MIPROv2`, `GEPA`, `Evaluate` |
| [08-rlm-mcp](./08-rlm-mcp.md) | `main.py`, `mcp_server.json` | `MCPClient`, `MemoryManager`, `ClassifyContent`, `ResearchReport`, `content_metric()` | `ChainOfThought`, `RLM`, `BootstrapFewShot`, `MIPROv2`, `GEPA`, `BetterTogether`, `BAMLAdapter` |
| [09-super-deep-research](./09-super-deep-research.md) | 16 files in 5 packages | `ResearchOrchestrator`, `ResearchFrontier`, `ResearchDirection`, `KnowledgeGraph`, `MemoryStore`, `MCPClient`, `LSEOptimizer`, `SkillConsolidator`, `SelfDistill`, 5 agent factories | `RLM`, `ChainOfThought` |
| [10-dapr-deep-research](./10-dapr-deep-research.md) | 15 files in 6 packages + shared | `ExplorerAgent`, `DeepReaderAgent`, `SynthesizerAgent`, `CriticAgent`, `ResearchWorkflow`, `InMemoryFrontier`, `DaprFrontier`, `ResearchFrontier` ABC, `ResearchDirection`, `NoopStore`, `LSEOptimizer`, `SkillConsolidator`, `MCPBridge`, `AssessBatchSaturation`, 10 DSPy signatures, shared compile constants | `RLM`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, `BootstrapFewShot`, `Evaluate`, `BAMLAdapter` |
| [11-meta-agent](./11-meta-agent.md) | 10 files in 5 packages | `MetaAgent`, `AgentStack`, `AgentEntry`, `AgentGenerator`, `AnalyzeTask`, `GenerateSignature`, `SelectNextAgent`, `InMemoryFrontier`, `LSEOptimizer`, `SkillConsolidator`, `MCPBridge` | `ChainOfThought`, `BootstrapFewShot` |
| [12-formal-evolution](./12-formal-evolution.md) | 10 files + config | MCP servers: z3-solver, lean-lsp, openrouter; agents auto-discover Z3/Lean4/OpenRouter tools | Same as lab 11 + tool integration |
| [13-autonomous-factory](./13-autonomous-factory.md) | 9 files + config | `AgentGenerator`, `MetaAgent`, `InMemoryFrontier`, `AgentStack`, `GFLPipeline`, `LSEOptimizer`, `SkillConsolidator`, `MCPBridge`, `ResourceBudget`, `AgentEntry`, `AnalyzeTask`, `SelectAgentCompare`, `ImproveAgentPrompt`, `ExtractRules`, `QualityEvaluation`, `ExtractPatterns` | `ChainOfThought`, `BestOfN`, `MultiChainComparison`, `Refine`, `RLM`, `ReAct`, `CodeAct`, `BootstrapFewShot`, `MIPROv2`, `GEPA`, `Evaluate` |
| [14-durable-meta-agent](./14-durable-meta-agent.md) | 35 files + config + dapr resources + swarm | `DurableMetaAgent`, `DurableMetaConfig` (Continue-as-New via `max_iterations_per_segment`), `GeneratedDurableAgent`, `wrap_module()`, `DaprFrontier` (dirty-flag persistence), `DaprLSEOptimizer`, `DaprAgentStack` (delta-update per-entry keys), `AssessBatchSaturation`, `SwarmCoordinator`, `SwarmMetaAgent`, `SwarmTask`, `SwarmDiscovery`, `SwarmHeartbeat`, `SwarmInquiry`, `SwarmResponse` + all lab 13 modules | `dspy.RLM` (REPL code+tools), `dspy.ReAct`, `dspy.CodeAct`, `dspy.ChainOfThought`, `BAMLAdapter` (Pydantic structured output), same as lab 13 + `workflow_entry`, `DaprChatClient`, `DurableAgent`, `@message_router`, `call_agent()` |
| [shared](./shared.md) | `config.py`, `research.py` | `get_lm_model()`, `get_student_lm_model()`, `get_lm_temperature()`, `get_agent_port()`, `get_dapr_state_store()`, `get_dapr_pubsub()`, `project_root()`, `ResearchDirection` dataclass, `ResearchFrontier` ABC, `SATURATION_THRESHOLD`, `MAX_BOOTSTRAPPED_DEMOS`, `MAX_LABELED_DEMOS` | — |

## Setup

```bash
cp .env.example .env    # Configure DEEPSEEK_API_KEY, LLM_MODEL, etc.
uv sync                 # Install dependencies
docker compose -f lab/10_dapr_deep_research/docker-compose.yml up -d  # Crawl4AI
dapr init               # For lab 10 (optional)
ollama pull gemma4      # For distillation (optional)
```

## Quick Reference: DSPy 3.2 Module Catalog

| Module | Constructor | Purpose |
|--------|------------|---------|
| `dspy.Predict` | `Predict(signature)` | Direct prediction, no reasoning trace |
| `dspy.ChainOfThought` | `CoT(signature)` | Step-by-step reasoning before output |
| `dspy.ReAct` | `ReAct(signature, tools=[])` | Agentic loop: thought→action→observation |
| `dspy.RLM` | `RLM(signature, tools=[], max_iterations=20)` | REPL-based code-gen agent |
| `dspy.ProgramOfThought` | `PoT(signature)` | Code-generating module with execution |
| `dspy.MultiChainComparison` | `MultiChainComparison(signature, n=3)` | Compare N reasoning chains |
| `dspy.BestOfN` | `BestOfN(module, N=3, reward_fn=...)` | Sample N, pick best by metric |
| `dspy.Refine` | `Refine(module, N=3, reward_fn=...)` | Iterative refinement loop |
| `dspy.Parallel` | `Parallel(num_threads=...)` | Internal parallel execution (use `Module.batch()`) |
| `dspy.Ensemble` | `Ensemble()` + `.compile([modules])` | Combine predictions |
| `dspy.BootstrapFewShot` | `BootstrapFewShot(metric=...)` | Trace→demonstration pipeline |
| `dspy.MIPROv2` | `MIPROv2(metric=..., auto="light")` | Bayesian instruction+ demo search |
| `dspy.GEPA` | `GEPA(metric=...)` | Genetic prompt evolution (ICLR 2026) |
| `dspy.BetterTogether` | `BetterTogether(metric=..., **optimizers)` | Chain multiple optimizers |
| `dspy.streamify` | `streamify(module, async_streaming=False)` | Convert module to streaming |
| `dspy.asyncify` | `asyncify(module)` | Convert module to async |

## Quick Reference: Pydantic Models

Used across labs 08-10 for structured RLM output:

| Model | Fields | Used In |
|-------|--------|---------|
| `FoundDirection` | topic, relevance, seed_query | 09, 10 |
| `ExplorationResult` | directions: list[FoundDirection] | 09, 10 |
| `ExtractedFinding` | claim, evidence, source, confidence | 09, 10 |
| `DeepReadResult` | findings, summary | 09, 10 |
| `SynthesisReport` | synthesis, key_insights, gaps | 09, 10 |
| `Critique` | strengths, weaknesses, follow_ups | 09, 10 |
| `ScrapedContent` | url, category, summary, key_topics | 08 |
| `ResearchReport` | findings, synthesis | 08 |

## Quick Reference: DSPy Signatures

| Signature | Inputs | Outputs | Defined In |
|-----------|--------|---------|------------|
| `GenerateHypotheses` | topic | hypotheses: list[str] | 10/agents/research_agents.py |
| `CrossValidateFindings` | findings_summary | validated_claims, contradictions | 10/agents/research_agents.py |
| `SynthesizeAcrossSources` | task | synthesis, key_insights, gaps | 10/agents/research_agents.py |
| `SelectAgent` | exploration_depth, confidence, topic | selected_agent | 10/agents/research_agents.py |
| `ComputeConfidenceDelta` | topic, agent_type, num_findings, findings_summary, exploration_depth | confidence_delta, reasoning | 10/agents/research_agents.py |
| `AssessSaturation` | topic, confidence, exploration_depth, source_count | is_saturated, reasoning | 10/agents/research_agents.py |
| `CritiqueReasoning` | research_summary | critique | 10/agents/research_agents.py |
| `QualityEvaluation` | num_directions, num_findings, frontier_saturation | quality_score, explanation | 10/evolution/lse.py |
| `ExtractPatterns` | trajectory_context | error_patterns, success_patterns, improvement_suggestion | 10/evolution/trace2skill.py |
| `AssessBatchSaturation` | directions_json: str | saturated_indices: list[int] | 10/memory/dapr_frontier.py |
| `ClassifyContent` | chunk | category, key_topics | 08/main.py |
| `ClassifyIntent` | query | intent, confidence | 07/main.py |
| `QualityEvaluation` | — | — | 07/main.py |
