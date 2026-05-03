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
├── 99-sandbox/         Scratch space
└── shared/             Env config, LM helpers shared across labs
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
| [10-dapr-deep-research](./10-dapr-deep-research.md) | 15 files in 6 packages | `ExplorerAgent`, `DeepReaderAgent`, `SynthesizerAgent`, `CriticAgent`, `ResearchWorkflow`, `DaprFrontier`, `LSEOptimizer`, `SkillConsolidator`, `MCPBridge`, 10 DSPy signatures | `RLM`, `ChainOfThought`, `BestOfN`, `Refine`, `MultiChainComparison`, `BootstrapFewShot`, `Evaluate`, `BAMLAdapter` |
| [shared](./shared.md) | `config.py` | `get_env_or_raise()`, `get_env()`, `get_lm_model()`, `get_student_lm_model()`, `project_root()` | — |

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
