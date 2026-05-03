# 06 — Advanced DSPy Patterns

Streaming, async execution, MultiChainComparison, `Module.batch` (parallel), `Ensemble.compile()`, and adapter switching.

Run:
```bash
python lab/06-advanced/main.py
```

### DSPy 3.2 API notes
- `dspy.Parallel(Module, n=3)` → `prog.batch(examples, num_threads=3)`
- `dspy.Ensemble(a, b)` → `dspy.Ensemble().compile([a, b])`
- `module.streamify()` → `dspy.streamify(module, async_streaming=False)`
