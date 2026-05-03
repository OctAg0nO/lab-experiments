"""
Advanced DSPy patterns.

Covers:
- MultiChainComparison (compare multiple reasoning chains)
- Parallel execution of modules
- Ensemble (combine predictions)
- Adapter switching (JSONAdapter, XMLAdapter)
- Streaming via streamify()
- Async via acall() / asyncify()
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy
import asyncio

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# 1. MultiChainComparison — compare reasoning chains
# ---------------------------------------------------------------------------
print("=== MultiChainComparison ===")

class CompareAnswers(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

mc = dspy.MultiChainComparison(CompareAnswers, n=3)
result = mc(question="Which number is larger: 0.8 or 0.11?")
print(f"Answer after chain comparison: {result.answer}")
print()

# ---------------------------------------------------------------------------
# 2. Parallel execution via Module.batch
# ---------------------------------------------------------------------------
print("=== Module.batch (parallel) ===")

class Summarize(dspy.Signature):
    text: str = dspy.InputField()
    summary: str = dspy.OutputField()

texts = [
    "Python is a high-level general-purpose programming language.",
    "DSPy is a framework for programming language models.",
    "Dapr provides distributed application runtime capabilities.",
]

prog = dspy.ChainOfThought(Summarize)
examples = [dspy.Example(text=t).with_inputs("text") for t in texts]
results = prog.batch(examples, num_threads=3)
for i, r in enumerate(results):
    print(f"Summary {i+1}: {r.summary}")
print()

# ---------------------------------------------------------------------------
# 3. Ensemble — combine multiple predictions via compile()
# ---------------------------------------------------------------------------
print("=== Ensemble ===")

class Classify(dspy.Signature):
    text: str = dspy.InputField()
    label: str = dspy.OutputField()

ensemble = dspy.Ensemble()
compiled = ensemble.compile([dspy.ChainOfThought(Classify), dspy.Predict(Classify)])
result = compiled(text="This movie was surprisingly good!")
print(f"Ensemble label: {result.label}")
print()

# ---------------------------------------------------------------------------
# 4. JSONAdapter for structured JSON output
# ---------------------------------------------------------------------------
print("=== JSONAdapter ===")
dspy.configure(adapter=dspy.JSONAdapter())

class Person(dspy.Signature):
    description: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()

extract = dspy.Predict(Person)
result = extract(description="A 28-year-old software engineer named Maya living in Berlin")
print(f"JSONAdapter result: name={result.name}, age={result.age}")

# ---------------------------------------------------------------------------
# 5. Streaming
# ---------------------------------------------------------------------------
print("\n=== Streaming ===")
dspy.configure(adapter=dspy.ChatAdapter())  # reset adapter

stream_gen = dspy.streamify(dspy.ChainOfThought("topic -> haiku"), async_streaming=False)
for chunk in stream_gen(topic="programming"):
    print(chunk, end="", flush=True)
print("\n")

# ---------------------------------------------------------------------------
# 6. Async
# ---------------------------------------------------------------------------
print("=== Async ===")

async def main():
    prog = dspy.ChainOfThought("question -> answer")
    result = await prog.acall(question="What is the async equivalent in DSPy?")
    print(f"Async answer: {result.answer}")

asyncio.run(main())
