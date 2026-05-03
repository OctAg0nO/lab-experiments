"""
ReAct agent loop: thought → action → observation.

Demonstrates:
- dspy.ReAct with custom tools
- Tools as plain functions (docstring → LLM sees as tool description)
- Tools as dspy.Tool with explicit name/desc overrides
- PythonInterpreter as a built-in tool
- Multi-turn agent execution
"""

from pathlib import Path
from dotenv import load_dotenv
import dspy

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)


def search(query: str) -> list[str]:
    """Search the knowledge base for information."""
    db = {
        "population of paris": "2.1 million",
        "capital of france": "Paris",
        "height of eiffel tower": "330 meters",
        "currency of japan": "Japanese Yen",
    }
    query_lower = query.lower()
    for key, value in db.items():
        if query_lower in key:
            return [f"{key}: {value}"]
    return [f"No results found for '{query}'"]


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    allowed = {"+", "-", "*", "/", "(", ")", " ", ".", "0", "1", "2",
               "3", "4", "5", "6", "7", "8", "9"}
    if not all(c in allowed for c in expression):
        return "Error: invalid characters in expression"
    return str(eval(expression))


# ---------------------------------------------------------------------------
# Basic ReAct with function tools
# ---------------------------------------------------------------------------
agent = dspy.ReAct("question -> answer", tools=[search, calculator])
result = agent(question="What is the population of Paris multiplied by 2?")
print("Q: What is the population of Paris multiplied by 2?")
print(f"A: {result.answer}")
print()

# ---------------------------------------------------------------------------
# Expose the reasoning trace
# ---------------------------------------------------------------------------
dspy.inspect_history(n=1)
print("---")

# ---------------------------------------------------------------------------
# ReAct with PythonInterpreter for code execution
# ---------------------------------------------------------------------------
code_agent = dspy.ReAct(
    "question -> answer",
    tools=[dspy.PythonInterpreter]
)
code_result = code_agent(question="Compute the sum of squares from 1 to 10")
print("Q: Compute the sum of squares from 1 to 10")
print(f"A: {code_result.answer}")
