"""
DSPy foundations: signatures, prediction modules, and custom module composition.

Covers:
- Short-form and class-based signatures
- dspy.Predict (direct prediction, no reasoning trace)
- dspy.ChainOfThought (step-by-step reasoning)
- Custom dspy.Module with multi-stage pipeline
- Field types: str, int, Literal, list
"""

from typing import Literal
from pathlib import Path
from dotenv import load_dotenv
import dspy

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

lm = dspy.LM("deepseek/deepseek-v4-flash")
dspy.configure(lm=lm)

# ---------------------------------------------------------------------------
# 1. Short-form signature
# ---------------------------------------------------------------------------
math = dspy.ChainOfThought("question -> answer: float")
result = math(question="What is 15 * 37?")
print(f"Math answer: {result.answer}")

# ---------------------------------------------------------------------------
# 2. Class-based signature with typed fields
# ---------------------------------------------------------------------------

class Classify(dspy.Signature):
    """Classify the sentiment and topic of a message."""
    sentence: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral"] = dspy.OutputField()
    topic: str = dspy.OutputField()

classifier = dspy.ChainOfThought(Classify)
result = classifier(sentence="DSPy makes LLM programming elegant and reliable")
print(f"Sentiment: {result.sentiment} | Topic: {result.topic}")

# ---------------------------------------------------------------------------
# 3. Predict (no reasoning trace — faster, cheaper)
# ---------------------------------------------------------------------------

class Extract(dspy.Signature):
    """Extract structured info from text."""
    text: str = dspy.InputField()
    name: str = dspy.OutputField()
    age: int = dspy.OutputField()

extractor = dspy.Predict(Extract)
result = extractor(text="Alice is 32 years old and works at Acme Corp.")
print(f"Extracted: {result.name}, age {result.age}")

# ---------------------------------------------------------------------------
# 4. Custom multi-stage Module
# ---------------------------------------------------------------------------

class Outline(dspy.Signature):
    topic: str = dspy.InputField()
    title: str = dspy.OutputField()
    sections: list[str] = dspy.OutputField()

class DraftSection(dspy.Signature):
    topic: str = dspy.InputField()
    heading: str = dspy.InputField()
    content: str = dspy.OutputField()

class ArticleWriter(dspy.Module):
    def __init__(self):
        self.outline = dspy.ChainOfThought(Outline)
        self.draft = dspy.ChainOfThought(DraftSection)

    def forward(self, topic):
        o = self.outline(topic=topic)
        sections = [self.draft(topic=topic, heading=h) for h in o.sections]
        return dspy.Prediction(title=o.title, sections=sections)

writer = ArticleWriter()
article = writer(topic="The future of AI agents")
print(f"Title: {article.title}")
print(f"Sections: {len(article.sections)} drafted")
