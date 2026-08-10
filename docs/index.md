# Unchained

**A single-file agentic AI framework.** Tools, memory, RAG, multi-agent
orchestration and structured output - in one file, with two dependencies
(`requests` + `pydantic`), and it works with OpenAI, Anthropic, or a local
Ollama model.

## Install

```bash
pip install requests pydantic
# then drop unchained.py into your project, or: pip install -e ".[dev]"
```

## Try it with zero setup

`MockLLM` is a deterministic, no-network stand-in, so you can run an agent
without any API key or server:

```python
from unchained import Agent, MockLLM, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


llm = MockLLM(
    script=[
        {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}, "id": "1"}]},
        {"content": "2 + 3 = 5."},
    ]
)
print(Agent(llm, tools=[add]).run("What is 2 + 3?"))
```

Swap `MockLLM(...)` for `LLM(provider="ollama")` (or `openai` / `anthropic`) to
use a real model - the rest of the code is identical.

## Where next

- [User Manual](USER_MANUAL.md) - a friendly, task-by-task guide.
- [Architecture](ARCHITECTURE.md) - how the pieces fit together.
- [API Reference](api.md) - every public class and function.
