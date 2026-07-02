"""Zero-setup quickstart - runs with no API key and no server.

Uses `MockLLM`, a deterministic stand-in for a real provider, so you can see
the agent loop, streaming, and structured output work immediately:

    python examples/quickstart.py

Swap `MockLLM(...)` for `LLM(provider="ollama")` or `LLM(provider="openai")`
to run against a real model - the rest of the code is identical.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel

from unchained import Agent, MockLLM, tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def tool_calling_demo() -> None:
    # Script the mock: first turn requests the tool, second turn answers.
    llm = MockLLM(
        script=[
            {"tool_calls": [{"name": "add", "arguments": {"a": 2, "b": 3}, "id": "1"}]},
            {"content": "2 + 3 = 5."},
        ]
    )
    agent = Agent(llm, tools=[add])
    print("Tool calling :", agent.run("What is 2 + 3?"))
    print("Token usage  :", agent.usage)


def streaming_demo() -> None:
    agent = Agent(MockLLM(reply="Streaming works token by token."))
    print("Streaming    : ", end="")
    for token in agent.stream("say something"):
        print(token, end="", flush=True)
    print()


def structured_demo() -> None:
    class Sum(BaseModel):
        result: int

    out = Agent(MockLLM(reply='{"result": 5}')).run("add 2 and 3", response_format=Sum)
    print("Structured   :", out)


if __name__ == "__main__":
    tool_calling_demo()
    streaming_demo()
    structured_demo()
    print(
        "\nAll of this ran with no API key and no server. "
        "Swap MockLLM for LLM(provider=...) to use a real model."
    )
