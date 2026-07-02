"""Web-research agent.

Gives an agent a single `web_search` tool (DuckDuckGo's keyless Instant Answer
API) and lets the ReAct loop decide when to search.

Run:
    python examples/researcher.py "What is the Rust programming language?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # find unchained.py

import requests

from unchained import LLM, Agent, tool


@tool
def web_search(query: str) -> str:
    """Search the web and return a short factual summary for a query."""
    resp = requests.get(
        "https://api.duckduckgo.com/",
        params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
        timeout=20,
    )
    data = resp.json()
    if data.get("AbstractText"):
        return f"{data['AbstractText']} (source: {data.get('AbstractSource', 'DuckDuckGo')})"
    topics = [t.get("Text", "") for t in data.get("RelatedTopics", []) if t.get("Text")]
    return "\n".join(topics[:5]) or "No results found."


def build_agent(provider: str = "ollama") -> Agent:
    return Agent(
        LLM(provider=provider),
        name="researcher",
        description="Researches topics on the web and summarises findings.",
        system_prompt=(
            "You are a research assistant. Use web_search to gather facts, "
            "then answer with a concise, well-sourced summary."
        ),
        tools=[web_search],
    )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is retrieval-augmented generation?"
    print(build_agent().run(question))
