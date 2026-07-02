"""Framework comparison data for the PickMyStack TrendAgent / FitAgent.

Ships a small curated dataset of agent frameworks with the attributes that
matter when picking a stack: size, dependency weight, ease of use, and whether
multi-agent and RAG are supported out of the box.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unchained import tool

# ease/momentum are 1-5 (higher is better/hotter).
FRAMEWORKS = {
    "unchained": {
        "lines_of_code": 619,
        "dependencies": 2,
        "ease": 5,
        "multi_agent": True,
        "rag": True,
        "momentum": 3,
        "note": "Single-file, minimal, provider-agnostic.",
    },
    "langchain": {
        "lines_of_code": 400_000,
        "dependencies": 50,
        "ease": 2,
        "multi_agent": True,
        "rag": True,
        "momentum": 5,
        "note": "Huge ecosystem, many integrations, steep curve.",
    },
    "llamaindex": {
        "lines_of_code": 150_000,
        "dependencies": 30,
        "ease": 3,
        "multi_agent": False,
        "rag": True,
        "momentum": 4,
        "note": "Best-in-class RAG and indexing.",
    },
    "crewai": {
        "lines_of_code": 40_000,
        "dependencies": 20,
        "ease": 3,
        "multi_agent": True,
        "rag": True,
        "momentum": 4,
        "note": "Role-based multi-agent crews.",
    },
    "autogen": {
        "lines_of_code": 80_000,
        "dependencies": 25,
        "ease": 2,
        "multi_agent": True,
        "rag": False,
        "momentum": 4,
        "note": "Conversation-first multi-agent, strong async.",
    },
}


def framework_data(name: str) -> dict:
    """Return the raw data dict for a single framework (or an empty dict)."""
    return FRAMEWORKS.get((name or "").strip().lower(), {})


@tool
def compare_frameworks(names: str = "") -> str:
    """Compare agent frameworks side by side.

    Pass a comma-separated list of framework names to compare a subset, or an
    empty string to compare all of them. Reports lines of code, dependency
    count, ease of use, multi-agent and RAG support, and community momentum.
    """
    requested = [n.strip().lower() for n in names.split(",") if n.strip()]
    selected = requested or list(FRAMEWORKS)

    rows = [
        "framework      | LOC     | deps | ease | multi | rag | momentum",
        "---------------|---------|------|------|-------|-----|---------",
    ]
    for key in selected:
        data = FRAMEWORKS.get(key)
        if not data:
            rows.append(f"{key:<14} | (unknown framework)")
            continue
        rows.append(
            f"{key:<14} | {data['lines_of_code']:>7,} | {data['dependencies']:>4} | "
            f"{data['ease']:>4} | {str(data['multi_agent']):>5} | {str(data['rag']):>3} | "
            f"{data['momentum']:>8}"
        )
    notes = [f"- {k}: {FRAMEWORKS[k]['note']}" for k in selected if k in FRAMEWORKS]
    return "\n".join(rows) + "\n\nNotes:\n" + "\n".join(notes)


if __name__ == "__main__":
    print(compare_frameworks("unchained, langchain, crewai"))
