"""CSV data-analysis agent.

Exposes a `describe_csv` tool built on the standard library (no pandas) so the
agent can answer questions about a spreadsheet.

Run:
    python examples/data_analyst.py data.csv "Which column has the highest average?"
"""

import contextlib
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unchained import LLM, Agent, tool


@tool
def describe_csv(path: str) -> str:
    """Return row/column counts and per-column numeric statistics for a CSV file."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return "The file is empty."

    lines = [f"{len(rows)} rows, columns: {', '.join(rows[0])}"]
    for column in rows[0]:
        numbers = []
        for row in rows:
            with contextlib.suppress(ValueError, TypeError):
                numbers.append(float(row[column]))
        if numbers:
            lines.append(
                f"- {column}: min={min(numbers):.2f} max={max(numbers):.2f} "
                f"mean={statistics.mean(numbers):.2f}"
            )
    return "\n".join(lines)


def build_agent(provider: str = "ollama") -> Agent:
    return Agent(
        LLM(provider=provider),
        name="data_analyst",
        description="Analyses CSV files and answers questions about the data.",
        system_prompt=(
            "You are a data analyst. Use describe_csv to inspect the file, "
            "then answer the user's question with specific numbers."
        ),
        tools=[describe_csv],
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/data_analyst.py <file.csv> [question]")
        raise SystemExit(1)
    csv_path = sys.argv[1]
    question = " ".join(sys.argv[2:]) or "Summarise this dataset."
    print(build_agent().run(f"File path: {csv_path}\n\n{question}"))
