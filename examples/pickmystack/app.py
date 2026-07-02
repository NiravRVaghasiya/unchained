"""PickMyStack - CLI entry point.

Describe a use case and constraints; three specialist agents (cost, fit, trend)
evaluate AI-stack options in parallel and a synthesizer produces a ranked
recommendation.

Usage:
    python -m examples.pickmystack.app "Build a customer-support chatbot" \\
        --budget 200 --team 3 --timeline "6 weeks" --provider ollama

    # or, run the file directly:
    python examples/pickmystack/app.py "Summarise legal documents" --budget 500
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Optional

# Make the project root importable whether run as a module or a plain script.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unchained import LLM, Agent, Router

try:  # relative import when executed as `python -m examples.pickmystack.app`
    from .tools import compare_frameworks, estimate_cost, get_rag, search_knowledge
except ImportError:  # absolute import when executed as `python examples/pickmystack/app.py`
    from examples.pickmystack.tools import (
        compare_frameworks,
        estimate_cost,
        get_rag,
        search_knowledge,
    )

ProgressFn = Optional[Callable[[str, str], None]]


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------
def build_specialists(llm: LLM) -> list:
    """The three specialist evaluation agents, sharing the knowledge-base RAG."""
    rag = get_rag()

    cost_agent = Agent(
        llm,
        name="cost",
        description="Estimates budget and total cost of ownership for AI stacks.",
        system_prompt=(
            "You are a cost analyst. Estimate the monthly and yearly cost of the "
            "candidate AI stacks for the user's use case and volume using the "
            "estimate_cost tool. Consult the knowledge base for pricing. Flag the "
            "most cost-effective options and any budget risks. Be concrete with numbers."
        ),
        tools=[estimate_cost, search_knowledge],
        rag=rag,
        max_iterations=5,
    )
    fit_agent = Agent(
        llm,
        name="fit",
        description="Scores how well each option fits the use case, team and requirements.",
        system_prompt=(
            "You are a technical-fit analyst. Judge how well candidate frameworks "
            "and models match the use case, team size and timeline. Use "
            "compare_frameworks and the knowledge base. Give each strong option a "
            "fit score out of 100 with a one-line justification."
        ),
        tools=[compare_frameworks, search_knowledge],
        rag=rag,
        max_iterations=5,
    )
    trend_agent = Agent(
        llm,
        name="trend",
        description="Assesses community momentum, maturity and ecosystem health.",
        system_prompt=(
            "You are a technology-trend analyst. Assess adoption momentum, "
            "maturity and long-term viability of the candidate stacks using "
            "compare_frameworks and the knowledge base. Highlight what is gaining "
            "or losing traction and any longevity risks."
        ),
        tools=[compare_frameworks, search_knowledge],
        rag=rag,
        max_iterations=5,
    )
    return [cost_agent, fit_agent, trend_agent]


def build_synthesizer(llm: LLM) -> Agent:
    return Agent(
        llm,
        name="synthesizer",
        description="Combines specialist findings into a final ranked recommendation.",
        system_prompt=(
            "You are the lead architect. Combine the cost, fit and trend findings "
            "into a clear recommendation. Output the TOP 3 stacks, ranked, each "
            "with: the framework + model + hosting, an estimated monthly cost, a "
            "fit score, and 2-3 sentences of reasoning. End with a one-line pick "
            "for the user's specific constraints."
        ),
        max_iterations=3,
    )


def build_router(llm: LLM) -> Router:
    """A Router over the specialists with the synthesizer attached."""
    return Router(llm, agents=build_specialists(llm), synthesizer=build_synthesizer(llm))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def format_query(
    use_case: str,
    budget: Optional[float],
    team: Optional[int],
    timeline: Optional[str],
    monthly_requests: int,
) -> str:
    parts = [f"Use case: {use_case}"]
    if budget is not None:
        parts.append(f"Budget: about ${budget:.0f}/month.")
    if team is not None:
        parts.append(f"Team size: {team} people.")
    if timeline:
        parts.append(f"Timeline: {timeline}.")
    parts.append(f"Expected volume: about {monthly_requests:,} requests/month.")
    parts.append(
        "Recommend concrete AI stacks (framework + model + hosting) that fit these constraints."
    )
    return "\n".join(parts)


def recommend(
    use_case: str,
    budget: Optional[float] = None,
    team: Optional[int] = None,
    timeline: Optional[str] = None,
    monthly_requests: int = 50_000,
    provider: str = "ollama",
    model: Optional[str] = None,
    progress: ProgressFn = None,
) -> Dict[str, object]:
    """Run the full PickMyStack pipeline and return findings + recommendation.

    ``progress(agent_name, status)`` is called with status "running"/"done"
    so a UI can show live updates. Specialists run in parallel.
    """
    llm = LLM(provider=provider, model=model, temperature=0.3)
    specialists = build_specialists(llm)
    synthesizer = build_synthesizer(llm)
    query = format_query(use_case, budget, team, timeline, monthly_requests)

    def _run(agent: Agent) -> str:
        if progress:
            progress(agent.name, "running")
        try:
            result = agent.run(query)
        except Exception as exc:  # a failing specialist shouldn't sink the run
            result = f"(unavailable: {exc})"
        if progress:
            progress(agent.name, "done")
        return result

    findings: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(specialists)) as pool:
        futures = {pool.submit(_run, a): a for a in specialists}
        for future in as_completed(futures):
            agent = futures[future]
            findings[agent.name] = future.result()

    combined = "\n\n".join(
        f"### {name.upper()} FINDINGS\n{text}" for name, text in findings.items()
    )
    if progress:
        progress("synthesizer", "running")
    try:
        recommendation = synthesizer.run(
            f"Original request:\n{query}\n\nSpecialist findings:\n{combined}\n\n"
            "Produce the final ranked recommendation."
        )
    except Exception as exc:
        recommendation = f"(synthesizer unavailable: {exc})"
    if progress:
        progress("synthesizer", "done")

    return {"query": query, "findings": findings, "recommendation": recommendation}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli_progress(name: str, status: str) -> None:
    icon = "..." if status == "running" else "[done]"
    print(f"  {icon} {name} agent {status}")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pickmystack",
        description="Recommend an AI stack for your use case and constraints.",
    )
    parser.add_argument("use_case", help="What you want to build.")
    parser.add_argument("--budget", type=float, help="Monthly budget in USD.")
    parser.add_argument("--team", type=int, help="Team size (people).")
    parser.add_argument("--timeline", help="Delivery timeline, e.g. '6 weeks'.")
    parser.add_argument(
        "--requests",
        type=int,
        default=50_000,
        dest="requests",
        help="Expected requests per month (default: 50000).",
    )
    parser.add_argument("--provider", default="ollama", choices=["ollama", "openai", "anthropic"])
    parser.add_argument("--model", default=None, help="Override the model name.")
    args = parser.parse_args(argv)

    print("PickMyStack - evaluating options with 3 specialist agents\n")
    result = recommend(
        use_case=args.use_case,
        budget=args.budget,
        team=args.team,
        timeline=args.timeline,
        monthly_requests=args.requests,
        provider=args.provider,
        model=args.model,
        progress=_cli_progress,
    )

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(result["recommendation"])


if __name__ == "__main__":
    main()
