"""Tests for the example scripts under examples/.

These run fully offline, same as tests/test_unchained.py: real network calls
(researcher.web_search) are monkeypatched, and PickMyStack agents use
MockLLM/FakeLLM stand-ins instead of a real provider.

    pytest tests/test_examples.py
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unchained import MockLLM


# ---------------------------------------------------------------------------
# examples/data_analyst.py
# ---------------------------------------------------------------------------
def test_describe_csv_reports_counts_and_numeric_stats(tmp_path):
    from examples.data_analyst import describe_csv

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,score\na,1\nb,2\nc,notanumber\n", encoding="utf-8")

    out = describe_csv(str(csv_path))
    assert "3 rows, columns: name, score" in out
    assert "score: min=1.00 max=2.00 mean=1.50" in out
    assert "name:" not in out.split("\n")[-1]  # non-numeric column has no stats line


def test_describe_csv_empty_file(tmp_path):
    from examples.data_analyst import describe_csv

    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")
    assert describe_csv(str(csv_path)) == "The file is empty."


def test_data_analyst_build_agent_wires_the_tool():
    from examples.data_analyst import build_agent, describe_csv

    agent = build_agent(provider="ollama")
    assert agent.name == "data_analyst"
    assert agent.tools["describe_csv"] is describe_csv  # @tool already wraps it


# ---------------------------------------------------------------------------
# examples/researcher.py
# ---------------------------------------------------------------------------
def test_web_search_returns_abstract_when_present(monkeypatch):
    import examples.researcher as researcher

    class _Resp:
        def json(self):
            return {"AbstractText": "Rust is a systems language.", "AbstractSource": "Wikipedia"}

    monkeypatch.setattr(researcher.requests, "get", lambda *a, **k: _Resp())
    out = researcher.web_search("What is Rust?")
    assert out == "Rust is a systems language. (source: Wikipedia)"


def test_web_search_falls_back_to_related_topics(monkeypatch):
    import examples.researcher as researcher

    class _Resp:
        def json(self):
            return {
                "AbstractText": "",
                "RelatedTopics": [{"Text": "Topic one"}, {"Text": ""}, {"Text": "Topic two"}],
            }

    monkeypatch.setattr(researcher.requests, "get", lambda *a, **k: _Resp())
    out = researcher.web_search("obscure query")
    assert out == "Topic one\nTopic two"


def test_web_search_no_results(monkeypatch):
    import examples.researcher as researcher

    class _Resp:
        def json(self):
            return {}

    monkeypatch.setattr(researcher.requests, "get", lambda *a, **k: _Resp())
    assert researcher.web_search("nothing") == "No results found."


def test_researcher_build_agent_wires_the_tool():
    from examples.researcher import build_agent, web_search

    agent = build_agent(provider="ollama")
    assert agent.name == "researcher"
    assert agent.tools["web_search"] is web_search  # @tool already wraps it


# ---------------------------------------------------------------------------
# examples/quickstart.py
# ---------------------------------------------------------------------------
def test_quickstart_tool_calling_demo(capsys):
    from examples.quickstart import tool_calling_demo

    tool_calling_demo()
    out = capsys.readouterr().out
    assert "2 + 3 = 5." in out
    assert "Token usage" in out


def test_quickstart_streaming_demo(capsys):
    from examples.quickstart import streaming_demo

    streaming_demo()
    out = capsys.readouterr().out
    assert "Streaming works token by token." in out


def test_quickstart_structured_demo(capsys):
    from examples.quickstart import structured_demo

    structured_demo()
    out = capsys.readouterr().out
    assert "result=5" in out


# ---------------------------------------------------------------------------
# examples/pickmystack/tools/cost_estimator.py
# ---------------------------------------------------------------------------
def test_estimate_cost_matches_exact_model_name():
    from examples.pickmystack.tools.cost_estimator import estimate_cost

    out = estimate_cost("gpt-4o-mini", 1_000, avg_input_tokens=800, avg_output_tokens=400)
    assert "'gpt-4o-mini'" in out
    # 1000 * 800 / 1e6 * 0.15 + 1000 * 400 / 1e6 * 0.60 = 0.12 + 0.24 = 0.36
    assert "$0.36/mo" in out
    assert "ESTIMATED TOTAL: $0.36/month" in out


def test_estimate_cost_does_not_confuse_gpt4o_variants():
    # Regression test: a naive substring match previously resolved
    # "gpt-4o-mini" to the (16x pricier) "gpt-4o" pricing tier.
    from examples.pickmystack.tools.cost_estimator import _closest_model

    assert _closest_model("gpt-4o-mini") == "gpt-4o-mini"
    assert _closest_model("GPT-4o Mini") == "gpt-4o-mini"
    assert _closest_model("gpt-4o") == "gpt-4o"
    assert _closest_model("gpt4o") == "gpt-4o"


def test_estimate_cost_open_weight_model_gets_gpu_hosting_by_default():
    from examples.pickmystack.tools.cost_estimator import estimate_cost

    out = estimate_cost("llama-3.1-8b", 50_000)
    assert "open-weight: no token fees" in out
    assert "Self-hosted GPU instance" in out
    assert "$250.00/mo" in out


def test_estimate_cost_unknown_model_falls_back_to_default():
    from examples.pickmystack.tools.cost_estimator import _closest_model

    assert _closest_model("some-unheard-of-model") == "gpt-4o-mini"
    assert _closest_model("") == "gpt-4o-mini"


def test_estimate_cost_respects_explicit_hosting_choice():
    from examples.pickmystack.tools.cost_estimator import estimate_cost

    out = estimate_cost("gpt-4o-mini", 1_000, hosting="container")
    assert "Always-on small container" in out
    assert "$60.00/mo" in out


# ---------------------------------------------------------------------------
# examples/pickmystack/tools/benchmark_fetcher.py
# ---------------------------------------------------------------------------
def test_framework_data_known_and_unknown():
    from examples.pickmystack.tools.benchmark_fetcher import framework_data

    data = framework_data("LangChain")
    assert data["dependencies"] == 50
    assert data["multi_agent"] is True
    assert framework_data("not-a-real-framework") == {}


def test_compare_frameworks_all_vs_subset():
    from examples.pickmystack.tools.benchmark_fetcher import compare_frameworks

    all_out = compare_frameworks("")
    assert "unchained" in all_out and "langchain" in all_out and "autogen" in all_out

    subset = compare_frameworks("unchained, crewai")
    assert "unchained" in subset and "crewai" in subset
    assert "autogen" not in subset


def test_compare_frameworks_reports_unknown_names():
    from examples.pickmystack.tools.benchmark_fetcher import compare_frameworks

    out = compare_frameworks("unchained, not-a-real-one")
    assert "(unknown framework)" in out


def test_unchained_loc_is_measured_live_not_hardcoded():
    # Regression test: this used to be a hardcoded, quickly stale number.
    from examples.pickmystack.tools.benchmark_fetcher import FRAMEWORKS

    assert FRAMEWORKS["unchained"]["lines_of_code"] > 500  # sanity: not 0/placeholder


# ---------------------------------------------------------------------------
# examples/pickmystack/tools/doc_retriever.py
# ---------------------------------------------------------------------------
def test_get_rag_indexes_knowledge_base_and_is_cached():
    from examples.pickmystack.tools.doc_retriever import get_rag

    rag = get_rag()
    assert len(rag) > 0
    assert get_rag() is rag  # module-level singleton, not rebuilt each call


def test_search_knowledge_returns_relevant_passage_with_source():
    from examples.pickmystack.tools.doc_retriever import search_knowledge

    out = search_knowledge("cheapest way to run an agent on a tight budget")
    assert out != "No relevant knowledge found."
    assert ".md | score" in out


def test_search_knowledge_no_hits_message(monkeypatch):
    import examples.pickmystack.tools.doc_retriever as doc_retriever

    class _EmptyRAG:
        def search(self, query, top_k=3):
            return []

    monkeypatch.setattr(doc_retriever, "_rag", _EmptyRAG())
    assert doc_retriever.search_knowledge("anything") == "No relevant knowledge found."


# ---------------------------------------------------------------------------
# examples/pickmystack/app.py
# ---------------------------------------------------------------------------
def test_format_query_includes_all_provided_constraints():
    from examples.pickmystack.app import format_query

    q = format_query(
        "Build a support bot", budget=200.0, team=3, timeline="6 weeks", monthly_requests=50_000
    )
    assert "Use case: Build a support bot" in q
    assert "Budget: about $200/month." in q
    assert "Team size: 3 people." in q
    assert "Timeline: 6 weeks." in q
    assert "50,000 requests/month" in q


def test_format_query_omits_unset_optional_constraints():
    from examples.pickmystack.app import format_query

    q = format_query("Build a bot", budget=None, team=None, timeline=None, monthly_requests=1000)
    assert "Budget:" not in q
    assert "Team size:" not in q
    assert "Timeline:" not in q


def test_build_specialists_and_synthesizer_are_wired_correctly():
    from examples.pickmystack.app import build_specialists, build_synthesizer
    from unchained import LLM

    llm = LLM(provider="ollama")
    specialists = build_specialists(llm)
    names = {a.name for a in specialists}
    assert names == {"cost", "fit", "trend"}
    for agent in specialists:
        assert agent.rag is not None  # all share the knowledge-base RAG

    synth = build_synthesizer(llm)
    assert synth.name == "synthesizer"


def test_recommend_end_to_end_offline(monkeypatch):
    """Drive the full PickMyStack pipeline with MockLLM instead of a real provider."""
    import examples.pickmystack.app as app

    def _fake_llm(provider="ollama", model=None, temperature=0.7, **kwargs):
        return MockLLM(reply=f"finding from {provider}")

    monkeypatch.setattr(app, "LLM", _fake_llm)

    seen_progress = []
    result = app.recommend(
        use_case="Build a support bot",
        budget=100,
        team=2,
        monthly_requests=1_000,
        progress=lambda name, status: seen_progress.append((name, status)),
    )

    assert set(result["findings"]) == {"cost", "fit", "trend"}
    assert result["recommendation"]  # synthesizer produced something
    # every specialist and the synthesizer reported both running and done
    seen_names = {name for name, _ in seen_progress}
    assert seen_names == {"cost", "fit", "trend", "synthesizer"}
    assert ("synthesizer", "done") in seen_progress


def test_recommend_isolates_a_failing_specialist(monkeypatch):
    import examples.pickmystack.app as app

    class _BoomLLM:
        def chat(self, *args, **kwargs):
            raise RuntimeError("provider down")

        def stream(self, *args, **kwargs):
            raise RuntimeError("provider down")

    monkeypatch.setattr(app, "LLM", lambda **kwargs: _BoomLLM())
    result = app.recommend(use_case="Build a bot", monthly_requests=1_000)
    assert all(v.startswith("(unavailable:") for v in result["findings"].values())
    assert result["recommendation"].startswith("(synthesizer unavailable:")
