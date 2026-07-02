"""PickMyStack - Streamlit web interface.

Launch from the project root:

    streamlit run examples/pickmystack/ui/app_ui.py

Describe your use case and constraints; three specialist agents evaluate AI
stacks in parallel and a synthesizer ranks the best options.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

# Make the project root importable (this file lives 3 levels below root).
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from examples.pickmystack.app import build_specialists, build_synthesizer, format_query
from unchained import LLM

DEFAULT_MODELS = {
    "ollama": "llama3.1",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
}

st.set_page_config(page_title="PickMyStack", page_icon="🧩", layout="wide")


# ---------------------------------------------------------------------------
# Sidebar - model / provider configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    provider = st.selectbox("LLM provider", ["ollama", "openai", "anthropic"], index=0)
    model = st.text_input("Model", value=DEFAULT_MODELS[provider])
    api_key = ""
    if provider in ("openai", "anthropic"):
        api_key = st.text_input(
            "API key", type="password", help="Left blank uses the matching environment variable."
        )
    monthly_requests = st.slider("Expected requests / month", 1_000, 1_000_000, 50_000, step=1_000)
    st.caption("Ollama runs locally at http://localhost:11434 - no key needed.")


# ---------------------------------------------------------------------------
# Main - use case & constraints
# ---------------------------------------------------------------------------
st.title("🧩 PickMyStack")
st.write(
    "Tell us what you want to build. Three specialist agents (cost, fit, trend) "
    "evaluate the options and recommend a stack. Powered by **Unchained**."
)

col1, col2 = st.columns([2, 1])
with col1:
    use_case = st.text_area(
        "What are you building?",
        placeholder="e.g. A customer-support chatbot over our product docs",
        height=110,
    )
with col2:
    budget = st.number_input("Budget ($/month)", min_value=0, value=200, step=50)
    team = st.number_input("Team size", min_value=1, value=3, step=1)
    timeline = st.text_input("Timeline", value="6 weeks")

go = st.button("🚀 Recommend a stack", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
def _make_llm() -> LLM:
    return LLM(
        provider=provider,
        model=model or DEFAULT_MODELS[provider],
        api_key=api_key or None,
        temperature=0.3,
    )


if go:
    if not use_case.strip():
        st.warning("Please describe what you want to build first.")
        st.stop()

    if provider in ("openai", "anthropic") and not (
        api_key or os.getenv("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY")
    ):
        st.error(f"An API key is required for the {provider} provider.")
        st.stop()

    llm = _make_llm()
    specialists = build_specialists(llm)
    synthesizer = build_synthesizer(llm)
    query = format_query(use_case, budget, team, timeline, monthly_requests)

    findings: dict = {}
    labels = {"cost": "💰 Cost", "fit": "🎯 Fit", "trend": "📈 Trend"}

    with st.status("Consulting specialist agents in parallel...", expanded=True) as status:
        try:
            with ThreadPoolExecutor(max_workers=len(specialists)) as pool:
                futures = {pool.submit(a.run, query): a for a in specialists}
                st.write("Dispatched cost, fit and trend agents.")
                for future in as_completed(futures):
                    agent = futures[future]
                    try:
                        findings[agent.name] = future.result()
                    except Exception as exc:
                        findings[agent.name] = f"(unavailable: {exc})"
                    st.write(f"✓ {labels.get(agent.name, agent.name)} agent finished.")

            st.write("Synthesizing final recommendation...")
            combined = "\n\n".join(
                f"### {name.upper()} FINDINGS\n{text}" for name, text in findings.items()
            )
            recommendation = synthesizer.run(
                f"Original request:\n{query}\n\nSpecialist findings:\n{combined}\n\n"
                "Produce the final ranked recommendation."
            )
            status.update(label="Done!", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Something went wrong", state="error")
            st.error(
                f"Could not reach the '{provider}' provider: {exc}\n\n"
                "If you selected Ollama, make sure it is running locally."
            )
            st.stop()

    st.subheader("🏆 Recommendation")
    st.markdown(recommendation)

    st.subheader("🔍 Specialist findings")
    cols = st.columns(len(findings))
    for col, (name, text) in zip(cols, findings.items()):
        with col, st.expander(labels.get(name, name), expanded=False):
            st.markdown(text)
else:
    st.info("Fill in your use case and constraints, then click **Recommend a stack**.")
