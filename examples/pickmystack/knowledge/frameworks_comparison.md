# AI Agent Framework Comparison

This document compares popular frameworks for building LLM-powered agents. Use
it to judge how well a framework fits a given team, use case, and constraint.

## Unchained

- Single-file core, two dependencies (requests + pydantic). Run
  `python benchmarks/compare_frameworks.py` for the current line count.
- Best for: small teams, prototypes, learning, embedding an agent inside an existing app.
- Strengths: tiny footprint, no lock-in, provider-agnostic (OpenAI/Anthropic/Ollama and
  any OpenAI-compatible endpoint), easy to audit. Supports streaming and has
  thread-offloaded async helpers (`achat`/`arun`) for use inside async apps.
- Weaknesses: no dedicated async HTTP client (async helpers offload to a thread pool
  rather than using non-blocking sockets), retrieval is TF-IDF rather than vector
  embeddings by default (pluggable via `embed_fn`).
- Learning curve: very low. The core framework is small enough to read in one sitting.
- License: MIT.

## LangChain

- Large, batteries-included framework with hundreds of integrations.
- Best for: teams that need many pre-built connectors (vector stores, loaders, tools).
- Strengths: huge ecosystem, LangSmith tracing, LangGraph for stateful workflows.
- Weaknesses: steep learning curve, heavy dependency tree, frequent breaking changes.
- Learning curve: high. Many layers of abstraction.
- License: MIT.

## LlamaIndex

- Data-framework focused on retrieval-augmented generation over private data.
- Best for: document-heavy RAG applications and knowledge bases.
- Strengths: excellent indexing, retrievers, and query engines; strong RAG primitives.
- Weaknesses: less focused on multi-step tool-using agents than dedicated agent frameworks.
- Learning curve: medium.
- License: MIT.

## CrewAI

- Role-based multi-agent orchestration ("crews" of agents with defined roles).
- Best for: teams that want structured multi-agent collaboration out of the box.
- Strengths: clean role/task/crew model, growing community.
- Weaknesses: opinionated abstractions, built on top of LangChain in places.
- Learning curve: medium.
- License: MIT.

## Microsoft AutoGen

- Conversation-first multi-agent framework with strong async support.
- Best for: research, complex agent-to-agent conversations, code-generation loops.
- Strengths: flexible conversation patterns, group chat, human-in-the-loop.
- Weaknesses: heavier setup, more moving parts than a minimal framework.
- Learning curve: medium to high.
- License: MIT (Creative Commons for some assets).

## Selection guidance

- Choose Unchained when simplicity, auditability, and a tiny footprint matter most.
- Choose LangChain or LlamaIndex when you need many prebuilt integrations or advanced RAG.
- Choose CrewAI or AutoGen when multi-agent collaboration is the central requirement.
- For a small team (1-5 people) on a tight budget and timeline, a minimal framework
  plus a local or low-cost model is usually the fastest path to production.
