# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-07-02

### Added
- **`MockLLM`**: a deterministic, no-network stand-in for `LLM` so agents run
  with zero setup (fixed reply, scripted responses, or a custom handler).
- **Pluggable embeddings for RAG**: pass `embed_fn` to switch from TF-IDF to
  dense-vector cosine similarity; TF-IDF remains the zero-dependency default.
- **Opt-in response caching**: `LLM(cache=True)` memoizes identical requests.
- `examples/quickstart.py`: a no-API-key tour of the whole framework.
- MkDocs-material documentation site with an auto-generated API reference
  (mkdocstrings) and a GitHub Pages deploy workflow.

### Changed
- `examples/coder.py` now runs generated code in an isolated subprocess with a
  timeout instead of in-process `exec` (still not a full security sandbox).

## [0.2.0] - 2026-07-02

### Added
- **Streaming**: `LLM.stream()` for all three providers and `Agent.stream()`
  to yield the final answer token by token.
- **Reliability**: automatic retry with exponential backoff and jitter on
  connection errors and `429`/`5xx` responses, honouring `Retry-After`
  (`max_retries`, `backoff` on `LLM`).
- **Observability**: a `Callback` base class and ready-made `LoggingCallback`,
  plus a module logger (`logging.getLogger("unchained")`). Agents accept
  `callbacks=[...]` and emit `on_iteration`/`on_llm_call`/`on_tool_call`/`on_finish`.
- **Token usage tracking**: usage is normalised across providers to
  `{prompt_tokens, completion_tokens, total_tokens}` and accumulated on
  `Agent.usage`.
- **Structured-output repair**: on a Pydantic `ValidationError`, the agent
  feeds the error and schema back to the model to self-correct
  (`structured_retries`).
- `examples/sqlite_memory.py`: a persistent `SQLiteMemory` demonstrating the
  `Memory` extension point.
- Developer infrastructure: GitHub Actions CI (lint, type-check, test matrix on
  Python 3.9-3.13), PyPI publish workflow, `ruff`/`mypy`/`pre-commit` config,
  an expanded offline test suite, and contributor scaffolding.

### Changed
- Line-count claims replaced with "single file, two dependencies" after
  adopting `ruff format`.

## [0.1.0] - 2026-07-02

### Added
- Single-file core `unchained.py`: `@tool`/`Tool`, multi-provider `LLM`
  (OpenAI, Anthropic, Ollama), `Memory`, `RAG`, `Agent` (ReAct loop), `Router`,
  and Pydantic structured output.
- Examples: `researcher`, `coder`, `data_analyst`, and the flagship
  multi-agent `pickmystack` app with a CLI and a Streamlit UI.
- Benchmarks, unit tests, and documentation (README, architecture, user manual).

[Unreleased]: https://github.com/NiravRVaghasiya/unchained/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NiravRVaghasiya/unchained/releases/tag/v0.1.0
