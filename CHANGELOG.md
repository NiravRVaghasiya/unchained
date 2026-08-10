# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-08-10

### Added
- **Richer tool schemas**: `@tool` parameters typed as `Literal[...]`, `Enum`
  subclasses, or nested Pydantic `BaseModel`s now produce an accurate JSON
  Schema (`enum` values, or an inlined object schema) instead of falling back
  to a plain string.
- **Async tools and agents**: `@tool` functions may be `async def` (driven by
  `Tool.run` via `asyncio.run`); `Agent.arun()` and `LLM.achat()` offload a
  full turn / request to a worker thread (`asyncio.to_thread`) so they can be
  awaited from async frameworks (FastAPI, aiohttp, ...) without blocking the
  event loop. This does not add a non-blocking async HTTP client - the
  underlying request is still made with `requests`.
- **Concurrent tool-call execution**: when a model requests more than one
  tool call in the same turn, `Agent` now runs them concurrently on a thread
  pool instead of sequentially, then feeds results back in the original
  order. Most tools are I/O-bound, so this cuts wall-clock latency per turn.
- **Token-aware `Memory`**: an optional `max_tokens` argument (rough
  4-chars/token estimate, no tokenizer dependency) additionally shrinks the
  retained window when the message-count cap alone would still risk
  overflowing the model's context window.
- **Per-provider `base_url` environment variables**: `OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, `OLLAMA_BASE_URL` are consulted when `base_url` isn't
  passed explicitly - this is what lets `provider="openai"` target any
  OpenAI-compatible endpoint (Groq, Together, OpenRouter, vLLM, LM Studio,
  ...) purely via configuration.
- `LLM.close()` to explicitly close the instance's HTTP session.

### Changed
- `LLM` now reuses a single `requests.Session` per instance instead of a
  fresh connection per call, so repeated requests reuse the underlying
  TCP/TLS connection.
- `LLM(cache=True)` is now a bounded LRU cache (`cache_size`, default 256)
  with an optional `cache_ttl` in seconds, instead of an unbounded dict -
  long-running processes no longer accumulate cache entries indefinitely.
- Fixed documentation drift: dropped hardcoded, quickly-stale line-count
  claims ("417 lines", "~620 lines") from `docs/ARCHITECTURE.md` and the
  PickMyStack knowledge base in favour of measuring live via
  `benchmarks/compare_frameworks.py`. The root `ARCHITECTURE.md` (an
  unreferenced duplicate of `docs/ARCHITECTURE.md`) is now a short pointer to
  the canonical copy used by the MkDocs site.

### Notes
- Investigated shipping a PEP 561 `py.typed` marker so downstream type
  checkers trust the inline types on an installed `pip install unchained-ai`.
  PEP 561 has no supported mechanism for module-only (`py-modules`)
  distributions to ship it - only package (directory) distributions can, and
  that would require restructuring `unchained.py` into a package, which
  conflicts with the single-file design. Not changed; documented in
  `pyproject.toml`.

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

[Unreleased]: https://github.com/NiravRVaghasiya/unchained/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NiravRVaghasiya/unchained/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NiravRVaghasiya/unchained/releases/tag/v0.1.0
