# Contributing to Unchained

Thanks for your interest in improving Unchained. This project values a small,
readable core, so contributions that keep things simple are especially welcome.

## Ground rules

- Keep the core in `unchained.py` focused and dependency-light (requests + pydantic).
- New capabilities usually belong in `examples/`, not the core.
- Every change should keep lint, type-check, and tests green.

## Development setup

```bash
git clone https://github.com/NiravRVaghasiya/unchained.git
cd unchained
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## The checks CI runs

Run these locally before opening a pull request:

```bash
ruff check .            # lint
ruff format --check .   # formatting
mypy                    # type-check the core (needs Python 3.10+)
pytest --cov=unchained  # tests + coverage
```

`pre-commit run --all-files` runs the lint/format/type hooks in one go.

## Making a change

1. Create a branch: `git checkout -b feature/short-description`.
2. Make your change. Add or update tests in `tests/`.
3. Keep the public API stable, or call out breaking changes clearly.
4. Update `CHANGELOG.md` under `[Unreleased]`.
5. Ensure all checks pass, then open a PR against `main`.

## Tests

- Tests must run offline. Use the `FakeLLM` pattern (see `tests/test_unchained.py`) or monkeypatch `unchained.requests.post`. Never call a real provider in a test.
- Cover the behavior you add, including at least one error path.

## Coding style

- Formatting and imports are handled by `ruff format` and `ruff` (isort rules); don't hand-format.
- Type hints are required in the core. Keep annotations 3.9-compatible (use `Optional[X]`, `List[X]`, etc.).
- Prefer clarity over cleverness. If a comment is needed to explain "why", add it.

## Reporting bugs and requesting features

Open an issue using the templates in `.github/ISSUE_TEMPLATE`. For security
issues, please do not open a public issue; see `SECURITY.md` if present or
contact the maintainers directly.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
