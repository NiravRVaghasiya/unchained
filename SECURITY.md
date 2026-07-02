# Security Policy

## Supported versions

Unchained is pre-1.0. Security fixes are applied to the latest release on the
`main` branch.

## Reporting a vulnerability

Please do not open a public issue for security vulnerabilities. Instead, use
GitHub's private ["Report a vulnerability"](https://github.com/NiravRVaghasiya/unchained/security/advisories/new)
advisory flow, or contact the maintainers directly.

We aim to acknowledge reports within a few days and will keep you updated on
remediation progress.

## Notes for users

- **API keys:** never commit real keys. Use environment variables or a local
  `.env` (see `.env.example`), both of which are gitignored.
- **`examples/coder.py`** executes model-generated Python in-process. It is a
  demo only. Do not expose it to untrusted input without a proper sandbox
  (separate process, container, resource limits).
- **The Streamlit UI** has no built-in authentication. Put it behind access
  control before exposing it publicly, or it will spend your API quota.
- Treat all model output as untrusted when feeding it into tools, shells, or
  file operations.
