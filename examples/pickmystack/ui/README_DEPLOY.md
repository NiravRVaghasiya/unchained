# Deploying the PickMyStack UI

The UI is a single Streamlit app (`app_ui.py`) that drives the PickMyStack
multi-agent pipeline. Below are three ways to run it, from easiest to most
production-ready.

## 1. Run locally

From the **project root**:

```bash
pip install -r examples/pickmystack/ui/requirements.txt
streamlit run examples/pickmystack/ui/app_ui.py
```

Then open http://localhost:8501.

- **Ollama (default):** install [Ollama](https://ollama.com), run `ollama pull llama3.1`, and make sure the daemon is up at `http://localhost:11434`. No API key needed.
- **OpenAI / Anthropic:** paste your key in the sidebar, or export `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` before launching.

## 2. Docker

Build from the project root so the framework and example package are in the
build context:

```bash
docker build -f examples/pickmystack/ui/Dockerfile -t pickmystack .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... pickmystack
```

To talk to an Ollama instance on the host from inside the container, point the
app at `http://host.docker.internal:11434` (macOS/Windows) or run Ollama in a
sibling container on the same network.

## 3. Managed platforms

### Streamlit Community Cloud
1. Push this repository to GitHub.
2. Create a new app pointing at `examples/pickmystack/ui/app_ui.py`.
3. Add your model API key under **App settings -> Secrets**.

### Hugging Face Spaces
1. Create a new **Streamlit** Space.
2. Add `unchained.py` and the `examples/` folder, plus a `requirements.txt` mirroring `ui/requirements.txt`.
3. Set the app file to `examples/pickmystack/ui/app_ui.py` and add your key as a Space secret.

### Render / Fly.io / Cloud Run
Use the provided `Dockerfile`. Set the start command to the container entrypoint
and expose port `8501`. Provide the model API key as an environment variable.

## Configuration

- Theme and server settings live in [`.streamlit/config.toml`](.streamlit/config.toml).
- Never commit real API keys. Use platform secrets or environment variables; `.streamlit/secrets.toml` is gitignored.

## Security note

The recommendation pipeline calls out to your chosen LLM provider. When you
deploy publicly, put the app behind authentication (or a platform access
control) so you don't expose your API quota to the world.
