# PickMyStack Streamlit UI
#
# Build from the PROJECT ROOT (so unchained.py and the examples package are in
# the build context):
#
#   docker build -f examples/pickmystack/ui/Dockerfile -t pickmystack .
#   docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... pickmystack
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY examples/pickmystack/ui/requirements.txt ./ui-requirements.txt
RUN pip install --no-cache-dir -r ui-requirements.txt

# Copy the framework and the example app.
COPY unchained.py ./unchained.py
COPY examples ./examples

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "examples/pickmystack/ui/app_ui.py", \
            "--server.port=8501", "--server.address=0.0.0.0"]
