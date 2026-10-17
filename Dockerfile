# Chemical Safety RAG Assistant: application image (the LLM runs in a separate Ollama container).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    OLLAMA_BASE_URL=http://ollama:11434

WORKDIR /app
RUN useradd --create-home --uid 1000 app

# Dependencies first for layer caching. CPU-only torch keeps the image ~1.5 GB smaller.
COPY requirements.txt requirements.lock ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock

# Bake the embedding model into the image so the container needs no internet at runtime.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'])"

COPY --chown=app:app rag/ rag/
COPY --chown=app:app ui/ ui/
COPY --chown=app:app .streamlit/ .streamlit/
COPY --chown=app:app app.py ingest.py ./
COPY --chown=app:app data/ data/
COPY --chown=app:app docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/storage /app/data/uploads \
    && chown -R app:app /app

USER app
# Documents (bundled + uploads) and the generated index persist in volumes.
VOLUME ["/app/data", "/app/storage"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
