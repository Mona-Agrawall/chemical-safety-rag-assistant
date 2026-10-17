#!/bin/sh
# Bring the index up to date with the documents volume, then start the app.
# Ingestion problems are reported but do not stop the app: the UI shows the
# knowledge-base status and offers a rebuild.
set -u

python ingest.py || echo "entrypoint: ingestion reported problems (see above); starting the app anyway" >&2

exec streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}"
