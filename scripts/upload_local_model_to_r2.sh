#!/usr/bin/env bash
# Upload the generated local-model/ bundle to the R2 bucket that the dev
# Worker serves from (env.dev binding MODEL_BUCKET -> kirciuokle-models).
# Run after: uv run scripts/prepare_local_model.py ... (two-tier bundle).
set -euo pipefail

BUCKET="kirciuokle-models"
SRC="local-model"

[ -f .env ] && { set -a; . ./.env; set +a; }

cd "$(dirname "$0")/.."

find "$SRC" -type f | while read -r f; do
  key="${f#"$SRC"/}"
  case "$f" in
    *.onnx|*.model) ct="application/octet-stream" ;;
    *.wasm)         ct="application/wasm" ;;
    *.mjs|*.js)     ct="text/javascript" ;;
    *.json)         ct="application/json" ;;
    *)              ct="application/octet-stream" ;;
  esac
  echo "put $key ($ct)"
  # --remote is REQUIRED: without it wrangler writes to the local .wrangler
  # simulated bucket, which the deployed Worker never reads.
  npx wrangler r2 object put "$BUCKET/$key" --file "$f" --content-type "$ct" --remote >/dev/null
done
echo "uploaded $(find "$SRC" -type f | wc -l) files to r2://$BUCKET"
