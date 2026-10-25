#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace/models /workspace/outputs "$HF_HOME"
if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo 'Downloading CAD-Coder model to persistent workspace...'
  /opt/conda/envs/llava/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import os
snapshot_download(repo_id='CADCODER/CAD-Coder', local_dir=os.environ['MODEL_PATH'])
PY
fi
exec /opt/conda/envs/llava/bin/python /opt/app/web_gui.py
