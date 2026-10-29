#!/usr/bin/env bash
# Create the WSL venv (~/.venvs/wordle, kept off /mnt/c for speed) and install training deps.
# Run: wsl -d Ubuntu-24.04 -- bash scripts/setup_env.sh
set -euo pipefail
UV=~/.local/bin/uv
VENV=~/.venvs/wordle

if [ ! -x "$VENV/bin/python" ]; then
    $UV venv --python 3.12 "$VENV"
fi

# trl 1.16.0 / transformers 5.20 / peft 0.21 match the versions already working on this machine (lexp venv).
# vLLM brings its own matching torch build.
$UV pip install --python "$VENV/bin/python" \
    "trl[vllm]==1.16.0" \
    "peft==0.21.2" \
    "textarena" \
    "trackio" \
    "pytest"

"$VENV/bin/python" - <<'EOF'
import torch, trl, transformers, vllm, textarena
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("trl", trl.__version__, "transformers", transformers.__version__, "vllm", vllm.__version__)
print("textarena", getattr(textarena, "__version__", "?"))
EOF
