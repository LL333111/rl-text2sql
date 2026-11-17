#!/usr/bin/env bash
# One-time setup on the Linux GPU machine. Run from the repo root.
#
# Two virtualenvs, because vLLM pins its own torch build:
#   /workspace/venv-train  torch + transformers + peft + trl   (training)
#   /workspace/venv-eval   vllm                                (batched evaluation)
# Exact versions of both are frozen into requirements-*.lock on first install;
# later installs reuse the lock files.
set -euo pipefail
export HF_HOME=/workspace/hf
export PIP_CACHE_DIR=/workspace/pip-cache

install() {  # install <venv> <lockfile> <packages...>
  local venv=$1 lock=$2; shift 2
  [ -d "$venv" ] || python3 -m venv "$venv"
  "$venv/bin/pip" install -q --upgrade pip
  if [ -f "$lock" ]; then
    "$venv/bin/pip" install -q -r "$lock"
  else
    "$venv/bin/pip" install -q "$@" PyYAML numpy scipy pytest
    "$venv/bin/pip" freeze > "$lock"
  fi
}

install /workspace/venv-train requirements-train.lock torch transformers peft trl datasets accelerate matplotlib
install /workspace/venv-eval  requirements-eval.lock  vllm

/workspace/venv-train/bin/python -c "import torch, trl, peft, transformers; print('train:', torch.__version__, trl.__version__, peft.__version__, transformers.__version__, torch.cuda.is_available())"
/workspace/venv-eval/bin/python -c "import vllm, torch; print('eval:', vllm.__version__, torch.__version__)"
echo SETUP_DONE
