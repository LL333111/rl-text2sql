#!/usr/bin/env bash
# Stage 1 on the GPU machine: zero-shot rows on val, then the GRPO timing test.
# Each step writes a marker line so progress can be followed from the log.
set -uo pipefail
cd /workspace/rl-text2sql
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
EVAL=/workspace/venv-eval/bin/python
TRAIN=/workspace/venv-train/bin/python

step() {  # step <name> <command...>
  local name=$1; shift
  echo "=== START $name $(date -u +%T)"
  if "$@"; then echo "=== OK $name $(date -u +%T)"; else echo "=== FAIL $name $(date -u +%T)"; fi
}

step row1_val  $EVAL -m src.evaluate --split val --model Qwen/Qwen2.5-Coder-1.5B-Instruct --out results/row1_zeroshot/val
step ref7b_val $EVAL -m src.evaluate --split val --model Qwen/Qwen2.5-Coder-7B-Instruct   --out results/ref_7b_zeroshot/val
step sft_smoke $TRAIN -m src.train_sft --out results/sft_smoke --max-steps 20
step grpo_timing $TRAIN -m src.train_grpo --config configs/grpo.yaml --out results/grpo_timing --max-steps 20
echo STAGE1_DONE
