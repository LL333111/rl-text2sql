#!/usr/bin/env bash
# Stages 2 and 3 on the GPU machine, run unattended and in sequence:
#   SFT config A, GRPO config A, SFT config B, GRPO config B   (3.0 GPU-h each)
#   each followed by evaluation of its checkpoints on val,
#   then selection on val (gates G1, G2) and, if G2 passes, the partial-credit
#   ablation with the selected GRPO learning rate.
# Every step writes START/OK/FAIL markers; a failed step does not stop the queue.
# Finished steps are skipped on re-run, so the script can be restarted safely.
set -uo pipefail
cd /workspace/rl-text2sql
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
EVAL=/workspace/venv-eval/bin/python
TRAIN=/workspace/venv-train/bin/python
BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct

train() {  # train <run> <module> <config> <overrides...>
  local run=$1 mod=$2 cfg=$3; shift 3
  if [ -f "results/$run/run_metadata.json" ]; then echo "=== SKIP train $run"; return; fi
  echo "=== START train $run $(date -u +%F_%T)"
  if $TRAIN -m "$mod" --config "$cfg" --out "results/$run" --set "$@" > "/workspace/logs/train_$run.log" 2>&1
  then echo "=== OK train $run $(date -u +%F_%T)"
  else echo "=== FAIL train $run $(date -u +%F_%T)"; tail -n 5 "/workspace/logs/train_$run.log"; fi
}

evalval() {  # evalval <run>
  local run=$1
  echo "=== START eval $run $(date -u +%F_%T)"
  if $EVAL -m src.evaluate --split val --model $BASE --checkpoints-of "results/$run" > "/workspace/logs/eval_$run.log" 2>&1
  then echo "=== OK eval $run $(date -u +%F_%T)"
  else echo "=== FAIL eval $run $(date -u +%F_%T)"; tail -n 5 "/workspace/logs/eval_$run.log"; fi
}

train sft_a  src.train_sft  configs/sft.yaml  learning_rate=1e-4
evalval sft_a
train grpo_a src.train_grpo configs/grpo.yaml learning_rate=2e-5
evalval grpo_a
train sft_b  src.train_sft  configs/sft.yaml  learning_rate=3e-5
evalval sft_b
train grpo_b src.train_grpo configs/grpo.yaml learning_rate=5e-5
evalval grpo_b

echo "=== SELECT $(date -u +%F_%T)"
if $TRAIN scripts/select_best.py --require-g2; then
  LR=$($TRAIN -c "import json;s=json.load(open('results/selection.json'));r=s['arms']['grpo']['chosen_run'];print(json.load(open(f'results/{r}/run_metadata.json'))['trainer']['learning_rate'])")
  echo "=== G2 PASS; ablation with learning_rate=$LR"
  train grpo_partial src.train_grpo configs/grpo_partial.yaml learning_rate=$LR
  evalval grpo_partial
  $TRAIN scripts/select_best.py
else
  echo "=== G2 FAIL: ablation not run"
fi
echo ALL_TRAIN_DONE
