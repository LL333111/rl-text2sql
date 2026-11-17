#!/usr/bin/env bash
# Stage 4 on the GPU machine: the first and only time test-A (Spider dev) and
# test-B (BIRD dev) are read. Every row is evaluated once on each test set.
# Checkpoints come from results/selection.json, which was computed on val only.
# Finished evaluations are skipped, so nothing is ever evaluated twice.
set -uo pipefail
cd /workspace/rl-text2sql
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
EVAL=/workspace/venv-eval/bin/python
PY=/workspace/venv-train/bin/python
BASE=Qwen/Qwen2.5-Coder-1.5B-Instruct
REF=Qwen/Qwen2.5-Coder-7B-Instruct

ev() {  # ev <row> <model> [adapter]
  local row=$1 model=$2 adapter=${3:-}
  for split in test_a test_b; do
    local out=results/final/$row/$split
    if [ -f "$out/metrics.json" ]; then echo "=== SKIP $row $split"; continue; fi
    echo "=== START $row $split $(date -u +%F_%T)"
    if $EVAL -m src.evaluate --split $split --allow-test --model $model ${adapter:+--adapter $adapter} \
         --max-model-len 32768 --out $out > /workspace/logs/final_${row}_${split}.log 2>&1
    then echo "=== OK $row $split $(grep -E '"exec_acc"' $out/metrics.json)"
    else echo "=== FAIL $row $split"; tail -n 4 /workspace/logs/final_${row}_${split}.log; fi
  done
}

ckpt() {  # ckpt <arm> <best|end>  -> adapter path
  $PY -c "import json;s=json.load(open('results/selection.json'))['arms']['$1'];print('results/'+s['chosen_run']+'/'+s['$2']['checkpoint'])"
}

ev row1_zeroshot $BASE
ev row2_sft_best  $BASE "$(ckpt sft best)"
ev row2_sft_end   $BASE "$(ckpt sft end)"
ev row3_grpo_best $BASE "$(ckpt grpo best)"
ev row3_grpo_end  $BASE "$(ckpt grpo end)"
if $PY -c "import json,sys;sys.exit(0 if 'grpo_partial' in json.load(open('results/selection.json'))['arms'] else 1)"; then
  ev abl_partial_best $BASE "$(ckpt grpo_partial best)"
  ev abl_partial_end  $BASE "$(ckpt grpo_partial end)"
fi
ev ref_7b_zeroshot $REF
echo FINAL_DONE
