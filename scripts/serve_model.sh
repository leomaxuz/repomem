#!/usr/bin/env bash
# Umumiy vLLM serve: istalgan model yo'li uchun (matn rejimi, gemma4 override).
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
MODEL="$1"; NAME="$2"; PORT="${3:-8000}"; MAXLEN="${4:-32768}"; UTIL="${5:-0.90}"
VENV="$HOME/gemma4-paper/vllm-venv/bin"
if [ ! -x "$VENV/vllm" ]; then echo "vLLM yo'q"; exit 1; fi
exec "$VENV/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size 2 \
  --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$UTIL" \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}' \
  --port "$PORT"
