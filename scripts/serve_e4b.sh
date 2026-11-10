#!/usr/bin/env bash
# Gemma 4 E4B (w4a16) ni vLLM OpenAI-mos server sifatida ishga tushiradi (matn rejimi).
# tmux sessiya: vllm-serve. Log: logs/vllm_serve.log. Port: 8000.
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
MODEL="$HOME/gemma4-paper/models/models/google/gemma-4/other/gemma-4-e4b-it-qat-w4a16-ct/2"
VENV="$HOME/gemma4-paper/vllm-venv/bin"
PORT="${1:-8000}"
NAME="${2:-gemma-4-e4b}"
P="python3 bin/stage.py"

if [ ! -x "$VENV/vllm" ]; then echo "vLLM o'rnatilmagan"; exit 1; fi

$P set 2 running "E4B vLLM serveri ishga tushirilmoqda (port $PORT)"
# Matn rejimi: Gemma4ForCausalLM override (vision/audio towerlarsiz).
# gemma4 tool/reasoning parserlari musobaqa harnessidagidek.
exec "$VENV/vllm" serve "$MODEL" \
  --served-model-name "$NAME" \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}' \
  --port "$PORT"
