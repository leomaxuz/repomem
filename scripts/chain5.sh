#!/usr/bin/env bash
# 5-zanjir: PFK-agent ablations.
#  E4B: entity grounding by the model (model / model-short), noisy graph (edge-drop 0.1/0.3, +fallback)
#  E2B: PFK-agent (does it lift the smallest model to its oracle?)
# tmux: chain5 | Log: logs/chain5.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
R="code/repomem/repomem_out"; G="data/extracted/graphs"; BENCH="$R/tasks_full.jsonl"; LOG="logs/chain5.log"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
ready_check(){ python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0; }
served(){ python3 -c "import urllib.request,json;print(json.load(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5))['data'][0]['id'])" 2>/dev/null; }
free_gpu(){
  tmux kill-session -t vllm-serve 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-9999}" -lt 2500 ] && { say "GPU bo'sh"; return 0; }; sleep 3
  done
}
wait_ready(){ for i in $(seq 1 120); do [ "$(ready_check)" = "200" ] && { say "$1 TAYYOR"; return 0; }; sleep 10; done; say "XATO: $1 timeout"; return 1; }
pfk(){  # $1=model $2=out $3...=extra args
  local m="$1" o="$2"; shift 2
  say "PFK $m -> $o ($*)"
  python3 code/repomem/pfk_agent.py --tasks "$BENCH" --graphs "$G" --model "$m" --out "$R/$o" --concurrency 6 "$@" >> "$LOG" 2>&1
  say "  tugadi: $(tail -c 0 /dev/null)$(grep -a 'PFK JAMI' "$LOG" | tail -1)"
}

say "===== 5-ZANJIR BOSHLANDI ====="
if [ "$(served)" != "gemma-4-e4b" ]; then
  free_gpu
  tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
fi
wait_ready gemma-4-e4b || exit 1

pfk gemma-4-e4b pfk_e4b_entmodel.jsonl --entity-mode model
pfk gemma-4-e4b pfk_e4b_entshort.jsonl --entity-mode model-short
pfk gemma-4-e4b pfk_e4b_drop10.jsonl --edge-drop 0.1 --drop-seed 1
pfk gemma-4-e4b pfk_e4b_drop30.jsonl --edge-drop 0.3 --drop-seed 1
pfk gemma-4-e4b pfk_e4b_drop30_fb.jsonl --edge-drop 0.3 --drop-seed 1 --fallback-full

free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e2b.sh 8000 gemma-4-e2b > logs/vllm_serve.log 2>&1"
if wait_ready gemma-4-e2b; then pfk gemma-4-e2b pfk_e2b.jsonl; fi

free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
say "E4B qayta tiklanmoqda."
say "===== 5-ZANJIR TUGADI ====="
