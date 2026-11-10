#!/usr/bin/env bash
# 7-zanjir: RepoMem-OSS (graphs built with build_graph.py from 6 public repos) — E4B and E2B:
# baseline, compact oracle, PFK-agent.   tmux: chain7 | Log: logs/chain7.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
R="ext/out"; G="ext/graphs"; BENCH="ext/tasks_ext.jsonl"; LOG="logs/chain7.log"; mkdir -p "$R"
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
run_all(){  # $1 = served model name, $2 = tag
  say "$1 baseline"
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model "$1" --out "$R/$2_full.jsonl" --concurrency 6 --thinking off >> "$LOG" 2>&1
  say "$1 compact"
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model "$1" --out "$R/$2_compact.jsonl" --concurrency 6 --thinking off --compact --graphs "$G" >> "$LOG" 2>&1
  say "$1 PFK"
  python3 code/repomem/pfk_agent.py --tasks "$BENCH" --graphs "$G" --model "$1" --out "$R/$2_pfk.jsonl" --concurrency 6 >> "$LOG" 2>&1
  say "$1 tugadi"
}

say "===== 7-ZANJIR BOSHLANDI ====="
if [ "$(served)" != "gemma-4-e4b" ]; then
  free_gpu
  tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
fi
wait_ready gemma-4-e4b && run_all gemma-4-e4b e4b

free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e2b.sh 8000 gemma-4-e2b > logs/vllm_serve.log 2>&1"
wait_ready gemma-4-e2b && run_all gemma-4-e2b e2b

free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
say "===== 7-ZANJIR TUGADI ====="
