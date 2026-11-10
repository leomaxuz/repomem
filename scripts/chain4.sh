#!/usr/bin/env bash
# 4-zanjir: 12B (gemma4_unified arxitektura — hf-overrides'siz) RepoMem baholash.
# tmux: chain4 | Log: logs/chain4.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
R="code/repomem/repomem_out"; G="data/extracted/graphs"; BENCH="$R/tasks_full.jsonl"
LOG="logs/chain4.log"; VENV="$HOME/gemma4-paper/vllm-venv/bin"
D12=$(ls -d "$HOME"/gemma4-paper/models/models/google/gemma-4/other/gemma-4-12b-it-qat-w4a16-ct/*/ | head -1)
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
ready_check(){ python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0; }
free_gpu(){
  tmux kill-session -t vllm-serve 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-9999}" -lt 2500 ] && { say "GPU bo'sh"; return 0; }; sleep 3
  done
}
say "===== 4-ZANJIR (12B) BOSHLANDI ====="
free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && $VENV/vllm serve '${D12%/}' --served-model-name gemma-4-12b --tensor-parallel-size 2 --max-model-len 32768 --gpu-memory-utilization 0.90 --limit-mm-per-prompt '{\"image\":0,\"audio\":0,\"video\":0}' --port 8000 > logs/vllm_12b.log 2>&1"
OK=0
for i in $(seq 1 140); do
  [ "$(ready_check)" = "200" ] && { OK=1; say "12B TAYYOR"; break; }
  if ! tmux has-session -t vllm-serve 2>/dev/null; then say "12B server yiqildi"; break; fi
  sleep 15
done
if [ "$OK" = 1 ]; then
  say "12B baseline..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-12b --out "$R/e12b_full.jsonl" --concurrency 6 --thinking off >> "$LOG" 2>&1
  say "12B compact..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-12b --out "$R/e12b_compact.jsonl" --concurrency 6 --thinking off --compact --graphs "$G" >> "$LOG" 2>&1
  say "12B tugadi."
else
  say "XATO 12B:"; grep -a -E "Error|error" logs/vllm_12b.log | tail -8 | tee -a "$LOG"
fi
free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
say "===== 4-ZANJIR TUGADI ====="
