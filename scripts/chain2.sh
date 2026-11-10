#!/usr/bin/env bash
# 2-zanjir: (1) E2B size-scaling (baseline+compact) -> (2) E4B PFK-agent -> birlashgan hisobot.
# Bitta GPU to'plamida bitta model sig'adi, shuning uchun modellar ketma-ket almashtiriladi.
# tmux: chain2 | Log: logs/chain2.log | Natija: code/repomem/repomem_out/REPORT2.md
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
P="python3 bin/stage.py"
R="code/repomem/repomem_out"
G="data/extracted/graphs"
BENCH="$R/tasks_full.jsonl"
LOG="logs/chain2.log"
mkdir -p "$R"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

ready_check(){ python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0; }

free_gpu(){
  tmux kill-session -t vllm-serve 2>/dev/null
  pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 50); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-9999}" -lt 2500 ] && { say "GPU bo'shadi (${used}MiB)"; return 0; }
    sleep 3
  done
  say "OGOH: GPU to'liq bo'shamadi (${used:-?}MiB) — davom etaman"; return 0
}

start_serve(){  # $1=serve skript  $2=model nomi
  tmux kill-session -t vllm-serve 2>/dev/null
  tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/$1 8000 $2 > logs/vllm_serve.log 2>&1"
  for i in $(seq 1 120); do
    [ "$(ready_check)" = "200" ] && { say "$2 server TAYYOR"; return 0; }
    if grep -qa "CUDA out of memory\|RuntimeError\|Traceback (most recent" logs/vllm_serve.log 2>/dev/null; then
      say "XATO: $2 yuklashda xato:"; tail -15 logs/vllm_serve.log | tee -a "$LOG"; return 1
    fi
    sleep 15
  done
  say "XATO: $2 server tayyor bo'lmadi (timeout)"; return 1
}

say "===== 2-ZANJIR BOSHLANDI ====="
[ -x vllm-venv/bin/vllm ] || { say "XATO: vLLM yo'q"; exit 1; }
[ -s "$BENCH" ] || { say "XATO: benchmark yo'q ($BENCH)"; exit 1; }
say "Benchmark: $(wc -l < "$BENCH") savol"

# ================= 1-QISM: E2B size-scaling =================
say "--- 1-QISM: E2B (size-scaling) ---"
$P set 7 running "E2B size-scaling tajribasi (model almashtirilmoqda)"
free_gpu
if start_serve serve_e2b.sh gemma-4-e2b; then
  say "E2B baseline (to'liq kontekst)..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-e2b \
      --out "$R/e2b_full.jsonl" --concurrency 6 --thinking off >> "$LOG" 2>&1
  say "E2B baseline tugadi."
  say "E2B compact (ishchi-xotira)..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-e2b \
      --out "$R/e2b_compact.jsonl" --concurrency 6 --thinking off --compact --graphs "$G" >> "$LOG" 2>&1
  say "E2B compact tugadi."
  $P set 7 done "E4B+E2B RepoMem baholandi (size-scaling)"
else
  say "E2B serveri ishga tushmadi — 1-qismni o'tkazib yuboraman."
  $P set 7 done "E4B baholandi (E2B o'tkazib yuborildi)"
fi

# ================= 2-QISM: E4B PFK-agent =================
say "--- 2-QISM: E4B PFK-agent ---"
$P set 8 running "PFK-agent (graf-retrieval ishchi-xotira) ishlamoqda"
free_gpu
if start_serve serve_e4b.sh gemma-4-e4b; then
  say "PFK-agent baholanmoqda (E4B, graf-retrieval)..."
  python3 code/repomem/pfk_agent.py --tasks "$BENCH" --graphs "$G" --model gemma-4-e4b \
      --base-url http://127.0.0.1:8000/v1 --out "$R/pfk_e4b.jsonl" --concurrency 6 >> "$LOG" 2>&1
  PSUM=$(grep "PFK JAMI:" "$LOG" | tail -1)
  say "PFK-agent tugadi: $PSUM"
  $P set 8 done "PFK-agent tugadi — $PSUM"
else
  say "XATO: E4B serveri ishga tushmadi — PFK o'tkazib yuborildi."
  $P set 8 failed "E4B server ishga tushmadi"
fi

# ================= 3-QISM: birlashgan hisobot =================
say "--- 3-QISM: birlashgan hisobot ---"
$P set 9 running "baseline / compact / PFK / E2B taqqoslash"
python3 code/repomem/summarize.py --out "$R/REPORT2.md" \
  --add "E4B baseline:$R/exp1_e4b_full.jsonl" \
  --add "E4B compact (oracle):$R/exp2_e4b_compact.jsonl" \
  --add "E4B PFK-agent:$R/pfk_e4b.jsonl" \
  --add "E2B baseline:$R/e2b_full.jsonl" \
  --add "E2B compact (oracle):$R/e2b_compact.jsonl" >> "$LOG" 2>&1
cp "$R/REPORT2.md" "$HOME/gemma4-paper/REPORT2.md" 2>/dev/null
say "REPORT2.md tayyor."
$P set 9 done "Taqqoslash tayyor — REPORT2.md (baseline vs compact vs PFK vs E2B)"

say "===== 2-ZANJIR TUGADI ====="
