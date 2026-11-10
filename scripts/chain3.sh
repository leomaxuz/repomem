#!/usr/bin/env bash
# 3-zanjir: 12B va 31B modellarini RepoMem'da baholaydi (o'lcham-egri chizig'i: E2B->E4B->12B->31B).
# Download (dl-big) va seed batareya (seeds) tugashini kutadi, keyin modellarni ketma-ket almashtiradi.
# tmux: chain3 | Log: logs/chain3.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
R="code/repomem/repomem_out"
G="data/extracted/graphs"
BENCH="$R/tasks_full.jsonl"
LOG="logs/chain3.log"
OTHER="$HOME/gemma4-paper/models/models/google/gemma-4/other"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
ready_check(){ python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0; }

free_gpu(){
  tmux kill-session -t vllm-serve 2>/dev/null; pkill -f "vllm serve" 2>/dev/null
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    [ "${used:-9999}" -lt 2500 ] && { say "GPU bo'sh (${used}MiB)"; return 0; }
    sleep 3
  done
  say "OGOH: GPU to'liq bo'shamadi"; return 0
}
serve(){  # $1=model path $2=name $3=maxlen $4=util
  tmux kill-session -t vllm-serve 2>/dev/null
  tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_model.sh '$1' '$2' 8000 '$3' '$4' > logs/vllm_serve.log 2>&1"
  for i in $(seq 1 140); do
    [ "$(ready_check)" = "200" ] && { say "$2 TAYYOR"; return 0; }
    if grep -qa "CUDA out of memory\|RuntimeError\|ValueError\|Traceback (most recent" logs/vllm_serve.log 2>/dev/null; then
      say "XATO $2:"; tail -12 logs/vllm_serve.log | tee -a "$LOG"; return 1
    fi
    sleep 15
  done
  say "XATO: $2 timeout"; return 1
}
eval_model(){  # $1=served-name $2=tag
  say "$1 baseline..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model "$1" --out "$R/${2}_full.jsonl" \
      --concurrency 6 --thinking off >> "$LOG" 2>&1
  say "$1 compact..."
  python3 code/repomem/evaluate.py --tasks "$BENCH" --model "$1" --out "$R/${2}_compact.jsonl" \
      --concurrency 6 --thinking off --compact --graphs "$G" >> "$LOG" 2>&1
  say "$1 tugadi."
}

say "===== 3-ZANJIR BOSHLANDI ====="

# 1) download tugashini kutamiz (12B majburiy, 31B ixtiyoriy)
say "Download (dl-big) tugashini kutyapman..."
for i in $(seq 1 120); do
  tmux has-session -t dl-big 2>/dev/null || break
  sleep 30
done
D12=$(ls -d "$OTHER"/gemma-4-12b-it-qat-w4a16-ct/*/ 2>/dev/null | head -1)
D31=$(ls -d "$OTHER"/gemma-4-31b-it-qat-w4a16-ct/*/ 2>/dev/null | head -1)
say "12B yo'l: ${D12:-YOQ}"; say "31B yo'l: ${D31:-YOQ}"

# 2) seed batareya tugashini kutamiz (E4B serverni bo'shatish uchun)
say "Seed batareya (seeds) tugashini kutyapman..."
for i in $(seq 1 160); do
  tmux has-session -t seeds 2>/dev/null || break
  sleep 30
done
say "Seeds tugadi (yoki yo'q)."

# 3) 12B
if [ -n "$D12" ] && [ -f "${D12}config.json" ]; then
  free_gpu
  if serve "${D12%/}" gemma-4-12b 32768 0.90; then eval_model gemma-4-12b e12b; else say "12B o'tkazib yuborildi"; fi
else say "12B yo'q — o'tkazib yuborildi"; fi

# 4) 31B (ixtiyoriy; KV uchun maxlen pastroq, util yuqoriroq)
if [ -n "$D31" ] && [ -f "${D31}config.json" ]; then
  free_gpu
  if serve "${D31%/}" gemma-4-31b 16384 0.94; then eval_model gemma-4-31b e31b; else say "31B o'tkazib yuborildi"; fi
else say "31B yo'q — o'tkazib yuborildi"; fi

# 5) E4B serverni tiklaymiz
free_gpu
tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
say "E4B server qayta tiklanmoqda."

# 6) o'lcham-egri hisobot
say "Scaling hisobot..."
python3 code/repomem/summarize.py --out "$R/REPORT_scaling.md" \
  --add "E2B baseline:$R/e2b_full.jsonl" \
  --add "E4B baseline:$R/exp1_e4b_full.jsonl" \
  --add "12B baseline:$R/e12b_full.jsonl" \
  --add "31B baseline:$R/e31b_full.jsonl" \
  --add "E2B compact:$R/e2b_compact.jsonl" \
  --add "E4B compact:$R/exp2_e4b_compact.jsonl" \
  --add "12B compact:$R/e12b_compact.jsonl" \
  --add "31B compact:$R/e31b_compact.jsonl" >> "$LOG" 2>&1
cp "$R/REPORT_scaling.md" "$HOME/gemma4-paper/REPORT_scaling.md" 2>/dev/null
say "===== 3-ZANJIR TUGADI ====="
