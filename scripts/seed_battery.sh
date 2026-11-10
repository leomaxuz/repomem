#!/usr/bin/env bash
# Ko'p-urug' robustlik batareyasi: seed {1,2,3} uchun benchmark + baseline + compact + PFK (hammasi E4B).
# E4B server (port 8000) tirik bo'lishi kerak. tmux: seeds | Log: logs/seeds.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
R="code/repomem/repomem_out/seeds"
G="data/extracted/graphs"
LOG="logs/seeds.log"
mkdir -p "$R"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "===== SEED BATAREYA BOSHLANDI ====="
# server tayyormi
for i in $(seq 1 40); do
  c=$(python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0)
  [ "$c" = "200" ] && break; sleep 10
done
[ "$c" = "200" ] || { say "XATO: E4B server yo'q"; exit 1; }

for S in 1 2 3; do
  B="$R/bench_s$S.jsonl"
  say "seed $S: benchmark (per-repo 12)..."
  python3 code/repomem/generate.py --graphs "$G" --out "$B" --per-repo 12 --seed "$S" >> "$LOG" 2>&1
  say "seed $S: n=$(wc -l < "$B")"
  say "seed $S: baseline..."
  python3 code/repomem/evaluate.py --tasks "$B" --model gemma-4-e4b --out "$R/seed_s${S}_base.jsonl" \
      --concurrency 6 --thinking off >> "$LOG" 2>&1
  say "seed $S: compact..."
  python3 code/repomem/evaluate.py --tasks "$B" --model gemma-4-e4b --out "$R/seed_s${S}_compact.jsonl" \
      --concurrency 6 --thinking off --compact --graphs "$G" >> "$LOG" 2>&1
  say "seed $S: PFK..."
  python3 code/repomem/pfk_agent.py --tasks "$B" --graphs "$G" --model gemma-4-e4b \
      --base-url http://127.0.0.1:8000/v1 --out "$R/seed_s${S}_pfk.jsonl" --concurrency 6 >> "$LOG" 2>&1
  say "seed $S tugadi."
done

say "Agregatsiya..."
python3 code/repomem/aggregate_seeds.py --dir "$R" --seeds 1,2,3 --out "$R/REPORT_seeds.md" >> "$LOG" 2>&1
cp "$R/REPORT_seeds.md" "$HOME/gemma4-paper/REPORT_seeds.md" 2>/dev/null
say "===== SEED BATAREYA TUGADI ====="
