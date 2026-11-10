#!/usr/bin/env bash
# Tungi tajribalar batareyasi. E4B server (port 8000) ishlab turgan bo'lishi kerak.
# tmux: exp | Log: logs/experiments.log | Natija: code/repomem/repomem_out/
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
P="python3 bin/stage.py"
R="code/repomem/repomem_out"
G="data/extracted/graphs"
BENCH="$R/tasks_full.jsonl"
LOG="logs/experiments.log"
mkdir -p "$R"
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== TAJRIBALAR BOSHLANDI ==="

# 0) Server tayyormi?
ready=""
for i in $(seq 1 40); do
  c=$(python3 - <<'PY' 2>/dev/null
import urllib.request
try:
    print(urllib.request.urlopen("http://127.0.0.1:8000/v1/models",timeout=5).status)
except Exception:
    print(0)
PY
)
  [ "$c" = "200" ] && { ready="yes"; break; }
  sleep 10
done
if [ "$ready" != "yes" ]; then
  say "XATO: E4B server javob bermayapti. Qayta ishga tushiraman..."
  tmux kill-session -t vllm-serve 2>/dev/null
  tmux new-session -d -s vllm-serve "cd ~/gemma4-paper && bash bin/serve_e4b.sh 8000 gemma-4-e4b > logs/vllm_serve.log 2>&1"
  for i in $(seq 1 100); do
    c=$(python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=5).status)" 2>/dev/null || echo 0)
    [ "$c" = "200" ] && { ready="yes"; break; }
    sleep 15
  done
  [ "$ready" != "yes" ] && { say "XATO: server baribir tayyor emas. To'xtadim."; exit 1; }
fi
say "Server tayyor."

# 1) Kengaytirilgan benchmark
say "Benchmark yaratilmoqda (per-repo 20, barcha graflar, seed 1)..."
$P set 6 running "kengaytirilgan RepoMem benchmark"
python3 code/repomem/generate.py --graphs "$G" --out "$BENCH" --per-repo 20 --seed 1 >> "$LOG" 2>&1
NB=$(wc -l < "$BENCH" 2>/dev/null || echo 0)
say "Benchmark: $NB savol"
[ "$NB" -lt 1 ] && { $P set 6 failed "benchmark bo'sh"; exit 1; }
$P set 6 done "RepoMem benchmark: $NB savol (defn/chain/caller)"

$P set 7 running "E4B tajribalari ketmoqda (baseline + compact + thinking)"

# 2) Exp1 — baseline, to'liq kontekst
say "1-tajriba: E4B to'liq kontekst (baseline)..."
python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-e4b \
    --out "$R/exp1_e4b_full.jsonl" --concurrency 6 --thinking off >> "$LOG" 2>&1
say "1-tajriba tugadi."

# 3) Exp2 — compact / ishchi-xotira
say "2-tajriba: E4B compact (ishchi-xotira, distraktorsiz)..."
python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-e4b \
    --out "$R/exp2_e4b_compact.jsonl" --concurrency 6 --thinking off \
    --compact --graphs "$G" >> "$LOG" 2>&1
say "2-tajriba tugadi."

# 4) Exp3 — thinking ON (namuna)
say "3-tajriba: E4B thinking=ON (namuna 400)..."
python3 code/repomem/evaluate.py --tasks "$BENCH" --model gemma-4-e4b \
    --out "$R/exp3_e4b_think.jsonl" --concurrency 4 --thinking on --limit 400 >> "$LOG" 2>&1
say "3-tajriba tugadi."

# 5) Hisobot
say "Hisobot yaratilmoqda..."
python3 code/repomem/make_report.py --dir "$R" --out "$R/REPORT.md" >> "$LOG" 2>&1
cp "$R/REPORT.md" "$HOME/gemma4-paper/REPORT.md" 2>/dev/null

# dashboardga sarlavha raqamlarini chiqaramiz
SUMM=$(grep "JAMI:" "$LOG" | tail -3 | head -1)
$P set 7 done "E4B tajribalari tugadi — hisobot: REPORT.md"
say "=== TAJRIBALAR TUGADI. REPORT.md tayyor. ==="
