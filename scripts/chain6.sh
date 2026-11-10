#!/usr/bin/env bash
cd ~/gemma4-paper; R=code/repomem/repomem_out; G=data/extracted/graphs; B=$R/tasks_full.jsonl; L=logs/chain6.log
for i in $(seq 1 60); do python3 -c "import urllib.request;urllib.request.urlopen(\"http://127.0.0.1:8000/v1/models\",timeout=5)" 2>/dev/null && break; sleep 10; done
echo "[$(date -u +%T)] start" >> $L
python3 code/repomem/pfk_agent.py --tasks $B --graphs $G --model gemma-4-e4b --out $R/pfk_e4b_entmodel_v.jsonl --entity-mode model --verify-entity --fallback-full >> $L 2>&1
python3 code/repomem/pfk_agent.py --tasks $B --graphs $G --model gemma-4-e4b --out $R/pfk_e4b_entshort_v.jsonl --entity-mode model-short --verify-entity --fallback-full >> $L 2>&1
echo "[$(date -u +%T)] 6-ZANJIR TUGADI" >> $L
