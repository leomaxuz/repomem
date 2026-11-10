#!/usr/bin/env bash
# vLLM'ni uv (tez resolver) bilan o'rnatadi. Detached; log: logs/vllm_setup.log
set -uo pipefail
cd "$HOME/gemma4-paper" || exit 1
P="python3 bin/stage.py"

# tiqilib qolgan eski pip'ni to'xtatamiz
pkill -f "pip install vllm" 2>/dev/null; sleep 1

$P set 2 running "vLLM o'rnatilmoqda (uv, tez resolver)"
if [ ! -x vllm-venv/bin/python ]; then python3 -m venv vllm-venv; fi
./vllm-venv/bin/pip -q install --upgrade pip uv >/dev/null 2>&1
UV="./vllm-venv/bin/uv"

echo "=== uv pip install vllm boshlandi $(date -u +%H:%M:%S) ==="
# uv torch'ni avtomatik to'g'ri backend bilan oladi; tiqilib qolmaydi
$UV pip install --python ./vllm-venv/bin/python vllm 2>&1 | tail -25
echo "=== install tugadi $(date -u +%H:%M:%S) ==="

V=$(./vllm-venv/bin/python -c "import vllm;print(vllm.__version__)" 2>&1 | tail -1)
echo "vLLM versiya: $V"
echo "=== gemma4 arch tekshiruvi ==="
./vllm-venv/bin/python -c "
from vllm.model_executor.models.registry import ModelRegistry
a=ModelRegistry.get_supported_archs()
print('gemma4-bor:', [x for x in a if 'emma4' in x.lower() or 'Gemma4' in x])
print('barcha gemma:', [x for x in a if 'emma' in x.lower()])
" 2>&1 | tail -5
if [ -x vllm-venv/bin/vllm ]; then
  $P log "vLLM o'rnatildi: $V"
  $P set 2 running "vLLM o'rnatildi ($V) — E4B serveri ishga tushiriladi"
else
  $P set 2 failed "vLLM o'rnatilmadi (uv)"
fi
