# RepoMem-OSS

**A distance-controlled code-memory benchmark built entirely from public, permissively licensed Python repositories.**

RepoMem-OSS contains 1,534 exactly scorable questions. Each question asks about one function, which is buried among other functions from the same repository. The question and its answer stay fixed while `distance_tokens`, the amount of distractor code between the question and the needed fact, varies from 0 to ~15K tokens. Nothing here comes from the Gemma 4 Developer Agent competition data. The code graphs were built with [`build_graph.py`](../../code/repomem/build_graph.py) using only the Python `ast` module. Every `calls` edge is resolved from a call expression in the caller's own body, so the edge always points from caller to callee.

| repository | tag | license | nodes | calls edges | items |
|---|---|---|---|---|---|
| pallets/flask | 3.0.3 | BSD-3-Clause | 818 | 174 | 260 |
| pallets/click | 8.1.7 | BSD-3-Clause | 864 | 310 | 260 |
| pallets/werkzeug | 3.0.3 | BSD-3-Clause | 1,725 | 783 | 259 |
| pallets/jinja | 3.1.4 | BSD-3-Clause | 1,409 | 568 | 260 |
| python-attrs/attrs | 23.2.0 | MIT | 801 | 366 | 235 |
| marshmallow-code/marshmallow | 3.21.3 | MIT | 998 | 133 | 260 |

Exact commits are listed in [`MANIFEST.json`](MANIFEST.json).

## Files

- `repomem_oss.jsonl.gz`: the benchmark, one JSON object per line: `{id, repo, kind, k, distance_tokens, n_items, prompt, answer, meta}`.
- `graphs/*.json.gz`: the code graphs: `nodes` (`id`, `name`, `text`) and `edges` (`source`, `target`, `type: "calls"`).
- `MANIFEST.json`: repositories, tags, commits, licenses, and the generation command.
- `LICENSES/`: the license of each source repository. The function source code in the prompts and graphs is © its respective authors and is redistributed under those licenses.

## Probe kinds

- `defn` (341 items): list the parameters of a named function among *N* shown functions.
- `caller` (278 items): name the shown function that calls a named target.
- `chain` (915 items): follow a call chain for *k* = 1–4 steps.

283 of the chain items are single-hop (*k* = 1) and 632 are multi-hop. Multi-hop chains in these repositories are hard even with perfect retrieval, so report the two groups separately (`ext_numbers.py` does this).

## Usage

```bash
gunzip -k repomem_oss.jsonl.gz
python code/repomem/evaluate.py  --tasks repomem_oss.jsonl --model <served-model> --out base.jsonl
python code/repomem/evaluate.py  --tasks repomem_oss.jsonl --model <served-model> --out oracle.jsonl --compact --graphs graphs/
python code/repomem/pfk_agent.py --tasks repomem_oss.jsonl --model <served-model> --out pfk.jsonl --graphs graphs/
```

`--graphs` expects decompressed `graphs/*.json`. To rebuild everything from source:

```bash
python code/repomem/build_graph.py --repo flask --name flask_3.0.3 --out graphs   # one call per repository
python code/repomem/generate.py --graphs graphs --out repomem_oss.jsonl --per-repo 260 --rounds 14 --seed 1
```

## Reference results (Gemma 4 int4 QAT, vLLM, greedy)

Single-hop items (n = 902), by distance (tokens):

| model | 0–512 | 512–2K | 2K–8K | 8K–20K | all |
|---|---|---|---|---|---|
| E2B baseline | 91.7 | 85.9 | 77.0 | 65.0 | 81.5 |
| E2B + PFK-agent | 91.7 | 91.8 | 92.6 | 93.3 | 92.2 |
| E2B oracle | 93.1 | 91.8 | 93.5 | 93.3 | 92.9 |
| E4B baseline | 93.1 | 92.9 | 91.6 | 86.7 | 91.7 |
| E4B + PFK-agent | 91.7 | 92.5 | 94.8 | 98.3 | 93.9 |
| E4B oracle | 93.1 | 92.5 | 95.8 | 98.3 | 94.6 |

On multi-hop chains (n = 632), E4B scores 13.1% as the baseline, 25.0% with PFK-agent and 24.8% with the oracle; E2B scores 9.8%, 26.6% and 26.3%.

License of this compilation: Apache-2.0 (questions and graph structure). The embedded source code remains under its original licenses (see `LICENSES/`).
