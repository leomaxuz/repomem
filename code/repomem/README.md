# RepoMem

**A distance-controlled benchmark for the working memory of code language models.**

RepoMem measures a single thing precisely: *can a model still use a fact about a code repository once that fact is buried under thousands of tokens of other code?* It holds the question and the answer fixed and sweeps only the amount of distractor code between them (`distance_tokens`), so a model's accuracy-vs-distance curve isolates its working-memory span from its knowledge and reasoning.

It is built automatically from repository **code graphs** (nodes = function/class source, `calls` edges), so it can be regenerated for any repository that has such a graph — nothing is hand-written.

---

## Why RepoMem

Small and quantized models (e.g. Gemma 4 E2B/E4B) use short sliding-window attention. On a real repository the fact needed to answer a question often sits far from the question, and these models silently fail there — not because they lack the knowledge, but because they cannot hold it. Generic needle-in-a-haystack tests use a random sentence in prose; RepoMem makes the needle a **function**, the distractors **other plausible functions from the same project**, and the task a **structural relationship** (signature, call chain, caller). And it makes distance an explicit, swept variable rather than a fixed condition.

## Probes

| kind | question | answer |
|---|---|---|
| `defn` | Among *N* shown functions, list the parameters of function `X`. | parameter set (order-insensitive, `self`/`cls` excluded) |
| `chain` | Starting from `A`, follow the call chain exactly *k* steps. Which function is reached? | function name |
| `caller` | Which shown function calls `X`? | function name |

Every item is **well-posed by construction** (unique, graph-grounded answer): chains use only steps with a single in-graph callee; callers use only targets with a single in-graph caller and a call that appears *textually* in the caller's source; parameters are extracted with the Python AST; and the exact question string is stored so alternative contexts ask the identical question.

## Data format

One JSONL record per item:

```json
{"id": "...", "repo": "fastapi_<sha>", "kind": "defn|chain|caller",
 "k": 0, "distance_tokens": 8829, "n_items": 64,
 "prompt": "```python ...```\n\n<question>", "answer": "session, offset, limit",
 "meta": {"target": "...", "question": "...", "...": "..."}}
```

`distance_tokens` is the key independent variable: the distractor tokens between the question and the information needed to answer it.

## Usage

**Generate** (from a directory of `<repo>.json` code graphs):

```bash
python generate.py --graphs <graphs_dir> --out tasks.jsonl --per-repo 20 --seed 1
```

**Evaluate** any OpenAI-compatible endpoint (e.g. a vLLM-served Gemma 4):

```bash
# baseline (full context)
python evaluate.py --tasks tasks.jsonl --base-url http://127.0.0.1:8000/v1 \
    --model gemma-4-e4b --out results.jsonl --concurrency 6 --thinking off

# compact oracle (upper bound: only the needed function shown, same question)
python evaluate.py --tasks tasks.jsonl --model gemma-4-e4b --out oracle.jsonl \
    --compact --graphs <graphs_dir>
```

**Report** accuracy by distance bucket and question kind:

```bash
python summarize.py --out REPORT.md --add "baseline:results.jsonl" --add "oracle:oracle.jsonl"
```

Scoring is deterministic and kind-specific (parameter-set match for `defn`, function-name match for `chain`/`caller`) — **no LLM-as-judge**, fully reproducible.

## Files

| file | purpose |
|---|---|
| `generate.py` | benchmark generator (stdlib only) |
| `evaluate.py` | evaluator; baseline, `--compact` oracle, `--thinking` |
| `pfk_agent.py` | **PFK-agent** reference: graph-retrieval working-memory agent |
| `summarize.py` | combined multi-condition report |
| `aggregate_seeds.py` | multi-seed mean ± std |
| `analysis_extra.py` | position (lost-in-the-middle), latency, qualitative examples |
| `scaling_stats.py` | size scaling on the common subset that fits every model's window |
| `plot_figures_v2.py` | paper figures (from `experiments/paper_numbers.json`) |
| `recompute_all.py` | every number in the paper, on the well-posed item set |
| `router.py` | prompt-length router ablation |

## Reference result (Gemma 4, int4 QAT, vLLM, 2×RTX 4090)

Results on the 1,554 well-posed items show two regimes.

**Within the 32K window**, baseline accuracy decays with distance while the compact oracle stays flat. E2B falls from 90.9% (0–512 tok) to 71.7% (8–20K), 20.7 points below its oracle. E4B loses up to 7.2 points to its oracle. The overall gap is 9.3 points for E2B and 2.9 / 4.7 / 0.7 for E4B / 12B / 31B.

**Beyond the window** (2.7% of items, prompts > 32K), the baseline cannot run, while PFK-agent answers 81% (oracle 83%).

Overall, PFK-agent lifts E4B from 85.7% to 89.1% (oracle 90.8%) at ~10× lower latency, and E2B from 79.8% to 91.3% (oracle 91.4%). Results are replicated on three regenerated instances. `recompute_all.py` produces every number in the paper; see the paper for the full tables.

`generate.py` drops `chain` items whose call steps are not visible in the source. Some competition-graph `calls` edges point callee → caller, which makes such items unanswerable. `--no-textual-filter` restores the raw 1,651-item v1 set.

Note: requests rejected for exceeding the serving window are recorded with an `err` field. Report them separately from in-window errors, since they are not memory failures of the model.

## License & citation

Code released under the Apache License 2.0 (see `LICENSE`). The competition code graphs, and the benchmark items generated from them, are Competition Data and are not redistributed here: regenerate them with `generate.py` from the competition data (or from any repository code graph in the same format). If you use RepoMem, please cite the accompanying paper (see `../../PAPER.md`).
