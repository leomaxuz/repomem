#!/usr/bin/env python3
"""Hybrid router: answer from the full context when the prompt is short, from the PFK-agent
working-memory buffer when it is long. The routing signal is the PROMPT LENGTH, which an agent
knows before answering (unlike distance_tokens or the gold answer).

The threshold is selected on one benchmark instance (seed 1) and evaluated, frozen, on the main
set and on seeds 2 and 3 -- so no tuning on the reported data.

  python router.py --dir repomem_out
"""
import argparse, json, os

B = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]


def load(p):
    return {(r := json.loads(l))["id"]: r for l in open(p)}


def ptoks(item):
    return len(item["prompt"]) // 4          # same ~4 chars/token estimate as generate.py


def route(ids, tasks, base, pfk, T):
    return {i: (base[i] if ptoks(tasks[i]) <= T else pfk[i]) for i in ids}


def acc(rows):
    return round(100 * sum(r["correct"] for r in rows) / len(rows), 1) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    D = a.dir
    sets = {"main": ("tasks_full.jsonl", "exp1_e4b_full.jsonl", "pfk_e4b.jsonl", "exp2_e4b_compact.jsonl")}
    for s in (1, 2, 3):
        sets[f"seed{s}"] = (f"seeds/bench_s{s}.jsonl", f"seeds/seed_s{s}_base.jsonl",
                            f"seeds/seed_s{s}_pfk.jsonl", f"seeds/seed_s{s}_compact.jsonl")
    data = {}
    for k, (t, b, p, o) in sets.items():
        tp = os.path.join(D, t)
        if not os.path.exists(tp):
            tp = os.path.join(D, "..", "..", "..", t)   # bench files may live elsewhere
        data[k] = (load(tp), load(os.path.join(D, b)), load(os.path.join(D, p)), load(os.path.join(D, o)))

    # 1) select threshold on seed1
    tasks, base, pfk, _ = data["seed1"]
    ids = list(base)
    grid = [256, 512, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    scores = {T: acc(list(route(ids, tasks, base, pfk, T).values())) for T in grid}
    T = max(grid, key=lambda t: (scores[t], -t))
    out = {"selected_on": "seed1", "grid": scores, "T": T, "eval": {}}

    # 2) evaluate frozen T everywhere
    for k, (tasks, base, pfk, orc) in data.items():
        ids = [i for i in base if i in tasks and i in pfk and i in orc]
        r = route(ids, tasks, base, pfk, T)
        row = {"n": len(ids), "baseline": acc([base[i] for i in ids]), "pfk": acc([pfk[i] for i in ids]),
               "router": acc(list(r.values())), "oracle": acc([orc[i] for i in ids]),
               "frac_pfk": round(100 * sum(ptoks(tasks[i]) > T for i in ids) / len(ids), 1),
               "by_bucket": {}}
        for lo, hi in B:
            bi = [i for i in ids if lo <= base[i]["distance_tokens"] < hi]
            row["by_bucket"][f"{lo}-{hi}"] = [acc([base[i] for i in bi]), acc([pfk[i] for i in bi]),
                                             acc([r[i] for i in bi]), acc([orc[i] for i in bi])]
        out["eval"][k] = row
    json.dump(out, open(os.path.join(D, "router.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
