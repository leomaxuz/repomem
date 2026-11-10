#!/usr/bin/env python3
"""Bir nechta urug' (seed) natijalaridan mean +/- std hisoblaydi (umumiy va masofa bo'yicha)."""
import argparse
import json
import math
import os

BUCKETS = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]


def load(path):
    try:
        return [json.loads(l) for l in open(path)]
    except Exception:
        return []


def acc(rows):
    return (100.0 * sum(bool(r.get("correct")) for r in rows) / len(rows)) if rows else None


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if len(xs) > 1 else 0.0
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(",")]
    conds = [("baseline", "base"), ("compact (oracle)", "compact"), ("PFK-agent", "pfk")]

    out = ["# RepoMem — multi-seed robustness (E4B)", "",
           f"_Seeds: {', '.join(seeds)} · mean ± std across seeds_", ""]

    # umumiy
    out += ["## Overall accuracy", "", "| Condition | mean ± std |", "|---|---|"]
    cond_rows = {}
    for label, tag in conds:
        per_seed = []
        rows_all = {}
        for s in seeds:
            r = load(os.path.join(a.dir, f"seed_s{s}_{tag}.jsonl"))
            rows_all[s] = r
            per_seed.append(acc(r))
        cond_rows[tag] = rows_all
        m, sd = mean_std(per_seed)
        vals = ", ".join(f"{x:.1f}" for x in per_seed if x is not None)
        out.append(f"| {label} | **{m:.1f} ± {sd:.1f}%**  ({vals}) |" if m is not None else f"| {label} | — |")
    out.append("")

    # masofa bo'yicha
    out += ["## By distance (mean ± std across seeds)", "",
            "| Distance (tokens) | " + " | ".join(l for l, _ in conds) + " |",
            "|---" * (len(conds) + 1) + "|"]
    for lo, hi in BUCKETS:
        lbl = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        cells = []
        for _, tag in conds:
            per_seed = []
            for s in seeds:
                sub = [r for r in cond_rows[tag][s] if lo <= r.get("distance_tokens", 0) < hi]
                per_seed.append(acc(sub))
            m, sd = mean_std(per_seed)
            cells.append(f"{m:.1f} ± {sd:.1f}" if m is not None else "—")
        out.append(f"| {lbl} | " + " | ".join(cells) + " |")
    out.append("")

    open(a.out, "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
