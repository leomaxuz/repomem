#!/usr/bin/env python3
"""Bir nechta natija JSONL faylini birlashtirib taqqoslash hisobotini (markdown) yasaydi.
  python summarize.py --out REPORT2.md \
     --add "E4B baseline:exp1_e4b_full.jsonl" --add "E4B compact:exp2_e4b_compact.jsonl" \
     --add "E4B PFK-agent:pfk_e4b.jsonl" --add "E2B baseline:e2b_full.jsonl" ...
Har JSONL satri: {kind, distance_tokens, correct, ...}.
"""
import argparse
import json

BUCKETS = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
KINDS = ["defn", "chain", "caller"]


def load(path):
    try:
        return [json.loads(l) for l in open(path)]
    except Exception:
        return []


def acc(rows):
    return (100.0 * sum(bool(r.get("correct")) for r in rows) / len(rows)) if rows else None


def cell(x):
    return f"{x:.1f}%" if x is not None else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", action="append", default=[], help="LABEL:path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="RepoMem — to'liq taqqoslash")
    a = ap.parse_args()

    series = []
    for spec in a.add:
        label, path = spec.split(":", 1)
        rows = load(path)
        if rows:
            series.append((label, rows))

    L = [s[0] for s in series]
    out = [f"# {a.title}", "", "_Avtomatik taqqoslash hisoboti · barcha natijalar bir benchmarkda_", ""]

    # umumiy
    out += ["## Umumiy aniqlik", "", "| Tajriba | n | Aniqlik |", "|---|---|---|"]
    for label, rows in series:
        out.append(f"| {label} | {len(rows)} | {cell(acc(rows))} |")
    out.append("")

    # masofa bo'yicha
    out += ["## Masofa bo'yicha (asl distance_tokens)", "",
            "| Masofa (token) | " + " | ".join(L) + " |",
            "|---" * (len(L) + 1) + "|"]
    for lo, hi in BUCKETS:
        lbl = f"{lo}-{hi if hi < 10**9 else 'inf'}"
        cells = []
        for _, rows in series:
            sub = [r for r in rows if lo <= r.get("distance_tokens", 0) < hi]
            cells.append(cell(acc(sub)))
        out.append(f"| {lbl} | " + " | ".join(cells) + " |")
    out.append("")

    # tur bo'yicha
    out += ["## Savol turi bo'yicha", "",
            "| Tur | " + " | ".join(L) + " |",
            "|---" * (len(L) + 1) + "|"]
    for k in KINDS:
        cells = []
        for _, rows in series:
            sub = [r for r in rows if r.get("kind") == k]
            cells.append(cell(acc(sub)))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out.append("")

    open(a.out, "w").write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
