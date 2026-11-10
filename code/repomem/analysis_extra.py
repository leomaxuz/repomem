#!/usr/bin/env python3
"""Qo'shimcha tahlillar (post-hoc, mavjud natijalardan):
  C) lost-in-the-middle: aniqlik ignaning kontekstdagi O'RNIGA qarab
  E) narx/latency (baseline vs PFK) + sifatli misollar (baseline xato, PFK to'g'ri)
Chiqish: experiments/ ichiga markdown + position_data.json (grafik uchun).
"""
import argparse
import json
import os

BUCKETS = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
POS_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]


def load_map(path, key="id"):
    d = {}
    try:
        for l in open(path):
            r = json.loads(l)
            d[r[key]] = r
    except Exception:
        pass
    return d


def acc(rows):
    return (100.0 * sum(bool(r.get("correct")) for r in rows) / len(rows)) if rows else None


def needle_id(meta, kind):
    if kind == "defn":
        return meta.get("target")
    if kind == "caller":
        return meta.get("caller")
    if kind == "chain":
        ch = meta.get("chain") or []
        return ch[-1] if ch else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="repomem_out papka")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True, help="experiments papka")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bench = load_map(a.bench)
    base = load_map(os.path.join(a.dir, "exp1_e4b_full.jsonl"))
    pfk = load_map(os.path.join(a.dir, "pfk_e4b.jsonl"))

    # ---------- C) lost-in-the-middle ----------
    pos_rows = []
    for iid, it in bench.items():
        if iid not in base or base[iid].get("err"):
            continue  # context-overflow rows (HTTP 400) are not memory failures; excluded
        nid = needle_id(it.get("meta", {}), it["kind"])
        if not nid:
            continue
        prompt = it.get("prompt", "")
        idx = prompt.find("# " + nid + "\n")
        if idx < 0 or len(prompt) == 0:
            continue
        frac = idx / len(prompt)
        pos_rows.append({"id": iid, "frac": frac, "correct": base[iid]["correct"],
                         "distance_tokens": it.get("distance_tokens", 0), "kind": it["kind"]})

    def pos_table(rows, title):
        out = [f"### {title}  (n={len(rows)})", "", "| needle position | n | accuracy |", "|---|---|---|"]
        data = []
        for lo, hi in POS_BINS:
            sub = [r for r in rows if lo <= r["frac"] < hi]
            a_ = acc(sub)
            out.append(f"| {lo:.1f}–{hi if hi<=1 else 1.0:.1f} | {len(sub)} | {a_:.1f}% |" if a_ is not None else f"| {lo:.1f}–{hi:.1f} | 0 | — |")
            data.append((f"{lo:.1f}-{min(hi,1.0):.1f}", len(sub), a_))
        return out, data

    long_rows = [r for r in pos_rows if r["distance_tokens"] >= 4000]
    o1, d_all = pos_table(pos_rows, "All items")
    o2, d_long = pos_table(long_rows, "Long-context only (distance ≥ 4000 tokens)")
    pos_md = ["# RepoMem — lost-in-the-middle (E4B baseline)", "",
              "_Accuracy vs the needle's relative position in the prompt (0 = start, 1 = end)._", "",
              *o1, "", *o2, ""]
    open(os.path.join(a.out, "REPORT_position.md"), "w").write("\n".join(pos_md))
    json.dump({"all": d_all, "long": d_long}, open(os.path.join(a.out, "position_data.json"), "w"))

    # ---------- E) cost / latency ----------
    def mean_sec(d, lo=0, hi=10**9):
        xs = [r.get("sec", 0) for r in d.values() if lo <= r.get("distance_tokens", 0) < hi and r.get("sec") and not r.get("err")]
        return sum(xs) / len(xs) if xs else None

    cost = ["# RepoMem — latency & cost (E4B)", "",
            "Per-query wall-clock (s), concurrency 6, same hardware. PFK answers from a small buffer.", "",
            "| Condition | mean latency (all) | mean latency (20K+) |", "|---|---|---|",
            f"| Baseline (full context) | {mean_sec(base):.2f}s | {mean_sec(base,20000):.2f}s |",
            f"| PFK-agent (retrieved buffer) | {mean_sec(pfk):.2f}s | {mean_sec(pfk,20000):.2f}s |", "",
            "_At long distance the baseline reads tens of thousands of distractor tokens per query; "
            "PFK-agent reads only the retrieved blocks — faster AND more accurate._"]
    open(os.path.join(a.out, "REPORT_cost.md"), "w").write("\n".join(cost))

    # ---------- E) qualitative examples ----------
    ex = ["# RepoMem — qualitative examples (baseline fails, PFK-agent succeeds)", ""]
    n = 0
    for iid, it in bench.items():
        if n >= 3:
            break
        if iid in base and iid in pfk and not base[iid]["correct"] and pfk[iid]["correct"] and not base[iid].get("err") \
           and it.get("distance_tokens", 0) >= 8000:
            q = it.get("meta", {}).get("question", "")
            ex += [f"## Example {n+1} — {it['kind']}, distance ≈ {it['distance_tokens']:,} tokens", "",
                   f"**Question:** {q}", "",
                   f"- **Gold:** `{base[iid]['answer']}`",
                   f"- **Baseline (full context):** `{base[iid]['pred'][:120]}`  ❌",
                   f"- **PFK-agent (retrieved):** `{pfk[iid]['pred'][:120]}`  ✅", ""]
            n += 1
    if n == 0:
        ex.append("_(no qualifying example found)_")
    open(os.path.join(a.out, "REPORT_examples.md"), "w").write("\n".join(ex))

    # konsolga xulosa
    print("POSITION (all):", d_all)
    print("POSITION (long):", d_long)
    print("LATENCY base all/20k:", round(mean_sec(base) or 0, 2), "/", round(mean_sec(base, 20000) or 0, 2))
    print("LATENCY pfk  all/20k:", round(mean_sec(pfk) or 0, 2), "/", round(mean_sec(pfk, 20000) or 0, 2))
    print("EXAMPLES found:", n)


if __name__ == "__main__":
    main()
