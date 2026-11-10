#!/usr/bin/env python3
"""Tajriba natijalaridan (exp*.jsonl) markdown hisobot yasaydi.

  python make_report.py --dir repomem_out --out repomem_out/REPORT.md
"""
import argparse
import json
import os
import time

BUCKETS = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
KINDS = ("defn", "chain", "caller")


def load(path):
    if not path or not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def acc(sub):
    sub = [x for x in sub if x.get("err") != "compact-skip"]
    return (sum(x["correct"] for x in sub) / len(sub)) if sub else 0.0


def n_real(sub):
    return len([x for x in sub if x.get("err") != "compact-skip"])


def blabel(lo, hi):
    return "%d-%s" % (lo, hi if hi < 10**9 else "∞")


def by_bucket(rows):
    out = []
    for lo, hi in BUCKETS:
        sub = [r for r in rows if lo <= r["distance_tokens"] < hi]
        if n_real(sub):
            out.append((blabel(lo, hi), n_real(sub), acc(sub)))
    return out


def by_kind(rows):
    out = []
    for k in KINDS:
        sub = [r for r in rows if r["kind"] == k]
        if n_real(sub):
            out.append((k, n_real(sub), acc(sub)))
    return out


def pct(x):
    return "%.1f%%" % (100 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = a.dir
    base = load(os.path.join(d, "exp1_e4b_full.jsonl"))
    comp = load(os.path.join(d, "exp2_e4b_compact.jsonl"))
    think = load(os.path.join(d, "exp3_e4b_think.jsonl"))

    L = []
    W = L.append
    W("# RepoMem — Gemma 4 E4B ishchi-xotira tajribalari")
    W("")
    W("_Avtomatik hisobot · %s UTC_" % time.strftime("%Y-%m-%d %H:%M"))
    W("")
    W("**Model:** gemma-4-e4b-it-qat-w4a16-ct (int4, vLLM 0.33.0, 2×RTX 4090, max_model_len=32768)  ")
    W("**Benchmark:** RepoMem — fastapi/rich/requests/httpx kod graflaridan avtomatik yasalgan "
      "bir navbatli savollar (defn/chain/caller). Asosiy o'zgaruvchi: `distance_tokens` — "
      "savol bilan kerakli ma'lumot orasidagi distraktor kod miqdori.")
    W("")
    W("## Qisqa xulosa")
    if base and comp:
        b_far = acc([r for r in base if r["distance_tokens"] >= 20000])
        c_far = acc([r for r in comp if r["distance_tokens"] >= 20000])
        b_all, c_all = acc(base), acc(comp)
        W("- **To'liq kontekst (baseline):** umumiy aniqlik **%s**." % pct(b_all))
        W("- **Compact / ishchi-xotira (faqat kerakli funksiya):** umumiy aniqlik **%s**." % pct(c_all))
        W("- **Asosiy natija:** uzoq masofada (20K+ token) aniqlik **%s → %s** "
          "(ishchi-xotira aralashuvi bilan). Bilim modelда bor; muammo — uni uzoq "
          "kontekstдан _ushlab turish_. Bu PFK-agent (ishchi-xotira uyachalari) g'oyasini asoslaydi."
          % (pct(b_far), pct(c_far)))
    W("")

    def table(title, rows, extra=""):
        W("### " + title + extra)
        W("")
        W("Umumiy: **%s** (n=%d, xatolar=%d)" %
          (pct(acc(rows)), n_real(rows), len([r for r in rows if r.get("err") and r["err"] != "compact-skip"])))
        W("")
        W("| Masofa (token) | n | Aniqlik |")
        W("|---|---|---|")
        for lab, n, ac in by_bucket(rows):
            W("| %s | %d | %s |" % (lab, n, pct(ac)))
        W("")
        W("| Savol turi | n | Aniqlik |")
        W("|---|---|---|")
        for k, n, ac in by_kind(rows):
            W("| %s | %d | %s |" % (k, n, pct(ac)))
        W("")

    if base:
        table("1-tajriba — To'liq kontekst (baseline)", base)
    if comp:
        table("2-tajriba — Compact / ishchi-xotira", comp)

    if base and comp:
        W("### Solishtirish — baseline → compact (masofa bo'yicha)")
        W("")
        W("| Masofa (token) | Baseline | Compact | Farq |")
        W("|---|---|---|---|")
        for lo, hi in BUCKETS:
            bs = [r for r in base if lo <= r["distance_tokens"] < hi]
            cs = [r for r in comp if lo <= r["distance_tokens"] < hi]
            if n_real(bs) and n_real(cs):
                ba, ca = acc(bs), acc(cs)
                W("| %s | %s | %s | +%.1f pp |" % (blabel(lo, hi), pct(ba), pct(ca), 100 * (ca - ba)))
        W("")
        W("_pp = foiz punkti. Compact rejim masofaga deyarli bog'liq emas — bu muammoning "
          "ishchi-xotira (sliding window) ekanini ko'rsatadi._")
        W("")

    if think:
        table("3-tajriba — To'liq kontekst + thinking ON", think, " (namuna)")
        if base:
            W("_Taqqoslash: baseline (thinking OFF) umumiy %s, thinking ON umumiy %s._"
              % (pct(acc(base)), pct(acc(think))))
            W("")

    W("## Fayllar")
    W("- `exp1_e4b_full.jsonl` — baseline natijalar (har savol: pred/correct/sec)")
    W("- `exp2_e4b_compact.jsonl` — compact natijalar")
    if think:
        W("- `exp3_e4b_think.jsonl` — thinking ON natijalar")
    W("- `tasks_full.jsonl` — benchmark savollari")
    W("")
    W("## Keyingi qadamlar")
    W("- E2B (kichikroq model) bilan bir xil benchmarkда size-scaling solishtirish")
    W("- PFK-agent: get_code_neighbors/get_code_subgraph orqali ignalarni ishchi-xotira "
      "buferiga olib, uzoq-masofa pasayishini avtomatik yo'qotish")
    W("- 129 vazifali agent track baseline va trace tahlili")

    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("Hisobot yozildi:", a.out, "(%d qator)" % len(L))


if __name__ == "__main__":
    main()
