#!/usr/bin/env python3
"""Size-scaling stats on the common subset of items that fit every model's context window.
31B is served at max_model_len=16384 (KV budget on 2x24GB), so items whose prompt exceeds
16K are rejected for it; for a fair curve every model is scored on the same fitting subset."""
import json, statistics as st, sys
R = "code/repomem/repomem_out/"
L = lambda f: {(r := json.loads(l))["id"]: r for l in open(R + f)}
fit = {i for i, r in L("e31b_full.jsonl").items() if not r.get("err")}
B = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000)]
S = {"E2B": ("e2b_full.jsonl", "e2b_compact.jsonl"),
     "E4B": ("exp1_e4b_full.jsonl", "exp2_e4b_compact.jsonl"),
     "12B": ("e12b_full.jsonl", "e12b_compact.jsonl"),
     "31B": ("e31b_full.jsonl", "e31b_compact.jsonl")}
out = {"n_fit": len(fit), "buckets": [f"{a}-{b}" for a, b in B], "models": {}}
acc = lambda rs: round(100 * sum(r["correct"] for r in rs) / max(1, len(rs)), 1)
for m, (fb, fc) in S.items():
    try:
        db, dc = L(fb), L(fc)
    except FileNotFoundError:
        continue
    d = {}
    for tag, dd in (("base", db), ("compact", dc)):
        rs = [dd[i] for i in fit if i in dd]
        d[tag] = [acc(rs)] + [acc([r for r in rs if lo <= r["distance_tokens"] < hi]) for lo, hi in B]
        d[tag + "_n"] = len(rs)
    full = list(db.values())
    d["full_overall"] = acc(full)
    d["full_20k"] = acc([r for r in full if r["distance_tokens"] >= 20000])
    d["full_errs"] = sum(1 for r in full if r.get("err"))
    secs = [r["sec"] for r in full if not r.get("err") and r.get("sec") is not None]
    d["sec"] = round(st.mean(secs), 2) if secs else None
    out["models"][m] = d
json.dump(out, open(R + "scaling_subset.json", "w"), indent=1)
print(json.dumps(out, indent=1))
