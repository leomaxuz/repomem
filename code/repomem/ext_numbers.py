#!/usr/bin/env python3
"""Numbers for RepoMem-OSS (benchmark built with build_graph.py from public repositories).

  python ext_numbers.py --dir ext/out --tasks ext/tasks_ext.jsonl --out ext_numbers.json
"""
import argparse, json, os, statistics as st

B = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000)]


def load(p):
    return {(r := json.loads(l))["id"]: r for l in open(p)}


def acc(d, ids):
    ids = list(ids)
    return round(100 * sum(bool(d[i]["correct"]) for i in ids) / len(ids), 1) if ids else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = load(a.tasks)
    ids = list(T)
    dist = {i: T[i]["distance_tokens"] for i in ids}
    repos = sorted({T[i]["repo"] for i in ids})
    out = {"n": len(ids), "buckets_n": [sum(lo <= dist[i] < hi for i in ids) for lo, hi in B],
           "kinds": {k: sum(T[i]["kind"] == k for i in ids) for k in ("defn", "chain", "caller")},
           "repos": {r: sum(T[i]["repo"] == r for i in ids) for r in repos}, "models": {}}
    # single-hop items (defn, caller, 1-step chain) measure retrieval; multi-step chains add
    # multi-hop reasoning, which caps even the oracle on these repositories
    hop = {"single": [i for i in ids if T[i]["kind"] != "chain" or T[i]["k"] == 1],
           "multi": [i for i in ids if T[i]["kind"] == "chain" and T[i]["k"] >= 2]}
    out["hop_n"] = {h: len(v) for h, v in hop.items()}
    out["hop_buckets_n"] = {h: [sum(lo <= dist[i] < hi for i in v) for lo, hi in B] for h, v in hop.items()}
    for m in ("e4b", "e2b"):
        for c in ("full", "compact", "pfk"):
            p = os.path.join(a.dir, f"{m}_{c}.jsonl")
            if not os.path.exists(p):
                continue
            d = load(p)
            ok = [i for i in ids if i in d]
            secs = [d[i]["sec"] for i in ok if d[i].get("sec") and not d[i].get("err")]
            out["models"][f"{m}_{c}"] = {
                "n": len(ok), "errors": sum(1 for i in ok if d[i].get("err")),
                "all": acc(d, ok), "buckets": [acc(d, [i for i in ok if lo <= dist[i] < hi]) for lo, hi in B],
                "kinds": {k: acc(d, [i for i in ok if T[i]["kind"] == k]) for k in ("defn", "chain", "caller")},
                "chain_k": {k: acc(d, [i for i in ok if T[i]["kind"] == "chain" and T[i]["k"] == k]) for k in (1, 2, 3, 4)},
                "hop": {h: {"all": acc(d, [i for i in v if i in d]),
                            "buckets": [acc(d, [i for i in v if i in d and lo <= dist[i] < hi]) for lo, hi in B]}
                        for h, v in hop.items()},
                "repos": {r: acc(d, [i for i in ok if T[i]["repo"] == r]) for r in repos},
                "sec": round(st.mean(secs), 2) if secs else None}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
