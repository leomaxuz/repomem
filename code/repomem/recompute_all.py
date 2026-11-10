#!/usr/bin/env python3
"""Recompute every number in the paper from the per-item outputs, on the well-posed item set.

Well-posedness filter (same rule as generate.py): every `chain` step A -> B must appear as a call
`B(` in A's source. Items failing it are dropped from every condition. Requests rejected for
exceeding the serving window carry an `err` field and are reported separately ("beyond window").

  python recompute_all.py --dir repomem_out --graphs <graphs_dir> --out paper_numbers.json
"""
import argparse, json, os, statistics as st

B = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
BL = ["0-512", "512-2K", "2K-8K", "8K-20K", "20K+"]


def load(p):
    return {(r := json.loads(l))["id"]: r for l in open(p)}


def acc(d, ids):
    ids = list(ids)
    return round(100 * sum(bool(d[i]["correct"]) for i in ids) / len(ids), 1) if ids else None


def sd(xs):
    return round(st.stdev(xs), 1) if len(xs) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = a.dir
    P = lambda f: os.path.join(D, f)
    texts = {}

    def node_text(repo):
        if repo not in texts:
            g = json.load(open(os.path.join(a.graphs, repo + ".json")))
            texts[repo] = {n["id"]: n.get("text") or "" for n in g["nodes"]}
        return texts[repo]

    def well_posed(it):
        if it["kind"] != "chain":
            return True
        t = node_text(it["repo"]); ch = it["meta"]["chain"]
        return all(b.split(".")[-1] + "(" in t.get(x, "") for x, b in zip(ch, ch[1:]))

    out = {}
    tasks = load(P("tasks_full.jsonl"))
    V = [i for i, it in tasks.items() if well_posed(it)]
    Vs = set(V)
    base = load(P("exp1_e4b_full.jsonl"))
    ov = [i for i in V if base[i].get("err")]
    fit = [i for i in V if not base[i].get("err")]
    dist = {i: tasks[i]["distance_tokens"] for i in V}
    kinds = {}
    for i in V:
        kinds[tasks[i]["kind"]] = kinds.get(tasks[i]["kind"], 0) + 1
    out["composition"] = {
        "n_raw": len(tasks), "n": len(V), "dropped": len(tasks) - len(V), "kinds": kinds,
        "buckets": {l: sum(lo <= dist[i] < hi for i in V) for l, (lo, hi) in zip(BL, B)},
        "beyond_window": len(ov),
        "beyond_kinds": {k: sum(tasks[i]["kind"] == k for i in ov) for k in ("defn", "chain", "caller")},
        "beyond_in_20k": sum(dist[i] >= 20000 for i in ov),
        "in_window_buckets": {l: sum(lo <= dist[i] < hi for i in fit) for l, (lo, hi) in zip(BL, B)},
    }

    conds = {"E4B base": "exp1_e4b_full.jsonl", "E4B oracle": "exp2_e4b_compact.jsonl", "E4B PFK": "pfk_e4b.jsonl",
             "E2B base": "e2b_full.jsonl", "E2B oracle": "e2b_compact.jsonl", "E2B PFK": "pfk_e2b.jsonl",
             "12B base": "e12b_full.jsonl", "12B oracle": "e12b_compact.jsonl",
             "31B base": "e31b_full.jsonl", "31B oracle": "e31b_compact.jsonl"}
    R = {k: load(P(f)) for k, f in conds.items()}
    main = {}
    for k, d in R.items():
        main[k] = {"all": acc(d, V), "in_window": acc(d, fit), "beyond": acc(d, ov),
                   "buckets": [acc(d, [i for i in fit if lo <= dist[i] < hi]) for lo, hi in B],
                   "kinds": {kk: acc(d, [i for i in V if tasks[i]["kind"] == kk]) for kk in ("defn", "chain", "caller")}}
    out["main"] = main
    b, p, o = main["E4B base"], main["E4B PFK"], main["E4B oracle"]
    out["headroom_e4b"] = {"all": round(100 * (p["all"] - b["all"]) / (o["all"] - b["all"]), 0),
                           "in_window": round(100 * (p["in_window"] - b["in_window"]) / (o["in_window"] - b["in_window"]), 0)}

    # scaling: common subset that fits the 16K window of 31B
    m31 = R["31B base"]
    sub = [i for i in V if not m31[i].get("err")]
    sc = {"n": len(sub)}
    for m in ("E2B", "E4B", "12B", "31B"):
        bb, oo = R[f"{m} base"], R[f"{m} oracle"]
        bk = lambda d: [acc(d, [i for i in sub if lo <= dist[i] < hi]) for lo, hi in B[:4]]
        secs = [bb[i]["sec"] for i in V if not bb[i].get("err") and bb[i].get("sec")]
        sc[m] = {"base": acc(bb, sub), "oracle": acc(oo, sub), "base_b": bk(bb), "oracle_b": bk(oo),
                 "sec": round(st.mean(secs), 2)}
        sc[m]["gap_all"] = round(sc[m]["oracle"] - sc[m]["base"], 1)
        sc[m]["gap_2k8k"] = round(sc[m]["oracle_b"][2] - sc[m]["base_b"][2], 1)
        sc[m]["gap_8k20k"] = round(sc[m]["oracle_b"][3] - sc[m]["base_b"][3], 1)
    for m in ("E2B", "E4B"):
        d = R[f"{m} PFK"]
        sc[f"{m} PFK"] = {"all": acc(d, sub), "b": [acc(d, [i for i in sub if lo <= dist[i] < hi]) for lo, hi in B[:4]],
                          "sec": round(st.mean(d[i]["sec"] for i in V), 2)}
    out["scaling"] = sc

    # thinking
    th = load(P("exp3_e4b_think.jsonl"))
    tv = [i for i in th if i in Vs]
    tf = [i for i in tv if not th[i].get("err") and not base[i].get("err")]
    out["thinking"] = {"n": len(tv), "all": acc(th, tv), "base_same_all": acc(base, tv),
                       "n_fit": len(tf), "fit": acc(th, tf), "base_same_fit": acc(base, tf)}

    # position (in-window, distance >= 4K)
    bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0001)]
    def needle(it):
        m = it["meta"]
        return m.get("target") if it["kind"] == "defn" else (m.get("caller") if it["kind"] == "caller" else m["chain"][-1])
    pos = []
    for i in fit:
        it = tasks[i]; n = needle(it); idx = it["prompt"].find("# " + n + "\n")
        if idx >= 0 and dist[i] >= 4000:
            pos.append((idx / len(it["prompt"]), i))
    out["position_long"] = {"n": len(pos), "bins": [acc(base, [i for f, i in pos if lo <= f < hi]) for lo, hi in bins]}

    # latency (non-err)
    lat = lambda d, ids: round(st.mean(d[i]["sec"] for i in ids if not d[i].get("err") and d[i].get("sec")), 2)
    out["latency"] = {"E4B base": lat(base, V), "E4B base 20K+": lat(base, [i for i in V if dist[i] >= 20000]),
                      "E4B PFK": lat(R["E4B PFK"], V), "E2B PFK": lat(R["E2B PFK"], V),
                      "31B base": lat(R["31B base"], V)}

    # ablations
    abl = {}
    for f in ("pfk_e4b", "pfk_e2b", "pfk_e4b_entmodel", "pfk_e4b_entmodel_v", "pfk_e4b_entshort", "pfk_e4b_entshort_v",
              "pfk_e4b_drop10", "pfk_e4b_drop30", "pfk_e4b_drop30_fb"):
        d = load(P(f + ".jsonl"))
        abl[f] = {"all": acc(d, V), "in_window": acc(d, fit), "beyond": acc(d, ov),
                  "retrieval_ok": round(100 * sum(bool(d[i].get("retrieved_ok")) for i in V) / len(V), 1),
                  "fallback": round(100 * sum(bool(d[i].get("fallback")) for i in V) / len(V), 1),
                  "entity_ok": round(100 * sum(bool(d[i].get("entity_ok")) for i in V) / len(V), 1)}
    out["ablations"] = abl

    # seeds
    seeds = {"base": [], "pfk": [], "oracle": []}
    per = []
    for s in (1, 2, 3):
        t = load(P(f"seeds/bench_s{s}.jsonl"))
        vs = [i for i, it in t.items() if well_posed(it)]
        bb = load(P(f"seeds/seed_s{s}_base.jsonl")); pp = load(P(f"seeds/seed_s{s}_pfk.jsonl")); oo = load(P(f"seeds/seed_s{s}_compact.jsonl"))
        o_ = [i for i in vs if bb[i].get("err")]; f_ = [i for i in vs if not bb[i].get("err")]
        row = {"n": len(vs), "n_beyond": len(o_)}
        for tag, d in (("base", bb), ("pfk", pp), ("oracle", oo)):
            row[tag] = [acc(d, [i for i in f_ if lo <= t[i]["distance_tokens"] < hi]) for lo, hi in B[:4]] + \
                       [acc(d, o_), acc(d, vs)]
        per.append(row)
    agg = {"n": [r["n"] for r in per], "n_beyond": [r["n_beyond"] for r in per]}
    for tag in ("base", "pfk", "oracle"):
        cols = list(zip(*[r[tag] for r in per]))
        agg[tag] = [[round(st.mean(c), 1), sd(c)] for c in cols]   # 4 buckets, beyond, all
    out["seeds"] = {"per_seed": per, "agg": agg}

    # router: threshold picked on seed 1 (well-posed), frozen elsewhere
    def route_eval(t, bb, pp, oo, ids, T):
        r = {i: (bb[i] if len(t[i]["prompt"]) // 4 <= T else pp[i]) for i in ids}
        return {"base": acc(bb, ids), "pfk": acc(pp, ids), "router": acc(r, ids), "oracle": acc(oo, ids)}
    sets = {"main": (tasks, base, R["E4B PFK"], R["E4B oracle"], V)}
    for s in (1, 2, 3):
        t = load(P(f"seeds/bench_s{s}.jsonl"))
        sets[f"seed{s}"] = (t, load(P(f"seeds/seed_s{s}_base.jsonl")), load(P(f"seeds/seed_s{s}_pfk.jsonl")),
                            load(P(f"seeds/seed_s{s}_compact.jsonl")), [i for i, it in t.items() if well_posed(it)])
    grid = [256, 512, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000]
    t1 = sets["seed1"]
    T = max(grid, key=lambda g: (route_eval(*t1, g)["router"], -g))
    out["router"] = {"T": T, "eval": {k: route_eval(*v, T) for k, v in sets.items()}}

    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
