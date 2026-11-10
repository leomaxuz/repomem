#!/usr/bin/env python3
"""PFK-agent — grafdan retrieval qiladigan ishchi-xotira agenti.

G'oya: baseline (to'liq kontekst) uzoq masofada quyaydi, chunki model ignani distraktor
ichida ushlab tura olmaydi. Oracle-compact (meta'dagi igna ID'lari) buni tuzatadi, lekin u
"firibgar" — javob yo'lini oldindan biladi. PFK-agent HALOL: savolda ATALGAN funksiya nomidan
boshlab, musobaqaning kod-graf vositalari (get_code_neighbors / get_code_subgraph / callers)
orqali kerakli bloklarni O'ZI topib, qisqa ishchi-xotira buferiga soladi, keyin javob beradi.
meta'dagi javob-yo'lini (chain butun ro'yxati / caller) ISHLATMAYDI — faqat berilgan nomni.

Chiqish JSONL har savol uchun evaluate.py bilan bir xil maydonlar + retrieved_ok.
  python pfk_agent.py --tasks tasks_full.jsonl --graphs <dir> --model gemma-4-e4b \
      --base-url http://127.0.0.1:8000/v1 --out pfk_e4b.jsonl --concurrency 6
"""
import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


# ---------- graf (generate.py bilan bir xil) ----------
def load_graph(path, edge_drop=0.0, drop_seed=0):
    """edge_drop>0: har bir `calls` qirrasini shu ehtimollik bilan tashlab yuboradi
    (real graflardagi to'liqsizlikni — dinamik dispatch, reflection — simulyatsiya qiladi)."""
    import random
    rng = random.Random(f"{os.path.basename(path)}:{drop_seed}")
    g = json.loads(open(path).read())
    nodes = {n["id"]: n for n in g["nodes"] if n.get("text")}
    calls = {}
    for e in g["edges"]:
        if e.get("type") != "calls":
            continue
        if edge_drop > 0 and rng.random() < edge_drop:
            continue
        if e["source"] in nodes and e["target"] in nodes:
            calls.setdefault(e["source"], []).append(e["target"])
    callers = {}
    for s, ts in calls.items():
        for t in ts:
            callers.setdefault(t, []).append(s)
    return nodes, calls, callers


def short(nid):
    return nid.split(".")[-1]


def is_func(text):
    t = text.lstrip()
    return t.startswith("def ") or t.startswith("async def ")


def func_block(node):
    return f"```python\n# {node['id']}\n{node['text'].rstrip()}\n```"


# ---------- retrieval (musobaqa graf-vositalarini modellashtiradi) ----------
def named_entity(item):
    """Savolda ATALGAN entity (meta'dan) — 'meta' rejimi uchun."""
    meta = item.get("meta", {})
    return meta.get("chain", [None])[0] if item["kind"] == "chain" else meta.get("target")


def retrieve(item, nodes, calls, callers, entity=None):
    """Savolda atalgan nomdan boshlab graf bo'yicha kerakli bloklar ID'larini qaytaradi.
    (needle_ids, ok). meta'dagi JAVOB yo'li ishlatilmaydi — faqat berilgan boshlang'ich nom.
    entity berilsa (model o'zi ajratgan), meta o'rniga shu ishlatiladi."""
    kind = item["kind"]
    ent = entity if entity is not None else named_entity(item)
    if kind == "defn":
        # savol `target` imzosini so'raydi — target nomi savolda berilgan
        tid = ent
        return ([tid], tid in nodes)
    if kind == "chain":
        # faqat ATALGAN boshlang'ich (chain[0]) beriladi; qolgan yo'lni graf bilan O'ZIMIZ aylanamiz
        start = ent
        if start not in nodes:
            return ([], False)
        k = int(item.get("k", 0))
        cur, path = start, [start]
        for _ in range(k):
            nxt = sorted({t for t in calls.get(cur, []) if t in nodes and is_func(nodes[t]["text"])})
            if len(nxt) != 1 or nxt[0] in path:   # noaniq -> retrieval to'liq emas
                break
            cur = nxt[0]
            path.append(cur)
        return (path, len(path) == k + 1)
    if kind == "caller":
        # atalgan `target` -> uni kim chaqirishini teskari qirralardan topamiz (get_callers)
        tid = ent
        if tid not in nodes:
            return ([], False)
        cs = callers.get(tid, [])
        if not cs:
            return ([tid], False)
        return ([cs[0], tid], True)
    return ([], False)


# ---------- baholash (evaluate.py bilan bir xil) ----------
def norm(s):
    s = s.strip().strip("`").strip()
    return re.sub(r"\s+", " ", s).lower()


def norm_name(s):
    m = re.findall(r"[a-z_][a-z0-9_]*", norm(s))
    return m[-1] if m else norm(s)


def norm_params(s):
    parts = [re.sub(r"[^a-z0-9_*]", "", p) for p in norm(s).split(",")]
    return [p for p in parts if p]


def score_item(item, pred):
    gold, kind, p = item["answer"], item["kind"], pred.strip()
    if kind == "defn":
        gp, pp = norm_params(gold), norm_params(p)
        return set(gp) == set(pp) and len(gp) == len(pp)
    return norm_name(gold) == norm_name(p) or norm_name(gold) in [norm_name(x) for x in re.split(r"[,\s]+", p)]


FQN_PAREN = re.compile(r"\s*\(`[^`]*`\)")


def extract_entity(base_url, model, question, nodes, short_only=False, verify=False):
    """Entity grounding by the MODEL itself (no meta): ask Gemma which function the question is
    about, then resolve that name against the graph. short_only=True first strips the fully
    qualified id from the question, so only the bare function name is available (harder, may be
    ambiguous). Returns (node_id or None, raw_reply, n_candidates)."""
    q = FQN_PAREN.sub("", question) if short_only else question
    prompt = ("Here is a question about a code repository:\n\n" + q + "\n\n"
              "Which single function does this question start from or ask about? "
              "Reply with ONLY its identifier exactly as written in the question.")
    raw = call(base_url, model, prompt, max_tokens=48).strip().strip("`").strip()
    name = raw.split()[0].strip("`.,()") if raw else ""
    if name in nodes:
        return name, raw, 1
    cands = sorted(i for i in nodes if short(i) == short(name))
    if not cands:
        return None, raw, 0
    if verify:
        # verified grounding: keep only candidates whose identifier literally appears in the question;
        # if the name is still ambiguous, refuse to guess (caller may fall back to the full context)
        seen = [c for c in cands if f"`{c}`" in q]
        if len(seen) == 1:
            return seen[0], raw, len(cands)
        if len(cands) == 1:
            return cands[0], raw, 1
        return None, raw, len(cands)
    # naive: prefer functions; deterministic choice
    funcs = [c for c in cands if is_func(nodes[c]["text"])] or cands
    return funcs[0], raw, len(cands)


def call(base_url, model, prompt, timeout=120, max_tokens=320):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise code analysis assistant. Answer with ONLY the requested value, no explanation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read())
    return (j["choices"][0]["message"].get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entity-mode", choices=["meta", "model", "model-short"], default="meta",
                    help="meta: named entity from the item; model: Gemma extracts it from the question; "
                         "model-short: same, but the fully-qualified id is stripped from the question first")
    ap.add_argument("--verify-entity", action="store_true",
                    help="accept the model's entity only if unambiguous / literally present in the question")
    ap.add_argument("--edge-drop", type=float, default=0.0, help="drop this fraction of graph `calls` edges")
    ap.add_argument("--drop-seed", type=int, default=0)
    ap.add_argument("--fallback-full", action="store_true",
                    help="if graph retrieval is incomplete, answer from the full original prompt instead")
    a = ap.parse_args()

    items = [json.loads(l) for l in open(a.tasks)]
    if a.limit:
        items = items[: a.limit]
    gcache = {}

    def gget(repo):
        if repo not in gcache:
            gp = os.path.join(a.graphs, repo + ".json")
            gcache[repo] = (load_graph(gp, a.edge_drop, a.drop_seed) if os.path.exists(gp)
                            else ({}, {}, {}))
        return gcache[repo]

    def work(it):
        t0 = time.time()
        extra = {}
        try:
            nodes, calls, callers = gget(it["repo"])
            q = it.get("meta", {}).get("question") or it["prompt"].split("\n\n")[-1]
            entity = None
            if a.entity_mode != "meta":
                entity, raw, ncand = extract_entity(a.base_url, a.model, q, nodes,
                                                    short_only=(a.entity_mode == "model-short"),
                                                    verify=a.verify_entity)
                extra = {"entity": entity, "entity_raw": raw[:120], "entity_cands": ncand,
                         "entity_ok": entity == named_entity(it)}
                if entity is None:
                    entity = "__unresolved__"
            ids, ok = retrieve(it, nodes, calls, callers, entity=entity)
            blocks = [func_block(nodes[i]) for i in ids if i in nodes]
            used_fallback = False
            if a.fallback_full and not ok:
                prompt, used_fallback = it["prompt"], True
            elif not blocks:
                return {**{k: it[k] for k in ("id", "repo", "kind", "k", "distance_tokens", "answer")},
                        "pred": "", "correct": False, "retrieved_ok": False, "err": "retrieve-empty",
                        "sec": round(time.time() - t0, 1), "n_retrieved": 0, **extra}
            else:
                prompt = "\n\n".join(blocks) + "\n\n" + q
            pred = call(a.base_url, a.model, prompt)
            return {**{k: it[k] for k in ("id", "repo", "kind", "k", "distance_tokens", "answer")},
                    "pred": pred[:300], "correct": bool(score_item(it, pred)), "retrieved_ok": bool(ok),
                    "err": None, "sec": round(time.time() - t0, 1), "n_retrieved": len(blocks),
                    "fallback": used_fallback, **extra}
        except Exception as e:
            return {**{k: it[k] for k in ("id", "repo", "kind", "k", "distance_tokens", "answer")},
                    "pred": "", "correct": False, "retrieved_ok": False, "err": repr(e)[:200],
                    "sec": round(time.time() - t0, 1), "n_retrieved": 0}

    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, r in enumerate(ex.map(work, items), 1):
            results.append(r)
            if i % 50 == 0:
                acc = sum(x["correct"] for x in results) / len(results)
                print(f"  {i}/{len(items)}  joriy aniqlik: {acc:.1%}", flush=True)
    with open(a.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def acc(sub):
        return (sum(x["correct"] for x in sub) / len(sub)) if sub else 0.0
    rok = sum(x["retrieved_ok"] for x in results) / len(results)
    print(f"\n=== PFK JAMI: {len(results)} savol, aniqlik {acc(results):.1%}  (retrieval-ok: {rok:.1%})")
    print("--- tur bo'yicha:")
    for k in ("defn", "chain", "caller"):
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"    {k:7} n={len(sub):4}  aniqlik {acc(sub):.1%}")
    print("--- masofa bo'yicha (asl distance_tokens):")
    for lo, hi in [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]:
        sub = [r for r in results if lo <= r["distance_tokens"] < hi]
        if sub:
            lbl = f"{lo}-{hi if hi < 10**9 else 'inf'}"
            print(f"    {lbl:14} n={len(sub):4}  aniqlik {acc(sub):.1%}")


if __name__ == "__main__":
    main()
