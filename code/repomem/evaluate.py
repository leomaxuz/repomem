#!/usr/bin/env python3
"""RepoMem baholash — vLLM OpenAI-mos endpointga savollarni yuborib, javoblarni baholaydi.

Asosiy chiqish: aniqlik (accuracy) masofaga (distance_tokens) va savol turiga qarab.
Bu kichik Gemma'ning ishchi xotira cheklovini ko'rsatadi.

  python evaluate.py --tasks repomem_out/sample.jsonl --base-url http://127.0.0.1:8000/v1 \
      --model gemma-4-e4b --out repomem_out/e4b_results.jsonl --concurrency 8 --thinking off
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import urllib.request


# --- Compact (ishchi-xotira) rejimi: savolni faqat kerakli funksiya(lar) bilan qayta quradi ---
def func_block(node):
    return "```python\n# " + node["id"] + "\n" + node["text"].rstrip() + "\n```"


def load_nodes(graph_path):
    g = json.loads(open(graph_path, encoding="utf-8").read())
    return {n["id"]: n for n in g["nodes"] if n.get("text")}


def build_compact(item, graphs_dir, cache):
    """meta'dagi 'igna' ID'lari asosida distraktorlarsiz, faqat kerakli bloklardan prompt yasaydi.
    Savol matni baseline bilan AYNAN BIR XIL (meta['question']) — faqat kontekst farq qiladi.
    Igna grafda topilmasa None qaytaradi."""
    repo, kind, meta = item["repo"], item["kind"], item.get("meta", {})
    if repo not in cache:
        gp = os.path.join(graphs_dir, repo + ".json")
        cache[repo] = load_nodes(gp) if os.path.exists(gp) else {}
    nodes = cache[repo]
    if kind == "defn":
        ids = [meta["target"]]
    elif kind == "chain":
        ids = list(meta["chain"])
    elif kind == "caller":
        ids = [meta["caller"], meta["target"]]
    else:
        return None
    q = meta.get("question")
    if not q:
        return None  # eski (savolsiz) benchmark bilan compact ishlamaydi
    blocks = [func_block(nodes[i]) for i in ids if i in nodes]
    if len(blocks) < len(ids):
        return None
    return "\n\n".join(blocks) + "\n\n" + q


def norm(s):
    """Taqqoslash uchun: kichik harf, faqat harf/raqam/pastki chiziq/vergul."""
    s = s.strip().strip("`").strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def norm_name(s):
    s = norm(s)
    m = re.findall(r"[a-z_][a-z0-9_]*", s)
    return m[-1] if m else s  # oxirgi identifikator (ko'pincha javob)


def norm_params(s):
    s = norm(s)
    parts = [re.sub(r"[^a-z0-9_*]", "", p) for p in s.split(",")]
    # self/cls konvensiya — baholashda e'tiborsiz (gold ham ularsiz)
    return [p for p in parts if p and p not in ("self", "cls")]


def score_item(item, pred):
    """Javob to'g'rimi? Savol turiga qarab."""
    gold = item["answer"]
    kind = item["kind"]
    # modelning javobidan oxirgi mazmunli qatorni olamiz
    p = pred.strip()
    if kind == "defn":
        gp, pp = norm_params(gold), norm_params(p)
        # tartibsiz to'plam mosligi (parametrlar to'plami bir xilmi)
        return set(gp) == set(pp) and len(gp) == len(pp)
    else:  # chain, caller — funksiya nomi
        return norm_name(gold) == norm_name(p) or norm_name(gold) in [norm_name(x) for x in re.split(r"[,\s]+", p)]


def call(base_url, model, prompt, thinking, timeout=120):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": ("You are a precise code analysis assistant. "
                                           "Answer with ONLY the requested value, no explanation.")},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        # defn javoblari uzun bo'lishi mumkin (30+ parametr) — yetarli budjet beramiz
        "max_tokens": 4096 if thinking else 320,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data,
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read())
    msg = j["choices"][0]["message"]
    return (msg.get("content") or "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--compact", action="store_true",
                    help="ishchi-xotira rejimi: faqat kerakli funksiya(lar)ni beradi (distraktorlarsiz)")
    ap.add_argument("--graphs", default="", help="--compact uchun graphs/ papka")
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.tasks)]
    if a.limit:
        items = items[: a.limit]
    thinking = a.thinking == "on"
    gcache = {}

    def work(it):
        t0 = time.time()
        try:
            if a.compact:
                prompt = build_compact(it, a.graphs, gcache)
                if prompt is None:
                    return {**{k: it[k] for k in ("id", "repo", "kind", "k", "distance_tokens", "n_items", "answer")},
                            "pred": "", "correct": False, "err": "compact-skip", "sec": 0.0}
            else:
                prompt = it["prompt"]
            pred = call(a.base_url, a.model, prompt, thinking)
            ok = score_item(it, pred)
            err = None
        except Exception as e:
            pred, ok, err = "", False, repr(e)[:200]
        return {**{k: it[k] for k in ("id", "repo", "kind", "k", "distance_tokens", "n_items", "answer")},
                "pred": pred[:300], "correct": bool(ok), "err": err, "sec": round(time.time() - t0, 1)}

    results = []
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for i, r in enumerate(ex.map(work, items), 1):
            results.append(r)
            if i % 20 == 0:
                acc = sum(x["correct"] for x in results) / len(results)
                print(f"  {i}/{len(items)}  joriy aniqlik: {acc:.1%}", flush=True)
    with open(a.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # xulosa: tur bo'yicha va masofa bo'yicha
    def acc(sub):
        return (sum(x["correct"] for x in sub) / len(sub)) if sub else 0.0
    errs = [r for r in results if r["err"]]
    print(f"\n=== JAMI: {len(results)} savol, aniqlik {acc(results):.1%}  (xatolar: {len(errs)})")
    if errs:
        print("  birinchi xato:", errs[0]["err"])
    print("--- tur bo'yicha:")
    for k in ("defn", "chain", "caller"):
        sub = [r for r in results if r["kind"] == k]
        if sub:
            print(f"    {k:7} n={len(sub):4}  aniqlik {acc(sub):.1%}")
    print("--- masofa bo'yicha (token):")
    buckets = [(0, 512), (512, 2000), (2000, 8000), (8000, 20000), (20000, 10**9)]
    for lo, hi in buckets:
        sub = [r for r in results if lo <= r["distance_tokens"] < hi]
        if sub:
            label = f"{lo}-{hi if hi < 10**9 else '∞'}"
            print(f"    {label:14} n={len(sub):4}  aniqlik {acc(sub):.1%}")


if __name__ == "__main__":
    main()
