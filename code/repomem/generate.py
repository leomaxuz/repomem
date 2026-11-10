#!/usr/bin/env python3
"""RepoMem — kod bazasida ishchi xotira testlari generatori.

Musobaqa kod graflaridan (nodes = funksiya/klass manbasi, edges = `calls`) bir navbatli
savol-javob testlari yasaydi. Asosiy maqsad: savol bilan kerakli ma'lumot orasidagi
MASOFANI (distraktor kod miqdorini) o'zgartirib, modelning uzoq kontekstdagi xotirasini
o'lchash. Kichik Gemma (E2B/E4B) sliding window = 512 token bilan cheklangan; masofa shundan
oshganda natija tushishini kutamiz.

Probelar:
  defn   — ko'p funksiya ta'rifi berilgan, N-chisining qiymatini (return/parametr) so'raydi.
  chain  — A -> B -> C ... chaqiruv zanjiri; "A oxir-oqibat nimani chaqiradi (k qadam)?"
  caller — "Qaysi funksiya X ni chaqiradi?" (graf bo'yicha haqiqiy javob)

Chiqish: JSONL, har satr {id, repo, kind, k, distance_tokens, n_items, prompt, answer, meta}.

  python generate.py --graphs <dir> --out tasks.jsonl --per-repo 40 --seed 0
"""
import argparse
import ast
import json
import random
from pathlib import Path


def load_graph(path):
    g = json.loads(Path(path).read_text())
    nodes = {n["id"]: n for n in g["nodes"] if n.get("text")}
    edges = [e for e in g["edges"] if e.get("type") == "calls"]
    calls = {}
    for e in edges:
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
    """Node funksiyami (klass emas)? Chain/caller probelar faqat funksiyalar ustida ishlaydi."""
    t = text.lstrip()
    return t.startswith("def ") or t.startswith("async def ")


def approx_tokens(s):
    # qo'pol baho: ~4 belgi/token
    return max(1, len(s) // 4)


def func_block(node):
    return f"```python\n# {node['id']}\n{node['text'].rstrip()}\n```"


def signature_params(text):
    """Funksiya imzosidan parametr nomlarini AST bilan aniq chiqaradi (tip annotatsiyalari,
    ko'p qatorli imzolar, *args/**kwargs to'g'ri ishlanadi). Klass yoki noto'g'ri kod -> None."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            names = [p.arg for p in (a.posonlyargs + a.args)]
            if a.vararg:
                names.append("*" + a.vararg.arg)
            names += [p.arg for p in a.kwonlyargs]
            if a.kwarg:
                names.append("**" + a.kwarg.arg)
            return names
    return None


def pick_chain(calls, nodes, k, rng, tries=400):
    """k uzunlikdagi ANIQ (deterministik) chaqiruv zanjirini topadi: har bosqichda
    chaqiruvchi funksiya faqat BITTA funksiyani chaqiradi, shuning uchun javob yagona."""
    starts = [s for s in calls if s in nodes and is_func(nodes[s]["text"])]
    rng.shuffle(starts)
    for s in starts[:tries]:
        chain = [s]
        cur = s
        ok = True
        for _ in range(k):
            nxt = [t for t in calls.get(cur, []) if t in nodes and is_func(nodes[t]["text"])]
            # deterministik bo'lishi uchun: aynan bitta chaqiruv (takrorsiz)
            uniq = sorted(set(nxt))
            if len(uniq) != 1 or uniq[0] in chain:
                ok = False
                break
            cur = uniq[0]
            chain.append(cur)
        if ok and len(chain) == k + 1:
            return chain
    return None


def real_params(text):
    """Parametrlar (self/cls siz) — javob konvensiyaga bog'liq bo'lmasligi uchun."""
    p = signature_params(text)
    if p is None:
        return None
    return [x for x in p if x not in ("self", "cls")]


def make_defn(nodes, rng, n_items, target_pos):
    # aniq imzoga ega va (self/cls dan tashqari) kamida 2 parametrli funksiyalar
    ids = [nid for nid in nodes
           if (p := real_params(nodes[nid]["text"])) is not None and len(p) >= 2]
    if len(ids) < n_items:
        return None
    sel = rng.sample(ids, n_items)
    target = sel[target_pos]
    params = real_params(nodes[target]["text"])
    ans = ", ".join(params)
    blocks = [func_block(nodes[nid]) for nid in sel]
    before = sum(approx_tokens(b) for b in blocks[:target_pos])
    body = "\n\n".join(blocks)
    q = (f"Above are {n_items} function definitions from the repository. "
         f"List the parameter names of `{short(target)}` (`{target}`), in order, "
         f"comma-separated (exclude `self`). Answer with just the parameter names.")
    return dict(kind="defn", k=0, distance_tokens=before, n_items=n_items,
                prompt=body + "\n\n" + q, answer=ans,
                meta={"target": target, "target_pos": target_pos, "params": params, "question": q})


def make_chain(nodes, calls, rng, k, n_distract):
    chain = pick_chain(calls, nodes, k, rng)
    if not chain:
        return None
    relevant = [func_block(nodes[nid]) for nid in chain]
    distract_ids = [nid for nid in nodes if nid not in chain]
    if len(distract_ids) < n_distract:
        n_distract = len(distract_ids)
    distract = [func_block(nodes[nid]) for nid in rng.sample(distract_ids, n_distract)]
    blocks = relevant + distract
    rng.shuffle(blocks)
    body = "\n\n".join(blocks)
    ans = short(chain[-1])
    q = (f"The functions above are from one repository. Each function calls at most one of the "
         f"OTHER functions shown above. Starting from `{short(chain[0])}` (`{chain[0]}`), follow "
         f"the chain of calls exactly {k} step(s) deep (each function calls the next shown one). "
         f"Which function is reached at step {k}? Answer with just the function name.")
    return dict(kind="chain", k=k, distance_tokens=approx_tokens(body), n_items=len(blocks),
                prompt=body + "\n\n" + q, answer=ans,
                meta={"chain": chain, "question": q})


def make_caller(nodes, callers, rng, n_distract):
    # yagona chaqiruvchi + chaqiruv MATNDA aniq ko'rinishi shart (compact'da ham javob beriladigan)
    # yagona chaqiruvchi; chaqiruvchi/chaqiriluvchi ikkalasi ham funksiya; chaqiruv matnda aniq
    cands = [t for t, ss in callers.items()
             if len(ss) == 1 and t in nodes and ss[0] in nodes
             and is_func(nodes[t]["text"]) and is_func(nodes[ss[0]]["text"])
             and (short(t) + "(") in nodes[ss[0]]["text"]]
    if not cands:
        return None
    target = rng.choice(cands)
    caller = callers[target][0]
    relevant = [func_block(nodes[caller]), func_block(nodes[target])]
    distract_ids = [nid for nid in nodes if nid not in (caller, target)]
    n_distract = min(n_distract, len(distract_ids))
    distract = [func_block(nodes[nid]) for nid in rng.sample(distract_ids, n_distract)]
    blocks = relevant + distract
    rng.shuffle(blocks)
    body = "\n\n".join(blocks)
    q = (f"Among the functions shown above, which one calls `{short(target)}` (`{target}`)? "
         f"Answer with just the calling function's name.")
    return dict(kind="caller", k=1, distance_tokens=approx_tokens(body), n_items=len(blocks),
                prompt=body + "\n\n" + q, answer=short(caller),
                meta={"target": target, "caller": caller, "question": q})


def gen_for_graph(path, rng, per_repo, rounds=1):
    nodes, calls, callers = load_graph(path)
    if len(nodes) < 30:
        return []
    out = []
    seen = set()
    repo = Path(path).stem
    # distance darajalari: funksiya bloklari soni (token masofasiga ta'sir qiladi)
    defn_sizes = [4, 8, 16, 32, 64]
    chain_ks = [1, 2, 3, 4]
    distracts = [0, 8, 24, 64]
    budget = per_repo
    plan = []
    for n in defn_sizes:
        plan.append(("defn", n))
    for k in chain_ks:
        for d in distracts:
            plan.append(("chain", (k, d)))
    for d in distracts:
        plan.append(("caller", d))
    if rounds > 1:
        # few large repositories instead of many snapshots: repeat the plan, drop duplicates
        plan = plan * rounds
    rng.shuffle(plan)
    for kind, param in plan:
        if len(out) >= budget:
            break
        if kind == "defn":
            n = param
            item = make_defn(nodes, rng, n, rng.randrange(n))
        elif kind == "chain":
            k, d = param
            item = make_chain(nodes, calls, rng, k, d)
        else:
            item = make_caller(nodes, callers, rng, param)
        if item and rounds > 1:
            key = (item["meta"]["question"], item["prompt"][:2000])
            if key in seen:
                continue
            seen.add(key)
        if item:
            item["id"] = f"{repo}:{kind}:{len(out)}"
            item["repo"] = repo
            out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", required=True, help="graphs/ papka yoki bitta .json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-repo", type=int, default=40)
    ap.add_argument("--max-graphs", type=int, default=0, help="0 = hammasi")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the per-graph probe plan this many times (for few, large repositories)")
    ap.add_argument("--no-textual-filter", action="store_true",
                    help="keep chain items whose call steps are not visible in the source (ill-posed; v1 behaviour)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    p = Path(a.graphs)
    files = [p] if p.is_file() else sorted(p.glob("*.json"))
    if a.max_graphs:
        files = files[: a.max_graphs]
    total = []
    dropped = 0
    for f in files:
        try:
            items = gen_for_graph(f, rng, a.per_repo, a.rounds)
            if not a.no_textual_filter:
                # Well-posedness: every chain step A -> B must be visible as a call `B(` in A's source.
                # Some competition-graph `calls` edges point the wrong way (callee -> caller); such items
                # have an unanswerable gold. Filtering AFTER generation keeps the RNG stream, so the
                # surviving items are identical with or without the filter.
                g = json.loads(Path(f).read_text())
                text = {n["id"]: n.get("text") or "" for n in g["nodes"]}
                keep = [it for it in items if it["kind"] != "chain" or all(
                    short(b) + "(" in text.get(x, "") for x, b in zip(it["meta"]["chain"], it["meta"]["chain"][1:]))]
                dropped += len(items) - len(keep)
                items = keep
            total.extend(items)
        except Exception as e:
            print(f"XATO {f.name}: {e!r}")
    if dropped:
        print(f"well-posedness filter: dropped {dropped} chain items with a non-textual call step")
    with open(a.out, "w") as fh:
        for it in total:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    # qisqa statistika
    by_kind = {}
    for it in total:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    print(f"jami: {len(total)} test, {len(files)} grafdan")
    print("turlari:", by_kind)
    if total:
        ds = sorted(it["distance_tokens"] for it in total)
        print(f"masofa (token): min {ds[0]}, median {ds[len(ds)//2]}, max {ds[-1]}")


if __name__ == "__main__":
    main()
