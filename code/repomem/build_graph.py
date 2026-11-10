#!/usr/bin/env python3
"""Build a RepoMem-compatible code graph from ANY Python repository, using only the stdlib `ast`.

Output format (the same one generate.py / pfk_agent.py read):
  {"directed": true, "nodes": [{"id", "name", "text"}...], "edges": [{"source", "target", "type": "calls"}...]}

Nodes are module-level functions and class methods (id = dotted module path + qualname,
text = dedented source of the def, without decorators). A `calls` edge A -> B is added only when
the call in A's body resolves *statically and unambiguously* to B:
  * `foo(...)`        -> function `foo` defined in the same module, or imported by
                         `from <module> import foo [as alias]` from a module of this repository;
  * `self.bar(...)` / `cls.bar(...)` -> method `bar` of the enclosing class;
  * `mod.foo(...)`    -> `foo` in a repository module imported as `import pkg.mod [as mod]`.
Every edge is therefore backed by a literal call in the caller's source, in the right direction.

  python build_graph.py --repo path/to/flask --name flask_3.0.3 --out graphs/
"""
import argparse
import ast
import json
import os
import textwrap


def module_name(root, path):
    rel = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
    if rel.endswith(".__init__"):
        rel = rel[: -len(".__init__")]
    if rel.startswith("src."):          # src-layout packages
        rel = rel[4:]
    return rel


def resolve_from(mod, is_pkg, level, target):
    """Absolute module for `from <level dots><target> import ...` inside module `mod`."""
    if level == 0:
        return target or ""
    parts = mod.split(".")
    base = parts if is_pkg else parts[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([p for p in base + ([target] if target else []) if p])


def collect(root):
    files = []
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if not x.startswith(".") and x not in ("build", "dist", "node_modules", "__pycache__")]
        files += [os.path.join(d, f) for f in fs if f.endswith(".py")]
    mods = {}
    for p in sorted(files):
        try:
            src = open(p, encoding="utf-8").read()
            tree = ast.parse(src)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        mods[module_name(root, p)] = (src, tree, p.endswith("__init__.py"))
    return mods


def build(root):
    mods = collect(root)
    nodes, defs = {}, []          # defs: (node_id, module, class_name or None, ast_node, src)
    for mod, (src, tree, _) in mods.items():
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append((f"{mod}.{top.name}", mod, None, top, src))
            elif isinstance(top, ast.ClassDef):
                for m in top.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defs.append((f"{mod}.{top.name}.{m.name}", mod, top.name, m, src))
    for nid, mod, cls, fn, src in defs:
        seg = ast.get_source_segment(src, fn)
        if seg:
            nodes[nid] = {"id": nid, "name": nid, "text": textwrap.dedent(" " * fn.col_offset + seg)}

    # per-module name resolution tables
    imports = {}
    for mod, (src, tree, is_pkg) in mods.items():
        names, aliases = {}, {}
        for st in tree.body:
            if isinstance(st, ast.ImportFrom):
                base = resolve_from(mod, is_pkg, st.level, st.module)
                for a in st.names:
                    names[a.asname or a.name] = f"{base}.{a.name}" if base else a.name
            elif isinstance(st, ast.Import):
                for a in st.names:
                    if a.asname:
                        aliases[a.asname] = a.name
                    else:
                        aliases[a.name.split(".")[0]] = a.name.split(".")[0]
        imports[mod] = (names, aliases)

    edges = set()
    for nid, mod, cls, fn, _ in defs:
        if nid not in nodes:
            continue
        names, aliases = imports[mod]
        # walk the signature and body only: decorators are not part of the node's text,
        # so a call inside a decorator would create an edge the reader cannot see
        roots = list(fn.body) + list(fn.args.defaults) + [d for d in fn.args.kw_defaults if d is not None]
        for c in (n for r in roots for n in ast.walk(r)):
            if not isinstance(c, ast.Call):
                continue
            f, tgt = c.func, None
            if isinstance(f, ast.Name):
                cand = f"{mod}.{f.id}"
                if cand in nodes:
                    tgt = cand
                elif f.id in names and names[f.id] in nodes:
                    tgt = names[f.id]
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                v = f.value.id
                if v in ("self", "cls") and cls:
                    cand = f"{mod}.{cls}.{f.attr}"
                    tgt = cand if cand in nodes else None
                elif v in aliases:
                    cand = f"{aliases[v]}.{f.attr}"
                    tgt = cand if cand in nodes else None
            if tgt and tgt != nid:
                edges.add((nid, tgt))
    return {"directed": True, "multigraph": False, "graph": {"source": os.path.basename(os.path.abspath(root))},
            "nodes": list(nodes.values()),
            "edges": [{"source": s, "target": t, "type": "calls"} for s, t in sorted(edges)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--name", required=True, help="graph file stem, e.g. flask_3.0.3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = build(a.repo)
    os.makedirs(a.out, exist_ok=True)
    json.dump(g, open(os.path.join(a.out, a.name + ".json"), "w"), ensure_ascii=False)
    print(f"{a.name}: {len(g['nodes'])} nodes, {len(g['edges'])} calls edges")


if __name__ == "__main__":
    main()
