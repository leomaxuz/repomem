#!/usr/bin/env python3
"""RepoMem paper figures, drawn directly from experiments/paper_numbers.json (recompute_all.py).
Colorblind-safe (Okabe-Ito), white background.

Fig 1  repomem_distance.png  E4B/E2B: in-window distance curve (left) + beyond-window items (right)
Fig 2  repomem_scaling.png   size scaling E2B/E4B/12B/31B on the common 16K-fitting subset (+ E2B with PFK)
Fig 3  repomem_seeds.png     three regenerated instances: mean ± std (in-window curve + beyond-window)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(PROJ, "figures")
os.makedirs(OUT, exist_ok=True)
N = json.load(open(os.path.join(PROJ, "experiments", "paper_numbers.json")))
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                     "figure.dpi": 160, "savefig.dpi": 200, "font.family": "DejaVu Sans"})
C_BASE, C_PFK, C_ORC, C_E2B = "#D55E00", "#0072B2", "#009E73", "#E69F00"
B4 = ["0–512", "512–2K", "2K–8K", "8K–20K"]


def style(ax, labels=None, xlabel=True):
    if labels:
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    if xlabel:
        ax.set_xlabel("Distance between question and needed fact (tokens)")
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color="0.9", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def beyond_panel(ax, vals, errs=None, title=""):
    cols = [C_BASE, C_PFK, C_ORC]
    ax.bar(range(3), vals, color=cols, width=0.62, yerr=errs, capsize=4 if errs else 0)
    for i, v in enumerate(vals):
        ax.text(i, v + (errs[i] if errs else 0) + 2, "0%\n(cannot run)" if v == 0 else f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks(range(3)); ax.set_xticklabels(["baseline", "PFK-agent", "oracle"], fontsize=10.5)
    ax.set_ylim(0, 112); style(ax, xlabel=False)
    ax.set_title(title, fontsize=11.5)


# ---------- Figure 1 ----------
M = N["main"]; comp = N["composition"]
n_in = sum(comp["in_window_buckets"].values())
fig, (a, b) = plt.subplots(1, 2, figsize=(10.2, 4.5), gridspec_kw={"width_ratios": [2.3, 1]})
a.plot(range(4), M["E4B oracle"]["buckets"][:4], color=C_ORC, lw=2, ls=(0, (4, 3)), marker="D", ms=5, label="E4B compact (oracle)")
a.plot(range(4), M["E4B PFK"]["buckets"][:4], color=C_PFK, lw=2.6, marker="o", ms=6, label="E4B PFK-agent (ours)")
a.plot(range(4), M["E4B base"]["buckets"][:4], color=C_BASE, lw=2.6, marker="s", ms=6, label="E4B baseline (full context)")
a.plot(range(4), M["E2B base"]["buckets"][:4], color=C_E2B, lw=2.2, marker="v", ms=6, ls=(0, (1, 1.5)), label="E2B baseline")
style(a, B4); a.set_ylabel("Accuracy"); a.set_ylim(65, 97)
a.set_title(f"Within the context window (n = {n_in:,})")
a.legend(frameon=False, loc="lower left", fontsize=10)
beyond_panel(b, [M["E4B base"]["beyond"], M["E4B PFK"]["beyond"], M["E4B oracle"]["beyond"]],
             title=f"Prompt exceeds 32K window\n(n = {comp['beyond_window']})")
fig.suptitle("RepoMem: accuracy decays with distance, then hits a hard window ceiling", fontsize=13.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "repomem_distance.png"), bbox_inches="tight")
print("saved repomem_distance.png")

# ---------- Figure 2: size scaling ----------
S = N["scaling"]
seq = {"E2B": ("#9ECAE1", "v"), "E4B": ("#4292C6", "s"), "12B": ("#2171B5", "o"), "31B": ("#08306B", "D")}
fig2, ax2 = plt.subplots(figsize=(7.0, 4.5))
for m, (c, mk) in seq.items():
    ax2.plot(range(4), S[m]["base_b"], color=c, lw=2.6, marker=mk, ms=6, label=f"{m} baseline")
mean_orc = [sum(v) / 4 for v in zip(*[S[m]["oracle_b"] for m in seq])]
ax2.plot(range(4), mean_orc, color=C_ORC, lw=1.8, ls=(0, (4, 3)), label="compact oracle (mean over sizes)")
ax2.plot(range(4), S["E2B PFK"]["b"], color=C_E2B, lw=2.8, marker="*", ms=11, label="E2B + PFK-agent (ours)")
style(ax2, B4); ax2.set_ylabel("Accuracy"); ax2.set_ylim(65, 97)
ax2.set_title(f"Size scaling, common subset that fits every model (n = {S['n']:,})", fontsize=12)
ax2.legend(frameon=False, loc="lower left", fontsize=10)
fig2.tight_layout(); fig2.savefig(os.path.join(OUT, "repomem_scaling.png"), bbox_inches="tight")
print("saved repomem_scaling.png")

# ---------- Figure 3: seeds ----------
A = N["seeds"]["agg"]
fig3, (a3, b3) = plt.subplots(1, 2, figsize=(10.2, 4.5), gridspec_kw={"width_ratios": [2.3, 1]})
for i, (tag, lab, c, mk, ls) in enumerate([("oracle", "E4B compact (oracle)", C_ORC, "D", (0, (4, 3))),
                                           ("pfk", "E4B PFK-agent (ours)", C_PFK, "o", "-"),
                                           ("base", "E4B baseline", C_BASE, "s", "-")]):
    mu = [x[0] for x in A[tag][:4]]; sd = [x[1] for x in A[tag][:4]]
    a3.errorbar([x + (i - 1) * 0.06 for x in range(4)], mu, yerr=sd, color=c, lw=2.4, ls=ls,
                marker=mk, ms=6, capsize=4, label=lab)
style(a3, B4); a3.set_ylabel("Accuracy"); a3.set_ylim(80, 97)
a3.set_title("Within the window, 3 regenerated instances (mean ± std)", fontsize=12)
a3.legend(frameon=False, loc="lower left", fontsize=10)
nb = A["n_beyond"]
beyond_panel(b3, [A["base"][4][0], A["pfk"][4][0], A["oracle"][4][0]], errs=[0.0, A["pfk"][4][1], A["oracle"][4][1]],
             title=f"Prompt exceeds 32K window\n({min(nb)}–{max(nb)} items / instance)")
fig3.tight_layout(); fig3.savefig(os.path.join(OUT, "repomem_seeds.png"), bbox_inches="tight")
print("saved repomem_seeds.png")

# ---------- Figure 4: RepoMem-OSS (graphs built with build_graph.py from 6 public repositories) ----------
EP = os.path.join(PROJ, "experiments", "ext_numbers.json")
if os.path.exists(EP):
    E = json.load(open(EP)); EM = E["models"]
    fig4, (a4, b4) = plt.subplots(1, 2, figsize=(10.2, 4.5), gridspec_kw={"width_ratios": [2.3, 1]})
    orc = [(x + y) / 2 for x, y in zip(EM["e2b_compact"]["hop"]["single"]["buckets"], EM["e4b_compact"]["hop"]["single"]["buckets"])]
    a4.plot(range(4), orc, color=C_ORC, lw=2, ls=(0, (4, 3)), marker="D", ms=5, label="compact oracle (mean E2B/E4B)")
    a4.plot(range(4), EM["e4b_pfk"]["hop"]["single"]["buckets"], color=C_PFK, lw=2.6, marker="o", ms=6, label="E4B PFK-agent")
    a4.plot(range(4), EM["e2b_pfk"]["hop"]["single"]["buckets"], color=C_PFK, lw=2.2, marker="o", ms=6, ls=(0, (1, 1.5)), label="E2B PFK-agent")
    a4.plot(range(4), EM["e4b_full"]["hop"]["single"]["buckets"], color=C_BASE, lw=2.6, marker="s", ms=6, label="E4B baseline")
    a4.plot(range(4), EM["e2b_full"]["hop"]["single"]["buckets"], color=C_E2B, lw=2.2, marker="v", ms=6, ls=(0, (1, 1.5)), label="E2B baseline")
    style(a4, B4); a4.set_ylabel("Accuracy"); a4.set_ylim(60, 100)
    a4.set_title(f"Single-hop items (n = {E['hop_n']['single']:,})", fontsize=12)
    a4.legend(frameon=False, loc="lower left", fontsize=9.5)
    w = 0.26
    for j, (tag, c) in enumerate([("full", C_BASE), ("pfk", C_PFK), ("compact", C_ORC)]):
        vals = [EM[f"{m}_{tag}"]["hop"]["multi"]["all"] for m in ("e4b", "e2b")]
        b4.bar([x + (j - 1) * w for x in range(2)], vals, width=w, color=c,
               label={"full": "baseline", "pfk": "PFK-agent", "compact": "oracle"}[tag])
        for x, v in zip(range(2), vals):
            b4.text(x + (j - 1) * w, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
    b4.set_xticks(range(2)); b4.set_xticklabels(["E4B", "E2B"]); b4.set_ylim(0, 100); style(b4, xlabel=False)
    b4.set_title(f"Multi-hop chains, k = 2–4\n(n = {E['hop_n']['multi']}): a reasoning ceiling", fontsize=11)
    b4.legend(frameon=False, loc="upper right", fontsize=9)
    fig4.suptitle("RepoMem-OSS: 6 public repositories outside the competition (flask, click, werkzeug, jinja, attrs, marshmallow)",
                  fontsize=12)
    fig4.tight_layout(); fig4.savefig(os.path.join(OUT, "repomem_oss.png"), bbox_inches="tight")
    print("saved repomem_oss.png")
