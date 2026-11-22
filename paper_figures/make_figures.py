"""Build the paper's Figures 1-21 as vector PDFs from the cached results in this repository.

What it does
    Reads data/results/*.json and data/cache/{nq,tqa}_{generations,entailment}.json and
    plots them. Nothing is recomputed: no generation, entailment scoring, classifier
    training or 500-split evaluation is run. Each fig* function names the original
    plot_results.py / plot_synthetic_final.py function whose quantities it reproduces.
    Every number printed in or plotted by a figure is checked against the value printed in
    the paper (the expected values are embedded in the chk() calls below); the script
    exits non-zero if any check fails.

How to run (from the repository root)
    python paper_figures/make_figures.py                     # writes paper_figures/out/
    python paper_figures/make_figures.py --out-dir <dir>     # writes <dir>/fig01_...pdf ...
    python paper_figures/check_fonts.py <dir>                # embedded-font check

Requirements
    Python: matplotlib, numpy, scipy, scikit-learn.
    TeX: a TeX installation with latex, dvips and kpsewhich on PATH and the XCharter and
    newtx packages. All text is typeset by LaTeX (text.usetex) in the paper's fonts,
    XCharter text and newtxmath (xcharter) maths. --preview PNGs also need dvipng.
"""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter

REPO = Path(__file__).resolve().parents[1]  # same root as ds_sgen.utils.REPO_ROOT (avoids its yaml import)
RES = REPO / "data" / "results"
CACHE = REPO / "data" / "cache"
OUT = REPO / "paper_figures" / "out"  # default; --out-dir overrides

FULL, HALF = 6.8, 3.3  # inches; text block of the target template
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # Paper body fonts. The custom preamble is loaded after matplotlib's default font
    # preamble, so XCharter replaces \rmdefault and \rmfamily (font.family serif) resolves to it.
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{XCharter}"
                           r"\usepackage[xcharter,bigdelims,vvarbb]{newtxmath}",
    "font.family": "serif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "legend.frameon": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "lines.markersize": 4, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.4, "grid.alpha": 0.35, "hatch.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "figure.constrained_layout.use": True,  # fill figsize so output width = target width
})

# Okabe-Ito palette; one fixed colour per method / dataset in every figure.
M1, M2, M3 = "#0072B2", "#E69F00", "#009E73"
TQA, NQ, POPQA = "#D55E00", "#56B4E9", "#CC79A7"
PASS, FAIL = "#009E73", "#D55E00"
CORRECT, WRONG = "#009E73", "#CC79A7"
GREY = "#777777"
FM1, FM2, EPS = r"$f_{M1}$", r"$f_{M2}$", r"$\epsilon$"
ARROW = r"$\rightarrow$"

PREVIEW = None


def m(x):
    """Signed number placed in free text: typeset in maths so '-' is a true minus."""
    return f"${x}$"


CHECKS = []
_TEX_SPECIAL = re.compile(r"(?<!\\)([%&#])")


def tex_escape(fig):
    """Escape LaTeX specials (% & #) in every text of the figure, once, just before saving.
    Already-escaped ones (e.g. PercentFormatter's \\% under usetex) are left alone."""
    for t in fig.findobj(Text):
        s = t.get_text()
        if s:
            t.set_text(_TEX_SPECIAL.sub(r"\\\1", s))


def assert_ticks_escaped(fig, name):
    """Tick labels are regenerated from their formatter at draw time, so tex_escape cannot
    reach custom tick lists; those must be escaped at source. An unescaped % would make LaTeX
    silently drop the rest of the label, so check what was actually drawn."""
    for ax in fig.axes:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            assert not _TEX_SPECIAL.search(t.get_text()), f"{name}: unescaped tick label {t.get_text()!r}"


def chk(fig, label, got, exp, tol):
    ok = abs(float(got) - float(exp)) <= tol
    CHECKS.append((fig, label, float(got), exp, ok))
    print(f"  [{'ok' if ok else 'MISMATCH'}] Fig {fig:>2} {label}: data={got:.6g} paper={exp}")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    tex_escape(fig)
    fig.savefig(OUT / f"{name}.pdf")
    assert_ticks_escaped(fig, name)
    if PREVIEW:
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  saved {name}.pdf")


def qa_arrays(ds):
    gen, ent = load(CACHE / f"{ds}_generations.json"), load(CACHE / f"{ds}_entailment.json")
    assert len(gen) == len(ent) and all(g["idx"] == e["idx"] for g, e in zip(gen, ent))
    return (np.array([g["mean_logprob"] for g in gen]),
            np.array([e["entail_label"] for e in ent]),
            np.array([e["entail_score"] for e in ent]),
            np.array([e["fM2"] for e in ent]))


# ─────────────────────────── main-text figures ───────────────────────────

def fig01(sweep):
    """plot_results.plot_epsilon_sweep_validity; epsilon_sweep_results.json."""
    eps = sweep["epsilons"]
    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    exp = {"method1": [12.4, 0, 0, 0], "method2_optC": [22.0, 0, 0, 0], "method3": [68.8, 11.0, 0.2, 0]}
    for key, lab, mk, c in [("method1", "M1: SGen-Semi-style", "o", M1),
                            ("method2_optC", "M2: Conservative (Option C)", "s", M2),
                            ("method3", "M3: DS-SGen", "D", M3)]:
        v = [sweep[key][str(e)]["shifted_validity"] * 100 for e in eps]
        for e, a, b in zip(eps, v, exp[key]):
            chk(1, f"{key} validity% eps={e}", a, b, 0.05)
        ax.plot(eps, v, marker=mk, color=c, label=lab)
    ax.axhline(98, color="k", ls="--", lw=0.8, label="PAC target (98%)")
    ax.set_xlabel(f"{EPS} (FDR-E target)")
    ax.set_ylabel("Shifted-domain validity (%)")
    ax.set_ylim(0, 105); ax.set_xticks(eps); ax.grid(True)
    ax.legend(loc="center right")
    save(fig, "fig01_nq_validity_vs_eps")


def fig02(wq):
    """plot_synthetic_final.plot_weight_quartile; synthetic_final_weight_quartile.json."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 1.9), sharey=True)
    exp = {"synthetic": ([0.860, 0.825, 0.815, 0.760], 0.100),
           "tqa_nq": ([0.645, 0.717, 0.743, 0.729], -0.084)}
    for ax, (lab, key) in zip(axes, [("Synthetic (covariate)", "synthetic"),
                                     (f"TQA{ARROW}NQ (concept)", "tqa_nq")]):
        qs, spread = wq[key]["quartile_accs"], wq[key]["Q1_minus_Q4"]
        for i, (a, b) in enumerate(zip(qs, exp[key][0])):
            chk(2, f"{key} Q{i+1}", a, b, 0.0005)
        chk(2, f"{key} Q1-Q4", spread, exp[key][1], 0.0005)
        tag = "covariate" if spread >= 0.05 else "concept"
        bars = ax.bar(["Q1\n(source-like)", "Q2", "Q3", "Q4\n(target-like)"], qs,
                      color=PASS if spread >= 0.05 else FAIL, width=0.7)
        for bar, v in zip(bars, qs):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_title(f"{lab}\nQ1 $-$ Q4 = {m(f'{spread:+.3f}')} ({tag} signature)")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Source accuracy")
    save(fig, "fig02_weight_quartiles")


def _curves(rows, field):
    return ([r["epsilon"] for r in rows], [r["m1"][field] for r in rows], [r["m3"][field] for r in rows])


def fig03(sw):
    """plot_synthetic_final.plot_validity_vs_eps; synthetic_final_eps_sweep.json."""
    fig, ax = plt.subplots(figsize=(FULL, 2.3))
    e, m1, m3 = _curves(sw["synthetic"], "validity_rate")
    for a, b in zip(m1, [1, 1, 0.922, 0.996, 1]): chk(3, "M1 synth validity", a, b, 0.0005)
    for a, b in zip(m3, [1, 1, 1, 1, 1]): chk(3, "M3 synth validity", a, b, 0.0005)
    ax.plot(e, m1, "o-", color=M1, label="M1 synthetic (covariate)")
    ax.plot(e, m3, "s-", color=M3, label="M3 synthetic (covariate)")
    et, m1t, m3t = _curves(sw["tqa_nq"], "validity_rate")
    vac = [r["m3"]["vacuous_frac"] for r in sw["tqa_nq"]]
    for a, b in zip(m1t, [1, 1, 1, 0.982, 0.124]): chk(3, "M1 TQA-NQ validity", a, b, 0.0005)
    for a, b in zip(m3t, [1, 1, 1, 0.994, 0.688]): chk(3, "M3 TQA-NQ validity", a, b, 0.0005)
    for a, b in zip(vac, [1.00, 1.00, 1.00, 0.99, 0.69]): chk(3, "M3 TQA-NQ vac label", round(a, 2), b, 1e-9)
    ax.plot(et, m1t, "o--", color=M1, mfc="white", lw=1.0, label=f"M1 TQA{ARROW}NQ (concept)")
    ax.plot(et, m3t, "s--", color=M3, mfc="white", lw=1.0, label=f"M3 TQA{ARROW}NQ (concept)")
    for x, y, v in zip(et, m3t, vac):
        # last point: label to the right, clear of the incoming dashed line; others above
        off, ha, va = ((6, 0), "left", "center") if x == et[-1] else ((0, 5), "center", "bottom")
        ax.annotate(f"vac={v:.2f}", (x, y), xytext=off, textcoords="offset points",
                    fontsize=6, color=M3, ha=ha, va=va)
    ax.axhline(0.98, ls=":", color="k", lw=0.8, label=r"PAC target ($1-\delta=0.98$)")
    ax.set_xlabel(f"{EPS} (FDR-E target)"); ax.set_ylabel("Shifted validity rate")
    ax.set_xlim(0.035, 0.28)
    ax.set_ylim(0, 1.1); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_xticks(e); ax.grid(True)
    ax.legend(loc="lower left")
    save(fig, "fig03_synthetic_validity_vs_eps")


def fig04(wq, pop, nq, tqa):
    """No plotting code in the repo for this figure. Rebuilt from: T1/T2a/T2b/T3 for TQA->NQ
    recomputed from cached labels and f_M1 with screening.run_screening_tests' formulas;
    T4/T5/T6 TQA->NQ from synthetic_final_weight_quartile.json['tqa_nq'];
    PopQA from screening_popqa_results.json."""
    fm_nq, y_nq = nq[0], nq[1]
    y_tqa = tqa[1]
    k = max(1, len(fm_nq) // 20)
    t = wq["tqa_nq"]
    tn = [y_tqa.mean(), y_nq.mean(), y_nq[np.argsort(-fm_nq)[:k]].mean(),
          y_tqa.mean() - y_nq.mean(), t["acc_clf"], t["ess_ratio"], t["Q1_minus_Q4"]]
    pq = [pop[x] for x in ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio", "quartile_spread"]]
    for a, b in zip(tn, [0.708, 0.431, 0.761, 0.277, 0.917, 0.072, -0.084]): chk(4, "TQA-NQ", a, b, 0.0005)
    for a, b in zip(pq, [0.437, 0.241, 0.640, 0.196, 0.692, 0.423, -0.092]): chk(4, "PopQA", a, b, 0.0005)
    thr = [(0.80, None), (0.75, None), (0.80, None), (0.03, 0.15), (0.55, 0.78), (0.50, None), (0.05, None)]
    tn_pass = [v >= lo if hi is None else lo <= v <= hi for v, (lo, hi) in zip(tn, thr)]
    tn_flag = ["P" if p else "F" for p in tn_pass]
    pq_flag = ["P" if pop[f"pass_{s}"] else ("S" if pop.get(f"soft_{s}") else "F")
               for s in ["1", "2a", "2b", "3", "4", "5", "6"]]
    chk(4, "TQA-NQ number of P", tn_flag.count("P"), 0, 0)
    chk(4, "PopQA T4 is P, T5 is S (soft 0.35 <= 0.423 < 0.50)",
        (pq_flag[4] == "P") + (pq_flag[5] == "S") + sum(f == "F" for f in pq_flag), 7, 0)
    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    x, w = np.arange(7), 0.36
    for off, vals, fl, c, lab in [(-w / 2, tn, tn_flag, NQ, f"TQA{ARROW}NQ"),
                                  (w / 2, pq, pq_flag, POPQA, f"PopQA head{ARROW}tail")]:
        ax.bar(x + off, vals, w, color=c, label=lab)
        _flag_labels(ax, x + off, vals, fl)
    _thresholds(ax, thr)
    ax.set_xticks(x, SCREEN_TICKS)
    ax.set_ylabel("Observed value"); ax.set_ylim(-0.2, 1.0)
    ax.legend(handles=ax.get_legend_handles_labels()[0] + _threshold_handles() + [_KEY],
              labels=[f"TQA{ARROW}NQ", f"PopQA head{ARROW}tail", "pass threshold", "pass band",
                      "P / F / S = pass / fail / soft pass"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, columnspacing=1.2)
    save(fig, "fig04_screening_scorecard")


SCREEN_THR = [(0.80, None), (0.75, None), (0.80, None), (0.03, 0.15), (0.55, 0.78), (0.50, None), (0.05, None)]
SCREEN_TICKS = ["T1\nacc$_S$", "T2a\nacc$_T$", "T2b\ntop-5\\%", "T3\ngap", "T4\nclf",
                "T5\nESS/$n$", "T6\nQ1$-$Q4"]  # % pre-escaped: tick text bypasses tex_escape
_KEY = plt.Line2D([], [], ls="none")


def _thresholds(ax, thr):
    """Pass thresholds (dashed) and bands (T3, T4) of Eq. (13), shared by Figs 4 and 20."""
    assert thr == SCREEN_THR
    for i, (lo, hi) in enumerate(thr):
        if hi is None:
            ax.hlines(lo, i - 0.45, i + 0.45, colors=PASS, linestyles="--", lw=0.9)
        else:
            ax.fill_between([i - 0.45, i + 0.45], lo, hi, color=PASS, alpha=0.18, lw=0)
    ax.axhline(0, color="k", lw=0.5)


def _threshold_handles():
    return [plt.Line2D([], [], color=PASS, ls="--", lw=0.9), Patch(color=PASS, alpha=0.18)]


def _flag_labels(ax, xs, vals, flags, above_gap=0.015):
    """P/F/S letters: inside tall bars, above short bars and above T3 (index 3) so they clear
    the [0.03, 0.15] band, below negative bars."""
    for i, (xi, v, f) in enumerate(zip(xs, vals, flags)):
        if v < 0:
            y, va = v - 0.015, "top"
        elif v > 0.15 and i != 3:
            y, va = v - 0.02, "top"
        else:
            y, va = v + above_gap, "bottom"
        ax.text(xi, y, rf"\textbf{{{f}}}", ha="center", va=va, fontsize=6.5)


# ─────────────────────────── appendix C / E: dataset-level ───────────────────────────

FM1_XLIM = (-0.5, 0.01)  # Figs 5 and 15(a); asserted to contain every f_M1 value in main()


def fig05(nq, tqa):
    """plot_results.plot_fm1_histograms; nq/tqa_generations.json."""
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bins = np.linspace(-1.0, 0.0, 50)
    for a, c, lab, mexp in [(nq[0], NQ, "NQ", -0.088), (tqa[0], TQA, "TQA", -0.057)]:
        chk(5, f"{lab} mean fM1", a.mean(), mexp, 0.0005); chk(5, f"{lab} n", len(a), 3610, 0)
        ax.hist(a, bins=bins, alpha=0.6, color=c, density=True, label=f"{lab} ($n$={len(a)})")
    for a, c, lab in [(nq[0], NQ, "NQ"), (tqa[0], TQA, "TQA")]:
        ax.axvline(a.mean(), color=c, ls="--", lw=1.0, label=f"{lab} mean = {m(f'{a.mean():.3f}')}")
    ax.set_xlim(*FM1_XLIM)
    ax.set_xlabel(f"{FM1} (mean log-probability)"); ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    save(fig, "fig05_fm1_histogram")


def fig06(nq, tqa):
    """plot_results.plot_fm1_cdf_comparison (y = linspace(0,1,n), as in the original)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    for a, c, lab in [(nq[0], NQ, "NQ"), (tqa[0], TQA, "TQA")]:
        s = np.sort(a)
        ax.plot(s, np.linspace(0, 1, len(s)), color=c, label=f"{lab} ($n$={len(s)})")
    ax.set_xlabel(f"{FM1} (mean log-probability)"); ax.set_ylabel("Cumulative probability")
    ax.grid(True); ax.legend(loc="upper left")
    save(fig, "fig06_fm1_ecdf")


def fig07(nq, tqa):
    """plot_results.plot_entailment_scores; nq/tqa_entailment.json."""
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bins = np.linspace(0, 1, 50)
    for a, c, lab in [(nq[2], NQ, "NQ"), (tqa[2], TQA, "TQA")]:
        ax.hist(a, bins=bins, alpha=0.6, color=c, density=True, label=f"{lab} ($n$={len(a)})")
    ax.set_xlabel("Entailment score $P(\\mathrm{ENT})$"); ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    save(fig, "fig07_entailment_histogram")


def fig08(nq, tqa):
    """plot_results.plot_correctness_by_fm1_decile (argsort deciles of n//10)."""
    fig, axes = plt.subplots(2, 1, figsize=(FULL, 3.6), sharey=True)
    exp = {"NQ": [16, 24, 31, 35, 43, 42, 48, 56, 61, 75], "TQA": [33, 53, 66, 69, 71, 78, 83, 83, 86, 87]}
    for ax, (fm, y), c, lab in [(axes[0], nq[:2], NQ, "NQ"), (axes[1], tqa[:2], TQA, "TQA")]:
        order, n = np.argsort(fm), len(fm)
        size = n // 10
        rates = []
        for d in range(10):
            idx = order[d * size:(d + 1) * size if d < 9 else n]
            rates.append(y[idx].mean())
        ticks = [f"D{d+1}" for d in range(10)]
        for d, (r, e) in enumerate(zip(rates, exp[lab])):
            chk(8, f"{lab} D{d+1} %", round(r * 100), e, 0)
        bars = ax.bar(range(10), rates, color=c, edgecolor="k", lw=0.4)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.015, f"{r:.0%}", ha="center", fontsize=6.5)
        ax.set_xticks(range(10), ticks)
        ax.set_ylim(0, 1.1); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel(f"{lab} correctness")
    axes[1].set_xlabel(f"{FM1} decile (low $\\rightarrow$ high confidence)")
    save(fig, "fig08_correctness_by_fm1_decile")


# ─────────────────────────── appendix D: method comparison ───────────────────────────

def fig09(sweep):
    """plot_results.plot_three_method_comparison; epsilon_sweep_results.json (eps 0.25 and 0.35)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.2))
    x, w, cols = np.arange(3), 0.35, [M1, M2, M3]
    keys = ["method1", "method2_optC", "method3"]
    exp_v = {0.25: [12, 22, 69], 0.35: [0, 0, 0]}
    exp_e = {0.25: [22.9, 18.5, 8.1], 0.35: [88.0, 85.9, 80.0]}
    for i, eps in enumerate([0.25, 0.35]):
        vals = [sweep[k][str(eps)]["shifted_validity"] * 100 for k in keys]
        effs = [sweep[k][str(eps)]["shifted_mean_efficiency"] * 100 for k in keys]
        alpha = 1.0 if i == 0 else 0.45
        for ax, data, fmt, show in [(a1, vals, "{:.0f}%", lambda v: True), (a2, effs, "{:.1f}%", lambda v: v > 0.5)]:
            bars = ax.bar(x + (i - 0.5) * w, data, w, color=cols, alpha=alpha, edgecolor="k", lw=0.4)
            for b, v in zip(bars, data):
                if show(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, fmt.format(v), ha="center", fontsize=6.5)
        for v, e in zip(vals, exp_v[eps]): chk(9, f"validity label eps={eps}", round(v), e, 0)
        for v, e in zip(effs, exp_e[eps]): chk(9, f"efficiency label eps={eps}", round(v, 1), e, 1e-9)
    a1.axhline(98, color="k", ls="--", lw=0.8)
    a1.text(2.55, 99.5, "PAC target 98%", ha="right", va="bottom", fontsize=6.5)
    a1.set_ylabel("Shifted-domain validity (%)"); a1.set_ylim(0, 110)
    a2.set_ylabel("Shifted-domain efficiency (%)"); a2.set_ylim(0, 100)
    labels = ["M1\nSGen-Semi-style", "M2\nConservative", "M3\nDS-SGen"]
    handles = [Patch(facecolor=GREY, edgecolor="k", lw=0.4, label=f"{EPS} = 0.25"),
               Patch(facecolor=GREY, alpha=0.45, edgecolor="k", lw=0.4, label=f"{EPS} = 0.35")]
    for ax in (a1, a2):
        ax.set_xticks(x, labels)
    a1.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.8))
    save(fig, "fig09_three_method_comparison")


def fig10(base):
    """plot_results.plot_fdr_distribution; baseline_results.json (M1 only, TQA vs NQ)."""
    eps = base["config"]["epsilon"]
    idf = np.array([s["indomain_test"]["fdr_e"] for s in base["per_split"]])
    shf = np.array([s["shifted_test"]["fdr_e"] for s in base["per_split"]])
    chk(10, "TQA valid %", round(100 * np.mean(idf <= eps)), 100, 0)
    chk(10, "NQ valid %", round(100 * np.mean(shf <= eps)), 12, 0)
    chk(10, "NQ splits with FDR-E = 0", int((shf == 0).sum()), 62, 0)
    chk(10, "NQ splits n_selected = 0", sum(s["shifted_test"]["n_selected"] == 0 for s in base["per_split"]), 62, 0)
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bins = np.linspace(0, 0.6, 40)
    ax.hist(idf, bins=bins, alpha=0.6, color=TQA, label=f"TQA in-domain (valid = {np.mean(idf <= eps):.0%})")
    ax.hist(shf, bins=bins, alpha=0.6, color=NQ, label=f"NQ shifted (valid = {np.mean(shf <= eps):.0%})")
    ax.axvline(eps, color="k", ls="--", lw=0.9, label=f"{EPS} = {eps}")
    nv = _vacuous(base, 10)
    ax.annotate(f"{nv} vacuous splits\n(both domains,\nbars overlap)", (0.008, nv), xytext=(0.03, 92),
                fontsize=6, va="bottom", arrowprops=dict(arrowstyle="->", lw=0.5, color="k"))
    ax.set_xlabel("FDR-E (M1, per split)"); ax.set_ylabel("Count (of 500 splits)")
    ax.legend(loc="upper right")
    save(fig, "fig10_fdr_distribution")


def _vacuous(base, fig):
    """Splits where M1 finds no tau1: n_selected = 0 on BOTH TQA and NQ, so FDR-E = 0 and
    efficiency = 0 in both domains -> the two histograms overlap in the first bin."""
    ps = base["per_split"]
    v_id = {i for i, s in enumerate(ps) if s["indomain_test"]["n_selected"] == 0}
    v_sh = {i for i, s in enumerate(ps) if s["shifted_test"]["n_selected"] == 0}
    v_tau = {i for i, s in enumerate(ps) if s["tau1"] is None}
    chk(fig, "vacuous splits identical in TQA, NQ and tau1=None", (v_id == v_sh == v_tau) * len(v_id), 62, 0)
    assert all(ps[i][d]["fdr_e"] == 0 and ps[i][d]["efficiency"] == 0
               for i in v_id for d in ("indomain_test", "shifted_test"))
    return len(v_id)


def fig11(base):
    """plot_results.plot_efficiency_distribution; baseline_results.json (M1 only)."""
    ide = np.array([s["indomain_test"]["efficiency"] for s in base["per_split"]])
    she = np.array([s["shifted_test"]["efficiency"] for s in base["per_split"]])
    chk(11, "TQA mean eff %", round(100 * ide.mean(), 1), 40.8, 1e-9)
    chk(11, "NQ mean eff %", round(100 * she.mean(), 1), 22.9, 1e-9)
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bins = np.linspace(0, 1, 30)
    ax.hist(ide, bins=bins, alpha=0.6, color=TQA, label=f"TQA in-domain (mean = {ide.mean():.1%})")
    ax.hist(she, bins=bins, alpha=0.6, color=NQ, label=f"NQ shifted (mean = {she.mean():.1%})")
    nv = _vacuous(base, 11)
    ax.annotate(f"{nv} vacuous splits\n(both domains,\nbars overlap)", (0.017, nv), xytext=(0.03, 96),
                fontsize=6, va="bottom", arrowprops=dict(arrowstyle="->", lw=0.5, color="k"))
    ax.set_ylim(0, 135)
    ax.set_xlabel("Efficiency (M1, fraction answered)"); ax.set_ylabel("Count (of 500 splits)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(loc="upper right")
    save(fig, "fig11_efficiency_distribution")


def fig12(iw):
    """plot_results.plot_weight_analysis; importance_weighted_results.json."""
    d, ws, ps = iw["diagnostics"], iw["diagnostics"]["weight_stats"], iw["per_split"]
    for lab, a, b, tol in [("n", ws["n"], 3610, 0), ("n_eff", ws["n_eff"], 1112.5, 0.05),
                           ("n_eff ratio %", 100 * ws["n_eff_ratio"], 30.8, 0.05),
                           ("clip pctl", ws["clip_percentile"], 95, 0), ("clip value", ws["clip_value"], 1.403, 0.0005),
                           ("min", ws["weight_min"], 0.041, 0.0005), ("median", ws["weight_median"], 0.332, 0.0005),
                           ("max", ws["weight_max"], 5.692, 0.0005), ("std", ws["weight_std"], 1.498, 0.0005),
                           ("raw max", ws["raw_weight_max"], 32.659, 0.0005),
                           ("CV acc", d["classifier_cv_accuracy"], 0.917, 0.0005)]:
        chk(12, lab, a, b, tol)
    neff = np.array([r["n_eff_total"] for r in ps])
    eff = np.array([r["shifted_test"]["efficiency"] for r in ps])
    n_abs = sum(r["shifted_test"]["n_selected"] == 0 for r in ps)
    n_va = sum(r["shifted_test"]["valid"] and r["shifted_test"]["n_selected"] > 0 for r in ps)
    n_inv = sum(not r["shifted_test"]["valid"] for r in ps)
    for lab, a, b, tol in [("ESS mean", neff.mean(), 584.3, 0.05), ("vacuous", n_abs, 344, 0),
                           ("valid answering", n_va, 0, 0), ("invalid", n_inv, 156, 0),
                           ("eff mean", eff.mean(), 0.081, 0.0005)]:
        chk(12, lab, a, b, tol)
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 3.8), gridspec_kw={"width_ratios": [0.8, 1.2]})
    ax = axes[0, 0]; ax.axis("off")
    n = ws["n"]
    # importance_weighted.compute_importance_weights: raw odds p/(1-p), capped at the 95th pct of
    # the raw odds, then rescaled so the weights sum to n; min/median/max/std are post-rescaling.
    txt = (f"Weights on the {n} TQA questions\n"
           f"raw max odds = {ws['raw_weight_max']:.3f}\n"
           f"cap = {ws['clip_percentile']}th pct of raw odds = {ws['clip_value']:.3f}\n"
           f"after normalisation ($\\Sigma\\hat{{w}}$ = {n}):\n"
           f"\\quad min {ws['weight_min']:.3f}, median {ws['weight_median']:.3f},\n"
           f"\\quad max {ws['weight_max']:.3f}, std {ws['weight_std']:.3f}\n"
           f"$n_{{\\mathrm{{eff}}}}$ = {ws['n_eff']:.1f} ({100*ws['n_eff_ratio']:.1f}% of {n})\n"
           f"domain classifier CV acc = {d['classifier_cv_accuracy']:.3f}")
    ax.text(0.0, 0.45, txt, transform=ax.transAxes, va="center", ha="left", fontsize=7, linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="#F2F2F2", edgecolor=GREY, lw=0.5))
    ax.set_title("(a) Weight diagnostics", loc="left")
    ax = axes[0, 1]
    ax.hist(neff, bins=30, color=M3, alpha=0.75, edgecolor="k", lw=0.3)
    ax.axvline(neff.mean(), color="k", ls="--", lw=0.9, label=f"mean = {neff.mean():.1f}")
    ax.set_xlabel(r"$n_{\mathrm{eff}}$ of $Z_U$ (per split, $|Z_U|=1895$)"); ax.set_ylabel("Count")
    ax.set_title("(b) Effective sample size across splits", loc="left"); ax.legend(loc="upper left")
    ax = axes[1, 0]
    vals = [n_abs, n_va, n_inv]
    bars = ax.bar(["Abstain\n(vacuous)", "Valid\n(answering)", "Invalid"], vals,
                  color=[GREY, M3, WRONG], edgecolor="k", lw=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(v), ha="center", fontsize=7)
    ax.set_ylabel(f"Splits (of {len(ps)})"); ax.set_ylim(0, 400)
    ax.set_title(f"(c) M3 split outcomes on NQ, {EPS} = 0.25", loc="left")
    ax = axes[1, 1]
    ax.hist(eff, bins=30, color=M3, alpha=0.75, edgecolor="k", lw=0.3)
    ax.axvline(eff.mean(), color="k", ls="--", lw=0.9, label=f"mean = {eff.mean():.3f}")
    ax.set_xlabel("M3 efficiency on NQ (per split)"); ax.set_ylabel("Count")
    ax.set_title("(d) Selection efficiency across splits", loc="left"); ax.legend(loc="upper right")
    save(fig, "fig12_m3_weight_diagnostics")


def fig13(base, nq):
    """plot_results.plot_abstention_analysis; baseline_results.json (M1 median tau1) + NQ cache.
    Question-level: all 3610 NQ questions thresholded at the median M1 tau1."""
    fm, y = nq[0], nq[1]
    tau = np.median([s["tau1"] for s in base["per_split"] if s["tau1"] is not None])
    sel = fm >= tau
    sc, sw_ = int((sel & (y == 1)).sum()), int((sel & (y == 0)).sum())
    ac, aw = int((~sel & (y == 1)).sum()), int((~sel & (y == 0)).sum())
    ns, na, n = sc + sw_, ac + aw, len(y)
    for lab, a, b, tol in [("median tau1", tau, -0.0356, 0.00005), ("selected", ns, 950, 0), ("sel correct", sc, 622, 0),
                           ("sel wrong", sw_, 328, 0), ("abstained", na, 2660, 0), ("abs correct", ac, 934, 0),
                           ("abs wrong", aw, 1726, 0), ("overall acc %", round(100 * y.mean(), 1), 43.1, 1e-9),
                           ("sel %", round(100 * ns / n, 1), 26.3, 1e-9), ("sel acc %", round(100 * sc / ns, 1), 65.5, 1e-9),
                           ("abs acc %", round(100 * ac / na, 1), 35.1, 1e-9)]:
        chk(13, lab, a, b, tol)
    fig, (ax, at) = plt.subplots(1, 2, figsize=(FULL, 2.3), gridspec_kw={"width_ratios": [1.5, 1]})
    ax.bar(["Selected", "Abstained"], [sc, ac], color=CORRECT, label="Correct")
    ax.bar(["Selected", "Abstained"], [sw_, aw], bottom=[sc, ac], color=WRONG, label="Wrong")
    for i, (c, tot) in enumerate([(sc, ns), (ac, na)]):
        ax.text(i, tot + 40, f"$n$={tot}\n({c/tot:.0%} correct)", ha="center", va="bottom", fontsize=6.5)
    ax.set_ylabel("Number of NQ questions"); ax.set_ylim(0, 3200); ax.legend(loc="upper left")
    ax.set_title(f"NQ (shifted), $\\hat\\tau_1$ = {m(f'{tau:.4f}')} (median M1 threshold)", loc="left")
    at.axis("off")
    txt = (f"NQ (shifted); threshold = median M1 $\\hat\\tau_1$ = {m(f'{tau:.4f}')}\n\n"
           f"Selected: {ns}/{n} ({ns/n:.1%})\n"
           f"\\quad Correct: {sc}/{ns} ({sc/ns:.1%})\n"
           f"\\quad Wrong: {sw_}/{ns} ({sw_/ns:.1%})  $\\leftarrow$ FDR\n\n"
           f"Abstained: {na}/{n} ({na/n:.1%})\n"
           f"\\quad Correct: {ac}/{na} ({ac/na:.1%})  $\\leftarrow$ wasted\n"
           f"\\quad Wrong: {aw}/{na} ({aw/na:.1%})\n\n"
           f"Overall accuracy: {y.mean():.1%}\n"
           f"Accuracy if selected: {sc/ns:.1%}\n"
           f"Accuracy if abstained: {ac/na:.1%}")
    at.text(0.02, 0.5, txt, transform=at.transAxes, va="center", fontsize=7, linespacing=1.3,
            bbox=dict(boxstyle="round", facecolor="#F2F2F2", edgecolor=GREY, lw=0.5))
    save(fig, "fig13_abstention_analysis")


def fig14(nq, tqa):
    """plot_results.plot_correctness_rate_by_domain."""
    r = [nq[1].mean(), tqa[1].mean()]
    chk(14, "NQ acc %", round(100 * r[0], 1), 43.1, 1e-9); chk(14, "TQA acc %", round(100 * r[1], 1), 70.8, 1e-9)
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    bars = ax.bar(["NQ", "TQA"], r, color=[NQ, TQA], width=0.6)
    for b, v in zip(bars, r):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=7)
    ax.set_ylabel("Correctness rate (entailment)"); ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    save(fig, "fig14_correctness_by_dataset")


def fig15(nq, tqa):
    """plot_results.plot_domain_shift_diagnostic (2x2)."""
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.0))
    ax = axes[0, 0]
    b = np.linspace(-1, 0, 40)
    ax.hist(nq[0], bins=b, alpha=0.6, color=NQ, label="NQ", density=True)
    ax.hist(tqa[0], bins=b, alpha=0.6, color=TQA, label="TQA", density=True)
    ax.set_xlabel(f"{FM1} (mean log-probability)"); ax.set_ylabel("Density")
    ax.set_xlim(*FM1_XLIM)
    ax.set_title("(a) Confidence distributions", loc="left"); ax.legend(loc="upper left")
    ax = axes[0, 1]

    def wilson(p, n, z=1.96):
        den = 1 + z**2 / n
        c = (p + z**2 / (2 * n)) / den
        s = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
        return c - s, c + s
    r = [nq[1].mean(), tqa[1].mean()]
    ci = [wilson(r[0], len(nq[1])), wilson(r[1], len(tqa[1]))]
    bars = ax.bar(["NQ", "TQA"], r, color=[NQ, TQA], width=0.6)
    ax.errorbar([0, 1], r, yerr=[[r[0] - ci[0][0], r[1] - ci[1][0]], [ci[0][1] - r[0], ci[1][1] - r[1]]],
                fmt="none", color="k", capsize=3, lw=0.8)
    for bb, v in zip(bars, r):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.04, f"{v:.1%}", ha="center", fontsize=7)
    print(f"    Fig 15 Wilson 95% CI: NQ [{ci[0][0]:.4f}, {ci[0][1]:.4f}], TQA [{ci[1][0]:.4f}, {ci[1][1]:.4f}]")
    chk(15, "NQ acc %", round(100 * r[0], 1), 43.1, 1e-9); chk(15, "TQA acc %", round(100 * r[1], 1), 70.8, 1e-9)
    ax.set_ylabel("Correctness rate"); ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("(b) Model accuracy (95% Wilson CI)", loc="left")
    ax = axes[1, 0]
    b2 = np.linspace(0, 1, 20)
    ax.hist(nq[3], bins=b2, alpha=0.6, color=NQ, label="NQ", density=True)
    ax.hist(tqa[3], bins=b2, alpha=0.6, color=TQA, label="TQA", density=True)
    ax.set_xlabel(f"{FM2} (self-consistency)"); ax.set_ylabel("Density")
    ax.set_title("(c) Self-consistency distributions", loc="left"); ax.legend(loc="upper left")
    ax = axes[1, 1]
    corr = [np.corrcoef(nq[0], nq[1])[0, 1], np.corrcoef(tqa[0], tqa[1])[0, 1],
            np.corrcoef(nq[3], nq[1])[0, 1], np.corrcoef(tqa[3], tqa[1])[0, 1]]
    for lab, a, e in zip(["NQ fM1", "TQA fM1", "NQ fM2", "TQA fM2"], corr, [0.318, 0.349, 0.348, 0.370]):
        chk(15, f"Pearson r {lab}", a, e, 0.0005)
    names = [f"NQ: {FM1}", f"TQA: {FM1}", f"NQ: {FM2}", f"TQA: {FM2}"]
    bars = ax.barh(range(4), corr, color=[NQ, TQA, NQ, TQA], edgecolor="k", lw=0.4)
    for bb, h in zip(bars, ["", "", "////", "////"]):
        bb.set_hatch(h)
    for i, v in enumerate(corr):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(4), names); ax.set_xlim(-0.1, max(corr) + 0.15)
    ax.set_xlabel("Pearson correlation with correctness")
    ax.set_title("(d) Feature-correctness correlations", loc="left")
    save(fig, "fig15_domain_shift_diagnostic")


def fig16(nq, tqa):
    """plot_results.plot_fm1_roc_curve: original 500-threshold subsample + trapezoid AUC."""
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(HALF, 2.6))
    for (fm, y), c, lab, e in [(nq[:2], NQ, "NQ", 0.695), (tqa[:2], TQA, "TQA", 0.707)]:
        th = np.sort(np.unique(fm))
        if len(th) > 500:
            th = th[np.linspace(0, len(th) - 1, 500, dtype=int)]
        npos, nneg = y.sum(), len(y) - y.sum()
        tpr = np.array([((fm >= t) & (y == 1)).sum() / npos for t in th])
        fpr = np.array([((fm >= t) & (y == 0)).sum() / nneg for t in th])
        o = np.argsort(fpr)
        fpr, tpr = fpr[o], tpr[o]
        auc = np.trapezoid(tpr, fpr)
        chk(16, f"{lab} AUC (original 500-threshold method)", round(auc, 3), e, 1e-9)
        print(f"    Fig 16 {lab}: exact sklearn AUC = {roc_auc_score(y, fm):.4f}")
        ax.plot(fpr, tpr, color=c, label=f"{lab} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="Random (AUC = 0.500)")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_aspect("equal"); ax.grid(True); ax.legend(loc="lower right")
    save(fig, "fig16_fm1_roc")


def fig17(nq, tqa):
    """plot_results.plot_calibration_curve (equal-frequency deciles of f_M1)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.2))
    for (fm, y), c, mk, lab in [(nq[:2], NQ, "o", "NQ"), (tqa[:2], TQA, "s", "TQA")]:
        edges = np.percentile(fm, np.linspace(0, 100, 11))
        xs, ys, ns = [], [], []
        for i in range(10):
            m = (fm >= edges[i]) & ((fm < edges[i + 1]) if i < 9 else (fm <= edges[i + 1]))
            if m.sum():
                xs.append(fm[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
        print(f"    Fig 17 {lab} bin counts: {ns}")
        chk(17, f"{lab} number of bins", len(ns), 10, 0)
        for k in ns:
            chk(17, f"{lab} bin count", k, 361, 0)
        assert len(set(ns)) == 1
        ax.plot(xs, ys, f"-{mk}", color=c, label=f"{lab} ($n$={len(fm)}; {len(ns)} bins of {ns[0]})")
    ax.set_xlabel(f"{FM1} (mean log-probability, decile mean)"); ax.set_ylabel("Observed correctness rate")
    ax.set_ylim(-0.02, 1.02); ax.grid(True); ax.legend(loc="upper left")
    save(fig, "fig17_calibration_curve")


def fig18(cons):
    """plot_results.plot_validity_efficiency_tradeoff; conservative_results.json.
    Layout fix: y-axis 0-1.05 (original 0.5-1.05 hid the 0.124-0.22 points)."""
    fig, ax = plt.subplots(figsize=(HALF, 2.4))
    # A, B and C share points (identity settings = M1; A/B non-trivial settings all at eff 0,
    # validity 1), so markers are hollow at decreasing sizes and coincident labels are merged.
    cfg = [("option_a", r"A: safety factor $\gamma$", "o", "#0072B2", 46),
           ("option_b", r"B: reduced $\epsilon/\kappa$", "s", "#E69F00", 22),
           ("option_c", r"C: $\delta$-budget frac", "^", "#CC79A7", 14)]
    # Table 6 (validity, efficiency). The original report printed 0.214 / 0.201 for frac
    # 0.25 / 0.50, which conservative_results.json does not support; the paper now prints 0.221 / 0.211.
    exp_c = {"0.0": (0.124, 0.229), "0.25": (0.138, 0.221), "0.5": (0.154, 0.211), "0.75": (0.220, 0.185)}
    allv = []
    for key, lab, mk, c, size in cfg:
        items = sorted(cons[key].items())
        e = [s["shifted"]["mean_efficiency"] for _, s in items]
        v = [s["shifted"]["validity_rate"] for _, s in items]
        allv += v
        ax.plot(e, v, ls="--", color=c, alpha=0.5, lw=0.8)
        ax.scatter(e, v, marker=mk, s=size, facecolors="none" if key != "option_c" else c,
                   edgecolors=c, lw=0.9, label=lab, zorder=5)
        if key == "option_c":
            for (p, s) in items:
                chk(18, f"C frac={p} validity", s["shifted"]["validity_rate"], exp_c[p][0], 0.0005)
                chk(18, f"C frac={p} efficiency (Table 6)", s["shifted"]["mean_efficiency"], exp_c[p][1], 0.0005)
                # frac labels placed in empty space (data coords) with leader lines
                pos = {"0.75": (0.13, 0.31), "0.5": (0.155, 0.05), "0.25": (0.222, 0.02)}
                if p in pos:
                    ax.annotate(f"frac {p}" if p == "0.75" else p,
                                (s["shifted"]["mean_efficiency"], s["shifted"]["validity_rate"]),
                                xytext=pos[p], textcoords="data", fontsize=6, ha="center", va="bottom",
                                color=c, arrowprops=dict(arrowstyle="-", lw=0.5, color=c, shrinkB=3))
    a, b = cons["option_a"], cons["option_b"]
    assert all(a[k]["shifted"]["mean_efficiency"] == 0 and a[k]["shifted"]["validity_rate"] == 1
               for k in a if k != "1.0")
    assert all(b[k]["shifted"]["mean_efficiency"] == 0 and b[k]["shifted"]["validity_rate"] == 1
               for k in b if k != "1.0")
    ax.annotate(r"A $\gamma$ = 1.2, 1.5, 2.0; B $\kappa$ = 1.5, 2, 3, 4", (0, 1.0), xytext=(8, -12),
                textcoords="offset points", fontsize=6)
    ax.annotate("A $\\gamma$=1.0, B $\\kappa$=1.0,\nC frac 0.0 (all = M1)", (a["1.0"]["shifted"]["mean_efficiency"],
                a["1.0"]["shifted"]["validity_rate"]), xytext=(0.255, 0.25), textcoords="data",
                fontsize=6, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY, shrinkB=5))
    ax.axhline(0.98, color="k", ls="--", lw=0.8, label="PAC target (98%)")
    assert min(allv) >= 0 and max(allv) <= 1.05
    ax.set_xlabel("NQ efficiency (fraction answered)"); ax.set_ylabel("NQ validity rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(-0.02, 0.38); ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.84))  # below the A/B label at y~0.9
    save(fig, "fig18_m2_validity_efficiency")


def fig19(base):
    """plot_results.plot_threshold_stability; baseline_results.json (M1 tau1)."""
    t = np.array([s["tau1"] for s in base["per_split"] if s["tau1"] is not None])
    iqr = np.percentile(t, 75) - np.percentile(t, 25)
    for lab, a, e in [("mean", t.mean(), -0.0352), ("median", np.median(t), -0.0356),
                      ("std", np.std(t), 0.0106), ("IQR", iqr, 0.0128)]:
        chk(19, lab, a, e, 0.00005)
    chk(19, "n splits with tau1 (62 of 500 have none)", len(t), 438, 0)
    fig, ax = plt.subplots(figsize=(HALF, 2.0))
    ax.hist(t, bins=30, color="#999999", edgecolor="k", lw=0.3)
    ax.axvline(t.mean(), color=TQA, ls="--", lw=1.0, label=f"Mean = {m(f'{t.mean():.4f}')}")
    ax.axvline(np.median(t), color=M1, ls=":", lw=1.2, label=f"Median = {m(f'{np.median(t):.4f}')}")
    ax.text(0.98, 0.97, f"Std = {np.std(t):.4f}\nIQR = {iqr:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7)
    ax.set_xlabel(r"Learned threshold $\hat\tau_1$ ($f_{M1}$), M1"); ax.set_ylabel(f"Count ({len(t)} splits)")
    ax.legend(loc="upper left")
    save(fig, "fig19_tau1_stability")


def fig20(sc):
    """plot_synthetic_final.plot_scorecard; synthetic_final_screening.json."""
    keys = ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio", "quartile_spread"]
    tests = ["1", "2a", "2b", "3", "4", "5", "6"]
    vals = [sc[k] for k in keys]
    passes = [sc[f"pass_{t}"] for t in tests]
    for a, e in zip(vals, [0.815, 0.765, 0.925, 0.050, 0.690, 0.526, 0.100]): chk(20, "value", a, e, 0.0005)
    chk(20, "n pass", sum(passes), 7, 0)
    fig, ax = plt.subplots(figsize=(FULL, 2.1))
    x = np.arange(7)
    ax.bar(x, vals, color=[PASS if p else FAIL for p in passes], width=0.7)
    _thresholds(ax, SCREEN_THR)
    flags = ["P" if p else "F" for p in passes]
    for i, (v, f) in enumerate(zip(vals, flags)):
        txt = f"{v:+.3f}" if i == 6 else f"{v:.3f}"  # sign only on the T6 spread
        ax.text(i, v + 0.015, txt, ha="center", va="bottom", fontsize=7)
        if v > 0.15 and i != 3:  # P at mid-bar (clear of the threshold lines), else above the value label
            ax.text(i, v / 2, rf"\textbf{{{f}}}", ha="center", va="center", fontsize=6.5)
        else:
            ax.text(i, v + 0.09, rf"\textbf{{{f}}}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, SCREEN_TICKS)
    ax.set_ylabel("Observed value"); ax.set_ylim(0, 1.05)
    ax.legend(handles=_threshold_handles() + [_KEY],
              labels=["pass threshold", "pass band", f"P = pass (all {sum(passes)}/7)"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, "fig20_synthetic_scorecard")


def fig21(sw):
    """plot_synthetic_final.plot_efficiency_vs_eps; synthetic_final_eps_sweep.json."""
    fig, ax = plt.subplots(figsize=(FULL, 2.2))
    e, m1, m3 = _curves(sw["synthetic"], "mean_efficiency")
    et, m1t, m3t = _curves(sw["tqa_nq"], "mean_efficiency")
    for a, b in zip(m1, [0, 0, 0.079, 0.549, 0.829]): chk(21, "M1 synth eff", a, b, 0.0005)
    for a, b in zip(m3, [0, 0, 0.009, 0.137, 0.480]): chk(21, "M3 synth eff", a, b, 0.0005)
    for a, b in zip(m1t, [0, 0, 0, 0.002, 0.229]): chk(21, "M1 TQA-NQ eff", a, b, 0.0005)
    for a, b in zip(m3t, [0, 0, 0, 0.001, 0.081]): chk(21, "M3 TQA-NQ eff", a, b, 0.0005)
    ax.plot(e, m1, "o-", color=M1, label="M1 synthetic (covariate)")
    ax.plot(e, m3, "s-", color=M3, label="M3 synthetic (covariate)")
    ax.plot(et, m1t, "o--", color=M1, mfc="white", lw=1.0, label=f"M1 TQA{ARROW}NQ (concept)")
    ax.plot(et, m3t, "s--", color=M3, mfc="white", lw=1.0, label=f"M3 TQA{ARROW}NQ (concept)")
    ax.set_xlabel(f"{EPS} (FDR-E target)"); ax.set_ylabel("Mean shifted efficiency")
    ax.set_ylim(0, 1.0); ax.set_xticks(e); ax.grid(True); ax.legend(loc="upper left")
    save(fig, "fig21_synthetic_efficiency_vs_eps")


def main():
    global PREVIEW, OUT
    ap = argparse.ArgumentParser(description="Build the paper's Figures 1-21 from cached results.")
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help=f"directory for fig*.pdf (default: {OUT.relative_to(REPO)})")
    ap.add_argument("--preview", help="also write PNG previews to this directory (needs dvipng)")
    args = ap.parse_args()
    OUT, PREVIEW = args.out_dir.resolve(), args.preview
    missing = [t for t in ("latex", "dvips", "kpsewhich") if not shutil.which(t)]
    if missing:
        sys.exit(f"TeX tools not on PATH: {', '.join(missing)}. Figures are typeset with "
                 "text.usetex (XCharter + newtxmath); install a TeX distribution and retry.")
    if PREVIEW and not shutil.which("dvipng"):
        sys.exit("--preview needs dvipng for usetex PNGs (not installed); omit --preview")
    print(f"  output directory: {OUT}")
    sweep = load(RES / "epsilon_sweep_results.json")
    wq = load(RES / "synthetic_final_weight_quartile.json")
    sw = load(RES / "synthetic_final_eps_sweep.json")
    base = load(RES / "baseline_results.json")
    iw = load(RES / "importance_weighted_results.json")
    cons = load(RES / "conservative_results.json")
    pop = load(RES / "screening_popqa_results.json")
    sc = load(RES / "synthetic_final_screening.json")
    nq, tqa = qa_arrays("nq"), qa_arrays("tqa")
    lo = min(nq[0].min(), tqa[0].min()); hi = max(nq[0].max(), tqa[0].max())
    assert FM1_XLIM[0] < lo and hi <= 0 < FM1_XLIM[1], (lo, hi)  # xlim of Figs 5/15a hides no data
    print(f"  f_M1 range over both datasets: [{lo:.4f}, {hi:.4f}]")
    fig01(sweep); fig02(wq); fig03(sw); fig04(wq, pop, nq, tqa)
    fig05(nq, tqa); fig06(nq, tqa); fig07(nq, tqa); fig08(nq, tqa)
    fig09(sweep); fig10(base); fig11(base); fig12(iw); fig13(base, nq)
    fig14(nq, tqa); fig15(nq, tqa); fig16(nq, tqa); fig17(nq, tqa)
    fig18(cons); fig19(base); fig20(sc); fig21(sw)
    bad = [c for c in CHECKS if not c[4]]
    print(f"\n{len(CHECKS)} checks, {len(bad)} mismatches")
    for c in bad:
        print("  MISMATCH", c)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
