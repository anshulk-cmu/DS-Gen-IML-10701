"""Design A plots. Run AFTER run_synthetic_a.py.

Four figures:
  1. Alpha sweep: n_pass and T3/T4/T5/T6 trajectories vs alpha
  2. Screening scorecard (7/7 test pass/fail bars)
  3. Three-method comparison: validity + efficiency
  4. Per-split FDR-E distribution (M1 vs M3 overlaid)
  5. Per-(topic, tier) accuracy heatmap on the synthetic pool
"""

import json
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

CACHE = "/data/user_data/anshulk/dsgen/cache"
RESULTS = "/data/user_data/anshulk/dsgen/results"
OUT = "/home/anshulk/ds-gen-10701/plots"


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_alpha_sweep():
    sw = _load(f"{RESULTS}/synthetic_a_screening_sweep.json")["sweep"]
    alphas = [e["alpha"] for e in sw]
    n_pass = [e["n_pass"] for e in sw]
    t3 = [e["scorecard"]["gap"] for e in sw]
    t4 = [e["scorecard"]["acc_clf"] for e in sw]
    t5 = [e["scorecard"]["ess_ratio"] for e in sw]
    t6 = [e["scorecard"]["quartile_spread"] for e in sw]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    ax = axes[0]
    ax.bar(alphas, n_pass, width=0.04, color="#2a9d8f")
    ax.axhline(6, ls="--", color="red", lw=1, label="target (>=6)")
    ax.set_xlabel("alpha"); ax.set_ylabel("screening tests passed")
    ax.set_ylim(0, 8); ax.set_title("Screening tests passed vs alpha")
    ax.legend()

    ax = axes[1]
    ax.plot(alphas, t3, "o-", label="T3 gap", color="#e76f51")
    ax.plot(alphas, t4, "s-", label="T4 acc_clf", color="#4a90d9")
    ax.plot(alphas, t5, "^-", label="T5 ESS ratio", color="#7bb661")
    ax.plot(alphas, t6, "d-", label="T6 quartile spread", color="#9b59b6")
    ax.axhspan(0.55, 0.78, color="#4a90d9", alpha=0.1)
    ax.axhspan(0.03, 0.15, color="#e76f51", alpha=0.1)
    ax.set_xlabel("alpha"); ax.set_ylabel("test value")
    ax.set_title("Screening metrics vs alpha")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_alpha_sweep.png", dpi=140)
    plt.close()


def plot_scorecard():
    sc = _load(f"{RESULTS}/synthetic_a_screening.json")
    tests = ["1", "2a", "2b", "3", "4", "5", "6"]
    keys = ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio", "quartile_spread"]
    vals = [sc[k] for k in keys]
    passes = [sc[f"pass_{t}"] for t in tests]
    colors = ["#2a9d8f" if p else "#e76f51" for p in passes]
    plt.figure(figsize=(8, 3.5))
    bars = plt.bar(tests, vals, color=colors)
    for bar, v, p in zip(bars, vals, passes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:+.3f}", ha="center", fontsize=8)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("screening test"); plt.ylabel("value")
    n_pass = sum(passes)
    plt.title(f"Design A — Screening scorecard ({n_pass}/7 pass)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_scorecard.png", dpi=140)
    plt.close()


def plot_three_method():
    m1 = _load(f"{RESULTS}/synthetic_a_m1_results.json")
    m3 = _load(f"{RESULTS}/synthetic_a_m3_results.json")
    labels = ["M1 in-dom", "M1 shifted", "M3 in-dom", "M3 shifted"]
    colors = ["#4a90d9", "#4a90d9", "#7bb661", "#7bb661"]

    val = [m1["indomain"]["validity_rate"], m1["shifted"]["validity_rate"],
           m3["indomain"]["validity_rate"], m3["shifted"]["validity_rate"]]
    eff = [m1["indomain"]["mean_efficiency"], m1["shifted"]["mean_efficiency"],
           m3["indomain"]["mean_efficiency"], m3["shifted"]["mean_efficiency"]]
    fdr = [m1["indomain"]["mean_fdr_e"], m1["shifted"]["mean_fdr_e"],
           m3["indomain"]["mean_fdr_e"], m3["shifted"]["mean_fdr_e"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].bar(labels, val, color=colors); axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.98, ls="--", color="red", lw=1, label="PAC target")
    axes[0].set_title("Validity rate"); axes[0].legend(loc="lower right", fontsize=8)

    axes[1].bar(labels, eff, color=colors); axes[1].set_ylim(0, 1.0)
    axes[1].set_title("Efficiency")

    axes[2].bar(labels, fdr, color=colors); axes[2].set_ylim(0, 0.3)
    axes[2].axhline(0.25, ls="--", color="red", lw=1, label="epsilon=0.25")
    axes[2].set_title("Mean FDR-E"); axes[2].legend(loc="upper left", fontsize=8)

    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_three_method_comparison.png", dpi=140)
    plt.close()


def plot_fdr_dist():
    m1 = _load(f"{RESULTS}/synthetic_a_m1_results.json")
    m3 = _load(f"{RESULTS}/synthetic_a_m3_results.json")
    m1_fdr = [r["shifted_test"]["fdr_e"] for r in m1["per_split"]]
    m3_fdr = [r["shifted_test"]["fdr_e"] for r in m3["per_split"]]
    plt.figure(figsize=(7, 3))
    plt.hist(m1_fdr, bins=40, alpha=0.6, label=f"M1  mean={np.mean(m1_fdr):.3f}", color="#4a90d9")
    plt.hist(m3_fdr, bins=40, alpha=0.6, label=f"M3  mean={np.mean(m3_fdr):.3f}", color="#7bb661")
    plt.axvline(0.25, ls="--", color="red", lw=1, label="epsilon=0.25")
    plt.xlabel("FDR-E (shifted)"); plt.ylabel("count (splits)")
    plt.legend(fontsize=8)
    plt.title("Design A — Shifted FDR-E per split")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_fdr_distribution.png", dpi=140)
    plt.close()


def plot_pool_heatmap():
    records = _load(f"{CACHE}/synth_qa_data.json")
    ents = _load(f"{CACHE}/synth_qa_entailment.json")
    topics = sorted({r["topic"] for r in records})
    tiers = sorted({r["tier"] for r in records})
    mat = np.zeros((len(topics), len(tiers)))
    counts = np.zeros_like(mat, dtype=int)
    for r, e in zip(records, ents):
        i = topics.index(r["topic"]); j = tiers.index(r["tier"])
        mat[i, j] += e["entail_label"]
        counts[i, j] += 1
    acc = mat / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(acc, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"tier {t}" for t in tiers])
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels([t[:25] for t in topics])
    for i in range(len(topics)):
        for j in range(len(tiers)):
            ax.text(j, i, f"{acc[i,j]:.2f}\n(n={counts[i,j]})",
                    ha="center", va="center", fontsize=7,
                    color="black" if acc[i,j] > 0.6 else "white")
    plt.colorbar(im, ax=ax, label="entailment accuracy")
    plt.title("Design A — synthetic pool accuracy by (topic, tier)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_pool_heatmap.png", dpi=140)
    plt.close()


def plot_topic_hist():
    pair = _load(f"{CACHE}/synthetic_a_pair_indices.json")
    s = np.array(pair["source_topic_hist"])
    t = np.array(pair["target_topic_hist"])
    K = len(s)
    x = np.arange(K)
    plt.figure(figsize=(9, 3))
    plt.bar(x - 0.2, s, 0.4, label="source (A)", color="#2a9d8f")
    plt.bar(x + 0.2, t, 0.4, label="target (B)", color="#e76f51")
    plt.xlabel("K-means cluster id"); plt.ylabel("count")
    plt.legend()
    plt.title(f"Design A — cluster distribution (alpha={pair['alpha']}, K={K})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_a_topic_distribution.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_alpha_sweep()
    plot_scorecard()
    plot_three_method()
    plot_fdr_dist()
    plot_pool_heatmap()
    plot_topic_hist()
    print(f"plots written to {OUT}")
