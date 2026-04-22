"""Synthetic experiment plots. Run AFTER run_synthetic.py."""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

CACHE = "/data/user_data/anshulk/dsgen/cache"
RESULTS = "/data/user_data/anshulk/dsgen/results"
OUT = "/home/anshulk/ds-gen-10701/plots"


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_topic_hist():
    pair = _load(f"{CACHE}/synthetic_pair_indices.json")
    s = np.array(pair["source_topic_hist"])
    t = np.array(pair["target_topic_hist"])
    K = len(s)
    x = np.arange(K)
    plt.figure(figsize=(9, 3))
    plt.bar(x - 0.2, s, 0.4, label="source")
    plt.bar(x + 0.2, t, 0.4, label="target")
    plt.xlabel("topic id")
    plt.ylabel("count")
    plt.legend()
    plt.title(f"Synthetic topic distribution (alpha={pair['alpha']}, K={K})")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_topic_distribution.png", dpi=140)
    plt.close()


def plot_scorecard():
    sc = _load(f"{RESULTS}/synthetic_screening.json")
    tests = ["1", "2a", "2b", "3", "4", "5", "6"]
    vals = [sc["acc_S"], sc["acc_T"], sc["acc_top5"],
            sc["gap"], sc["acc_clf"], sc["ess_ratio"], sc["quartile_spread"]]
    passes = [sc[f"pass_{k}"] for k in tests]
    colors = ["#2a9d8f" if p else "#e76f51" for p in passes]
    plt.figure(figsize=(8, 3))
    plt.bar(tests, vals, color=colors)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("screening test")
    plt.ylabel("value")
    plt.title("Synthetic screening scorecard")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_scorecard.png", dpi=140)
    plt.close()


def plot_three_method():
    m1 = _load(f"{RESULTS}/synthetic_m1_results.json")
    m3 = _load(f"{RESULTS}/synthetic_m3_results.json")
    labels = ["M1 in-dom", "M1 shifted", "M3 in-dom", "M3 shifted"]
    vals_val = [m1["indomain"]["validity_rate"], m1["shifted"]["validity_rate"],
                m3["indomain"]["validity_rate"], m3["shifted"]["validity_rate"]]
    vals_eff = [m1["indomain"]["mean_efficiency"], m1["shifted"]["mean_efficiency"],
                m3["indomain"]["mean_efficiency"], m3["shifted"]["mean_efficiency"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].bar(labels, vals_val, color=["#4a90d9", "#4a90d9", "#7bb661", "#7bb661"])
    axes[0].set_title("Validity rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.98, ls="--", color="red", lw=1, label="PAC target")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].bar(labels, vals_eff, color=["#4a90d9", "#4a90d9", "#7bb661", "#7bb661"])
    axes[1].set_title("Efficiency")
    axes[1].set_ylim(0, 1.0)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_three_method_comparison.png", dpi=140)
    plt.close()


def plot_fdr_dist():
    m1 = _load(f"{RESULTS}/synthetic_m1_results.json")
    m3 = _load(f"{RESULTS}/synthetic_m3_results.json")
    m1_fdr = [r["shifted_test"]["fdr_e"] for r in m1["per_split"]]
    m3_fdr = [r["shifted_test"]["fdr_e"] for r in m3["per_split"]]
    plt.figure(figsize=(7, 3))
    plt.hist(m1_fdr, bins=40, alpha=0.6, label="M1", color="#4a90d9")
    plt.hist(m3_fdr, bins=40, alpha=0.6, label="M3", color="#7bb661")
    plt.axvline(0.25, ls="--", color="red", lw=1, label="epsilon = 0.25")
    plt.xlabel("FDR-E (shifted)")
    plt.ylabel("count (splits)")
    plt.legend()
    plt.title("Synthetic shifted FDR-E per split")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_fdr_distribution.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_topic_hist()
    plot_scorecard()
    plot_three_method()
    plot_fdr_dist()
    print(f"plots written to {OUT}")
