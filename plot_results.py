"""Visualization script for DS-SGen experiment results.

Generates all plots for the paper. Can be run incrementally — each plot function
checks whether the required data exists before plotting.

Usage:
    python plot_results.py                          # all available plots
    python plot_results.py --stage generation       # generation stats only
    python plot_results.py --stage baseline         # baseline results only
    python plot_results.py --stage conservative     # Method 2 results only
    python plot_results.py --stage all              # everything

Output: plots/ directory (PNG files, 300 DPI)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ds_sgen.utils import load_config, get_cache_path, load_cache


PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
NQ_COLOR = "#2196F3"   # Blue
TQA_COLOR = "#FF5722"  # Red-orange
METHOD_COLORS = {"A": "#4CAF50", "B": "#FF9800", "C": "#9C27B0"}


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Generation statistics (available after Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fm1_histograms(nq_gen, tqa_gen):
    """Plot 1: fM1 (mean log-prob) distribution comparison between NQ and TQA."""
    nq_fm1 = [r["mean_logprob"] for r in nq_gen]
    tqa_fm1 = [r["mean_logprob"] for r in tqa_gen]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 0.0, 50)
    ax.hist(nq_fm1, bins=bins, alpha=0.6, label=f"NQ (n={len(nq_fm1)})", color=NQ_COLOR, density=True)
    ax.hist(tqa_fm1, bins=bins, alpha=0.6, label=f"TQA (n={len(tqa_fm1)})", color=TQA_COLOR, density=True)
    ax.axvline(np.mean(nq_fm1), color=NQ_COLOR, linestyle="--", linewidth=1.5, label=f"NQ mean={np.mean(nq_fm1):.3f}")
    ax.axvline(np.mean(tqa_fm1), color=TQA_COLOR, linestyle="--", linewidth=1.5, label=f"TQA mean={np.mean(tqa_fm1):.3f}")
    ax.set_xlabel("fM1 (Mean Log-Probability)")
    ax.set_ylabel("Density")
    ax.set_title("Generation Confidence (fM1) by Domain")
    ax.legend()
    _save(fig, "fm1_histogram")


def plot_answer_length_comparison(nq_gen, tqa_gen):
    """Plot 2: Answer length distributions."""
    nq_lens = [len(r["greedy_answer"]) for r in nq_gen]
    tqa_lens = [len(r["greedy_answer"]) for r in tqa_gen]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 500, 50)
    ax.hist(nq_lens, bins=bins, alpha=0.6, label=f"NQ (mean={np.mean(nq_lens):.0f} chars)", color=NQ_COLOR, density=True)
    ax.hist(tqa_lens, bins=bins, alpha=0.6, label=f"TQA (mean={np.mean(tqa_lens):.0f} chars)", color=TQA_COLOR, density=True)
    ax.set_xlabel("Greedy Answer Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("Answer Length Distribution by Domain")
    ax.legend()
    _save(fig, "answer_length_histogram")


def plot_fm1_boxplot(nq_gen, tqa_gen):
    """Plot 3: Side-by-side boxplot of fM1."""
    nq_fm1 = [r["mean_logprob"] for r in nq_gen]
    tqa_fm1 = [r["mean_logprob"] for r in tqa_gen]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([nq_fm1, tqa_fm1], tick_labels=["NQ", "TQA"],
                     patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor(NQ_COLOR + "80")
    bp["boxes"][1].set_facecolor(TQA_COLOR + "80")
    ax.set_ylabel("fM1 (Mean Log-Probability)")
    ax.set_title("Generation Confidence by Domain")
    _save(fig, "fm1_boxplot")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Entailment statistics (available after Stage 3)
# ═══════════════════════════════════════════════════════════════════════════

def plot_entailment_scores(nq_ent, tqa_ent):
    """Plot 4: Entailment score (P(ENTAILMENT)) distribution."""
    nq_scores = [r["entail_score"] for r in nq_ent]
    tqa_scores = [r["entail_score"] for r in tqa_ent]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(nq_scores, bins=bins, alpha=0.6, label=f"NQ (n={len(nq_scores)})", color=NQ_COLOR, density=True)
    ax.hist(tqa_scores, bins=bins, alpha=0.6, label=f"TQA (n={len(tqa_scores)})", color=TQA_COLOR, density=True)
    ax.set_xlabel("Entailment Score P(ENTAILMENT)")
    ax.set_ylabel("Density")
    ax.set_title("Correctness Score Distribution by Domain")
    ax.legend()
    _save(fig, "entailment_score_histogram")


def plot_fm2_distribution(nq_ent, tqa_ent):
    """Plot 5: fM2 (self-consistency) distribution — discrete bar chart."""
    nq_fm2 = [r["fM2"] for r in nq_ent]
    tqa_fm2 = [r["fM2"] for r in tqa_ent]

    bins = np.arange(0, 1.1, 0.1)
    nq_counts, _ = np.histogram(nq_fm2, bins=bins)
    tqa_counts, _ = np.histogram(tqa_fm2, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bins) - 1)
    width = 0.35
    ax.bar(x - width/2, nq_counts / len(nq_fm2), width, label="NQ", color=NQ_COLOR, alpha=0.8)
    ax.bar(x + width/2, tqa_counts / len(tqa_fm2), width, label="TQA", color=TQA_COLOR, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in bins[:-1]])
    ax.set_xlabel("fM2 (Self-Consistency)")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title("Self-Consistency Score Distribution by Domain")
    ax.legend()
    _save(fig, "fm2_distribution")


def plot_fm1_vs_fm2_scatter(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """Plot 6: 2D scatter of (fM1, fM2) colored by domain and correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, gen, ent, name, color in [
        (axes[0], nq_gen, nq_ent, "NQ", NQ_COLOR),
        (axes[1], tqa_gen, tqa_ent, "TQA", TQA_COLOR),
    ]:
        fm1 = [g["mean_logprob"] for g in gen]
        fm2 = [e["fM2"] for e in ent]
        correct = [e["entail_label"] for e in ent]

        correct_mask = np.array(correct) == 1
        ax.scatter(np.array(fm1)[correct_mask], np.array(fm2)[correct_mask],
                   alpha=0.15, s=8, color="#4CAF50", label="Correct")
        ax.scatter(np.array(fm1)[~correct_mask], np.array(fm2)[~correct_mask],
                   alpha=0.3, s=12, color="#F44336", label="Wrong", marker="x")
        ax.set_xlabel("fM1 (Mean Log-Probability)")
        ax.set_title(name)
        ax.legend(loc="upper left")

    axes[0].set_ylabel("fM2 (Self-Consistency)")
    fig.suptitle("Confidence Landscape: fM1 vs fM2, Colored by Correctness", fontsize=14)
    plt.tight_layout()
    _save(fig, "fm1_vs_fm2_scatter")


def plot_correctness_rate_by_domain(nq_ent, tqa_ent):
    """Plot 7: Bar chart of correctness rate (entail_label=1) per domain."""
    nq_rate = np.mean([e["entail_label"] for e in nq_ent])
    tqa_rate = np.mean([e["entail_label"] for e in tqa_ent])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["NQ", "TQA"], [nq_rate, tqa_rate],
                   color=[NQ_COLOR, TQA_COLOR], alpha=0.8)
    ax.set_ylabel("Correctness Rate (Entailment)")
    ax.set_title("Model Accuracy by Domain")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    for bar, rate in zip(bars, [nq_rate, tqa_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.1%}", ha="center", fontsize=12)
    _save(fig, "correctness_rate_by_domain")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Baseline SGen-Semi results (available after Stage 4)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fdr_distribution(results):
    """Plot 8: FDR-E distribution across 100 splits for in-domain vs shifted."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    id_fdrs = [s["indomain_test"]["fdr_e"] for s in results["per_split"]]
    sh_fdrs = [s["shifted_test"]["fdr_e"] for s in results["per_split"]]
    epsilon = results["config"]["epsilon"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 0.6, 40)
    ax.hist(id_fdrs, bins=bins, alpha=0.6,
            label=f"{id_label} in-domain (valid={np.mean(np.array(id_fdrs) <= epsilon):.0%})",
            color=TQA_COLOR)
    ax.hist(sh_fdrs, bins=bins, alpha=0.6,
            label=f"{sh_label} shifted (valid={np.mean(np.array(sh_fdrs) <= epsilon):.0%})",
            color=NQ_COLOR)
    ax.axvline(epsilon, color="red", linestyle="--", linewidth=2, label=f"ε = {epsilon}")
    ax.set_xlabel("FDR-E")
    ax.set_ylabel("Count (out of 100 splits)")
    ax.set_title("FDR-E Distribution Across Calibration Splits")
    ax.legend()
    _save(fig, "fdr_distribution")


def plot_efficiency_distribution(results):
    """Plot 9: Efficiency distribution across splits."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    id_eff = [s["indomain_test"]["efficiency"] for s in results["per_split"]]
    sh_eff = [s["shifted_test"]["efficiency"] for s in results["per_split"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(id_eff, bins=bins, alpha=0.6, label=f"{id_label} in-domain (mean={np.mean(id_eff):.1%})", color=TQA_COLOR)
    ax.hist(sh_eff, bins=bins, alpha=0.6, label=f"{sh_label} shifted (mean={np.mean(sh_eff):.1%})", color=NQ_COLOR)
    ax.set_xlabel("Efficiency (fraction of questions answered)")
    ax.set_ylabel("Count (out of 100 splits)")
    ax.set_title("Selection Efficiency Distribution Across Splits")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    _save(fig, "efficiency_distribution")


def plot_validity_bar(results):
    """Plot 10: Validity rate bar chart (the headline result)."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    id_val = results["indomain"]["validity_rate"]
    sh_val = results["shifted"]["validity_rate"]
    target = 1 - results["config"]["delta"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"{id_label} (in-domain)", f"{sh_label} (shifted)"], [id_val, sh_val],
                   color=[TQA_COLOR, NQ_COLOR], alpha=0.8)
    ax.axhline(target, color="red", linestyle="--", linewidth=2, label=f"PAC target (1-δ = {target:.0%})")
    ax.set_ylabel("Validity Rate")
    ax.set_title("PAC Guarantee Validity: In-Domain vs Shifted")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    for bar, val in zip(bars, [id_val, sh_val]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=14, fontweight="bold")
    _save(fig, "validity_rate_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Conservative threshold results (Method 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_validity_efficiency_tradeoff(cons_results):
    """Plot 11: Pareto frontier — validity vs efficiency on shifted domain across all options."""
    # Get shifted label from first result
    first_opt = next(iter(cons_results["option_a"].values()))
    sh_label = first_opt["shifted"]["label"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for option_key, option_label, marker in [
        ("option_a", "A: Safety Factor", "o"),
        ("option_b", "B: Reduced ε", "s"),
        ("option_c", "C: Delta Budget", "^"),
    ]:
        if option_key not in cons_results:
            continue
        option = cons_results[option_key]
        validities = []
        efficiencies = []
        labels = []
        for param_key, sweep in sorted(option.items()):
            validities.append(sweep["shifted"]["validity_rate"])
            efficiencies.append(sweep["shifted"]["mean_efficiency"])
            labels.append(param_key)

        color = METHOD_COLORS[option_label.split(":")[0].strip()]
        ax.scatter(efficiencies, validities, color=color, marker=marker, s=80,
                   label=option_label, zorder=5)
        ax.plot(efficiencies, validities, color=color, alpha=0.4, linestyle="--")

        for eff, val, lbl in zip(efficiencies, validities, labels):
            ax.annotate(lbl, (eff, val), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=color)

    ax.axhline(0.98, color="red", linestyle="--", linewidth=1.5, label="PAC target (98%)")
    ax.set_xlabel(f"{sh_label} Efficiency (fraction answered)")
    ax.set_ylabel(f"{sh_label} Validity Rate")
    ax.set_title(f"Method 2: Validity-Efficiency Tradeoff on {sh_label} (shifted)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(0.5, 1.05)
    _save(fig, "validity_efficiency_tradeoff")


def plot_method_comparison_table(baseline_results, cons_results):
    """Plot 12: Summary comparison table as a figure (for paper)."""
    id_label = baseline_results["indomain"]["label"]
    sh_label = baseline_results["shifted"]["label"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    rows = [["Method 1 (Baseline)", "", "",
             f"{baseline_results['indomain']['validity_rate']:.0%}",
             f"{baseline_results['indomain']['mean_fdr_e']:.3f}",
             f"{baseline_results['indomain']['mean_efficiency']:.1%}",
             f"{baseline_results['shifted']['validity_rate']:.0%}",
             f"{baseline_results['shifted']['mean_fdr_e']:.3f}",
             f"{baseline_results['shifted']['mean_efficiency']:.1%}"]]

    for option_key, option_name in [("option_a", "A"), ("option_b", "B"), ("option_c", "C")]:
        if option_key not in cons_results:
            continue
        for param_key, sweep in sorted(cons_results[option_key].items()):
            rows.append([
                f"Method 2{option_name}", param_key, "",
                f"{sweep['indomain']['validity_rate']:.0%}",
                f"{sweep['indomain']['mean_fdr_e']:.3f}",
                f"{sweep['indomain']['mean_efficiency']:.1%}",
                f"{sweep['shifted']['validity_rate']:.0%}",
                f"{sweep['shifted']['mean_fdr_e']:.3f}",
                f"{sweep['shifted']['mean_efficiency']:.1%}",
            ])

    cols = ["Method", "Param", "",
            f"{id_label} Valid", f"{id_label} FDR-E", f"{id_label} Eff",
            f"{sh_label} Valid", f"{sh_label} FDR-E", f"{sh_label} Eff"]
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.4)
    ax.set_title("DS-SGen Results Summary", fontsize=14, pad=20)
    _save(fig, "results_summary_table")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate DS-SGen plots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "generation", "entailment", "baseline", "conservative"])
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    cfg = load_config(args.config)
    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]

    print("DS-SGen Plot Generator")
    print(f"  Plot output: {PLOT_DIR}/")
    print()

    # --- Generation plots (Stage 2 data) ---
    if args.stage in ("all", "generation"):
        nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
        tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))

        if nq_gen and tqa_gen:
            print(f"Generation plots (NQ={len(nq_gen)}, TQA={len(tqa_gen)}):")
            plot_fm1_histograms(nq_gen, tqa_gen)
            plot_answer_length_comparison(nq_gen, tqa_gen)
            plot_fm1_boxplot(nq_gen, tqa_gen)
        else:
            print("Skipping generation plots — cache incomplete")
            if nq_gen:
                print(f"  NQ generations: {len(nq_gen)} (complete)")
            if tqa_gen:
                print(f"  TQA generations: {len(tqa_gen)} (partial)")

    # --- Entailment plots (Stage 3 data) ---
    if args.stage in ("all", "entailment"):
        nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
        tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))
        nq_ent = load_cache(get_cache_path(cache_dir, "nq_entailment"))
        tqa_ent = load_cache(get_cache_path(cache_dir, "tqa_entailment"))

        if nq_ent and tqa_ent:
            print(f"Entailment plots (NQ={len(nq_ent)}, TQA={len(tqa_ent)}):")
            plot_entailment_scores(nq_ent, tqa_ent)
            plot_fm2_distribution(nq_ent, tqa_ent)
            plot_correctness_rate_by_domain(nq_ent, tqa_ent)
            if nq_gen and tqa_gen:
                plot_fm1_vs_fm2_scatter(nq_gen, nq_ent, tqa_gen, tqa_ent)
        else:
            print("Skipping entailment plots — cache not yet available")

    # --- Baseline results plots (Stage 4 data) ---
    if args.stage in ("all", "baseline"):
        baseline_path = os.path.join(results_dir, "baseline_results.json")
        baseline = load_cache(baseline_path)

        if baseline:
            print("Baseline plots:")
            plot_fdr_distribution(baseline)
            plot_efficiency_distribution(baseline)
            plot_validity_bar(baseline)
        else:
            print("Skipping baseline plots — results not yet available")

    # --- Conservative threshold plots (Method 2) ---
    if args.stage in ("all", "conservative"):
        cons_path = os.path.join(results_dir, "conservative_results.json")
        cons = load_cache(cons_path)
        baseline_path = os.path.join(results_dir, "baseline_results.json")
        baseline = load_cache(baseline_path)

        if cons:
            print("Conservative threshold plots:")
            plot_validity_efficiency_tradeoff(cons)
            if baseline:
                plot_method_comparison_table(baseline, cons)
        else:
            print("Skipping conservative plots — results not yet available")

    print("\nDone.")


if __name__ == "__main__":
    main()
