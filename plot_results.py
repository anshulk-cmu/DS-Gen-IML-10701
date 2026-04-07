"""Visualization script for DS-SGen experiment results.

Generates all plots for the paper. Each plot function checks whether the
required data exists before plotting. Plots are grouped by pipeline stage.

Usage:
    python plot_results.py                          # all available plots
    python plot_results.py --stage generation       # generation stats only
    python plot_results.py --stage entailment       # entailment stats only
    python plot_results.py --stage baseline         # baseline SGen results
    python plot_results.py --stage conservative     # Method 2 results
    python plot_results.py --stage method3          # Method 3 results
    python plot_results.py --stage epsilon_sweep    # epsilon sweep comparison
    python plot_results.py --stage all              # everything

Output: plots/ directory (PNG files, 300 DPI)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.gridspec as gridspec

from ds_sgen.utils import load_config, get_cache_path, load_cache


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

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
    "axes.spines.top": False,
    "axes.spines.right": False,
}
NQ_COLOR = "#2196F3"    # Blue
TQA_COLOR = "#FF5722"   # Red-orange
CORRECT_COLOR = "#4CAF50"  # Green
WRONG_COLOR = "#F44336"    # Red
M3_COLOR = "#4CAF50"    # Green for Method 3
METHOD_COLORS = {"A": "#4CAF50", "B": "#FF9800", "C": "#9C27B0"}


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def _validate_lengths(*args, names=None):
    """Validate that all data lists have matching lengths."""
    lengths = [len(a) for a in args]
    if len(set(lengths)) > 1:
        label = ", ".join(names) if names else "inputs"
        raise ValueError(f"Length mismatch in {label}: {lengths}")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Generation statistics (available after Stage 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fm1_histograms(nq_gen, tqa_gen):
    """fM1 (mean log-prob) distribution comparison between NQ and TQA.

    Shows the confidence distribution for both domains. TQA typically has
    higher confidence (less negative logprobs) because trivia questions are
    more factual. The gap between distributions is the domain shift."""
    nq_fm1 = [r["mean_logprob"] for r in nq_gen]
    tqa_fm1 = [r["mean_logprob"] for r in tqa_gen]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-1.0, 0.0, 50)
    ax.hist(nq_fm1, bins=bins, alpha=0.6, label=f"NQ (n={len(nq_fm1)})",
            color=NQ_COLOR, density=True)
    ax.hist(tqa_fm1, bins=bins, alpha=0.6, label=f"TQA (n={len(tqa_fm1)})",
            color=TQA_COLOR, density=True)
    ax.axvline(np.mean(nq_fm1), color=NQ_COLOR, linestyle="--", linewidth=1.5,
               label=f"NQ mean={np.mean(nq_fm1):.3f}")
    ax.axvline(np.mean(tqa_fm1), color=TQA_COLOR, linestyle="--", linewidth=1.5,
               label=f"TQA mean={np.mean(tqa_fm1):.3f}")
    ax.set_xlabel("fM1 (Mean Log-Probability)")
    ax.set_ylabel("Density")
    ax.set_title("Generation Confidence (fM1) by Domain")
    ax.legend()
    _save(fig, "fm1_histogram")


def plot_answer_length_comparison(nq_gen, tqa_gen):
    """Answer length distributions by domain."""
    nq_lens = [len(r["greedy_answer"]) for r in nq_gen]
    tqa_lens = [len(r["greedy_answer"]) for r in tqa_gen]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 500, 50)
    ax.hist(nq_lens, bins=bins, alpha=0.6,
            label=f"NQ (mean={np.mean(nq_lens):.0f} chars)", color=NQ_COLOR, density=True)
    ax.hist(tqa_lens, bins=bins, alpha=0.6,
            label=f"TQA (mean={np.mean(tqa_lens):.0f} chars)", color=TQA_COLOR, density=True)
    ax.set_xlabel("Greedy Answer Length (characters)")
    ax.set_ylabel("Density")
    ax.set_title("Answer Length Distribution by Domain")
    ax.legend()
    _save(fig, "answer_length_histogram")


def plot_fm1_boxplot(nq_gen, tqa_gen):
    """Side-by-side boxplot of fM1 showing median, IQR, and outliers."""
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


def plot_fm1_cdf_comparison(nq_gen, tqa_gen):
    """Overlaid CDFs of fM1 for NQ vs TQA.

    More precise than histograms for showing exactly where the distributions
    diverge. The horizontal gap at any quantile is the domain shift magnitude
    at that confidence level."""
    nq_fm1 = np.sort([r["mean_logprob"] for r in nq_gen])
    tqa_fm1 = np.sort([r["mean_logprob"] for r in tqa_gen])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nq_fm1, np.linspace(0, 1, len(nq_fm1)),
            color=NQ_COLOR, linewidth=2, label=f"NQ (n={len(nq_fm1)})")
    ax.plot(tqa_fm1, np.linspace(0, 1, len(tqa_fm1)),
            color=TQA_COLOR, linewidth=2, label=f"TQA (n={len(tqa_fm1)})")
    ax.set_xlabel("fM1 (Mean Log-Probability)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution of fM1: NQ vs TQA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "fm1_cdf_comparison")


def plot_sampled_answer_diversity(nq_gen, tqa_gen):
    """Distribution of unique sampled answer count per question.

    Uses exact string matching to count distinct answers among K=5 samples.
    If most questions have 1 unique answer, the model is very consistent.
    If 5, it's guessing randomly each time. This validates the sampling
    strategy and shows the information content in fM2."""
    def _count_unique(gen_list):
        counts = []
        for r in gen_list:
            answers = r.get("sampled_answers", [])
            # Normalize: lowercase, strip whitespace
            normalized = [a.strip().lower() for a in answers]
            counts.append(len(set(normalized)))
        return counts

    nq_unique = _count_unique(nq_gen)
    tqa_unique = _count_unique(tqa_gen)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.5, 7, 1)  # 1,2,3,4,5
    ax.hist(nq_unique, bins=bins, alpha=0.6, label=f"NQ (mean={np.mean(nq_unique):.2f})",
            color=NQ_COLOR, density=True)
    ax.hist(tqa_unique, bins=bins, alpha=0.6, label=f"TQA (mean={np.mean(tqa_unique):.2f})",
            color=TQA_COLOR, density=True)
    ax.set_xlabel("Number of Unique Sampled Answers (out of K=5)")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title("Sampled Answer Diversity by Domain")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend()
    _save(fig, "sampled_answer_diversity")


def plot_sampled_logprob_spread(nq_gen, tqa_gen):
    """Spread (std) of sampled log-probs per question.

    High spread means the model's confidence varies across samples — a sign
    of uncertainty. Low spread means it's consistently confident (or not)."""
    def _spreads(gen_list):
        stds = []
        for r in gen_list:
            lps = r.get("sampled_mean_logprobs", [])
            if len(lps) > 1:
                stds.append(float(np.std(lps)))
        return stds

    nq_spread = _spreads(nq_gen)
    tqa_spread = _spreads(tqa_gen)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 0.5, 50)
    ax.hist(nq_spread, bins=bins, alpha=0.6,
            label=f"NQ (mean={np.mean(nq_spread):.3f})", color=NQ_COLOR, density=True)
    ax.hist(tqa_spread, bins=bins, alpha=0.6,
            label=f"TQA (mean={np.mean(tqa_spread):.3f})", color=TQA_COLOR, density=True)
    ax.set_xlabel("Std Dev of Sampled Log-Probabilities")
    ax.set_ylabel("Density")
    ax.set_title("Sampled Answer Confidence Spread by Domain")
    ax.legend()
    _save(fig, "sampled_logprob_spread")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Entailment statistics (available after Stage 3)
# ═══════════════════════════════════════════════════════════════════════════

def plot_entailment_scores(nq_ent, tqa_ent):
    """P(ENTAILMENT) score distribution by domain."""
    nq_scores = [r["entail_score"] for r in nq_ent]
    tqa_scores = [r["entail_score"] for r in tqa_ent]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 50)
    ax.hist(nq_scores, bins=bins, alpha=0.6,
            label=f"NQ (n={len(nq_scores)})", color=NQ_COLOR, density=True)
    ax.hist(tqa_scores, bins=bins, alpha=0.6,
            label=f"TQA (n={len(tqa_scores)})", color=TQA_COLOR, density=True)
    ax.set_xlabel("Entailment Score P(ENTAILMENT)")
    ax.set_ylabel("Density")
    ax.set_title("Correctness Score Distribution by Domain")
    ax.legend()
    _save(fig, "entailment_score_histogram")


def plot_fm2_distribution(nq_ent, tqa_ent):
    """fM2 (self-consistency) distribution — discrete bar chart."""
    nq_fm2 = [r["fM2"] for r in nq_ent]
    tqa_fm2 = [r["fM2"] for r in tqa_ent]

    bins = np.arange(0, 1.1, 0.1)
    nq_counts, _ = np.histogram(nq_fm2, bins=bins)
    tqa_counts, _ = np.histogram(tqa_fm2, bins=bins)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bins) - 1)
    width = 0.35
    ax.bar(x - width/2, nq_counts / len(nq_fm2), width,
           label="NQ", color=NQ_COLOR, alpha=0.8)
    ax.bar(x + width/2, tqa_counts / len(tqa_fm2), width,
           label="TQA", color=TQA_COLOR, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in bins[:-1]])
    ax.set_xlabel("fM2 (Self-Consistency)")
    ax.set_ylabel("Fraction of Questions")
    ax.set_title("Self-Consistency Score Distribution by Domain")
    ax.legend()
    _save(fig, "fm2_distribution")


def plot_fm1_vs_fm2_scatter(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """2D scatter of (fM1, fM2) colored by correctness.

    This is the feature space that the SGen algorithm operates on.
    Correct answers should cluster at high fM1 + high fM2 (top-right).
    If the two classes separate well, threshold selection will work."""
    _validate_lengths(nq_gen, nq_ent, names=["nq_gen", "nq_ent"])
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, gen, ent, name in [
        (axes[0], nq_gen, nq_ent, "NQ"),
        (axes[1], tqa_gen, tqa_ent, "TQA"),
    ]:
        fm1 = np.array([g["mean_logprob"] for g in gen])
        fm2 = np.array([e["fM2"] for e in ent])
        correct = np.array([e["entail_label"] for e in ent])

        correct_mask = correct == 1
        n_correct = int(correct_mask.sum())
        n_wrong = int((~correct_mask).sum())

        ax.scatter(fm1[correct_mask], fm2[correct_mask],
                   alpha=0.15, s=8, color=CORRECT_COLOR,
                   label=f"Correct (n={n_correct})")
        ax.scatter(fm1[~correct_mask], fm2[~correct_mask],
                   alpha=0.3, s=12, color=WRONG_COLOR,
                   label=f"Wrong (n={n_wrong})", marker="x")
        ax.set_xlabel("fM1 (Mean Log-Probability)")
        ax.set_title(f"{name} (accuracy={n_correct/len(correct):.1%})")
        ax.legend(loc="upper left")

    axes[0].set_ylabel("fM2 (Self-Consistency)")
    fig.suptitle("Confidence Landscape: fM1 vs fM2, Colored by Correctness", fontsize=14)
    plt.tight_layout()
    _save(fig, "fm1_vs_fm2_scatter")


def plot_correctness_rate_by_domain(nq_ent, tqa_ent):
    """Bar chart of correctness rate (entail_label=1) per domain."""
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


def plot_fm1_roc_curve(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """ROC curve: how well fM1 separates correct from wrong answers.

    This is the most important feature-validation plot. If AUC is close to
    0.5, fM1 is useless and threshold selection can't work. AUC > 0.7
    means fM1 is a solid discriminator."""
    _validate_lengths(nq_gen, nq_ent, names=["nq_gen", "nq_ent"])
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    fig, ax = plt.subplots(figsize=(7, 7))

    for gen, ent, name, color in [
        (nq_gen, nq_ent, "NQ", NQ_COLOR),
        (tqa_gen, tqa_ent, "TQA", TQA_COLOR),
    ]:
        fm1 = np.array([g["mean_logprob"] for g in gen])
        labels = np.array([e["entail_label"] for e in ent])

        # Manually compute ROC (no sklearn dependency)
        thresholds = np.sort(np.unique(fm1))
        # Sample thresholds for efficiency
        if len(thresholds) > 500:
            idx = np.linspace(0, len(thresholds) - 1, 500, dtype=int)
            thresholds = thresholds[idx]

        tpr_list, fpr_list = [], []
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            continue

        for t in thresholds:
            predicted_pos = fm1 >= t
            tp = (predicted_pos & (labels == 1)).sum()
            fp = (predicted_pos & (labels == 0)).sum()
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)

        fpr_list, tpr_list = np.array(fpr_list), np.array(tpr_list)
        # Sort by FPR for proper curve
        sort_idx = np.argsort(fpr_list)
        fpr_list, tpr_list = fpr_list[sort_idx], tpr_list[sort_idx]

        # AUC via trapezoidal rule
        auc = np.trapz(tpr_list, fpr_list)

        ax.plot(fpr_list, tpr_list, color=color, linewidth=2,
                label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve: fM1 as Correctness Predictor")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    _save(fig, "fm1_roc_curve")


def plot_calibration_curve(nq_gen, nq_ent, tqa_gen, tqa_ent, n_bins=10):
    """Calibration curve: binned fM1 vs actual correctness rate.

    A perfectly calibrated model follows the diagonal. Points above the
    diagonal mean the model is better than it thinks (underconfident);
    below means overconfident. Essential for understanding if the threshold
    will generalize."""
    _validate_lengths(nq_gen, nq_ent, names=["nq_gen", "nq_ent"])
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    fig, ax = plt.subplots(figsize=(8, 6))

    for gen, ent, name, color, marker in [
        (nq_gen, nq_ent, "NQ", NQ_COLOR, "o"),
        (tqa_gen, tqa_ent, "TQA", TQA_COLOR, "s"),
    ]:
        fm1 = np.array([g["mean_logprob"] for g in gen])
        labels = np.array([e["entail_label"] for e in ent])

        # Create equal-frequency bins (deciles)
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(fm1, percentiles)

        bin_centers = []
        bin_rates = []
        bin_counts = []

        for i in range(n_bins):
            if i < n_bins - 1:
                mask = (fm1 >= bin_edges[i]) & (fm1 < bin_edges[i + 1])
            else:
                mask = (fm1 >= bin_edges[i]) & (fm1 <= bin_edges[i + 1])
            n_in_bin = mask.sum()
            if n_in_bin == 0:
                continue
            bin_centers.append(np.mean(fm1[mask]))
            bin_rates.append(labels[mask].mean())
            bin_counts.append(n_in_bin)

        ax.plot(bin_centers, bin_rates, f"-{marker}", color=color, linewidth=2,
                markersize=8, label=f"{name} (n={len(gen)})")
        # Add count annotations
        for x, y, c in zip(bin_centers, bin_rates, bin_counts):
            ax.annotate(str(c), (x, y), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, ha="center", color=color, alpha=0.7)

    ax.set_xlabel("fM1 (Mean Log-Probability, binned)")
    ax.set_ylabel("Actual Correctness Rate")
    ax.set_title("Calibration: fM1 vs Actual Correctness")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    _save(fig, "calibration_curve")


def plot_correctness_by_fm1_decile(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """Correctness rate by fM1 decile — the most intuitive feature validation.

    Groups questions into 10 equal-frequency bins by confidence. Shows a
    monotonically increasing bar chart if fM1 is a good feature. The
    steepness of the increase shows discriminative power."""
    _validate_lengths(nq_gen, nq_ent, names=["nq_gen", "nq_ent"])
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, gen, ent, name, color in [
        (axes[0], nq_gen, nq_ent, "NQ", NQ_COLOR),
        (axes[1], tqa_gen, tqa_ent, "TQA", TQA_COLOR),
    ]:
        fm1 = np.array([g["mean_logprob"] for g in gen])
        labels = np.array([e["entail_label"] for e in ent])

        decile_idx = np.argsort(fm1)
        n = len(fm1)
        decile_size = n // 10

        rates = []
        decile_labels = []
        for d in range(10):
            start = d * decile_size
            end = start + decile_size if d < 9 else n
            idx = decile_idx[start:end]
            rate = labels[idx].mean()
            rates.append(rate)
            lo, hi = fm1[idx].min(), fm1[idx].max()
            decile_labels.append(f"D{d+1}\n[{lo:.2f},{hi:.2f}]")

        bars = ax.bar(range(10), rates, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{rate:.0%}", ha="center", fontsize=8)
        ax.set_xticks(range(10))
        ax.set_xticklabels(decile_labels, fontsize=7)
        ax.set_xlabel("fM1 Decile (low confidence → high confidence)")
        ax.set_title(f"{name}: Correctness by fM1 Decile")
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    axes[0].set_ylabel("Correctness Rate")
    fig.suptitle("Feature Validation: Does Higher Confidence Mean Higher Accuracy?", fontsize=14)
    plt.tight_layout()
    _save(fig, "correctness_by_fm1_decile")


def plot_fm2_conditional_on_fm1(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """For borderline fM1 (middle deciles), does fM2 add discriminative power?

    Splits questions into three fM1 bands (low/mid/high), then within each
    band compares fM2 distributions for correct vs wrong answers. If fM2
    separates in the middle band, it justifies using both features."""
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    # Use TQA (larger dataset, calibration set)
    fm1 = np.array([g["mean_logprob"] for g in tqa_gen])
    fm2 = np.array([e["fM2"] for e in tqa_ent])
    labels = np.array([e["entail_label"] for e in tqa_ent])

    p33, p66 = np.percentile(fm1, [33, 66])
    bands = [
        ("Low fM1\n(bottom third)", fm1 < p33),
        ("Mid fM1\n(middle third)", (fm1 >= p33) & (fm1 < p66)),
        ("High fM1\n(top third)", fm1 >= p66),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    bins = np.linspace(0, 1, 15)

    for ax, (band_name, mask) in zip(axes, bands):
        correct_mask = mask & (labels == 1)
        wrong_mask = mask & (labels == 0)

        fm2_correct = fm2[correct_mask]
        fm2_wrong = fm2[wrong_mask]

        ax.hist(fm2_correct, bins=bins, alpha=0.6, color=CORRECT_COLOR,
                label=f"Correct (n={len(fm2_correct)})", density=True)
        ax.hist(fm2_wrong, bins=bins, alpha=0.6, color=WRONG_COLOR,
                label=f"Wrong (n={len(fm2_wrong)})", density=True)
        ax.set_xlabel("fM2 (Self-Consistency)")
        ax.set_title(band_name)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density")
    fig.suptitle("TQA: Does fM2 Help Discriminate Within fM1 Bands?", fontsize=14)
    plt.tight_layout()
    _save(fig, "fm2_conditional_on_fm1")


def plot_domain_shift_diagnostic(nq_gen, nq_ent, tqa_gen, tqa_ent):
    """2x2 diagnostic panel summarizing the entire domain shift problem.

    (a) fM1 distributions overlaid
    (b) Correctness rates
    (c) fM2 distributions overlaid
    (d) Feature-correctness correlation coefficients

    One figure that gives reviewers the complete picture."""
    _validate_lengths(nq_gen, nq_ent, names=["nq_gen", "nq_ent"])
    _validate_lengths(tqa_gen, tqa_ent, names=["tqa_gen", "tqa_ent"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) fM1 distributions
    ax = axes[0, 0]
    nq_fm1 = np.array([g["mean_logprob"] for g in nq_gen])
    tqa_fm1 = np.array([g["mean_logprob"] for g in tqa_gen])
    bins = np.linspace(-1, 0, 40)
    ax.hist(nq_fm1, bins=bins, alpha=0.6, color=NQ_COLOR, label="NQ", density=True)
    ax.hist(tqa_fm1, bins=bins, alpha=0.6, color=TQA_COLOR, label="TQA", density=True)
    ax.set_xlabel("fM1 (Mean Log-Probability)")
    ax.set_ylabel("Density")
    ax.set_title("(a) Confidence Distributions")
    ax.legend()

    # (b) Correctness rates with confidence intervals
    ax = axes[0, 1]
    nq_labels = np.array([e["entail_label"] for e in nq_ent])
    tqa_labels = np.array([e["entail_label"] for e in tqa_ent])
    nq_rate = nq_labels.mean()
    tqa_rate = tqa_labels.mean()
    # Wilson confidence interval
    def _wilson_ci(p, n, z=1.96):
        denom = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denom
        spread = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
        return center - spread, center + spread
    nq_lo, nq_hi = _wilson_ci(nq_rate, len(nq_labels))
    tqa_lo, tqa_hi = _wilson_ci(tqa_rate, len(tqa_labels))

    bars = ax.bar(["NQ", "TQA"], [nq_rate, tqa_rate],
                   color=[NQ_COLOR, TQA_COLOR], alpha=0.8)
    ax.errorbar(["NQ", "TQA"], [nq_rate, tqa_rate],
                yerr=[[nq_rate-nq_lo, tqa_rate-tqa_lo],
                      [nq_hi-nq_rate, tqa_hi-tqa_rate]],
                fmt="none", color="black", capsize=5)
    for bar, rate in zip(bars, [nq_rate, tqa_rate]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{rate:.1%}", ha="center", fontsize=11)
    ax.set_ylabel("Correctness Rate")
    ax.set_title("(b) Model Accuracy (with 95% CI)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    # (c) fM2 distributions
    ax = axes[1, 0]
    nq_fm2 = np.array([e["fM2"] for e in nq_ent])
    tqa_fm2 = np.array([e["fM2"] for e in tqa_ent])
    bins_fm2 = np.linspace(0, 1, 20)
    ax.hist(nq_fm2, bins=bins_fm2, alpha=0.6, color=NQ_COLOR, label="NQ", density=True)
    ax.hist(tqa_fm2, bins=bins_fm2, alpha=0.6, color=TQA_COLOR, label="TQA", density=True)
    ax.set_xlabel("fM2 (Self-Consistency)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Self-Consistency Distributions")
    ax.legend()

    # (d) Feature-correctness correlations
    ax = axes[1, 1]
    # Point-biserial correlation (Pearson between continuous and binary)
    def _corr(x, y):
        if len(x) < 2:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    corrs = {
        "NQ: fM1↔correct": _corr(nq_fm1, nq_labels),
        "TQA: fM1↔correct": _corr(tqa_fm1, tqa_labels),
        "NQ: fM2↔correct": _corr(nq_fm2, nq_labels),
        "TQA: fM2↔correct": _corr(tqa_fm2, tqa_labels),
    }
    names = list(corrs.keys())
    vals = list(corrs.values())
    colors = [NQ_COLOR, TQA_COLOR, NQ_COLOR, TQA_COLOR]
    hatches = ["", "", "///", "///"]

    bars = ax.barh(range(len(names)), vals, color=colors, alpha=0.8, edgecolor="black")
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Pearson Correlation")
    ax.set_title("(d) Feature-Correctness Correlations")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlim(-0.1, max(vals) + 0.15)

    fig.suptitle("Domain Shift Diagnostic: NQ vs TQA", fontsize=15, y=1.01)
    fig.tight_layout()
    _save(fig, "domain_shift_diagnostic")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Baseline SGen-Semi results (available after Stage 4)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fdr_distribution(results):
    """FDR-E distribution across splits for in-domain vs shifted."""
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
    ax.axvline(epsilon, color="red", linestyle="--", linewidth=2, label=f"epsilon = {epsilon}")
    ax.set_xlabel("FDR-E")
    ax.set_ylabel("Count (out of splits)")
    ax.set_title("FDR-E Distribution Across Calibration Splits")
    ax.legend()
    _save(fig, "fdr_distribution")


def plot_efficiency_distribution(results):
    """Efficiency distribution across splits."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    id_eff = [s["indomain_test"]["efficiency"] for s in results["per_split"]]
    sh_eff = [s["shifted_test"]["efficiency"] for s in results["per_split"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 30)
    ax.hist(id_eff, bins=bins, alpha=0.6,
            label=f"{id_label} in-domain (mean={np.mean(id_eff):.1%})", color=TQA_COLOR)
    ax.hist(sh_eff, bins=bins, alpha=0.6,
            label=f"{sh_label} shifted (mean={np.mean(sh_eff):.1%})", color=NQ_COLOR)
    ax.set_xlabel("Efficiency (fraction of questions answered)")
    ax.set_ylabel("Count (out of splits)")
    ax.set_title("Selection Efficiency Distribution Across Splits")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    _save(fig, "efficiency_distribution")


def plot_validity_bar(results):
    """Validity rate bar chart — the headline result."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    id_val = results["indomain"]["validity_rate"]
    sh_val = results["shifted"]["validity_rate"]
    target = 1 - results["config"]["delta"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([f"{id_label}\n(in-domain)", f"{sh_label}\n(shifted)"],
                   [id_val, sh_val], color=[TQA_COLOR, NQ_COLOR], alpha=0.8)
    ax.axhline(target, color="red", linestyle="--", linewidth=2,
               label=f"PAC target (1-delta = {target:.0%})")
    ax.set_ylabel("Validity Rate")
    ax.set_title("PAC Guarantee Validity: In-Domain vs Shifted")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    for bar, val in zip(bars, [id_val, sh_val]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=14, fontweight="bold")
    _save(fig, "validity_rate_comparison")


def plot_fdr_vs_efficiency_scatter(results):
    """Each split as a (efficiency, FDR) dot.

    Shows the tradeoff cloud. Tight clusters mean the algorithm is stable.
    The epsilon line separates valid from invalid splits."""
    id_label = results["indomain"]["label"]
    sh_label = results["shifted"]["label"]
    epsilon = results["config"]["epsilon"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, domain, label, color in [
        (axes[0], "indomain_test", id_label, TQA_COLOR),
        (axes[1], "shifted_test", sh_label, NQ_COLOR),
    ]:
        fdrs = [s[domain]["fdr_e"] for s in results["per_split"]]
        effs = [s[domain]["efficiency"] for s in results["per_split"]]
        valid = [s[domain]["valid"] for s in results["per_split"]]

        valid_mask = np.array(valid)
        fdrs, effs = np.array(fdrs), np.array(effs)

        ax.scatter(effs[valid_mask], fdrs[valid_mask], alpha=0.5, s=30,
                   color=CORRECT_COLOR, label=f"Valid (n={valid_mask.sum()})", zorder=3)
        ax.scatter(effs[~valid_mask], fdrs[~valid_mask], alpha=0.7, s=40,
                   color=WRONG_COLOR, marker="x",
                   label=f"Invalid (n={(~valid_mask).sum()})", zorder=3)
        ax.axhline(epsilon, color="red", linestyle="--", linewidth=1.5,
                   label=f"epsilon = {epsilon}")
        ax.set_xlabel("Efficiency (fraction answered)")
        ax.set_ylabel("FDR-E")
        ax.set_title(f"{label} ({'in-domain' if domain == 'indomain_test' else 'shifted'})")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(True, alpha=0.2)

    fig.suptitle("FDR-E vs Efficiency: Per-Split Operating Points", fontsize=14)
    plt.tight_layout()
    _save(fig, "fdr_vs_efficiency_scatter")


def plot_threshold_stability(results):
    """Distribution of chosen tau1 across splits.

    Shows how stable the threshold selection is. A tight distribution means
    the calibration is robust; a wide spread means it's sensitive to the split."""
    tau1s = [s["tau1"] for s in results["per_split"] if s["tau1"] is not None]

    if len(tau1s) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tau1s, bins=30, color="#607D8B", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axvline(np.mean(tau1s), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(tau1s):.4f}")
    ax.axvline(np.median(tau1s), color="orange", linestyle="--", linewidth=2,
               label=f"Median = {np.median(tau1s):.4f}")
    ax.set_xlabel("Chosen Threshold tau1 (fM1)")
    ax.set_ylabel("Count (out of splits)")
    ax.set_title("Threshold Stability Across Random Splits")
    ax.legend()

    # Add std annotation
    ax.text(0.98, 0.95, f"Std = {np.std(tau1s):.4f}\nIQR = {np.percentile(tau1s, 75) - np.percentile(tau1s, 25):.4f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    _save(fig, "threshold_stability")


def plot_cumulative_fdr_curve(results, nq_gen, nq_ent, tqa_gen, tqa_ent):
    """Cumulative FDR as you lower the threshold (include more questions).

    Sort all questions by fM1 descending. Sweep from most confident to least,
    plotting (fraction included, cumulative FDR). Shows the 'cost of greed':
    how FDR degrades as you try to answer more questions."""
    cal_dataset = results.get("cal_dataset", "tqa")

    # Use shifted domain for the curve (the harder case)
    if cal_dataset == "tqa":
        gen, ent = nq_gen, nq_ent
        domain_label = "NQ (shifted)"
        color = NQ_COLOR
    else:
        gen, ent = tqa_gen, tqa_ent
        domain_label = "TQA (shifted)"
        color = TQA_COLOR

    _validate_lengths(gen, ent, names=["gen", "ent"])

    fm1 = np.array([g["mean_logprob"] for g in gen])
    labels = np.array([e["entail_label"] for e in ent])
    epsilon = results["config"]["epsilon"]

    # Sort by fM1 descending (most confident first)
    order = np.argsort(-fm1)
    sorted_labels = labels[order]

    n = len(sorted_labels)
    cum_wrong = np.cumsum(sorted_labels == 0)
    cum_total = np.arange(1, n + 1)
    cum_fdr = cum_wrong / cum_total
    fractions = cum_total / n

    fig, ax = plt.subplots(figsize=(8, 5))
    # Subsample for clean plotting
    step = max(1, n // 500)
    ax.plot(fractions[::step], cum_fdr[::step], color=color, linewidth=2,
            label=domain_label)
    ax.axhline(epsilon, color="red", linestyle="--", linewidth=1.5,
               label=f"epsilon = {epsilon}")
    ax.fill_between(fractions[::step], 0, cum_fdr[::step], alpha=0.1, color=color)

    # Mark the crossover point
    crossover = np.where(cum_fdr > epsilon)[0]
    if len(crossover) > 0:
        ci = crossover[0]
        ax.axvline(fractions[ci], color="gray", linestyle=":", alpha=0.7)
        ax.annotate(f"Max safe: {fractions[ci]:.1%}",
                    xy=(fractions[ci], epsilon), xytext=(fractions[ci]+0.05, epsilon+0.05),
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    fontsize=10, color="gray")

    ax.set_xlabel("Fraction of Questions Answered (sorted by confidence)")
    ax.set_ylabel("Cumulative FDR-E")
    ax.set_title(f"Cost of Greed: FDR vs Coverage on {domain_label}")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(0.6, cum_fdr.max() + 0.05))
    _save(fig, "cumulative_fdr_curve")


def plot_abstention_analysis(results, nq_gen, nq_ent, tqa_gen, tqa_ent):
    """Of the questions filtered out, what fraction were actually correct?

    This is the 'cost of safety' — how many right answers you're throwing
    away. Uses the median threshold across splits."""
    cal_dataset = results.get("cal_dataset", "tqa")

    if cal_dataset == "tqa":
        gen, ent = nq_gen, nq_ent
        domain_label = "NQ (shifted)"
    else:
        gen, ent = tqa_gen, tqa_ent
        domain_label = "TQA (shifted)"

    _validate_lengths(gen, ent, names=["gen", "ent"])

    fm1 = np.array([g["mean_logprob"] for g in gen])
    labels = np.array([e["entail_label"] for e in ent])

    # Get median threshold
    tau1s = [s["tau1"] for s in results["per_split"] if s["tau1"] is not None]
    if len(tau1s) == 0:
        return
    median_tau = np.median(tau1s)

    selected = fm1 >= median_tau
    abstained = ~selected

    n_total = len(labels)
    n_selected = selected.sum()
    n_abstained = abstained.sum()

    # Among selected
    sel_correct = (selected & (labels == 1)).sum()
    sel_wrong = (selected & (labels == 0)).sum()
    # Among abstained
    abs_correct = (abstained & (labels == 1)).sum()
    abs_wrong = (abstained & (labels == 0)).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: stacked bar
    ax = axes[0]
    bars_sel = ax.bar(["Selected", "Abstained"],
                       [sel_correct, abs_correct],
                       color=CORRECT_COLOR, alpha=0.8, label="Correct")
    bars_wrong = ax.bar(["Selected", "Abstained"],
                         [sel_wrong, abs_wrong],
                         bottom=[sel_correct, abs_correct],
                         color=WRONG_COLOR, alpha=0.8, label="Wrong")
    ax.set_ylabel("Number of Questions")
    ax.set_title(f"{domain_label}: Selection Breakdown\n(tau1={median_tau:.4f}, median across splits)")
    ax.legend()
    # Annotate
    for i, (c, w, tot) in enumerate([(sel_correct, sel_wrong, n_selected),
                                      (abs_correct, abs_wrong, n_abstained)]):
        if tot > 0:
            ax.text(i, tot + 5, f"n={tot}\n({c/tot:.0%} correct)", ha="center", fontsize=9)

    # Right: summary stats
    ax = axes[1]
    ax.axis("off")
    if n_selected > 0 and n_abstained > 0:
        stats_text = (
            f"Domain: {domain_label}\n"
            f"Threshold (median tau1): {median_tau:.4f}\n"
            f"{'─' * 40}\n"
            f"Selected:  {n_selected}/{n_total} ({n_selected/n_total:.1%})\n"
            f"  Correct: {sel_correct}/{n_selected} ({sel_correct/n_selected:.1%})\n"
            f"  Wrong:   {sel_wrong}/{n_selected} ({sel_wrong/n_selected:.1%}) ← FDR\n"
            f"{'─' * 40}\n"
            f"Abstained: {n_abstained}/{n_total} ({n_abstained/n_total:.1%})\n"
            f"  Correct: {abs_correct}/{n_abstained} ({abs_correct/n_abstained:.1%}) ← wasted\n"
            f"  Wrong:   {abs_wrong}/{n_abstained} ({abs_wrong/n_abstained:.1%})\n"
            f"{'─' * 40}\n"
            f"Overall accuracy: {labels.mean():.1%}\n"
            f"Accuracy if selected: {sel_correct/n_selected:.1%}\n"
            f"Accuracy if abstained: {abs_correct/n_abstained:.1%}\n"
        )
    else:
        stats_text = "Insufficient data for abstention analysis."
    ax.text(0.05, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment="center", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Abstention Statistics")

    fig.suptitle("Cost of Safety: What Are We Filtering Out?", fontsize=14)
    plt.tight_layout()
    _save(fig, "abstention_analysis")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Conservative threshold results (Method 2)
# ═══════════════════════════════════════════════════════════════════════════

def plot_validity_efficiency_tradeoff(cons_results):
    """Pareto frontier: validity vs efficiency on shifted domain across all options."""
    first_opt = next(iter(cons_results["option_a"].values()))
    sh_label = first_opt["shifted"]["label"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for option_key, option_label, marker in [
        ("option_a", "A: Safety Factor", "o"),
        ("option_b", "B: Reduced epsilon", "s"),
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

    ax.axhline(0.98, color="red", linestyle="--", linewidth=1.5,
               label="PAC target (98%)")
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
    """Summary comparison table as a figure (for paper)."""
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
# STAGE: Method 3 — Importance Weighted
# ═══════════════════════════════════════════════════════════════════════════

def plot_method3_fdr_distribution(baseline, iw_results):
    """Overlaid FDR-E histograms: Method 1 vs Method 3 on shifted domain."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 0.8, 40)

    if baseline and "per_split" in baseline:
        m1_fdr = [r["shifted_test"]["fdr_e"] for r in baseline["per_split"]]
        ax.hist(m1_fdr, bins=bins, alpha=0.5,
                label=f"M1: Vanilla SGen (n={len(m1_fdr)})",
                color=NQ_COLOR, density=True)

    m3_fdr = [r["shifted_test"]["fdr_e"] for r in iw_results["per_split"]]
    ax.hist(m3_fdr, bins=bins, alpha=0.5,
            label=f"M3: DS-SGen (n={len(m3_fdr)})",
            color=M3_COLOR, density=True)

    eps = iw_results["config"]["sgen"]["epsilon"]
    ax.axvline(eps, color="red", linestyle="--", linewidth=2, label=f"epsilon = {eps}")
    ax.set_xlabel("FDR-E (Shifted Domain)")
    ax.set_ylabel("Density")
    ax.set_title("FDR-E Distribution: Method 1 vs Method 3 (Shifted Domain)")
    ax.legend()
    _save(fig, "method3_fdr_distribution")


def plot_weight_analysis(iw_results):
    """2x2 weight analysis: diagnostics, n_eff, split outcomes, efficiency."""
    diag = iw_results["diagnostics"]
    ws = diag["weight_stats"]
    per_split = iw_results["per_split"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: weight stats text
    ax = axes[0, 0]
    stats_text = (
        f"Weight Statistics\n"
        f"n = {ws['n']}\n"
        f"n_eff = {ws['n_eff']:.1f} ({100*ws['n_eff_ratio']:.1f}%)\n"
        f"clip pctl = {ws['clip_percentile']}%\n"
        f"clip value = {ws['clip_value']:.3f}\n"
        f"min = {ws['weight_min']:.3f}\n"
        f"median = {ws['weight_median']:.3f}\n"
        f"max = {ws['weight_max']:.3f}\n"
        f"std = {ws['weight_std']:.3f}\n"
        f"raw max = {ws['raw_weight_max']:.3f}\n"
        f"\nClassifier CV acc = {diag['classifier_cv_accuracy']:.3f}"
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title("Weight Diagnostics")

    # Top-right: n_eff across splits
    ax = axes[0, 1]
    n_effs = [r["n_eff_total"] for r in per_split]
    ax.hist(n_effs, bins=30, color=M3_COLOR, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(n_effs), color="red", linestyle="--",
               label=f"mean = {np.mean(n_effs):.1f}")
    ax.set_xlabel("n_eff (per split)")
    ax.set_ylabel("Count")
    ax.set_title("Effective Sample Size Across Splits")
    ax.legend()

    # Bottom-left: split outcomes
    ax = axes[1, 0]
    n_abstain = sum(1 for r in per_split if r["shifted_test"]["n_selected"] == 0)
    n_valid_answering = sum(1 for r in per_split
                           if r["shifted_test"]["valid"] and r["shifted_test"]["n_selected"] > 0)
    n_invalid = sum(1 for r in per_split if not r["shifted_test"]["valid"])
    bars = ax.bar(["Abstain\n(vacuous)", "Valid\n(answering)", "Invalid"],
                  [n_abstain, n_valid_answering, n_invalid],
                  color=["gray", M3_COLOR, TQA_COLOR], alpha=0.8, edgecolor="black")
    for bar, val in zip(bars, [n_abstain, n_valid_answering, n_invalid]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', fontsize=11)
    ax.set_ylabel("Number of Splits (out of 100)")
    ax.set_title("Method 3: Split Outcomes on Shifted Domain")

    # Bottom-right: efficiency
    ax = axes[1, 1]
    shifted_eff = [r["shifted_test"]["efficiency"] for r in per_split]
    ax.hist(shifted_eff, bins=30, color=M3_COLOR, alpha=0.7, edgecolor="black")
    ax.axvline(np.mean(shifted_eff), color="red", linestyle="--",
               label=f"mean = {np.mean(shifted_eff):.3f}")
    ax.set_xlabel("Efficiency (Shifted Domain)")
    ax.set_ylabel("Count")
    ax.set_title("Selection Efficiency Across Splits")
    ax.legend()

    fig.suptitle("Method 3: Weight Analysis", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "method3_weight_analysis")


# ═══════════════════════════════════════════════════════════════════════════
# STAGE: Epsilon Sweep
# ═══════════════════════════════════════════════════════════════════════════

def plot_epsilon_sweep_validity(sweep):
    """THE HEADLINE FIGURE: Validity vs epsilon for all three methods."""
    epsilons = sweep["epsilons"]
    m1_vals = [sweep["method1"][str(e)]["shifted_validity"] for e in epsilons]
    m2_vals = [sweep["method2_optC"][str(e)]["shifted_validity"] for e in epsilons]
    m3_vals = [sweep["method3"][str(e)]["shifted_validity"] for e in epsilons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, [v * 100 for v in m1_vals], "o-", color=NQ_COLOR,
            linewidth=2, markersize=8, label="M1: Vanilla SGen")
    ax.plot(epsilons, [v * 100 for v in m2_vals], "s-", color=TQA_COLOR,
            linewidth=2, markersize=8, label="M2: Conservative (Option C)")
    ax.plot(epsilons, [v * 100 for v in m3_vals], "D-", color=M3_COLOR,
            linewidth=2, markersize=8, label="M3: DS-SGen (Ours)")
    ax.axhline(98, color="black", linestyle="--", linewidth=1.5,
               label="PAC target (98%)", alpha=0.7)

    ax.set_xlabel("epsilon (FDR-E target)", fontsize=12)
    ax.set_ylabel("Shifted Domain Validity Rate (%)", fontsize=12)
    ax.set_title("PAC Validity Under Domain Shift vs. Operating Point", fontsize=13)
    ax.set_ylim(0, 105)
    ax.set_xticks(epsilons)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    _save(fig, "epsilon_sweep_validity")


def plot_epsilon_sweep_efficiency(sweep):
    """Efficiency vs epsilon for all three methods."""
    epsilons = sweep["epsilons"]
    m1_eff = [sweep["method1"][str(e)]["shifted_mean_efficiency"] for e in epsilons]
    m2_eff = [sweep["method2_optC"][str(e)]["shifted_mean_efficiency"] for e in epsilons]
    m3_eff = [sweep["method3"][str(e)]["shifted_mean_efficiency"] for e in epsilons]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epsilons, [v * 100 for v in m1_eff], "o-", color=NQ_COLOR,
            linewidth=2, markersize=8, label="M1: Vanilla SGen")
    ax.plot(epsilons, [v * 100 for v in m2_eff], "s-", color=TQA_COLOR,
            linewidth=2, markersize=8, label="M2: Conservative (Option C)")
    ax.plot(epsilons, [v * 100 for v in m3_eff], "D-", color=M3_COLOR,
            linewidth=2, markersize=8, label="M3: DS-SGen (Ours)")

    ax.set_xlabel("epsilon (FDR-E target)", fontsize=12)
    ax.set_ylabel("Shifted Domain Efficiency (%)", fontsize=12)
    ax.set_title("Selection Efficiency Under Domain Shift vs. Operating Point", fontsize=13)
    ax.set_xticks(epsilons)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    _save(fig, "epsilon_sweep_efficiency")


def plot_three_method_comparison(sweep):
    """Two-panel: validity (left) and efficiency (right) at eps=0.25 and 0.35."""
    eps_vals = [0.25, 0.35]
    methods = ["M1\nVanilla", "M2\nConserv.", "M3\nDS-SGen"]
    colors = [NQ_COLOR, TQA_COLOR, M3_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(methods))
    width = 0.35

    for i, eps in enumerate(eps_vals):
        ek = str(eps)
        vals = [
            sweep["method1"].get(ek, {}).get("shifted_validity", 0) * 100,
            sweep["method2_optC"].get(ek, {}).get("shifted_validity", 0) * 100,
            sweep["method3"].get(ek, {}).get("shifted_validity", 0) * 100,
        ]
        offset = (i - 0.5) * width
        bars = ax1.bar(x + offset, vals, width, label=f"eps={eps}",
                       color=[colors[j] for j in range(3)],
                       edgecolor="black", alpha=0.85 if i == 0 else 0.55)
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.0f}%", ha='center', fontsize=9)

    ax1.axhline(98, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.set_ylabel("Shifted Domain Validity Rate (%)")
    ax1.set_title("Validity")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0, 110)
    ax1.legend()

    for i, eps in enumerate(eps_vals):
        ek = str(eps)
        effs = [
            sweep["method1"].get(ek, {}).get("shifted_mean_efficiency", 0) * 100,
            sweep["method2_optC"].get(ek, {}).get("shifted_mean_efficiency", 0) * 100,
            sweep["method3"].get(ek, {}).get("shifted_mean_efficiency", 0) * 100,
        ]
        offset = (i - 0.5) * width
        bars = ax2.bar(x + offset, effs, width, label=f"eps={eps}",
                       color=[colors[j] for j in range(3)],
                       edgecolor="black", alpha=0.85 if i == 0 else 0.55)
        for bar, val in zip(bars, effs):
            if val > 0.5:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha='center', fontsize=9)

    ax2.set_ylabel("Shifted Domain Efficiency (%)")
    ax2.set_title("Efficiency")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.legend()

    fig.suptitle("Three-Method Comparison Under Domain Shift", fontsize=14)
    fig.tight_layout()
    _save(fig, "three_method_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Main Dispatcher
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate DS-SGen plots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "generation", "entailment", "baseline",
                                 "conservative", "method3", "epsilon_sweep"])
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    cfg = load_config(args.config)
    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]

    print("DS-SGen Plot Generator")
    print(f"  Plot output: {PLOT_DIR}/")
    print()

    # Track counts
    total_plots = 0
    skipped_stages = []

    # --- Generation plots ---
    if args.stage in ("all", "generation"):
        nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
        tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))

        if nq_gen and tqa_gen:
            print(f"Generation plots (NQ={len(nq_gen)}, TQA={len(tqa_gen)}):")
            plot_fm1_histograms(nq_gen, tqa_gen)
            plot_answer_length_comparison(nq_gen, tqa_gen)
            plot_fm1_boxplot(nq_gen, tqa_gen)
            plot_fm1_cdf_comparison(nq_gen, tqa_gen)
            plot_sampled_answer_diversity(nq_gen, tqa_gen)
            plot_sampled_logprob_spread(nq_gen, tqa_gen)
            total_plots += 6
        else:
            skipped_stages.append("generation (cache incomplete)")
            if nq_gen:
                print(f"  NQ generations: {len(nq_gen)} (complete)")
            if tqa_gen:
                print(f"  TQA generations: {len(tqa_gen)} (partial)")

    # --- Entailment plots ---
    if args.stage in ("all", "entailment"):
        nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
        tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))
        nq_ent = load_cache(get_cache_path(cache_dir, "nq_entailment"))
        tqa_ent = load_cache(get_cache_path(cache_dir, "tqa_entailment"))

        if nq_ent and tqa_ent:
            print(f"\nEntailment plots (NQ={len(nq_ent)}, TQA={len(tqa_ent)}):")
            plot_entailment_scores(nq_ent, tqa_ent)
            plot_fm2_distribution(nq_ent, tqa_ent)
            plot_correctness_rate_by_domain(nq_ent, tqa_ent)
            total_plots += 3

            if nq_gen and tqa_gen:
                plot_fm1_vs_fm2_scatter(nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_fm1_roc_curve(nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_calibration_curve(nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_correctness_by_fm1_decile(nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_fm2_conditional_on_fm1(nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_domain_shift_diagnostic(nq_gen, nq_ent, tqa_gen, tqa_ent)
                total_plots += 6
        else:
            skipped_stages.append("entailment (cache not yet available)")

    # --- Baseline results plots ---
    if args.stage in ("all", "baseline"):
        baseline_path = os.path.join(results_dir, "baseline_results.json")
        baseline = load_cache(baseline_path)

        if baseline:
            print("\nBaseline plots:")
            plot_fdr_distribution(baseline)
            plot_efficiency_distribution(baseline)
            plot_validity_bar(baseline)
            plot_fdr_vs_efficiency_scatter(baseline)
            plot_threshold_stability(baseline)
            total_plots += 5

            # Plots that need generation + entailment data too
            nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
            tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))
            nq_ent = load_cache(get_cache_path(cache_dir, "nq_entailment"))
            tqa_ent = load_cache(get_cache_path(cache_dir, "tqa_entailment"))

            if nq_gen and tqa_gen and nq_ent and tqa_ent:
                plot_cumulative_fdr_curve(baseline, nq_gen, nq_ent, tqa_gen, tqa_ent)
                plot_abstention_analysis(baseline, nq_gen, nq_ent, tqa_gen, tqa_ent)
                total_plots += 2
        else:
            skipped_stages.append("baseline (results not yet available)")

    # --- Conservative threshold plots (Method 2) ---
    if args.stage in ("all", "conservative"):
        cons_path = os.path.join(results_dir, "conservative_results.json")
        cons = load_cache(cons_path)
        baseline_path = os.path.join(results_dir, "baseline_results.json")
        baseline = load_cache(baseline_path)

        if cons:
            print("\nConservative threshold plots:")
            plot_validity_efficiency_tradeoff(cons)
            total_plots += 1
            if baseline:
                plot_method_comparison_table(baseline, cons)
                total_plots += 1
        else:
            skipped_stages.append("conservative (results not yet available)")

    # --- Method 3: Importance Weighted plots ---
    if args.stage in ("all", "method3"):
        iw_path = os.path.join(results_dir, "importance_weighted_results.json")
        iw_results = load_cache(iw_path)
        baseline_path = os.path.join(results_dir, "baseline_results.json")
        baseline = load_cache(baseline_path)

        if iw_results:
            print("\nMethod 3 plots:")
            plot_method3_fdr_distribution(baseline, iw_results)
            plot_weight_analysis(iw_results)
            total_plots += 2
        else:
            skipped_stages.append("method3 (results not yet available)")

    # --- Epsilon sweep plots ---
    if args.stage in ("all", "epsilon_sweep"):
        sweep_path = os.path.join(results_dir, "epsilon_sweep_results.json")
        sweep = load_cache(sweep_path)

        if sweep:
            print("\nEpsilon sweep plots:")
            plot_epsilon_sweep_validity(sweep)
            plot_epsilon_sweep_efficiency(sweep)
            plot_three_method_comparison(sweep)
            total_plots += 3
        else:
            skipped_stages.append("epsilon_sweep (results not yet available)")

    # --- Summary ---
    print(f"\n{'=' * 50}")
    print(f"Total plots generated: {total_plots}")
    if skipped_stages:
        print(f"Skipped stages: {', '.join(skipped_stages)}")
    print("Done.")


if __name__ == "__main__":
    main()
