"""Final covariate-shift plots. Run after run_synthetic_eps.py."""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS = "/data/user_data/anshulk/dsgen/results"
OUT = "/home/anshulk/ds-gen-10701/plots"


def _load(path):
    with open(path) as f:
        return json.load(f)


def plot_scorecard():
    sc = _load(f"{RESULTS}/synthetic_final_screening.json")
    tests = ["1", "2a", "2b", "3", "4", "5", "6"]
    keys = ["acc_S", "acc_T", "acc_top5", "gap", "acc_clf", "ess_ratio", "quartile_spread"]
    vals = [sc[k] for k in keys]
    passes = [sc[f"pass_{t}"] for t in tests]
    colors = ["#2a9d8f" if p else "#e76f51" for p in passes]
    plt.figure(figsize=(8, 3.5))
    bars = plt.bar(tests, vals, color=colors)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{v:+.3f}", ha="center", fontsize=8)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("screening test"); plt.ylabel("value")
    n_pass = sum(passes)
    plt.title(f"Synthetic pair — screening scorecard ({n_pass}/7 pass)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_final_scorecard.png", dpi=140)
    plt.close()


def plot_weight_quartile():
    wq = _load(f"{RESULTS}/synthetic_final_weight_quartile.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)

    for ax, (label, key) in zip(axes, [("Synthetic (covariate)", "synthetic"),
                                        ("TQA → NQ (concept)", "tqa_nq")]):
        if key not in wq:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(label)
            continue
        qs = wq[key]["quartile_accs"]
        spread = wq[key]["Q1_minus_Q4"]
        labels = ["Q1\n(source-like)", "Q2", "Q3", "Q4\n(target-like)"]
        color = "#2a9d8f" if spread >= 0.05 else "#e76f51"
        bars = ax.bar(labels, qs, color=color, alpha=0.85)
        for bar, v in zip(bars, qs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", fontsize=8)
        tag = "covariate" if spread >= 0.05 else "concept"
        ax.set_title(f"{label}\nQ1 − Q4 = {spread:+.3f}  ({tag} signature)")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy")

    plt.suptitle("Weight-quartile accuracy: covariate vs concept shift", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_final_weight_quartile.png", dpi=140,
                bbox_inches="tight")
    plt.close()


def _extract_curves(sweep):
    """Return (epsilons, m1_valid, m1_eff, m3_valid, m3_eff, m3_vac)."""
    eps = [e["epsilon"] for e in sweep]
    m1_v = [e["m1"]["validity_rate"] for e in sweep]
    m1_e = [e["m1"]["mean_efficiency"] for e in sweep]
    m3_v = [e["m3"]["validity_rate"] for e in sweep]
    m3_e = [e["m3"]["mean_efficiency"] for e in sweep]
    m3_vac = [e["m3"]["vacuous_frac"] for e in sweep]
    return eps, m1_v, m1_e, m3_v, m3_e, m3_vac


def plot_validity_vs_eps():
    sw = _load(f"{RESULTS}/synthetic_final_eps_sweep.json")

    plt.figure(figsize=(8, 4.2))
    eps_s, m1_v_s, _, m3_v_s, _, _ = _extract_curves(sw["synthetic"])
    plt.plot(eps_s, m1_v_s, "o-", label="M1 synthetic (covariate)",
             color="#4a90d9", lw=2)
    plt.plot(eps_s, m3_v_s, "s-", label="M3 synthetic (covariate)",
             color="#2a9d8f", lw=2)

    if sw.get("tqa_nq"):
        eps_t, m1_v_t, _, m3_v_t, _, m3_vac_t = _extract_curves(sw["tqa_nq"])
        plt.plot(eps_t, m1_v_t, "o--", label="M1 TQA→NQ (concept)",
                 color="#4a90d9", lw=1.5, alpha=0.6)
        plt.plot(eps_t, m3_v_t, "s--", label="M3 TQA→NQ (concept)",
                 color="#e76f51", lw=1.5, alpha=0.6)
        for e, v, vac in zip(eps_t, m3_v_t, m3_vac_t):
            plt.annotate(f"vac={vac:.2f}", xy=(e, v), xytext=(5, -12),
                         textcoords="offset points", fontsize=7, alpha=0.7,
                         color="#e76f51")

    plt.axhline(0.98, ls=":", color="red", lw=1, label="PAC target (1−δ=0.98)")
    plt.xlabel("epsilon (FDR-E target)")
    plt.ylabel("shifted validity rate")
    plt.title("Validity vs epsilon — the crossover")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_final_validity_vs_eps.png", dpi=140)
    plt.close()


def plot_efficiency_vs_eps():
    sw = _load(f"{RESULTS}/synthetic_final_eps_sweep.json")

    plt.figure(figsize=(8, 4.2))
    eps_s, _, m1_e_s, _, m3_e_s, _ = _extract_curves(sw["synthetic"])
    plt.plot(eps_s, m1_e_s, "o-", label="M1 synthetic (covariate)",
             color="#4a90d9", lw=2)
    plt.plot(eps_s, m3_e_s, "s-", label="M3 synthetic (covariate)",
             color="#2a9d8f", lw=2)

    if sw.get("tqa_nq"):
        eps_t, _, m1_e_t, _, m3_e_t, _ = _extract_curves(sw["tqa_nq"])
        plt.plot(eps_t, m1_e_t, "o--", label="M1 TQA→NQ (concept)",
                 color="#4a90d9", lw=1.5, alpha=0.6)
        plt.plot(eps_t, m3_e_t, "s--", label="M3 TQA→NQ (concept)",
                 color="#e76f51", lw=1.5, alpha=0.6)

    plt.xlabel("epsilon (FDR-E target)")
    plt.ylabel("mean shifted efficiency")
    plt.title("Efficiency vs epsilon")
    plt.ylim(0, 1.0)
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/synthetic_final_efficiency_vs_eps.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_scorecard()
    plot_weight_quartile()
    plot_validity_vs_eps()
    plot_efficiency_vs_eps()
    print(f"plots written to {OUT}")
