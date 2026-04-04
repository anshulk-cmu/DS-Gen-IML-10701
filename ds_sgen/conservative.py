"""Method 2: Conservative Threshold — naive domain-shift fix for SGen-Semi.

This module is self-contained: it reimplements the SGen-Semi split logic with
three conservative options, each designed to restore PAC FDR-E validity under
domain shift at the cost of reduced selection efficiency.

Options:
  A — Safety factor on (tau1, tau2) after grid search.
      tau1 += log(gamma)  (fM1 is log-scale)
      tau2 *= gamma        (fM2 is in [0,1])

  B — Reduced epsilon in the grid search constraint.
      Use epsilon_eff = epsilon / k instead of epsilon.
      Evaluate validity against the *original* epsilon for fair comparison.

  C — Delta budget allocation for potential domain shift.
      Reserve a fraction of delta for shift: delta_cp = delta - delta_p - delta_s.
      Smaller delta_adj widens Clopper-Pearson bounds, making selection stricter.

Paper reference:
  Lee et al., "Selective Generation for Controllable LMs" (NeurIPS 2024)
  Algorithm 2 (SGen-Semi) — extended here with conservative modifications.
"""

import logging
import time

import numpy as np
from scipy.stats import beta as beta_dist

from ds_sgen.utils import save_cache

logger = logging.getLogger(__name__)


# ── Helpers (same math as sgen_semi.py, kept here for self-containment) ──────

def _merge_records(records, generations, entailments):
    """Merge data, generation, and entailment results into unified records."""
    merged = []
    for rec, gen, ent in zip(records, generations, entailments):
        merged.append({
            "idx": rec["idx"],
            "question": rec["question"],
            "reference_answer": rec["reference_answer"],
            "greedy_answer": gen["greedy_answer"],
            "fM1": gen["mean_logprob"],
            "fM2": ent["fM2"],
            "entail_score": ent["entail_score"],
            "entail_label": ent["entail_label"],
            "dataset": rec["dataset"],
        })
    return merged


def _compute_conformal_threshold(correct_scores: np.ndarray, epsilon_e: float) -> float:
    """Conformal threshold from correct answers' scores: epsilon_e quantile."""
    n = len(correct_scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * epsilon_e))
    if k < 1:
        k = 1
    if k > n:
        return float(correct_scores.max())
    sorted_scores = np.sort(correct_scores)
    return float(sorted_scores[k - 1])


def _build_percentile_grid(values: np.ndarray, n_grid: int) -> np.ndarray:
    """Build threshold grid from percentiles of observed values."""
    percentiles = np.linspace(0, 100, n_grid)
    grid = np.percentile(values, percentiles)
    return np.unique(grid)


def _clopper_pearson_upper(failures: int, total: int, alpha: float) -> float:
    """Clopper-Pearson upper confidence bound on P(failure)."""
    if total == 0:
        return 0.0
    if failures == total:
        return 1.0
    return float(beta_dist.ppf(1 - alpha, failures + 1, total - failures))


# ── Core: single-split with conservative overrides ──────────────────────────

def _run_single_split(
    cal_merged: list[dict],
    shifted_merged: list[dict],
    split_seed: int,
    sgen_cfg: dict,
    *,
    epsilon_effective: float | None = None,
    delta_shift: float = 0.0,
    tau_safety_factor: float = 1.0,
) -> dict:
    """Run one calibration/test split with conservative modifications.

    This mirrors sgen_semi._run_single_split() but injects three knobs:
      epsilon_effective  — Option B (reduced epsilon in grid constraint)
      delta_shift        — Option C (delta budget reserved for shift)
      tau_safety_factor  — Option A (post-hoc threshold inflation)

    The evaluation always uses the *original* epsilon for fair validity checks.
    """
    epsilon = sgen_cfg["epsilon"]
    if epsilon_effective is None:
        epsilon_effective = epsilon

    delta = sgen_cfg["delta"]
    delta_p = sgen_cfg["delta_p"]
    cal_frac = sgen_cfg["cal_frac"]
    zu_frac = sgen_cfg["zu_frac"]
    epsilon_e = sgen_cfg["epsilon_e"]
    n_grid = sgen_cfg["n_grid"]
    selection_mode = sgen_cfg.get("selection_mode", "fm1_only")

    n_cal = len(cal_merged)
    rng = np.random.RandomState(split_seed)

    # Step 1: Split calibration dataset into cal and in-domain test
    indices = rng.permutation(n_cal)
    cal_size = int(np.floor(n_cal * cal_frac))
    cal_idx = indices[:cal_size]
    test_idx = indices[cal_size:]

    cal_data = [cal_merged[i] for i in cal_idx]
    indomain_test = [cal_merged[i] for i in test_idx]

    # Step 2: Split calibration into Z_U (unlabeled) and Z_E (labeled)
    zu_size = int(np.floor(len(cal_data) * zu_frac))
    z_u = cal_data[:zu_size]
    z_e = cal_data[zu_size:]

    # Step 3: Conformal threshold from Z_E correct answers' scores
    ze_correct_scores = np.array([r["entail_score"] for r in z_e if r["entail_label"] == 1])
    tau_cp = _compute_conformal_threshold(ze_correct_scores, epsilon_e)

    # Step 4: Pseudo-label Z_U
    for r in z_u:
        r["pseudo_label"] = 1 if r["entail_score"] >= tau_cp else 0

    # Step 5: Grid search (fM1-only by default)
    zu_fM1 = np.array([r["fM1"] for r in z_u])
    zu_fM2 = np.array([r["fM2"] for r in z_u])
    zu_pseudo = np.array([r["pseudo_label"] for r in z_u])

    tau1_grid = _build_percentile_grid(zu_fM1, n_grid)
    tau2_grid = _build_percentile_grid(zu_fM2, n_grid)

    if selection_mode == "fm1_only":
        H = len(tau1_grid)
    elif selection_mode == "fm2_only":
        H = len(tau2_grid)
    else:
        H = len(tau1_grid) * len(tau2_grid)

    # ── Option C: reduced delta budget ──
    delta_cp = delta - delta_p - delta_shift
    if delta_cp <= 0:
        abstain = {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
                   "n_selected": 0, "n_total": 0}
        return {
            "split_seed": split_seed, "cal_size": len(cal_data),
            "zu_size": len(z_u), "ze_size": len(z_e), "tau_cp": tau_cp,
            "tau1": None, "tau2": None, "grid_size_H": H,
            "epsilon_effective": epsilon_effective,
            "delta_shift": delta_shift, "tau_safety_factor": tau_safety_factor,
            "indomain_test": {**abstain, "n_total": len(indomain_test)},
            "shifted_test": {**abstain, "n_total": len(shifted_merged)},
        }
    delta_adj = delta_cp / H if H > 0 else delta_cp

    best_tau1, best_tau2 = None, None
    best_efficiency = -1.0

    if selection_mode == "fm1_only":
        for t1 in tau1_grid:
            sel = zu_fM1 >= t1
            m = int(sel.sum())
            if m == 0:
                continue
            failures = int((sel & (zu_pseudo == 0)).sum())
            cp_upper = _clopper_pearson_upper(failures, m, delta_adj)
            if cp_upper <= epsilon_effective:
                efficiency = m / len(z_u)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_tau1 = t1
                    best_tau2 = 0.0
    else:
        for t1 in tau1_grid:
            selected = zu_fM1 >= t1
            for t2 in tau2_grid:
                sel = selected & (zu_fM2 >= t2)
                m = int(sel.sum())
                if m == 0:
                    continue
                failures = int((sel & (zu_pseudo == 0)).sum())
                cp_upper = _clopper_pearson_upper(failures, m, delta_adj)
                if cp_upper <= epsilon_effective:
                    efficiency = m / len(z_u)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_tau1 = t1
                        best_tau2 = t2

    # ── Option A: inflate thresholds by safety factor ──
    if best_tau1 is not None and tau_safety_factor != 1.0:
        best_tau1 = best_tau1 + np.log(tau_safety_factor)  # fM1 is log-scale

    # Step 6: Evaluate on test sets (always against original epsilon)
    def _evaluate(data, tau1):
        if tau1 is None:
            return {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
                    "n_selected": 0, "n_total": len(data)}
        fM1 = np.array([r["fM1"] for r in data])
        labels = np.array([r["entail_label"] for r in data])

        if selection_mode == "fm1_only":
            selected = fM1 >= tau1
        else:
            fM2 = np.array([r["fM2"] for r in data])
            selected = (fM1 >= tau1) & (fM2 >= best_tau2)

        n_selected = int(selected.sum())
        if n_selected == 0:
            return {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
                    "n_selected": 0, "n_total": len(data)}

        n_wrong = int((selected & (labels == 0)).sum())
        fdr_e = n_wrong / n_selected
        efficiency = n_selected / len(data)
        valid = fdr_e <= epsilon
        return {"fdr_e": fdr_e, "efficiency": efficiency, "valid": valid,
                "n_selected": n_selected, "n_total": len(data)}

    indomain_result = _evaluate(indomain_test, best_tau1)
    shifted_result = _evaluate(shifted_merged, best_tau1)

    return {
        "split_seed": split_seed,
        "cal_size": len(cal_data),
        "zu_size": len(z_u),
        "ze_size": len(z_e),
        "tau_cp": tau_cp,
        "tau1": best_tau1,
        "tau2": best_tau2,
        "grid_size_H": H,
        "epsilon_effective": epsilon_effective,
        "delta_shift": delta_shift,
        "tau_safety_factor": tau_safety_factor,
        "indomain_test": indomain_result,
        "shifted_test": shifted_result,
    }


# ── Sweep runner for a single option ────────────────────────────────────────

def _run_sweep(
    cal_merged: list[dict],
    shifted_merged: list[dict],
    sgen_cfg: dict,
    base_seed: int,
    n_splits: int,
    cal_label: str,
    shifted_label: str,
    *,
    epsilon_effective: float | None = None,
    delta_shift: float = 0.0,
    tau_safety_factor: float = 1.0,
    label: str = "",
) -> dict:
    """Run n_splits with given conservative overrides, return aggregated metrics."""
    per_split = []
    for s in range(n_splits):
        split_seed = base_seed + s
        result = _run_single_split(
            cal_merged, shifted_merged, split_seed, sgen_cfg,
            epsilon_effective=epsilon_effective,
            delta_shift=delta_shift,
            tau_safety_factor=tau_safety_factor,
        )
        per_split.append(result)

    id_fdr = [r["indomain_test"]["fdr_e"] for r in per_split]
    id_eff = [r["indomain_test"]["efficiency"] for r in per_split]
    id_val = [r["indomain_test"]["valid"] for r in per_split]
    sh_fdr = [r["shifted_test"]["fdr_e"] for r in per_split]
    sh_eff = [r["shifted_test"]["efficiency"] for r in per_split]
    sh_val = [r["shifted_test"]["valid"] for r in per_split]

    summary = {
        "indomain": {
            "label": cal_label,
            "validity_rate": float(np.mean(id_val)),
            "mean_fdr_e": float(np.mean(id_fdr)),
            "std_fdr_e": float(np.std(id_fdr)),
            "mean_efficiency": float(np.mean(id_eff)),
            "std_efficiency": float(np.std(id_eff)),
        },
        "shifted": {
            "label": shifted_label,
            "validity_rate": float(np.mean(sh_val)),
            "mean_fdr_e": float(np.mean(sh_fdr)),
            "std_fdr_e": float(np.std(sh_fdr)),
            "mean_efficiency": float(np.mean(sh_eff)),
            "std_efficiency": float(np.std(sh_eff)),
        },
        "per_split": per_split,
    }

    logger.info("  %-30s | %s valid=%.2f eff=%.3f | %s valid=%.2f eff=%.3f",
                label, cal_label,
                summary["indomain"]["validity_rate"], summary["indomain"]["mean_efficiency"],
                shifted_label,
                summary["shifted"]["validity_rate"], summary["shifted"]["mean_efficiency"])

    return summary


# ── Public API ──────────────────────────────────────────────────────────────

def run_conservative_experiment(
    cfg: dict,
    nq_records: list[dict], nq_gen: list[dict], nq_ent: list[dict],
    tqa_records: list[dict], tqa_gen: list[dict], tqa_ent: list[dict],
) -> dict:
    """Run Method 2: Conservative Threshold with all three options.

    Each option is swept over multiple parameter values. Results are saved
    to {results_dir}/conservative_results.json.

    Returns dict with keys "option_a", "option_b", "option_c", each
    containing sub-dicts keyed by the sweep parameter value.
    """
    sgen_cfg = cfg["sgen"]
    cons_cfg = cfg["conservative"]
    base_seed = cfg["seed"]
    n_splits = cons_cfg.get("n_splits") or sgen_cfg["n_splits"]
    epsilon = sgen_cfg["epsilon"]
    delta = sgen_cfg["delta"]
    delta_p = sgen_cfg["delta_p"]

    logger.info("=" * 60)
    logger.info("Method 2: Conservative Threshold")
    logger.info("=" * 60)
    logger.info("  Base epsilon=%.3f, delta=%.4f, n_splits=%d", epsilon, delta, n_splits)

    t0 = time.time()

    # Merge records once
    nq_merged = _merge_records(nq_records, nq_gen, nq_ent)
    tqa_merged = _merge_records(tqa_records, tqa_gen, tqa_ent)
    logger.info("  Merged %d NQ + %d TQA records", len(nq_merged), len(tqa_merged))

    # Determine calibration direction
    cal_dataset = sgen_cfg.get("cal_dataset", "tqa")
    if cal_dataset == "tqa":
        cal_merged, shifted_merged = tqa_merged, nq_merged
        cal_label, shifted_label = "TQA", "NQ"
    else:
        cal_merged, shifted_merged = nq_merged, tqa_merged
        cal_label, shifted_label = "NQ", "TQA"
    logger.info("  Cal: %s (%d), Shifted: %s (%d)",
                cal_label, len(cal_merged), shifted_label, len(shifted_merged))

    results = {}

    # ── Option A: Safety Factor on Thresholds ──
    safety_factors = cons_cfg["safety_factors"]
    logger.info("")
    logger.info("Option A: Safety Factor on Thresholds (gamma)")
    logger.info("  tau1 += log(gamma)")
    logger.info("  Sweep: %s", safety_factors)

    option_a = {}
    for gamma in safety_factors:
        label = f"gamma={gamma:.1f}"
        option_a[str(gamma)] = _run_sweep(
            cal_merged, shifted_merged, sgen_cfg, base_seed, n_splits,
            cal_label, shifted_label,
            tau_safety_factor=gamma,
            label=label,
        )
    results["option_a"] = option_a

    # ── Option B: Reduced Epsilon ──
    epsilon_divisors = cons_cfg["epsilon_divisors"]
    logger.info("")
    logger.info("Option B: Reduced Epsilon in Grid Search")

    option_b = {}
    for k in epsilon_divisors:
        eps_eff = epsilon / k
        label = f"eps_div={k:.1f} (eps_eff={eps_eff:.3f})"
        option_b[str(k)] = _run_sweep(
            cal_merged, shifted_merged, sgen_cfg, base_seed, n_splits,
            cal_label, shifted_label,
            epsilon_effective=eps_eff,
            label=label,
        )
    results["option_b"] = option_b

    # ── Option C: Delta Budget Allocation ──
    delta_shift_fracs = cons_cfg["delta_shift_fracs"]
    logger.info("")
    logger.info("Option C: Delta Budget Allocation for Shift")

    option_c = {}
    for frac in delta_shift_fracs:
        ds = frac * (delta - delta_p)
        label = f"frac={frac:.2f} (delta_s={ds:.6f})"
        option_c[str(frac)] = _run_sweep(
            cal_merged, shifted_merged, sgen_cfg, base_seed, n_splits,
            cal_label, shifted_label,
            delta_shift=ds,
            label=label,
        )
    results["option_c"] = option_c

    elapsed = time.time() - t0
    logger.info("")
    logger.info("Method 2 complete in %.1f seconds", elapsed)

    # Save results
    results_path = f"{cfg['paths']['results_dir']}/conservative_results.json"
    # Strip per_split data for the saved file to keep it manageable
    save_data = {
        "config": {
            "sgen": sgen_cfg,
            "conservative": cons_cfg,
            "seed": cfg["seed"],
        },
        "option_a": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                     for k, v in option_a.items()},
        "option_b": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                     for k, v in option_b.items()},
        "option_c": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                     for k, v in option_c.items()},
    }
    save_cache(save_data, results_path)
    logger.info("  Results saved to %s", results_path)

    return results


def print_conservative_summary(results: dict):
    """Print formatted comparison table for all conservative options."""
    print()
    print("=" * 90)
    print("METHOD 2: CONSERVATIVE THRESHOLD RESULTS")
    print("=" * 90)

    # Get labels from first option
    first_opt = next(iter(results["option_a"].values()))
    id_label = first_opt["indomain"]["label"]
    sh_label = first_opt["shifted"]["label"]

    header = (f"  {'Setting':<32} | {id_label+' Vld':>8} {id_label+' FDR':>8} {id_label+' Eff':>8}"
              f" | {sh_label+' Vld':>8} {sh_label+' FDR':>8} {sh_label+' Eff':>8}")
    sep = "  " + "-" * 86

    # Option A
    print(f"\n  Option A: Safety Factor on Thresholds (tau1 += log(gamma))")
    print(header)
    print(sep)
    for key, val in results["option_a"].items():
        idr, shr = val["indomain"], val["shifted"]
        label = f"gamma = {key}"
        print(f"  {label:<32} | {idr['validity_rate']:>7.1%} {idr['mean_fdr_e']:>8.4f}"
              f" {idr['mean_efficiency']:>8.4f}"
              f" | {shr['validity_rate']:>7.1%} {shr['mean_fdr_e']:>8.4f}"
              f" {shr['mean_efficiency']:>8.4f}")

    # Option B
    print(f"\n  Option B: Reduced Epsilon (grid uses eps/k, evaluate against original)")
    print(header)
    print(sep)
    for key, val in results["option_b"].items():
        idr, shr = val["indomain"], val["shifted"]
        k = float(key)
        eps_eff = 0.25 / k
        label = f"eps/{key} = {eps_eff:.3f}"
        print(f"  {label:<32} | {idr['validity_rate']:>7.1%} {idr['mean_fdr_e']:>8.4f}"
              f" {idr['mean_efficiency']:>8.4f}"
              f" | {shr['validity_rate']:>7.1%} {shr['mean_fdr_e']:>8.4f}"
              f" {shr['mean_efficiency']:>8.4f}")

    # Option C
    print(f"\n  Option C: Delta Budget Allocation (reserve frac of delta for shift)")
    print(header)
    print(sep)
    for key, val in results["option_c"].items():
        idr, shr = val["indomain"], val["shifted"]
        label = f"frac = {key}"
        print(f"  {label:<32} | {idr['validity_rate']:>7.1%} {idr['mean_fdr_e']:>8.4f}"
              f" {idr['mean_efficiency']:>8.4f}"
              f" | {shr['validity_rate']:>7.1%} {shr['mean_fdr_e']:>8.4f}"
              f" {shr['mean_efficiency']:>8.4f}")

    print()
    print("=" * 90)
    print("  Key: Valid = P(FDR-E <= 0.25) across 100 splits | "
          "FDR-E = mean empirical error | Eff = selection rate")
    print("=" * 90)
    print()
