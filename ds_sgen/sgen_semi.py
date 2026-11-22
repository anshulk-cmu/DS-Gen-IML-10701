"""Method 1 (M1): a simplified SGen-Semi-style baseline.

This is NOT a faithful reimplementation of Algorithm 2 (SGen-Semi) of Lee et al.,
"Selective Generation for Controllable Language Models" (NeurIPS 2024). It keeps
their idea (conformal pseudo-labels on unlabelled data, then a PAC grid search
for the selection threshold) but implements only one bound. Per split:

  1. tau_CP = conformal lower quantile, at level epsilon_e, of the NLI entailment
     probability s = P(ENT) over the correct-answer subset of Z_E
     (k = ceil((n_c + 1) * epsilon_e), tau_CP = k-th smallest score). For a new
     correct answer exchangeable with those in Z_E this gives
     Pr(s < tau_CP | e = 1) <= epsilon_e, marginally over Z_E.
  2. Pseudo-labels on Z_U: e_hat = 1[s >= tau_CP].
  3. For each grid point tau in H (|H| = 20 percentiles of f_M1 on Z_U), one
     Clopper-Pearson upper bound, at level alpha = (delta - delta_p) / |H|, on
     NER(tau) = Pr(e_hat = 0 | f_M1 >= tau); accept tau if the bound is <= epsilon
     and return the accepted tau with the largest selection rate. If no tau is
     accepted, the split abstains on every question (counted as valid, FDR-E 0).

So the code bounds the selected pseudo-label non-entailment rate only (Bonferroni
over |H|). Lee et al.'s SGen-Semi additionally bounds FER and FNER with labelled
data, searches over epsilon_E values, and splits delta = delta_W + delta_S + delta_E;
none of that is implemented here, and delta_p is subtracted but used by no other
bound. Their Theorem 1 therefore does not transfer to this code, and M1's in-domain
validity is an empirical observation. Because FDR-E = FER - FNER + NER <= FER + NER
and e, e_hat are both thresholds of the same NLI output, the NER bound controls the
NLI-label FDR-E up to the FER term, which can be non-zero only when tau_CP <= 1/2
(s > 1/2 forces argmax = ENT, i.e. e = 1).

Selection modes: fM1 only (used in all reported results), fM2 only, or fM1 AND fM2.
The calibration dataset is configurable (reported runs: calibrate on TQA, shifted
test on NQ).
"""

import numpy as np
from scipy.stats import beta as beta_dist

from ds_sgen.utils import save_cache


def _merge_records(records: list[dict], generations: list[dict], entailments: list[dict]) -> list[dict]:
    """Merge data, generation, and entailment results into unified per-question records."""
    merged = []
    for rec, gen, ent in zip(records, generations, entailments):
        merged.append({
            "idx": rec["idx"],
            "question": rec["question"],
            "reference_answer": rec["reference_answer"],
            "greedy_answer": gen["greedy_answer"],
            "fM1": gen["mean_logprob"],        # Mean log-prob (higher = more confident)
            "fM2": ent["fM2"],                 # Self-consistency (higher = more consistent)
            "entail_score": ent["entail_score"],  # P(entailment), continuous
            "entail_label": ent["entail_label"],  # Binary correctness
            "dataset": rec["dataset"],
        })
    return merged


def _compute_conformal_threshold(correct_scores: np.ndarray, epsilon_e: float) -> float:
    """Compute split conformal prediction threshold from correct answers' scores.

    tau_CP = epsilon_e quantile of correct answers' entailment scores.
    A new correct answer exchangeable with these n scores falls below tau_CP
    exactly when its rank among the n + 1 scores is at most k, so (without ties)
    Pr(score < tau_CP) = k / (n + 1), which lies in [epsilon_e, epsilon_e + 1/(n+1)).
    The marginal guarantee is therefore score >= tau_CP with probability
    1 - k/(n+1), approximately (not at least) 1 - epsilon_e.

    tau_CP = sorted_correct_scores[k-1] where k = ceil((n+1) * epsilon_e).
    Points with score >= tau_CP are pseudo-labeled as "correct".
    """
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
    """Build a grid of thresholds from percentiles of observed values.

    Uses n_grid evenly-spaced percentiles (0th to 100th).
    Deduplicates and sorts the result.
    """
    percentiles = np.linspace(0, 100, n_grid)
    grid = np.percentile(values, percentiles)
    return np.unique(grid)


def _clopper_pearson_upper(failures: int, total: int, alpha: float) -> float:
    """Clopper-Pearson upper confidence bound for binomial proportion.

    Returns upper bound on P(failure) given `failures` out of `total` trials.
    """
    if total == 0:
        return 0.0
    if failures == total:
        return 1.0
    return float(beta_dist.ppf(1 - alpha, failures + 1, total - failures))


def _select(fM1: np.ndarray, fM2: np.ndarray, tau1, tau2, mode: str) -> np.ndarray:
    """Apply selection rule based on mode."""
    if mode == "fm1_only":
        return fM1 >= tau1
    elif mode == "fm2_only":
        return fM2 >= tau2
    else:  # "both"
        return (fM1 >= tau1) & (fM2 >= tau2)


def _run_single_split(
    cal_merged: list[dict],
    shifted_merged: list[dict],
    split_seed: int,
    sgen_cfg: dict,
) -> dict:
    """Run one calibration/test split of SGen-Semi.

    cal_merged: the calibration dataset (split into cal/in-domain-test)
    shifted_merged: the shifted test dataset (evaluated whole)
    """
    epsilon = sgen_cfg["epsilon"]
    delta = sgen_cfg["delta"]
    delta_p = sgen_cfg["delta_p"]
    cal_frac = sgen_cfg["cal_frac"]
    zu_frac = sgen_cfg["zu_frac"]
    epsilon_e = sgen_cfg["epsilon_e"]
    n_grid = sgen_cfg["n_grid"]
    selection_mode = sgen_cfg.get("selection_mode", "fm1_only")

    n_cal = len(cal_merged)
    rng = np.random.RandomState(split_seed)

    # Step 1: Split calibration dataset into calibration and in-domain test
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

    # Step 3: Conformal threshold from Z_E correct answers' entailment scores
    ze_correct_scores = np.array([r["entail_score"] for r in z_e if r["entail_label"] == 1])
    tau_cp = _compute_conformal_threshold(ze_correct_scores, epsilon_e)

    # Step 4: Pseudo-label Z_U
    for r in z_u:
        r["pseudo_label"] = 1 if r["entail_score"] >= tau_cp else 0

    # Step 5: Grid search for threshold(s) based on selection mode
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

    delta_cp = delta - delta_p
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
            if cp_upper <= epsilon:
                efficiency = m / len(z_u)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_tau1 = t1
                    best_tau2 = 0.0  # placeholder — not used in selection
    elif selection_mode == "fm2_only":
        for t2 in tau2_grid:
            sel = zu_fM2 >= t2
            m = int(sel.sum())
            if m == 0:
                continue
            failures = int((sel & (zu_pseudo == 0)).sum())
            cp_upper = _clopper_pearson_upper(failures, m, delta_adj)
            if cp_upper <= epsilon:
                efficiency = m / len(z_u)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_tau1 = -float("inf")  # placeholder
                    best_tau2 = t2
    else:  # "both"
        for t1 in tau1_grid:
            selected = zu_fM1 >= t1
            for t2 in tau2_grid:
                sel = selected & (zu_fM2 >= t2)
                m = int(sel.sum())
                if m == 0:
                    continue
                failures = int((sel & (zu_pseudo == 0)).sum())
                cp_upper = _clopper_pearson_upper(failures, m, delta_adj)
                if cp_upper <= epsilon:
                    efficiency = m / len(z_u)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_tau1 = t1
                        best_tau2 = t2

    # Step 6: Evaluate on test sets
    def _evaluate(data: list[dict], tau1, tau2):
        if tau1 is None:
            return {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
                    "n_selected": 0, "n_total": len(data)}

        fM1 = np.array([r["fM1"] for r in data])
        fM2 = np.array([r["fM2"] for r in data])
        labels = np.array([r["entail_label"] for r in data])

        selected = _select(fM1, fM2, tau1, tau2, selection_mode)
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

    indomain_result = _evaluate(indomain_test, best_tau1, best_tau2)
    shifted_result = _evaluate(shifted_merged, best_tau1, best_tau2)

    return {
        "split_seed": split_seed,
        "cal_size": len(cal_data),
        "zu_size": len(z_u),
        "ze_size": len(z_e),
        "tau_cp": tau_cp,
        "tau1": best_tau1,
        "tau2": best_tau2,
        "grid_size_H": H,
        "selection_mode": selection_mode,
        "indomain_test": indomain_result,
        "shifted_test": shifted_result,
    }


def run_experiment(
    cfg: dict,
    nq_records: list[dict], nq_generations: list[dict], nq_entailments: list[dict],
    tqa_records: list[dict], tqa_generations: list[dict], tqa_entailments: list[dict],
) -> dict:
    """Run SGen-Semi with n_splits random calibration splits.

    Calibrates on the dataset specified by sgen.cal_dataset (default: "tqa").
    Evaluates in-domain (held-out cal data) and shifted (the other dataset).
    """
    sgen_cfg = cfg["sgen"]
    n_splits = sgen_cfg["n_splits"]
    base_seed = cfg["seed"]
    cal_dataset = sgen_cfg.get("cal_dataset", "tqa")
    selection_mode = sgen_cfg.get("selection_mode", "fm1_only")

    print("Stage 4: Running SGen-Semi algorithm")
    print(f"  epsilon={sgen_cfg['epsilon']}, delta={sgen_cfg['delta']}, "
          f"n_splits={n_splits}, n_grid={sgen_cfg['n_grid']}")
    print(f"  cal_dataset={cal_dataset}, selection_mode={selection_mode}")

    nq_merged = _merge_records(nq_records, nq_generations, nq_entailments)
    tqa_merged = _merge_records(tqa_records, tqa_generations, tqa_entailments)

    # Determine which dataset is calibration vs shifted test
    if cal_dataset == "tqa":
        cal_merged, shifted_merged = tqa_merged, nq_merged
        cal_label, shifted_label = "TQA", "NQ"
    else:
        cal_merged, shifted_merged = nq_merged, tqa_merged
        cal_label, shifted_label = "NQ", "TQA"

    print(f"  Calibration: {cal_label} ({len(cal_merged)} questions)")
    print(f"  Shifted test: {shifted_label} ({len(shifted_merged)} questions)")

    per_split = []
    for s in range(n_splits):
        split_seed = base_seed + s
        result = _run_single_split(cal_merged, shifted_merged, split_seed, sgen_cfg)
        per_split.append(result)

        if (s + 1) % 10 == 0:
            indomain_vals = [r["indomain_test"]["valid"] for r in per_split]
            shifted_vals = [r["shifted_test"]["valid"] for r in per_split]
            print(f"  Split {s+1}/{n_splits}: "
                  f"{cal_label} validity={np.mean(indomain_vals):.2f}, "
                  f"{shifted_label} validity={np.mean(shifted_vals):.2f}")

    # Aggregate
    id_fdr = [r["indomain_test"]["fdr_e"] for r in per_split]
    id_eff = [r["indomain_test"]["efficiency"] for r in per_split]
    id_val = [r["indomain_test"]["valid"] for r in per_split]
    sh_fdr = [r["shifted_test"]["fdr_e"] for r in per_split]
    sh_eff = [r["shifted_test"]["efficiency"] for r in per_split]
    sh_val = [r["shifted_test"]["valid"] for r in per_split]

    summary = {
        "config": sgen_cfg,
        "cal_dataset": cal_dataset,
        "selection_mode": selection_mode,
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

    # Save results
    results_path = f"{cfg['paths']['results_dir']}/baseline_results.json"
    save_cache(summary, results_path)
    print(f"\n  Results saved to {results_path}")

    return summary
