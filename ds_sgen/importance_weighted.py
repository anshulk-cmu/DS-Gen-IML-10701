"""Method 3: DS-SGen with Importance Reweighting.

Extends SGen-Semi (Lee et al., NeurIPS 2024) with density-ratio-based importance
weighting from DS-CP (Lin et al., 2025). Calibration samples are reweighted by
estimated P_target(x)/P_cal(x) so that the conformal threshold and PAC bounds
are calibrated for the shifted test domain.

Pipeline:
  1. Embed all prompts with a sentence transformer (all-MiniLM-L6-v2).
  2. Train a logistic regression domain classifier on embeddings.
  3. Convert classifier probabilities to density-ratio importance weights.
  4. Use weighted conformal prediction for pseudo-labeling.
  5. Use weighted Clopper-Pearson bounds (via effective sample size) for PAC-FDR.

Paper references:
  - Lee et al., "Selective Generation for Controllable LMs" (NeurIPS 2024)
  - Lin et al., "Domain-Shift-Aware Conformal Prediction for LLMs" (arXiv 2025)
  - Tibshirani et al., "Conformal Prediction Under Covariate Shift" (NeurIPS 2019)
"""

import logging
import os

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ds_sgen.sgen_semi import (
    _merge_records,
    _build_percentile_grid,
    _clopper_pearson_upper,
)
from ds_sgen.utils import save_cache, load_cache

logger = logging.getLogger(__name__)


# ── Embedding ─────────────────────────────────────────────────────────────────

def compute_embeddings(
    questions: list[str],
    model_name: str,
    cache_folder: str | None = None,
) -> np.ndarray:
    """Embed questions using a sentence transformer.

    Returns (N, D) float32 array (D=384 for all-MiniLM-L6-v2).
    """
    from sentence_transformers import SentenceTransformer

    logger.info("  Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name, cache_folder=cache_folder)

    logger.info("  Encoding %d questions...", len(questions))
    embeddings = model.encode(
        questions,
        show_progress_bar=True,
        batch_size=256,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ── Domain classifier ─────────────────────────────────────────────────────────

def train_domain_classifier(
    cal_embeddings: np.ndarray,
    shifted_embeddings: np.ndarray,
    C: float = 1.0,
) -> tuple:
    """Train logistic regression to distinguish calibration (0) from shifted (1).

    Returns (fitted_classifier, 5-fold_cv_accuracy).
    """
    X = np.concatenate([cal_embeddings, shifted_embeddings], axis=0)
    y = np.concatenate([
        np.zeros(len(cal_embeddings)),
        np.ones(len(shifted_embeddings)),
    ])

    clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")

    # 5-fold cross-validation for diagnostic accuracy
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    cv_accuracy = float(cv_scores.mean())
    logger.info("  Domain classifier 5-fold CV accuracy: %.3f (+/- %.3f)",
                cv_accuracy, cv_scores.std())

    # Fit on all data for final classifier
    clf.fit(X, y)
    return clf, cv_accuracy


# ── Importance weights ────────────────────────────────────────────────────────

def compute_importance_weights(
    classifier,
    cal_embeddings: np.ndarray,
    clip_percentile: float = 95.0,
) -> tuple[np.ndarray, dict]:
    """Compute density-ratio importance weights for calibration samples.

    w(x) = p_hat(x) / (1 - p_hat(x))  where p_hat = P(shifted | x).

    Returns (weights, diagnostics_dict).
    """
    n = len(cal_embeddings)

    # Predicted probability of being from the shifted (target) domain
    p_hat = classifier.predict_proba(cal_embeddings)[:, 1]

    # Clamp to avoid division by zero
    p_hat = np.clip(p_hat, 0.01, 0.99)

    # Raw density ratio
    raw_weights = p_hat / (1.0 - p_hat)

    # Clip extreme weights
    clip_val = np.percentile(raw_weights, clip_percentile)
    clipped_weights = np.minimum(raw_weights, clip_val)

    # Normalize so weights sum to n
    weights = clipped_weights * (n / clipped_weights.sum())

    # Effective sample size
    n_eff = (weights.sum()) ** 2 / (weights ** 2).sum()

    diagnostics = {
        "n": n,
        "n_eff": float(n_eff),
        "n_eff_ratio": float(n_eff / n),
        "clip_percentile": clip_percentile,
        "clip_value": float(clip_val),
        "weight_min": float(weights.min()),
        "weight_median": float(np.median(weights)),
        "weight_max": float(weights.max()),
        "weight_std": float(weights.std()),
        "raw_weight_max": float(raw_weights.max()),
    }

    return weights, diagnostics


# ── Weighted conformal threshold ──────────────────────────────────────────────

def _weighted_conformal_threshold(
    correct_scores: np.ndarray,
    weights: np.ndarray,
    epsilon_e: float,
) -> float:
    """Weighted quantile threshold for conformal pseudo-labeling.

    Finds smallest q such that:
        sum(w_i * 1{s_i <= q}) / sum(w_i) >= epsilon_e

    This is the weighted analog of _compute_conformal_threshold from sgen_semi.
    """
    n = len(correct_scores)
    if n == 0:
        return float("inf")

    # Sort scores ascending, reorder weights to match
    order = np.argsort(correct_scores)
    sorted_scores = correct_scores[order]
    sorted_weights = weights[order]

    # Cumulative weighted fraction
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()

    # Find first index where cumulative >= epsilon_e
    idx = np.searchsorted(cumulative, epsilon_e, side="left")

    if idx >= n:
        # Cumulative never reaches epsilon_e — return max score
        return float(sorted_scores[-1])

    return float(sorted_scores[idx])


# ── Weighted Clopper-Pearson bound ────────────────────────────────────────────

def _weighted_clopper_pearson_upper(
    weighted_failure_rate: float,
    alpha: float,
    n_eff: float,
) -> float:
    """Clopper-Pearson upper bound using weighted failure rate and effective sample size.

    Args:
        weighted_failure_rate: sum(w_i for failures) / sum(w_i for selected),
            i.e. the importance-weighted estimate of the target-domain failure rate.
        alpha: significance level for one-sided bound.
        n_eff: effective sample size of the selected subset.
    """
    if n_eff < 5:
        return 1.0  # vacuous — too few effective samples
    if weighted_failure_rate <= 0.0:
        return float(beta_dist.ppf(1 - alpha, 1, n_eff))
    if weighted_failure_rate >= 1.0:
        return 1.0

    failures_eff = weighted_failure_rate * n_eff

    # Guard against float rounding
    failures_eff = max(0.0, min(n_eff - 0.001, failures_eff))
    successes_eff = n_eff - failures_eff

    return float(beta_dist.ppf(1 - alpha, failures_eff + 1, successes_eff))


# ── Single split ──────────────────────────────────────────────────────────────

def _run_single_split(
    cal_merged: list[dict],
    shifted_merged: list[dict],
    cal_weights: np.ndarray,
    split_seed: int,
    sgen_cfg: dict,
    iw_cfg: dict,
) -> dict:
    """Run one calibration/test split of importance-weighted SGen-Semi.

    Mirrors sgen_semi._run_single_split() with weighted conformal threshold
    and weighted Clopper-Pearson bounds.
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

    # ── Step 1: Split calibration into cal/in-domain-test ──
    indices = rng.permutation(n_cal)
    cal_size = int(np.floor(n_cal * cal_frac))
    cal_idx = indices[:cal_size]
    test_idx = indices[cal_size:]

    # Split BOTH data and weights using the SAME index array
    cal_data = [cal_merged[i] for i in cal_idx]
    cal_data_weights = cal_weights[cal_idx]

    indomain_test = [cal_merged[i] for i in test_idx]

    # ── Step 2: Split cal into Z_U (unlabeled) and Z_E (labeled) ──
    zu_size = int(np.floor(len(cal_data) * zu_frac))
    z_u = cal_data[:zu_size]
    z_u_weights = cal_data_weights[:zu_size]
    z_e = cal_data[zu_size:]
    z_e_weights = cal_data_weights[zu_size:]

    # ── Step 3: Weighted conformal threshold from Z_E correct answers ──
    ze_correct_mask = np.array([r["entail_label"] == 1 for r in z_e])
    ze_scores = np.array([r["entail_score"] for r in z_e])

    if ze_correct_mask.sum() == 0:
        tau_cp = float("inf")
    else:
        ze_correct_scores = ze_scores[ze_correct_mask]
        ze_correct_weights = z_e_weights[ze_correct_mask]
        tau_cp = _weighted_conformal_threshold(
            ze_correct_scores, ze_correct_weights, epsilon_e
        )

    # ── Step 4: Pseudo-label Z_U ──
    for r in z_u:
        r["pseudo_label"] = 1 if r["entail_score"] >= tau_cp else 0

    # ── Step 5: Grid search with weighted bounds ──
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
    best_n_eff_sel = 0.0

    if selection_mode == "fm1_only":
        for t1 in tau1_grid:
            sel = zu_fM1 >= t1
            m = int(sel.sum())
            if m == 0:
                continue
            fail_mask = sel & (zu_pseudo == 0)

            # Weighted failure rate and n_eff for the SELECTED subset
            sel_weights = z_u_weights[sel]
            fail_weights = z_u_weights[fail_mask]
            w_fail_rate = fail_weights.sum() / sel_weights.sum()
            n_eff_sel = (sel_weights.sum()) ** 2 / (sel_weights ** 2).sum()

            cp_upper = _weighted_clopper_pearson_upper(
                w_fail_rate, delta_adj, n_eff_sel
            )
            if cp_upper <= epsilon:
                efficiency = m / len(z_u)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_tau1 = t1
                    best_tau2 = 0.0
                    best_n_eff_sel = n_eff_sel

    elif selection_mode == "fm2_only":
        for t2 in tau2_grid:
            sel = zu_fM2 >= t2
            m = int(sel.sum())
            if m == 0:
                continue
            fail_mask = sel & (zu_pseudo == 0)

            sel_weights = z_u_weights[sel]
            fail_weights = z_u_weights[fail_mask]
            w_fail_rate = fail_weights.sum() / sel_weights.sum()
            n_eff_sel = (sel_weights.sum()) ** 2 / (sel_weights ** 2).sum()

            cp_upper = _weighted_clopper_pearson_upper(
                w_fail_rate, delta_adj, n_eff_sel
            )
            if cp_upper <= epsilon:
                efficiency = m / len(z_u)
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_tau1 = -float("inf")
                    best_tau2 = t2
                    best_n_eff_sel = n_eff_sel

    else:  # "both"
        for t1 in tau1_grid:
            pre_sel = zu_fM1 >= t1
            for t2 in tau2_grid:
                sel = pre_sel & (zu_fM2 >= t2)
                m = int(sel.sum())
                if m == 0:
                    continue
                fail_mask = sel & (zu_pseudo == 0)

                sel_weights = z_u_weights[sel]
                fail_weights = z_u_weights[fail_mask]
                w_fail_rate = fail_weights.sum() / sel_weights.sum()
                n_eff_sel = (sel_weights.sum()) ** 2 / (sel_weights ** 2).sum()

                cp_upper = _weighted_clopper_pearson_upper(
                    w_fail_rate, delta_adj, n_eff_sel
                )
                if cp_upper <= epsilon:
                    efficiency = m / len(z_u)
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_tau1 = t1
                        best_tau2 = t2
                        best_n_eff_sel = n_eff_sel

    # ── Step 6: Evaluate on test sets (unweighted — actual FDR-E) ──
    def _evaluate(data: list[dict], tau1, tau2):
        if tau1 is None:
            return {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
                    "n_selected": 0, "n_total": len(data)}

        fM1 = np.array([r["fM1"] for r in data])
        fM2 = np.array([r["fM2"] for r in data])
        labels = np.array([r["entail_label"] for r in data])

        if selection_mode == "fm1_only":
            selected = fM1 >= tau1
        elif selection_mode == "fm2_only":
            selected = fM2 >= tau2
        else:
            selected = (fM1 >= tau1) & (fM2 >= tau2)

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

    # n_eff for the full Z_U (diagnostic)
    n_eff_total = (z_u_weights.sum()) ** 2 / (z_u_weights ** 2).sum()

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
        "n_eff_total": float(n_eff_total),
        "n_eff_selected": float(best_n_eff_sel),
        "mean_weight": float(z_u_weights.mean()),
        "indomain_test": indomain_result,
        "shifted_test": shifted_result,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_experiment(
    cfg: dict,
    nq_records: list[dict], nq_generations: list[dict], nq_entailments: list[dict],
    tqa_records: list[dict], tqa_generations: list[dict], tqa_entailments: list[dict],
) -> dict:
    """Run Method 3: Importance-Weighted SGen-Semi.

    1. Embed all prompts.
    2. Train domain classifier.
    3. Compute importance weights for calibration dataset.
    4. Run n_splits splits with weighted conformal + weighted PAC bounds.
    """
    sgen_cfg = cfg["sgen"]
    iw_cfg = cfg["importance_weighted"]
    n_splits = iw_cfg.get("n_splits") or sgen_cfg["n_splits"]
    base_seed = cfg["seed"]
    cal_dataset = sgen_cfg.get("cal_dataset", "tqa")

    logger.info("=" * 60)
    logger.info("Method 3: DS-SGen with Importance Reweighting")
    logger.info("=" * 60)
    logger.info("  epsilon=%.3f, delta=%.4f, n_splits=%d",
                sgen_cfg["epsilon"], sgen_cfg["delta"], n_splits)
    logger.info("  embedding_model=%s, classifier=%s, clip=%d%%",
                iw_cfg["embedding_model"], iw_cfg["classifier"],
                iw_cfg["weight_clip_percentile"])

    # ── Merge records ──
    nq_merged = _merge_records(nq_records, nq_generations, nq_entailments)
    tqa_merged = _merge_records(tqa_records, tqa_generations, tqa_entailments)

    if cal_dataset == "tqa":
        cal_merged, shifted_merged = tqa_merged, nq_merged
        cal_label, shifted_label = "TQA", "NQ"
        cal_questions = [r["question"] for r in tqa_records]
        shifted_questions = [r["question"] for r in nq_records]
    else:
        cal_merged, shifted_merged = nq_merged, tqa_merged
        cal_label, shifted_label = "NQ", "TQA"
        cal_questions = [r["question"] for r in nq_records]
        shifted_questions = [r["question"] for r in tqa_records]

    logger.info("  Calibration: %s (%d), Shifted: %s (%d)",
                cal_label, len(cal_merged), shifted_label, len(shifted_merged))

    # ── Step 1: Compute embeddings (with caching) ──
    cache_dir = cfg["paths"]["cache_dir"]
    model_name = iw_cfg["embedding_model"]
    hf_cache = cfg["paths"].get("hf_cache")

    cal_emb_path = os.path.join(cache_dir, f"{cal_label.lower()}_embeddings.npy")
    shifted_emb_path = os.path.join(cache_dir, f"{shifted_label.lower()}_embeddings.npy")

    logger.info("")
    logger.info("Step 1: Computing embeddings")

    if os.path.exists(cal_emb_path) and os.path.exists(shifted_emb_path):
        logger.info("  Loading cached embeddings")
        cal_embeddings = np.load(cal_emb_path)
        shifted_embeddings = np.load(shifted_emb_path)
    else:
        cal_embeddings = compute_embeddings(cal_questions, model_name, hf_cache)
        shifted_embeddings = compute_embeddings(shifted_questions, model_name, hf_cache)
        np.save(cal_emb_path, cal_embeddings)
        np.save(shifted_emb_path, shifted_embeddings)
        logger.info("  Cached embeddings to %s, %s", cal_emb_path, shifted_emb_path)

    logger.info("  Cal embeddings: %s, Shifted embeddings: %s",
                cal_embeddings.shape, shifted_embeddings.shape)

    # ── Step 2: Train domain classifier ──
    logger.info("")
    logger.info("Step 2: Training domain classifier")
    classifier, cv_accuracy = train_domain_classifier(
        cal_embeddings, shifted_embeddings, C=iw_cfg["classifier_C"]
    )

    # ── Step 3: Compute importance weights ──
    logger.info("")
    logger.info("Step 3: Computing importance weights")
    weights, weight_diag = compute_importance_weights(
        classifier, cal_embeddings, iw_cfg["weight_clip_percentile"]
    )

    logger.info("  n_eff = %.1f / %d (%.1f%%)",
                weight_diag["n_eff"], weight_diag["n"],
                100 * weight_diag["n_eff_ratio"])
    logger.info("  Weights: min=%.3f, median=%.3f, max=%.3f, std=%.3f",
                weight_diag["weight_min"], weight_diag["weight_median"],
                weight_diag["weight_max"], weight_diag["weight_std"])

    # ── Step 4: Run splits ──
    logger.info("")
    logger.info("Step 4: Running %d splits", n_splits)

    per_split = []
    for s in range(n_splits):
        split_seed = base_seed + s
        result = _run_single_split(
            cal_merged, shifted_merged, weights,
            split_seed, sgen_cfg, iw_cfg,
        )
        per_split.append(result)

        if (s + 1) % 10 == 0:
            id_vals = [r["indomain_test"]["valid"] for r in per_split]
            sh_vals = [r["shifted_test"]["valid"] for r in per_split]
            logger.info("  Split %d/%d: %s validity=%.2f, %s validity=%.2f",
                        s + 1, n_splits,
                        cal_label, np.mean(id_vals),
                        shifted_label, np.mean(sh_vals))

    # ── Aggregate ──
    id_fdr = [r["indomain_test"]["fdr_e"] for r in per_split]
    id_eff = [r["indomain_test"]["efficiency"] for r in per_split]
    id_val = [r["indomain_test"]["valid"] for r in per_split]
    sh_fdr = [r["shifted_test"]["fdr_e"] for r in per_split]
    sh_eff = [r["shifted_test"]["efficiency"] for r in per_split]
    sh_val = [r["shifted_test"]["valid"] for r in per_split]
    n_effs = [r["n_eff_total"] for r in per_split]

    summary = {
        "config": {
            "sgen": sgen_cfg,
            "importance_weighted": iw_cfg,
        },
        "cal_dataset": cal_dataset,
        "selection_mode": sgen_cfg.get("selection_mode", "fm1_only"),
        "diagnostics": {
            "classifier_cv_accuracy": cv_accuracy,
            "weight_stats": weight_diag,
            "mean_n_eff_across_splits": float(np.mean(n_effs)),
        },
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

    # ── Save results ──
    results_path = f"{cfg['paths']['results_dir']}/importance_weighted_results.json"
    save_cache(summary, results_path)
    logger.info("\n  Results saved to %s", results_path)

    return summary


def print_importance_weighted_summary(results: dict):
    """Print formatted summary of Method 3 results."""
    diag = results["diagnostics"]
    id_r = results["indomain"]
    sh_r = results["shifted"]

    print()
    print("=" * 70)
    print("METHOD 3: DS-SGen WITH IMPORTANCE REWEIGHTING")
    print("=" * 70)

    print(f"\n  Diagnostics:")
    print(f"    Domain classifier CV accuracy: {diag['classifier_cv_accuracy']:.3f}")
    ws = diag["weight_stats"]
    print(f"    n_eff = {ws['n_eff']:.1f} / {ws['n']} ({100*ws['n_eff_ratio']:.1f}%)")
    print(f"    Weights: min={ws['weight_min']:.3f}, median={ws['weight_median']:.3f}, "
          f"max={ws['weight_max']:.3f}, std={ws['weight_std']:.3f}")

    print(f"\n  {id_r['label']} (in-domain, calibration):")
    print(f"    Validity rate:   {id_r['validity_rate']:>7.2%}  (target: >= 98%)")
    print(f"    Mean FDR-E:      {id_r['mean_fdr_e']:.4f} +/- {id_r['std_fdr_e']:.4f}")
    print(f"    Mean efficiency: {id_r['mean_efficiency']:.4f} +/- {id_r['std_efficiency']:.4f}")

    print(f"\n  {sh_r['label']} (shifted test):")
    print(f"    Validity rate:   {sh_r['validity_rate']:>7.2%}  (target: >= 98%)")
    print(f"    Mean FDR-E:      {sh_r['mean_fdr_e']:.4f} +/- {sh_r['std_fdr_e']:.4f}")
    print(f"    Mean efficiency: {sh_r['mean_efficiency']:.4f} +/- {sh_r['std_efficiency']:.4f}")

    print()
    print("=" * 70)
