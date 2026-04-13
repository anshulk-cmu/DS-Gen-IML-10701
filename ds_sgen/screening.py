"""Dataset pair screening protocol for DS-SGen.

Pre-flight 6-test battery to determine whether a candidate (source, target)
dataset pair has the right shift structure for importance-weighted SGen to
succeed.  Each test has a go/no-go threshold calibrated against the known-bad
TQA→NQ case.

Tests:
  1. Source accuracy floor         — acc_S ≥ 1 − ε + 0.05
  2a. Target accuracy floor        — acc_T ≥ 1 − ε
  2b. Reachable floor (top-5%)     — acc_top5 ≥ 1 − ε + 0.05
  3. Accuracy gap                  — 0.03 ≤ gap ≤ 0.15
  4. Domain classifier accuracy    — 0.55 ≤ acc_clf ≤ 0.78
  5. Effective sample size ratio   — ess_ratio ≥ 0.50
  6. Quartile spread (concept test)— spread ≥ 0.05

Also includes a PopQA data loader that splits by subject popularity into
head (top 60%) and tail (bottom 20%) domains.
"""

import logging
import os

import numpy as np
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ds_sgen.utils import get_cache_path, load_cache, save_cache

logger = logging.getLogger(__name__)


def _setup_file_logger(log_dir: str):
    """Add a file handler to the module logger if one doesn't exist yet."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "screening.log"))
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


# ── PopQA data loading ───────────────────────────────────────────────────────

def load_popqa(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load PopQA and split into head (source) and tail (target) by popularity.

    Head = top 60% by s_pop (subject Wikipedia page views).
    Tail = bottom 20% by s_pop.

    Each split is sampled to screening.sample_size (default 1000).
    Returns (head_records, tail_records) in the project's normalized schema.
    """
    _setup_file_logger(cfg.get("log_dir", "logs"))

    scfg = cfg["screening"]
    cache_dir = cfg["paths"]["cache_dir"]

    logger.info("=" * 60)
    logger.info("[SCREENING] Loading PopQA dataset")
    logger.info("  dataset=%s, split=%s", scfg["dataset"], scfg["split"])
    logger.info("  head_cutoff=%.2f, tail_cutoff=%.2f, sample_size=%d",
                scfg["head_cutoff"], scfg["tail_cutoff"], scfg["sample_size"])

    head_cache = get_cache_path(cache_dir, "popqa_head_data")
    tail_cache = get_cache_path(cache_dir, "popqa_tail_data")

    cached_head = load_cache(head_cache)
    cached_tail = load_cache(tail_cache)
    if cached_head is not None and cached_tail is not None:
        logger.info("PopQA: loaded %d head + %d tail from cache (%s, %s)",
                     len(cached_head), len(cached_tail), head_cache, tail_cache)
        print(f"  PopQA: loaded {len(cached_head)} head + {len(cached_tail)} tail from cache")
        return cached_head, cached_tail

    from datasets import load_dataset

    logger.info("PopQA: downloading from HuggingFace (dataset=%s, split=%s)",
                scfg["dataset"], scfg["split"])
    print("  PopQA: downloading from HuggingFace...")
    ds = load_dataset(
        scfg["dataset"],
        split=scfg["split"],
        cache_dir=cfg["paths"]["hf_cache"],
    )
    logger.info("PopQA: downloaded %d raw examples", len(ds))

    # Sort by subject popularity (s_pop)
    pops = np.array([float(ex["s_pop"]) for ex in ds])
    sorted_idx = np.argsort(pops)

    n = len(ds)
    head_cutoff = scfg["head_cutoff"]   # top 60%
    tail_cutoff = scfg["tail_cutoff"]   # bottom 20%

    tail_end = int(n * tail_cutoff)
    head_start = int(n * (1.0 - head_cutoff))

    tail_indices = sorted_idx[:tail_end]
    head_indices = sorted_idx[head_start:]

    logger.info("PopQA: popularity split complete")
    logger.info("  Total examples: %d", n)
    logger.info("  Head (top %.0f%%): %d examples (indices %d-%d)",
                head_cutoff * 100, len(head_indices), head_start, n - 1)
    logger.info("  Tail (bottom %.0f%%): %d examples (indices 0-%d)",
                tail_cutoff * 100, len(tail_indices), tail_end - 1)

    # Log popularity ranges
    head_pops = pops[head_indices]
    tail_pops = pops[tail_indices]
    logger.info("  Head s_pop: min=%.1f, max=%.1f, median=%.1f, mean=%.1f",
                head_pops.min(), head_pops.max(), np.median(head_pops), head_pops.mean())
    logger.info("  Tail s_pop: min=%.1f, max=%.1f, median=%.1f, mean=%.1f",
                tail_pops.min(), tail_pops.max(), np.median(tail_pops), tail_pops.mean())
    logger.info("  Popularity ratio (head median / tail median): %.1f",
                np.median(head_pops) / max(np.median(tail_pops), 1))
    print(f"  Head s_pop: [{head_pops.min():.1f}, {head_pops.max():.1f}], "
          f"median={np.median(head_pops):.1f}")
    print(f"  Tail s_pop: [{tail_pops.min():.1f}, {tail_pops.max():.1f}], "
          f"median={np.median(tail_pops):.1f}")

    # Sample
    sample_size = scfg["sample_size"]
    seed = cfg["seed"]
    rng = np.random.RandomState(seed)

    logger.info("  Sampling %d per domain (seed=%d)", sample_size, seed)
    if len(head_indices) > sample_size:
        head_indices = rng.choice(head_indices, sample_size, replace=False)
    if len(tail_indices) > sample_size:
        tail_indices = rng.choice(tail_indices, sample_size, replace=False)

    def _to_records(indices, dataset_label):
        records = []
        for i, idx in enumerate(indices):
            ex = ds[int(idx)]
            records.append({
                "idx": i,
                "question": ex["question"],
                "reference_answer": ex["obj"],
                "all_answers": [ex["obj"]],
                "dataset": dataset_label,
                "s_pop": float(ex["s_pop"]),
                "subj": ex["subj"],
                "prop": ex["prop"],
            })
        return records

    head_records = _to_records(head_indices, "popqa_head")
    tail_records = _to_records(tail_indices, "popqa_tail")

    # Log sample statistics
    head_sample_pops = np.array([r["s_pop"] for r in head_records])
    tail_sample_pops = np.array([r["s_pop"] for r in tail_records])
    logger.info("  Sampled head s_pop: min=%.1f, max=%.1f, median=%.1f",
                head_sample_pops.min(), head_sample_pops.max(), np.median(head_sample_pops))
    logger.info("  Sampled tail s_pop: min=%.1f, max=%.1f, median=%.1f",
                tail_sample_pops.min(), tail_sample_pops.max(), np.median(tail_sample_pops))

    # Log a few example questions from each domain
    for label, recs in [("HEAD", head_records), ("TAIL", tail_records)]:
        logger.debug("  Sample %s questions:", label)
        for r in recs[:5]:
            logger.debug("    [s_pop=%.0f] Q: '%s' A: '%s'",
                         r["s_pop"], r["question"][:80], r["reference_answer"][:40])

    save_cache(head_records, head_cache)
    save_cache(tail_records, tail_cache)
    logger.info("PopQA: cached %d head to %s", len(head_records), head_cache)
    logger.info("PopQA: cached %d tail to %s", len(tail_records), tail_cache)
    print(f"  PopQA: sampled {len(head_records)} head + {len(tail_records)} tail")

    return head_records, tail_records


# ── Screening tests ──────────────────────────────────────────────────────────

def run_screening_tests(
    y_S: np.ndarray,
    y_T: np.ndarray,
    fM_S: np.ndarray,
    fM_T: np.ndarray,
    emb_S: np.ndarray,
    emb_T: np.ndarray,
    epsilon: float = 0.25,
    classifier_C: float = 1.0,
    w_clip: tuple[float, float] = (0.01, 100.0),
) -> dict:
    """Run the 6-test screening battery on a candidate dataset pair.

    Args:
        y_S: Binary correctness labels for source (1=correct, 0=wrong).
        y_T: Binary correctness labels for target.
        fM_S: Generator confidence scores for source (mean logprob / fM1).
        fM_T: Generator confidence scores for target.
        emb_S: Sentence embeddings for source questions (n_S, D).
        emb_T: Sentence embeddings for target questions (n_T, D).
        epsilon: Target FDR-E level.
        classifier_C: Logistic regression regularization (inverse).
        w_clip: (min, max) for clipping raw density ratios.

    Returns:
        Dict with all test values, pass/fail flags, and diagnostics.
    """
    n_S = len(y_S)
    n_T = len(y_T)

    logger.info("=" * 60)
    logger.info("[SCREENING] Running 6-test screening battery")
    logger.info("  epsilon=%.3f, n_S=%d, n_T=%d, emb_dim=%d",
                epsilon, n_S, n_T, emb_S.shape[1])
    logger.info("  classifier_C=%.2f, w_clip=(%.2f, %.2f)",
                classifier_C, w_clip[0], w_clip[1])

    threshold_1 = 1.0 - epsilon + 0.05
    threshold_2 = 1.0 - epsilon

    # ── Test 1: Source accuracy ──
    acc_S = float(y_S.mean())
    pass_1 = acc_S >= threshold_1
    logger.info("")
    logger.info("Test 1 — Source accuracy floor")
    logger.info("  acc_S = %.3f (correct: %d / %d)", acc_S, int(y_S.sum()), n_S)
    logger.info("  threshold: >= %.2f", threshold_1)
    logger.info("  result: %s", "PASS" if pass_1 else "FAIL")

    # ── Test 2a: Target accuracy ──
    acc_T = float(y_T.mean())
    pass_2a = acc_T >= threshold_2
    soft_2a = acc_T >= threshold_2 - 0.05
    logger.info("")
    logger.info("Test 2a — Target accuracy floor")
    logger.info("  acc_T = %.3f (correct: %d / %d)", acc_T, int(y_T.sum()), n_T)
    logger.info("  threshold: >= %.2f (soft: >= %.2f)", threshold_2, threshold_2 - 0.05)
    logger.info("  result: %s", "PASS" if pass_2a else ("SOFT" if soft_2a else "FAIL"))

    # ── Test 2b: Reachable floor (top-5% of target by confidence) ──
    top5_k = max(1, len(fM_T) // 20)
    top5_idx = np.argsort(-fM_T)[:top5_k]
    acc_top5 = float(y_T[top5_idx].mean())
    pass_2b = acc_top5 >= threshold_1
    logger.info("")
    logger.info("Test 2b — Reachable floor (top-5%% of target by fM1)")
    logger.info("  top5_k = %d, acc_top5 = %.3f (correct: %d / %d)",
                top5_k, acc_top5, int(y_T[top5_idx].sum()), top5_k)
    logger.info("  fM1 range in top-5%%: [%.4f, %.4f]",
                float(fM_T[top5_idx].min()), float(fM_T[top5_idx].max()))
    logger.info("  threshold: >= %.2f", threshold_1)
    logger.info("  result: %s", "PASS" if pass_2b else "FAIL")

    # ── Test 3: Accuracy gap ──
    gap = acc_S - acc_T
    pass_3 = 0.03 <= gap <= 0.15
    logger.info("")
    logger.info("Test 3 — Accuracy gap (shift severity)")
    logger.info("  gap = acc_S - acc_T = %.3f - %.3f = %.3f", acc_S, acc_T, gap)
    logger.info("  threshold: [0.03, 0.15]")
    if gap < 0.03:
        logger.info("  note: gap too small — shift may be too mild for a story")
    elif gap > 0.15:
        logger.info("  note: gap too large — may indicate concept shift")
    logger.info("  result: %s", "PASS" if pass_3 else "FAIL")

    # ── Test 4: Domain classifier ──
    logger.info("")
    logger.info("Test 4 — Domain classifier separability")
    X = np.vstack([emb_S, emb_T])
    d = np.concatenate([np.zeros(len(emb_S)), np.ones(len(emb_T))])

    clf = LogisticRegression(C=classifier_C, max_iter=1000, solver="lbfgs")
    logger.info("  Training logistic regression (C=%.2f, max_iter=1000)", classifier_C)
    cv_scores = cross_val_score(clf, X, d, cv=5, scoring="accuracy")
    acc_clf = float(cv_scores.mean())
    acc_clf_std = float(cv_scores.std())
    logger.info("  5-fold CV scores: %s", [f"{s:.3f}" for s in cv_scores])
    logger.info("  acc_clf = %.3f +/- %.3f", acc_clf, acc_clf_std)
    logger.info("  threshold: [0.55, 0.78]")
    pass_4 = 0.55 <= acc_clf <= 0.78
    if acc_clf < 0.55:
        logger.info("  note: domains nearly indistinguishable — vanilla SGen may suffice")
    elif acc_clf > 0.78:
        logger.info("  note: domains too separable — weights will collapse")
    logger.info("  result: %s", "PASS" if pass_4 else "FAIL")

    # Fit on all data for weights
    clf.fit(X, d)
    logger.info("  Final classifier fitted on all %d samples", len(X))

    # ── Importance weights on source ──
    p_tgt = clf.predict_proba(emb_S)[:, 1]
    p_tgt = np.clip(p_tgt, 0.01, 0.99)
    raw_w = p_tgt / (1.0 - p_tgt)
    w = np.clip(raw_w, w_clip[0], w_clip[1])

    logger.info("")
    logger.info("Importance weights (source)")
    logger.info("  p_tgt (classifier P(target|x)): min=%.4f, median=%.4f, max=%.4f",
                float(p_tgt.min()), float(np.median(p_tgt)), float(p_tgt.max()))
    logger.info("  Raw weights: min=%.4f, median=%.4f, max=%.4f, std=%.4f",
                float(raw_w.min()), float(np.median(raw_w)), float(raw_w.max()),
                float(raw_w.std()))
    logger.info("  Clipped weights: min=%.4f, median=%.4f, max=%.4f, std=%.4f",
                float(w.min()), float(np.median(w)), float(w.max()), float(w.std()))
    n_clipped = int((raw_w > w_clip[1]).sum() + (raw_w < w_clip[0]).sum())
    logger.info("  Clipped %d / %d weights (%.1f%%)", n_clipped, n_S, 100 * n_clipped / n_S)

    # ── Test 5: Effective sample size ──
    n_eff = (w.sum() ** 2) / (w ** 2).sum()
    ess_ratio = float(n_eff / n_S)
    pass_5 = ess_ratio >= 0.50
    soft_5 = ess_ratio >= 0.35
    logger.info("")
    logger.info("Test 5 — Effective sample size")
    logger.info("  n_eff = %.1f / %d", n_eff, n_S)
    logger.info("  ess_ratio = %.3f", ess_ratio)
    logger.info("  threshold: >= 0.50 (soft: >= 0.35)")
    logger.info("  result: %s", "PASS" if pass_5 else ("SOFT" if soft_5 else "FAIL"))

    # ── Test 6: Quartile spread (concept vs covariate) ──
    q25, q75 = np.percentile(w, [25, 75])
    Q1 = w <= q25  # source-like (low weight)
    Q4 = w > q75   # target-like (high weight)

    acc_Q1 = float(y_S[Q1].mean()) if Q1.sum() > 0 else float("nan")
    acc_Q4 = float(y_S[Q4].mean()) if Q4.sum() > 0 else float("nan")
    spread = acc_Q1 - acc_Q4

    logger.info("")
    logger.info("Test 6 — Quartile spread (concept vs covariate shift)")
    logger.info("  Weight quartile boundaries: Q25=%.3f, Q75=%.3f", q25, q75)
    logger.info("  Q1 (source-like, n=%d): acc = %.3f", int(Q1.sum()), acc_Q1)
    logger.info("  Q4 (target-like, n=%d): acc = %.3f", int(Q4.sum()), acc_Q4)
    logger.info("  spread = Q1 - Q4 = %.3f", spread)
    logger.info("  threshold: >= 0.05")
    pass_6 = spread >= 0.05
    if spread < 0.03:
        logger.info("  note: flat accuracy across weight quartiles — concept shift dominant")
    elif spread >= 0.05:
        logger.info("  note: target-like source points are harder — covariate shift signal")
    logger.info("  result: %s", "PASS" if pass_6 else "FAIL")

    # ── Test 6 strong form: logistic regression of y_S on log(w) ──
    log_w = np.log(w + 1e-9)
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_w, y_S)
    logger.info("")
    logger.info("Test 6 strong form — Linear regression y_S ~ log(w)")
    logger.info("  slope = %.4f (std_err=%.4f)", slope, std_err)
    logger.info("  p-value = %.4f, R^2 = %.4f", p_value, r_value ** 2)
    logger.info("  interpretation: %s",
                "significant negative slope — covariate shift confirmed"
                if slope < -0.2 and p_value < 0.05
                else "weak/no slope — concept shift likely dominant")

    # ── Quartile detail (all four quartiles for diagnostic) ──
    q_bounds = np.percentile(w, [25, 50, 75])
    quartile_accs = []
    for qi, (lo, hi) in enumerate([
        (w.min() - 1, q_bounds[0]),
        (q_bounds[0], q_bounds[1]),
        (q_bounds[1], q_bounds[2]),
        (q_bounds[2], w.max() + 1),
    ]):
        mask = (w > lo) & (w <= hi) if qi > 0 else (w <= hi)
        q_acc = float(y_S[mask].mean()) if mask.sum() > 0 else float("nan")
        quartile_accs.append(q_acc)
        logger.debug("  Quartile Q%d (w in [%.3f, %.3f], n=%d): acc=%.3f",
                     qi + 1, lo, hi, int(mask.sum()), q_acc)

    # ── Summary ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("[SCREENING] Battery complete — summary")
    logger.info("  Test 1  (Source acc):     %.3f  %s", acc_S, "PASS" if pass_1 else "FAIL")
    logger.info("  Test 2a (Target acc):     %.3f  %s", acc_T,
                "PASS" if pass_2a else ("SOFT" if soft_2a else "FAIL"))
    logger.info("  Test 2b (Top-5%% acc):     %.3f  %s", acc_top5, "PASS" if pass_2b else "FAIL")
    logger.info("  Test 3  (Gap):            %.3f  %s", gap, "PASS" if pass_3 else "FAIL")
    logger.info("  Test 4  (Classifier):     %.3f  %s", acc_clf, "PASS" if pass_4 else "FAIL")
    logger.info("  Test 5  (ESS ratio):      %.3f  %s", ess_ratio,
                "PASS" if pass_5 else ("SOFT" if soft_5 else "FAIL"))
    logger.info("  Test 6  (Quartile spread):%.3f  %s", spread, "PASS" if pass_6 else "FAIL")
    n_pass = sum([pass_1, pass_2a, pass_2b, pass_3, pass_4, pass_5, pass_6])
    logger.info("  Total: %d / 7 pass", n_pass)
    logger.info("=" * 60)

    return {
        # Test values
        "acc_S": acc_S,
        "acc_T": acc_T,
        "acc_top5": acc_top5,
        "gap": gap,
        "acc_clf": acc_clf,
        "acc_clf_std": acc_clf_std,
        "ess_ratio": ess_ratio,
        "n_eff": float(n_eff),
        "quartile_spread": spread,
        "acc_Q1": acc_Q1,
        "acc_Q4": acc_Q4,
        "quartile_accs": quartile_accs,
        "slope_logw": float(slope),
        "slope_pvalue": float(p_value),

        # Pass/fail
        "pass_1": pass_1,
        "pass_2a": pass_2a,
        "pass_2b": pass_2b,
        "pass_3": pass_3,
        "pass_4": pass_4,
        "pass_5": pass_5,
        "pass_6": pass_6,

        # Soft passes
        "soft_2a": soft_2a,
        "soft_5": soft_5,

        # Config
        "epsilon": epsilon,
        "n_S": n_S,
        "n_T": n_T,
        "top5_k": top5_k,

        # Weight diagnostics
        "weight_min": float(w.min()),
        "weight_median": float(np.median(w)),
        "weight_max": float(w.max()),
        "weight_std": float(w.std()),
        "raw_weight_max": float(raw_w.max()),
    }


def print_scorecard(results: dict):
    """Print the screening scorecard in a readable table format."""
    eps = results["epsilon"]

    def _flag(passed, soft=None):
        if passed:
            return "PASS"
        if soft is not None and soft:
            return "SOFT"
        return "FAIL"

    print()
    print("=" * 72)
    print(f"  SCREENING SCORECARD (epsilon = {eps})")
    print("=" * 72)
    print()
    print(f"  Source: n={results['n_S']}, Target: n={results['n_T']}")
    print()

    rows = [
        ("1.  Source accuracy",
         f"acc_S = {results['acc_S']:.3f}",
         f">= {1 - eps + 0.05:.2f}",
         _flag(results["pass_1"])),

        ("2a. Target accuracy",
         f"acc_T = {results['acc_T']:.3f}",
         f">= {1 - eps:.2f}",
         _flag(results["pass_2a"], results["soft_2a"])),

        ("2b. Reachable floor (top-5%)",
         f"acc_top5 = {results['acc_top5']:.3f}",
         f">= {1 - eps + 0.05:.2f}",
         _flag(results["pass_2b"])),

        ("3.  Accuracy gap",
         f"gap = {results['gap']:.3f}",
         "[0.03, 0.15]",
         _flag(results["pass_3"])),

        ("4.  Domain classifier",
         f"acc_clf = {results['acc_clf']:.3f}",
         "[0.55, 0.78]",
         _flag(results["pass_4"])),

        ("5.  ESS ratio",
         f"n_eff/n = {results['ess_ratio']:.3f}",
         ">= 0.50",
         _flag(results["pass_5"], results["soft_5"])),

        ("6.  Quartile spread",
         f"Q1-Q4 = {results['quartile_spread']:.3f}",
         ">= 0.05",
         _flag(results["pass_6"])),
    ]

    print(f"  {'Test':<30} {'Value':<22} {'Threshold':<14} {'Result'}")
    print(f"  {'-'*30} {'-'*22} {'-'*14} {'-'*6}")
    for name, value, threshold, flag in rows:
        print(f"  {name:<30} {value:<22} {threshold:<14} {flag}")

    # Diagnostics
    print()
    print("  Diagnostics:")
    print(f"    Classifier CV: {results['acc_clf']:.3f} +/- {results['acc_clf_std']:.3f}")
    print(f"    n_eff: {results['n_eff']:.1f} / {results['n_S']}")
    print(f"    Weights: min={results['weight_min']:.3f}, "
          f"median={results['weight_median']:.3f}, "
          f"max={results['weight_max']:.3f}")
    print(f"    Quartile accuracies: "
          f"Q1={results['quartile_accs'][0]:.3f}, "
          f"Q2={results['quartile_accs'][1]:.3f}, "
          f"Q3={results['quartile_accs'][2]:.3f}, "
          f"Q4={results['quartile_accs'][3]:.3f}")
    print(f"    Slope(y_S ~ log w): {results['slope_logw']:.4f} "
          f"(p={results['slope_pvalue']:.4f})")

    # Verdict
    n_pass = sum(1 for k in ["pass_1", "pass_2a", "pass_2b", "pass_3",
                              "pass_4", "pass_5", "pass_6"]
                 if results[k])
    n_soft = sum(1 for k in ["soft_2a", "soft_5"]
                 if results[k] and not results[k.replace("soft_", "pass_")])

    print()
    if n_pass == 7:
        print("  VERDICT: ALL PASS — greenlight for full DS-SGen run.")
    elif n_pass + n_soft >= 6:
        print(f"  VERDICT: {n_pass} pass + {n_soft} soft — likely viable, "
              "check borderline tests.")
    else:
        print(f"  VERDICT: {n_pass}/7 pass — pair may not produce the "
              "M1/M2 fail -> M3 rescues story.")

    print("=" * 72)
    print()
