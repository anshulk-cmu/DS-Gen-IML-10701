"""Epsilon sweep: run all three methods at multiple epsilon values.

Produces the headline figure — validity vs. epsilon for Methods 1, 2, 3.
Each method's line crossing the 98% validity threshold tells you the minimum
epsilon where the PAC guarantee holds under domain shift.

Requires:
  - Cached Stages 1-3 data (from run_baseline.py)
  - Cached embeddings (from run_importance_weighted.py, or computed here)

Usage:
    python run_epsilon_sweep.py
    python run_epsilon_sweep.py --config configs/default.yaml
"""

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np

from ds_sgen.utils import load_config, set_seed, load_cache, get_cache_path, save_cache
from ds_sgen.sgen_semi import _merge_records, _run_single_split as m1_run_split
from ds_sgen.conservative import _run_single_split as m2_run_split
from ds_sgen.importance_weighted import (
    compute_embeddings,
    train_domain_classifier,
    compute_importance_weights,
    _run_single_split as m3_run_split,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)


# ── Cache loading ─────────────────────────────────────────────────────────────

def load_cached_stages(cfg: dict):
    """Load cached outputs from Stages 1-3."""
    cache_dir = cfg["paths"]["cache_dir"]
    required = {
        "nq_data": "Stage 1", "tqa_data": "Stage 1",
        "nq_generations": "Stage 2", "tqa_generations": "Stage 2",
        "nq_entailment": "Stage 3", "tqa_entailment": "Stage 3",
    }
    caches = {}
    for name, stage in required.items():
        path = get_cache_path(cache_dir, name)
        data = load_cache(path)
        if data is None:
            raise FileNotFoundError(f"Missing cache: {path} ({stage})")
        caches[name] = data
    return caches


# ── Sweep helpers ─────────────────────────────────────────────────────────────

def _aggregate_splits(per_split: list[dict]) -> dict:
    """Aggregate per-split results into summary statistics."""
    id_val = [r["indomain_test"]["valid"] for r in per_split]
    id_fdr = [r["indomain_test"]["fdr_e"] for r in per_split]
    id_eff = [r["indomain_test"]["efficiency"] for r in per_split]
    sh_val = [r["shifted_test"]["valid"] for r in per_split]
    sh_fdr = [r["shifted_test"]["fdr_e"] for r in per_split]
    sh_eff = [r["shifted_test"]["efficiency"] for r in per_split]

    return {
        "indomain_validity": float(np.mean(id_val)),
        "indomain_mean_fdr_e": float(np.mean(id_fdr)),
        "indomain_mean_efficiency": float(np.mean(id_eff)),
        "shifted_validity": float(np.mean(sh_val)),
        "shifted_mean_fdr_e": float(np.mean(sh_fdr)),
        "shifted_mean_efficiency": float(np.mean(sh_eff)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Epsilon sweep for all methods")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    sweep_cfg = cfg["epsilon_sweep"]
    epsilons = sweep_cfg["epsilons"]
    n_splits = sweep_cfg["n_splits"]
    base_seed = cfg["seed"]
    sgen_cfg = cfg["sgen"]
    iw_cfg = cfg["importance_weighted"]

    logger.info("=" * 60)
    logger.info("Epsilon Sweep: All Methods")
    logger.info("=" * 60)
    logger.info("  Epsilons: %s", epsilons)
    logger.info("  n_splits: %d", n_splits)

    # Load cached data
    caches = load_cached_stages(cfg)
    nq_merged = _merge_records(caches["nq_data"], caches["nq_generations"], caches["nq_entailment"])
    tqa_merged = _merge_records(caches["tqa_data"], caches["tqa_generations"], caches["tqa_entailment"])

    cal_dataset = sgen_cfg.get("cal_dataset", "tqa")
    if cal_dataset == "tqa":
        cal_merged, shifted_merged = tqa_merged, nq_merged
        cal_label, shifted_label = "TQA", "NQ"
        cal_questions = [r["question"] for r in caches["tqa_data"]]
        shifted_questions = [r["question"] for r in caches["nq_data"]]
    else:
        cal_merged, shifted_merged = nq_merged, tqa_merged
        cal_label, shifted_label = "NQ", "TQA"
        cal_questions = [r["question"] for r in caches["nq_data"]]
        shifted_questions = [r["question"] for r in caches["tqa_data"]]

    logger.info("  Cal: %s (%d), Shifted: %s (%d)",
                cal_label, len(cal_merged), shifted_label, len(shifted_merged))

    # ── Compute Method 3 weights ONCE (independent of epsilon) ──
    logger.info("")
    logger.info("Pre-computing Method 3 weights...")

    cache_dir = cfg["paths"]["cache_dir"]
    hf_cache = cfg["paths"].get("hf_cache")
    model_name = iw_cfg["embedding_model"]

    cal_emb_path = os.path.join(cache_dir, f"{cal_label.lower()}_embeddings.npy")
    shifted_emb_path = os.path.join(cache_dir, f"{shifted_label.lower()}_embeddings.npy")

    if os.path.exists(cal_emb_path) and os.path.exists(shifted_emb_path):
        cal_embeddings = np.load(cal_emb_path)
        shifted_embeddings = np.load(shifted_emb_path)
        logger.info("  Loaded cached embeddings")
    else:
        cal_embeddings = compute_embeddings(cal_questions, model_name, hf_cache)
        shifted_embeddings = compute_embeddings(shifted_questions, model_name, hf_cache)
        np.save(cal_emb_path, cal_embeddings)
        np.save(shifted_emb_path, shifted_embeddings)

    classifier, cv_acc = train_domain_classifier(
        cal_embeddings, shifted_embeddings, C=iw_cfg["classifier_C"]
    )
    weights, weight_diag = compute_importance_weights(
        classifier, cal_embeddings, iw_cfg["weight_clip_percentile"]
    )
    logger.info("  Classifier CV accuracy: %.3f, n_eff: %.1f / %d",
                cv_acc, weight_diag["n_eff"], weight_diag["n"])

    # ── Method 2: delta_shift for Option C frac=0.75 ──
    delta = sgen_cfg["delta"]
    delta_p = sgen_cfg["delta_p"]
    delta_shift_m2 = 0.75 * (delta - delta_p)

    # ── Run sweep ──
    results = {"epsilons": epsilons, "method1": {}, "method2_optC": {}, "method3": {}}

    t0 = time.time()

    for eps in epsilons:
        logger.info("")
        logger.info("─── epsilon = %.2f ───", eps)

        cfg_copy = copy.deepcopy(sgen_cfg)
        cfg_copy["epsilon"] = eps
        # epsilon_e stays at 0.05 — controls pseudo-labeling, NOT FDR target

        # Method 1: Vanilla SGen-Semi
        m1_splits = []
        for s in range(n_splits):
            r = m1_run_split(cal_merged, shifted_merged, base_seed + s, cfg_copy)
            m1_splits.append(r)
        m1_agg = _aggregate_splits(m1_splits)
        results["method1"][str(eps)] = m1_agg
        logger.info("  M1 (Vanilla):       %s valid=%.2f eff=%.3f",
                    shifted_label, m1_agg["shifted_validity"],
                    m1_agg["shifted_mean_efficiency"])

        # Method 2: Conservative Option C (frac=0.75)
        m2_splits = []
        for s in range(n_splits):
            r = m2_run_split(
                cal_merged, shifted_merged, base_seed + s, cfg_copy,
                delta_shift=delta_shift_m2,
            )
            m2_splits.append(r)
        m2_agg = _aggregate_splits(m2_splits)
        results["method2_optC"][str(eps)] = m2_agg
        logger.info("  M2 (Conservative):  %s valid=%.2f eff=%.3f",
                    shifted_label, m2_agg["shifted_validity"],
                    m2_agg["shifted_mean_efficiency"])

        # Method 3: Importance Weighted
        m3_splits = []
        for s in range(n_splits):
            r = m3_run_split(
                cal_merged, shifted_merged, weights,
                base_seed + s, cfg_copy, iw_cfg,
            )
            m3_splits.append(r)
        m3_agg = _aggregate_splits(m3_splits)
        results["method3"][str(eps)] = m3_agg
        logger.info("  M3 (DS-SGen):       %s valid=%.2f eff=%.3f",
                    shifted_label, m3_agg["shifted_validity"],
                    m3_agg["shifted_mean_efficiency"])

    elapsed = time.time() - t0

    # ── Save ──
    results["diagnostics"] = {
        "classifier_cv_accuracy": cv_acc,
        "weight_n_eff": weight_diag["n_eff"],
        "delta_shift_m2": delta_shift_m2,
    }

    results_path = f"{cfg['paths']['results_dir']}/epsilon_sweep_results.json"
    save_cache(results, results_path)
    logger.info("\nResults saved to %s", results_path)

    # ── Summary table ──
    print()
    print("=" * 80)
    print("EPSILON SWEEP RESULTS — Shifted Domain (%s) Validity" % shifted_label)
    print("=" * 80)
    print(f"  {'Epsilon':>8} | {'M1 Valid':>10} {'M1 Eff':>8} | "
          f"{'M2 Valid':>10} {'M2 Eff':>8} | "
          f"{'M3 Valid':>10} {'M3 Eff':>8}")
    print("  " + "-" * 74)
    for eps in epsilons:
        m1 = results["method1"][str(eps)]
        m2 = results["method2_optC"][str(eps)]
        m3 = results["method3"][str(eps)]
        print(f"  {eps:>8.2f} | {m1['shifted_validity']:>9.1%} {m1['shifted_mean_efficiency']:>8.3f} | "
              f"{m2['shifted_validity']:>9.1%} {m2['shifted_mean_efficiency']:>8.3f} | "
              f"{m3['shifted_validity']:>9.1%} {m3['shifted_mean_efficiency']:>8.3f}")
    print("=" * 80)
    print(f"  Target validity: >= 98%  (1 - delta = {1 - delta:.0%})")
    print(f"  Total sweep time: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
