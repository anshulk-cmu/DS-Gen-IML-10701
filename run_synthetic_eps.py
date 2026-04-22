"""Final covariate-shift experiment — epsilon sweep + concept-shift control.

Pipeline:
  Step 1: delete prior synthetic_a_*/synth_a_*/synthetic_a_pair_indices.json
          (keeps the generated 1831-item pool at cache/synth_qa_*)
  Step 2: reconstruct the synthetic source/target pair (accuracy_sorted partition)
  Step 3: epsilon sweep on the synthetic pair (M1 + M3)
  Step 4: epsilon sweep on TQA -> NQ (M1 + M3, concept-shift baseline)
  Step 5: persist results and compute weight-quartile diagnostic
"""

import argparse
import copy
import glob
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict

import numpy as np

from ds_sgen.utils import (
    load_config, set_seed, load_cache, save_cache, get_cache_path,
)
from ds_sgen.sgen_semi import _merge_records
from ds_sgen.synthetic_shift import build_synthetic_pair, _json_safe
from ds_sgen.screening import run_screening_tests

DATASET_NAME = "synth_qa"


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)


# ── Step 1: cleanup ─────────────────────────────────────────────────────────

def cleanup_prior_runs(cfg, logger):
    """Delete the previous Design-A artifacts; KEEP the synth_qa_* dataset caches."""
    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]
    plots_dir = "plots"

    targets = []
    targets += glob.glob(os.path.join(results_dir, "synthetic_a_*.json"))
    targets += glob.glob(os.path.join(results_dir, "synthetic_final_*.json"))
    targets += glob.glob(os.path.join(plots_dir, "synthetic_a_*.png"))
    targets += glob.glob(os.path.join(plots_dir, "synthetic_final_*.png"))
    targets += glob.glob(os.path.join(cache_dir, "synth_a_source_*"))
    targets += glob.glob(os.path.join(cache_dir, "synth_a_target_*"))
    targets += [os.path.join(cache_dir, "synthetic_a_pair_indices.json")]

    scratches = [
        os.path.join(results_dir, "_synth_a_scratch"),
        os.path.join(results_dir, "_synth_eps_scratch"),
    ]

    logger.info("Step 1: cleanup")
    n_removed = 0
    for t in targets:
        if os.path.isfile(t):
            os.remove(t)
            logger.info("  removed file %s", t)
            n_removed += 1
    for s in scratches:
        if os.path.isdir(s):
            shutil.rmtree(s)
            logger.info("  removed dir  %s", s)
            n_removed += 1
    logger.info("  removed %d items (dataset caches kept)", n_removed)


# ── Step 2: reconstruct pair ────────────────────────────────────────────────

def load_synthetic_pool(cfg, logger):
    cache_dir = cfg["paths"]["cache_dir"]
    records = load_cache(get_cache_path(cache_dir, f"{DATASET_NAME}_data"))
    gens = load_cache(get_cache_path(cache_dir, f"{DATASET_NAME}_generations"))
    ents = load_cache(get_cache_path(cache_dir, f"{DATASET_NAME}_entailment"))
    if records is None or gens is None or ents is None:
        raise FileNotFoundError(
            f"Synthetic dataset caches missing under {cache_dir}. "
            "Run run_synthetic_a.py first to generate synth_qa_*."
        )
    emb_path = os.path.join(cache_dir, f"{DATASET_NAME}_embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing {emb_path}")
    embeddings = np.load(emb_path)
    logger.info("  loaded %d records, %d gens, %d ents, embeddings %s",
                len(records), len(gens), len(ents), embeddings.shape)
    return records, gens, ents, embeddings


def slice_by_indices(records, gens, ents, idx, label):
    out_rec, out_gen, out_ent = [], [], []
    for new_i, orig_i in enumerate(idx):
        r = dict(records[orig_i]); r["idx"] = new_i; r["dataset"] = label
        out_rec.append(r)
        g = dict(gens[orig_i]); g["idx"] = new_i
        out_gen.append(g)
        e = dict(ents[orig_i]); e["idx"] = new_i
        out_ent.append(e)
    return out_rec, out_gen, out_ent


def reconstruct_pair(cfg, merged, embeddings, logger):
    sa = cfg.get("synthetic_a", {})
    K = sa.get("K", 10)
    n_side = sa.get("n_source", 800)
    fm1_quantile = sa.get("fm1_quantile", 0.00)
    partition_strategy = sa.get("partition_strategy", "accuracy_sorted")

    logger.info("")
    logger.info("Step 2: reconstruct synthetic pair (K=%d, n=%d, partition=%s)",
                K, n_side, partition_strategy)

    # Pick alpha = 0.75 (the screening-optimal from the accuracy_sorted sweep)
    alpha = sa.get("final_alpha", 0.75)
    pair = build_synthetic_pair(
        merged, embeddings,
        alpha=alpha, K=K, n_S=n_side, n_T=n_side,
        fm1_quantile=fm1_quantile, seed=cfg["seed"],
        partition_strategy=partition_strategy,
    )
    logger.info("  alpha=%.2f, source_acc=%.3f, target_acc=%.3f",
                alpha, pair.source_acc, pair.target_acc)
    return pair


# ── Step 3 & 4: epsilon sweep ───────────────────────────────────────────────

def _cfg_with_scratch(cfg: dict, epsilon: float):
    """Deep-copy cfg, override epsilon + cal_dataset + scratch results_dir."""
    c = copy.deepcopy(cfg)
    c["sgen"]["epsilon"] = epsilon
    c["sgen"]["cal_dataset"] = "tqa"
    scratch = os.path.join(cfg["paths"]["results_dir"], "_synth_eps_scratch")
    os.makedirs(scratch, exist_ok=True)
    c["paths"]["results_dir"] = scratch
    return c


def _vacuous_stats(per_split):
    """Return (vacuous_frac, non_vac_count, non_vac_validity)."""
    total = len(per_split)
    vacs = [r for r in per_split if r["shifted_test"]["efficiency"] == 0]
    nvs = [r for r in per_split if r["shifted_test"]["efficiency"] > 0]
    nv_val = (sum(1 for r in nvs if r["shifted_test"]["valid"]) / len(nvs)
              if nvs else None)
    return len(vacs) / total, len(nvs), nv_val


def _summarize_method(result):
    vac_frac, nv_count, nv_val = _vacuous_stats(result["per_split"])
    return {
        "validity_rate": result["shifted"]["validity_rate"],
        "mean_efficiency": result["shifted"]["mean_efficiency"],
        "mean_fdr_e": result["shifted"]["mean_fdr_e"],
        "vacuous_frac": vac_frac,
        "non_vacuous_count": nv_count,
        "non_vacuous_validity": nv_val,
        "indomain_validity": result["indomain"]["validity_rate"],
        "indomain_efficiency": result["indomain"]["mean_efficiency"],
    }


def run_m1_direct(cfg, nq_triple, tqa_triple):
    """Call M1 without any embedding swap (works for both synthetic and TQA->NQ)."""
    from ds_sgen.sgen_semi import run_experiment as m1_run
    t_rec, t_gen, t_ent = nq_triple
    s_rec, s_gen, s_ent = tqa_triple
    return m1_run(
        cfg,
        nq_records=t_rec, nq_generations=t_gen, nq_entailments=t_ent,
        tqa_records=s_rec, tqa_generations=s_gen, tqa_entailments=s_ent,
    )


def run_m3_synthetic(cfg, source_triple, target_triple, source_emb, target_emb):
    """M3 with embedding swap: synthetic embeddings temporarily replace tqa/nq caches."""
    from ds_sgen.importance_weighted import run_experiment as m3_run

    cache_dir = cfg["paths"]["cache_dir"]
    tqa_emb_path = os.path.join(cache_dir, "tqa_embeddings.npy")
    nq_emb_path = os.path.join(cache_dir, "nq_embeddings.npy")
    bak_tqa = tqa_emb_path + ".bak_synth_eps"
    bak_nq = nq_emb_path + ".bak_synth_eps"

    if not os.path.exists(tqa_emb_path):
        raise FileNotFoundError(f"Expected {tqa_emb_path} before backup")
    os.rename(tqa_emb_path, bak_tqa)
    nq_was_present = os.path.exists(nq_emb_path)
    if nq_was_present:
        os.rename(nq_emb_path, bak_nq)

    try:
        np.save(tqa_emb_path, source_emb)
        np.save(nq_emb_path, target_emb)
        t_rec, t_gen, t_ent = target_triple
        s_rec, s_gen, s_ent = source_triple
        result = m3_run(
            cfg,
            nq_records=t_rec, nq_generations=t_gen, nq_entailments=t_ent,
            tqa_records=s_rec, tqa_generations=s_gen, tqa_entailments=s_ent,
        )
    finally:
        if os.path.exists(tqa_emb_path):
            os.remove(tqa_emb_path)
        os.replace(bak_tqa, tqa_emb_path)
        if nq_was_present:
            if os.path.exists(nq_emb_path):
                os.remove(nq_emb_path)
            os.replace(bak_nq, nq_emb_path)
        elif os.path.exists(nq_emb_path):
            os.remove(nq_emb_path)
    return result


def run_m3_direct(cfg, nq_triple, tqa_triple):
    """M3 with no swap — used for the TQA->NQ concept-shift baseline."""
    from ds_sgen.importance_weighted import run_experiment as m3_run
    t_rec, t_gen, t_ent = nq_triple
    s_rec, s_gen, s_ent = tqa_triple
    return m3_run(
        cfg,
        nq_records=t_rec, nq_generations=t_gen, nq_entailments=t_ent,
        tqa_records=s_rec, tqa_generations=s_gen, tqa_entailments=s_ent,
    )


# ── Weight-quartile diagnostic ──────────────────────────────────────────────

def weight_quartile_block(scorecard):
    """Pull covariate-vs-concept diagnostic fields from a screening scorecard."""
    return {
        "Q1": float(scorecard["acc_Q1"]),
        "Q4": float(scorecard["acc_Q4"]),
        "Q1_minus_Q4": float(scorecard["quartile_spread"]),
        "quartile_accs": [float(x) for x in scorecard["quartile_accs"]],
        "ess_ratio": float(scorecard["ess_ratio"]),
        "acc_clf": float(scorecard["acc_clf"]),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epsilons", nargs="*", type=float, default=None)
    ap.add_argument("--skip-tqa-nq", action="store_true")
    ap.add_argument("--n-splits", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.n_splits is not None:
        cfg["sgen"]["n_splits"] = args.n_splits

    epsilons = args.epsilons or cfg.get("synthetic_eps", {}).get(
        "epsilons", [0.05, 0.10, 0.15, 0.20, 0.25]
    )
    include_tqa_nq = not args.skip_tqa_nq and cfg.get(
        "synthetic_eps", {}).get("include_tqa_nq", True)

    logger.info("=" * 60)
    logger.info("Perfect covariate-shift experiment — epsilon sweep")
    logger.info("epsilons = %s", epsilons)
    logger.info("include_tqa_nq = %s", include_tqa_nq)
    logger.info("=" * 60)
    t0 = time.time()

    cleanup_prior_runs(cfg, logger)

    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # ── Step 2 ──
    logger.info("")
    logger.info("Loading synthetic pool")
    records, gens, ents, embeddings = load_synthetic_pool(cfg, logger)
    merged = _merge_records(records, gens, ents)
    pair = reconstruct_pair(cfg, merged, embeddings, logger)

    source_idx = pair.source_idx
    target_idx = pair.target_idx
    save_cache(asdict(pair),
               os.path.join(cache_dir, "synthetic_a_pair_indices.json"))

    source_triple = slice_by_indices(records, gens, ents, source_idx, "tqa")
    target_triple = slice_by_indices(records, gens, ents, target_idx, "nq")
    save_cache(source_triple[0], get_cache_path(cache_dir, "synth_a_source_data"))
    save_cache(source_triple[1], get_cache_path(cache_dir, "synth_a_source_generations"))
    save_cache(source_triple[2], get_cache_path(cache_dir, "synth_a_source_entailment"))
    save_cache(target_triple[0], get_cache_path(cache_dir, "synth_a_target_data"))
    save_cache(target_triple[1], get_cache_path(cache_dir, "synth_a_target_generations"))
    save_cache(target_triple[2], get_cache_path(cache_dir, "synth_a_target_entailment"))
    source_emb = embeddings[source_idx]
    target_emb = embeddings[target_idx]
    np.save(os.path.join(cache_dir, "synth_a_source_embeddings.npy"), source_emb)
    np.save(os.path.join(cache_dir, "synth_a_target_embeddings.npy"), target_emb)

    # Screening on the synthetic pair (at default epsilon=0.25)
    logger.info("")
    logger.info("Screening on the synthetic pair")
    y_S = np.array([merged[i]["entail_label"] for i in source_idx])
    y_T = np.array([merged[i]["entail_label"] for i in target_idx])
    fM_S = np.array([merged[i]["fM1"] for i in source_idx])
    fM_T = np.array([merged[i]["fM1"] for i in target_idx])
    sc_synth = run_screening_tests(
        y_S=y_S, y_T=y_T, fM_S=fM_S, fM_T=fM_T,
        emb_S=source_emb, emb_T=target_emb,
        epsilon=cfg["sgen"]["epsilon"],
        classifier_C=cfg["importance_weighted"]["classifier_C"],
    )
    sc_synth_safe = _json_safe(sc_synth)
    save_cache(sc_synth_safe,
               os.path.join(results_dir, "synthetic_final_screening.json"))
    n_pass_synth = sum(sc_synth[f"pass_{k}"]
                       for k in ["1", "2a", "2b", "3", "4", "5", "6"])
    logger.info("  synthetic screening: %d/7 pass", n_pass_synth)

    # ── Step 3: epsilon sweep on synthetic ──
    synth_sweep = []
    logger.info("")
    logger.info("Step 3: epsilon sweep on SYNTHETIC pair")
    for eps in epsilons:
        logger.info("")
        logger.info("── synthetic @ epsilon = %.3f ──", eps)
        cfg_eps = _cfg_with_scratch(cfg, eps)
        m1_res = run_m1_direct(cfg_eps, target_triple, source_triple)
        m3_res = run_m3_synthetic(cfg_eps, source_triple, target_triple,
                                  source_emb, target_emb)
        synth_sweep.append({
            "epsilon": eps,
            "m1": _summarize_method(m1_res),
            "m3": _summarize_method(m3_res),
        })
        logger.info("  synthetic eps=%.3f  M1 val=%.3f eff=%.3f | M3 val=%.3f eff=%.3f vac=%.2f",
                    eps,
                    m1_res["shifted"]["validity_rate"],
                    m1_res["shifted"]["mean_efficiency"],
                    m3_res["shifted"]["validity_rate"],
                    m3_res["shifted"]["mean_efficiency"],
                    synth_sweep[-1]["m3"]["vacuous_frac"])

    # ── Step 4: epsilon sweep on TQA -> NQ ──
    tqa_nq_sweep = []
    sc_tqa_nq_safe = None
    if include_tqa_nq:
        logger.info("")
        logger.info("Step 4: epsilon sweep on TQA -> NQ (concept-shift baseline)")
        nq_rec = load_cache(get_cache_path(cache_dir, "nq_data"))
        nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
        nq_ent = load_cache(get_cache_path(cache_dir, "nq_entailment"))
        tqa_rec = load_cache(get_cache_path(cache_dir, "tqa_data"))
        tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))
        tqa_ent = load_cache(get_cache_path(cache_dir, "tqa_entailment"))

        if any(x is None for x in [nq_rec, nq_gen, nq_ent, tqa_rec, tqa_gen, tqa_ent]):
            logger.warning("  TQA/NQ caches missing, skipping concept-shift baseline")
            include_tqa_nq = False

    if include_tqa_nq:
        # Screening on TQA -> NQ
        tqa_emb = np.load(os.path.join(cache_dir, "tqa_embeddings.npy"))
        nq_emb = np.load(os.path.join(cache_dir, "nq_embeddings.npy"))
        tqa_merged = _merge_records(tqa_rec, tqa_gen, tqa_ent)
        nq_merged = _merge_records(nq_rec, nq_gen, nq_ent)
        y_S2 = np.array([m["entail_label"] for m in tqa_merged])
        y_T2 = np.array([m["entail_label"] for m in nq_merged])
        fM_S2 = np.array([m["fM1"] for m in tqa_merged])
        fM_T2 = np.array([m["fM1"] for m in nq_merged])
        sc_tqa_nq = run_screening_tests(
            y_S=y_S2, y_T=y_T2, fM_S=fM_S2, fM_T=fM_T2,
            emb_S=tqa_emb, emb_T=nq_emb,
            epsilon=cfg["sgen"]["epsilon"],
            classifier_C=cfg["importance_weighted"]["classifier_C"],
        )
        sc_tqa_nq_safe = _json_safe(sc_tqa_nq)
        n_pass_tqa = sum(sc_tqa_nq[f"pass_{k}"]
                         for k in ["1", "2a", "2b", "3", "4", "5", "6"])
        logger.info("  TQA -> NQ screening: %d/7 pass (concept-shift expected)", n_pass_tqa)

        nq_triple = (nq_rec, nq_gen, nq_ent)
        tqa_triple = (tqa_rec, tqa_gen, tqa_ent)

        for eps in epsilons:
            logger.info("")
            logger.info("── TQA->NQ @ epsilon = %.3f ──", eps)
            cfg_eps = _cfg_with_scratch(cfg, eps)
            m1_res = run_m1_direct(cfg_eps, nq_triple, tqa_triple)
            m3_res = run_m3_direct(cfg_eps, nq_triple, tqa_triple)
            tqa_nq_sweep.append({
                "epsilon": eps,
                "m1": _summarize_method(m1_res),
                "m3": _summarize_method(m3_res),
            })
            logger.info("  tqa->nq eps=%.3f  M1 val=%.3f eff=%.3f | M3 val=%.3f eff=%.3f vac=%.2f",
                        eps,
                        m1_res["shifted"]["validity_rate"],
                        m1_res["shifted"]["mean_efficiency"],
                        m3_res["shifted"]["validity_rate"],
                        m3_res["shifted"]["mean_efficiency"],
                        tqa_nq_sweep[-1]["m3"]["vacuous_frac"])

    # ── Step 5: persist ──
    save_cache(
        {"synthetic": synth_sweep, "tqa_nq": tqa_nq_sweep},
        os.path.join(results_dir, "synthetic_final_eps_sweep.json"),
    )

    wq = {"synthetic": weight_quartile_block(sc_synth_safe)}
    if sc_tqa_nq_safe is not None:
        wq["tqa_nq"] = weight_quartile_block(sc_tqa_nq_safe)
    save_cache(wq,
               os.path.join(results_dir, "synthetic_final_weight_quartile.json"))

    logger.info("")
    logger.info("Total time: %.1f minutes", (time.time() - t0) / 60)
    logger.info("Artifacts:")
    logger.info("  results/synthetic_final_screening.json (synthetic %d/7)", n_pass_synth)
    logger.info("  results/synthetic_final_eps_sweep.json (%d synthetic eps, %d tqa_nq eps)",
                len(synth_sweep), len(tqa_nq_sweep))
    logger.info("  results/synthetic_final_weight_quartile.json")


if __name__ == "__main__":
    main()
