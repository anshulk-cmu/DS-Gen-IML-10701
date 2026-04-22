"""Orchestrator for the synthetic covariate-shift experiment.

Reuses cached TQA data/generations/entailments/embeddings. Builds a
(source, target) pair with topic-mixture weighting, sweeps alpha by
screening scorecard, then invokes the existing M1/M2/M3 run_experiment
functions with synthetic source as "tqa" and synthetic target as "nq".

Usage:
    python run_synthetic.py
    python run_synthetic.py --alpha 0.75             # skip sweep
    python run_synthetic.py --skip-m2                # faster
    python run_synthetic.py --n-splits 10            # debug
"""

import argparse
import copy
import logging
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from ds_sgen.utils import (
    load_config, set_seed, load_cache, save_cache, get_cache_path,
)
from ds_sgen.sgen_semi import _merge_records
from ds_sgen.synthetic_shift import (
    sweep_alpha_with_screening, build_synthetic_pair, pick_best_alpha,
    _json_safe,
)


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)


def load_cached_tqa(cfg: dict):
    cache_dir = cfg["paths"]["cache_dir"]
    records = load_cache(get_cache_path(cache_dir, "tqa_data"))
    gens = load_cache(get_cache_path(cache_dir, "tqa_generations"))
    ents = load_cache(get_cache_path(cache_dir, "tqa_entailment"))
    if records is None or gens is None or ents is None:
        raise FileNotFoundError("Missing TQA caches. Run run_baseline.py first.")
    emb_path = os.path.join(cache_dir, "tqa_embeddings.npy")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Missing {emb_path}. Run run_importance_weighted.py first."
        )
    embeddings = np.load(emb_path)
    return records, gens, ents, embeddings


def slice_by_indices(records, gens, ents, idx, label):
    out_rec, out_gen, out_ent = [], [], []
    for new_i, orig_i in enumerate(idx):
        r = dict(records[orig_i])
        r["idx"] = new_i
        r["dataset"] = label
        out_rec.append(r)
        g = dict(gens[orig_i]); g["idx"] = new_i
        out_gen.append(g)
        e = dict(ents[orig_i]); e["idx"] = new_i
        out_ent.append(e)
    return out_rec, out_gen, out_ent


def _cfg_with_scratch_results(cfg: dict) -> tuple[dict, str]:
    """Deep-copy cfg and redirect results_dir to a scratch subdir.

    Prevents synthetic M1/M2/M3 runs from overwriting real baseline_results.json,
    conservative_results.json, importance_weighted_results.json (which are
    hardcoded save paths in those modules).
    """
    cfg_syn = copy.deepcopy(cfg)
    cfg_syn["sgen"]["cal_dataset"] = "tqa"
    scratch = os.path.join(cfg["paths"]["results_dir"], "_synth_scratch")
    os.makedirs(scratch, exist_ok=True)
    cfg_syn["paths"]["results_dir"] = scratch
    return cfg_syn, scratch


def run_m1_on_pair(cfg, source_triple, target_triple):
    from ds_sgen.sgen_semi import run_experiment as m1_run

    cfg_syn, _ = _cfg_with_scratch_results(cfg)
    s_rec, s_gen, s_ent = source_triple
    t_rec, t_gen, t_ent = target_triple

    results = m1_run(
        cfg_syn,
        nq_records=t_rec, nq_generations=t_gen, nq_entailments=t_ent,
        tqa_records=s_rec, tqa_generations=s_gen, tqa_entailments=s_ent,
    )
    save_cache(
        results,
        os.path.join(cfg["paths"]["results_dir"], "synthetic_m1_results.json"),
    )
    return results


def run_m2_on_pair(cfg, source_triple, target_triple):
    from ds_sgen.conservative import run_conservative_experiment

    cfg_syn, _ = _cfg_with_scratch_results(cfg)
    s_rec, s_gen, s_ent = source_triple
    t_rec, t_gen, t_ent = target_triple

    results = run_conservative_experiment(
        cfg_syn,
        nq_records=t_rec, nq_gen=t_gen, nq_ent=t_ent,
        tqa_records=s_rec, tqa_gen=s_gen, tqa_ent=s_ent,
    )
    save_cache(
        results,
        os.path.join(cfg["paths"]["results_dir"], "synthetic_m2_results.json"),
    )
    return results


def run_m3_on_pair(cfg, source_triple, target_triple, source_emb, target_emb):
    """Run M3 with synthetic embeddings temporarily placed at the paths M3 reads."""
    from ds_sgen.importance_weighted import run_experiment as m3_run

    cfg_syn, _ = _cfg_with_scratch_results(cfg)

    cache_dir = cfg["paths"]["cache_dir"]
    tqa_emb_path = os.path.join(cache_dir, "tqa_embeddings.npy")
    nq_emb_path = os.path.join(cache_dir, "nq_embeddings.npy")
    bak_tqa = tqa_emb_path + ".bak_synth"
    bak_nq = nq_emb_path + ".bak_synth"

    if not os.path.exists(tqa_emb_path):
        raise FileNotFoundError(f"Expected {tqa_emb_path} to exist before backup")
    os.rename(tqa_emb_path, bak_tqa)
    nq_was_present = os.path.exists(nq_emb_path)
    if nq_was_present:
        os.rename(nq_emb_path, bak_nq)

    try:
        np.save(tqa_emb_path, source_emb)
        np.save(nq_emb_path, target_emb)

        s_rec, s_gen, s_ent = source_triple
        t_rec, t_gen, t_ent = target_triple
        results = m3_run(
            cfg_syn,
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
        else:
            if os.path.exists(nq_emb_path):
                os.remove(nq_emb_path)

    save_cache(
        results,
        os.path.join(cfg["paths"]["results_dir"], "synthetic_m3_results.json"),
    )
    return results


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--n-splits", type=int, default=None)
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--skip-m2", action="store_true")
    ap.add_argument("--skip-m3", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.n_splits is not None:
        cfg["sgen"]["n_splits"] = args.n_splits

    syn_cfg = cfg.get("synthetic", {})
    K = syn_cfg.get("K", 20)
    n_S = syn_cfg.get("n_source", 1000)
    n_T = syn_cfg.get("n_target", 1000)
    fm1_quantile = syn_cfg.get("fm1_quantile", 0.40)
    alphas = args.alphas or syn_cfg.get(
        "alphas", [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    )
    target_clf_acc = syn_cfg.get("target_clf_acc", 0.66)

    logger.info("=" * 60)
    logger.info("Synthetic covariate-shift experiment")
    logger.info("=" * 60)
    t0 = time.time()

    logger.info("Stage 1: loading cached TQA")
    records, gens, ents, embeddings = load_cached_tqa(cfg)
    logger.info("  %d records, %d gens, %d ents, emb %s",
                len(records), len(gens), len(ents), embeddings.shape)

    merged = _merge_records(records, gens, ents)

    if args.alpha is None:
        logger.info("")
        logger.info("Stage 2: sweeping alpha = %s", alphas)
        sweep = sweep_alpha_with_screening(
            merged, embeddings,
            alphas=alphas, K=K, n_S=n_S, n_T=n_T,
            fm1_quantile=fm1_quantile,
            epsilon=cfg["sgen"]["epsilon"],
            classifier_C=cfg["importance_weighted"]["classifier_C"],
            seed=cfg["seed"],
        )
        save_cache(
            {"sweep": [
                {"alpha": e["alpha"], "n_pass": e["n_pass"],
                 "scorecard": e["scorecard"]}
                for e in sweep
            ]},
            os.path.join(cfg["paths"]["results_dir"],
                         "synthetic_screening_sweep.json"),
        )
        best = pick_best_alpha(sweep, target_clf_acc=target_clf_acc)
        logger.info("")
        logger.info("Best alpha = %.2f (n_pass=%d/7, T4=%.3f, T5=%.3f)",
                    best["alpha"], best["n_pass"],
                    best["scorecard"]["acc_clf"], best["scorecard"]["ess_ratio"])
        chosen_pair_dict = best["pair"]
        chosen_scorecard = best["scorecard"]
    else:
        logger.info("")
        logger.info("Stage 2 skipped — using alpha=%.2f", args.alpha)
        pair = build_synthetic_pair(
            merged, embeddings,
            alpha=args.alpha, K=K, n_S=n_S, n_T=n_T,
            fm1_quantile=fm1_quantile, seed=cfg["seed"],
        )
        from ds_sgen.screening import run_screening_tests
        y_S = np.array([merged[i]["entail_label"] for i in pair.source_idx])
        y_T = np.array([merged[i]["entail_label"] for i in pair.target_idx])
        fM_S = np.array([merged[i]["fM1"] for i in pair.source_idx])
        fM_T = np.array([merged[i]["fM1"] for i in pair.target_idx])
        emb_S = embeddings[pair.source_idx]
        emb_T = embeddings[pair.target_idx]
        sc = run_screening_tests(
            y_S=y_S, y_T=y_T, fM_S=fM_S, fM_T=fM_T,
            emb_S=emb_S, emb_T=emb_T,
            epsilon=cfg["sgen"]["epsilon"],
            classifier_C=cfg["importance_weighted"]["classifier_C"],
        )
        chosen_pair_dict = asdict(pair)
        chosen_scorecard = _json_safe(sc)

    save_cache(
        chosen_pair_dict,
        os.path.join(cfg["paths"]["cache_dir"], "synthetic_pair_indices.json"),
    )
    save_cache(
        chosen_scorecard,
        os.path.join(cfg["paths"]["results_dir"], "synthetic_screening.json"),
    )

    source_idx = chosen_pair_dict["source_idx"]
    target_idx = chosen_pair_dict["target_idx"]
    assert len(set(source_idx) & set(target_idx)) == 0, "source/target overlap"

    logger.info("")
    logger.info("Stage 3: slicing caches")
    source_triple = slice_by_indices(records, gens, ents, source_idx, "tqa")
    target_triple = slice_by_indices(records, gens, ents, target_idx, "nq")

    cache_dir = cfg["paths"]["cache_dir"]
    save_cache(source_triple[0], get_cache_path(cache_dir, "synth_source_data"))
    save_cache(source_triple[1], get_cache_path(cache_dir, "synth_source_generations"))
    save_cache(source_triple[2], get_cache_path(cache_dir, "synth_source_entailment"))
    save_cache(target_triple[0], get_cache_path(cache_dir, "synth_target_data"))
    save_cache(target_triple[1], get_cache_path(cache_dir, "synth_target_generations"))
    save_cache(target_triple[2], get_cache_path(cache_dir, "synth_target_entailment"))
    source_emb = embeddings[source_idx]
    target_emb = embeddings[target_idx]
    np.save(os.path.join(cache_dir, "synth_source_embeddings.npy"), source_emb)
    np.save(os.path.join(cache_dir, "synth_target_embeddings.npy"), target_emb)

    if not args.skip_m1:
        logger.info("")
        logger.info("Stage 4: Method 1 (SGen-Semi) on synthetic pair")
        m1 = run_m1_on_pair(cfg, source_triple, target_triple)
        logger.info("  M1 in-domain validity: %.3f, mean FDR-E: %.3f",
                    m1["indomain"]["validity_rate"], m1["indomain"]["mean_fdr_e"])
        logger.info("  M1 shifted  validity: %.3f, mean FDR-E: %.3f",
                    m1["shifted"]["validity_rate"], m1["shifted"]["mean_fdr_e"])

    if not args.skip_m2:
        logger.info("")
        logger.info("Stage 5: Method 2 (Conservative) on synthetic pair")
        m2 = run_m2_on_pair(cfg, source_triple, target_triple)
        logger.info("  M2 sweep complete (%d options)", len(m2))

    if not args.skip_m3:
        logger.info("")
        logger.info("Stage 6: Method 3 (DS-SGen) on synthetic pair")
        m3 = run_m3_on_pair(cfg, source_triple, target_triple,
                            source_emb, target_emb)
        logger.info("  M3 in-domain validity: %.3f, mean FDR-E: %.3f",
                    m3["indomain"]["validity_rate"], m3["indomain"]["mean_fdr_e"])
        logger.info("  M3 shifted  validity: %.3f, mean FDR-E: %.3f",
                    m3["shifted"]["validity_rate"], m3["shifted"]["mean_fdr_e"])
        logger.info("  M3 mean n_eff across splits: %.1f",
                    m3["diagnostics"]["mean_n_eff_across_splits"])

    logger.info("")
    logger.info("Total time: %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
