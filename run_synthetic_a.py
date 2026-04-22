"""Orchestrator for Design A: full synthetic QA generation.

Pipeline:
  Stage 0: Generate synthetic (Q, A) pool with GPT-4o-mini (10 topics x 3 tiers x 80 items).
  Stage 1: Run the existing generation pipeline on the synthetic questions.
  Stage 2: DeBERTa entailment scoring.
  Stage 3: MiniLM sentence embeddings.
  Stage 4: Topic-mixture resampling + alpha sweep (reuses synthetic_shift.py).
  Stage 5-7: M1 / M2 / M3 on the constructed pair.

All artifacts use the "synth_qa" dataset name (generation/entailment caches)
and "synthetic_a_" results prefix.

Usage:
    python run_synthetic_a.py
    python run_synthetic_a.py --skip-generation   # if synth_qa_data.json already cached
    python run_synthetic_a.py --n-splits 50       # debug
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
from ds_sgen.generate_synthetic_qa import generate_qa_pool
from ds_sgen.generate_responses import generate_and_cache_openai
from ds_sgen.entailment_scoring import score_and_cache
from ds_sgen.importance_weighted import compute_embeddings
from ds_sgen.synthetic_shift import (
    sweep_alpha_with_screening, build_synthetic_pair, pick_best_alpha,
    _json_safe,
)


DATASET_NAME = "synth_qa"


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)


def log_pool_diagnostics(records, gens, ents, logger):
    """Report per-(topic, tier) cell accuracy so we can verify tier discrimination."""
    by_cell = {}
    for r, e in zip(records, ents):
        key = (r["topic"], r["tier"])
        by_cell.setdefault(key, []).append(e["entail_label"])

    logger.info("")
    logger.info("Pool accuracy by (topic, tier):")
    topics = sorted({r["topic"] for r in records})
    tiers = sorted({r["tier"] for r in records})
    logger.info("  %-30s %s", "topic", "  ".join(f"tier{t}" for t in tiers))
    for topic in topics:
        row = [topic]
        for tier in tiers:
            labels = by_cell.get((topic, tier), [])
            if labels:
                row.append(f"{np.mean(labels):.2f}(n={len(labels)})")
            else:
                row.append("--")
        logger.info("  %-30s %s", row[0], "  ".join(row[1:]))

    logger.info("")
    logger.info("Pool accuracy by tier (marginalized over topic):")
    for tier in tiers:
        labels = [e["entail_label"] for r, e in zip(records, ents) if r["tier"] == tier]
        logger.info("  tier %d: acc=%.3f over n=%d", tier, np.mean(labels), len(labels))


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


def _cfg_with_scratch_results(cfg: dict) -> dict:
    cfg_syn = copy.deepcopy(cfg)
    cfg_syn["sgen"]["cal_dataset"] = "tqa"
    scratch = os.path.join(cfg["paths"]["results_dir"], "_synth_a_scratch")
    os.makedirs(scratch, exist_ok=True)
    cfg_syn["paths"]["results_dir"] = scratch
    return cfg_syn


def run_m1_on_pair(cfg, source_triple, target_triple):
    from ds_sgen.sgen_semi import run_experiment as m1_run
    cfg_syn = _cfg_with_scratch_results(cfg)
    s_rec, s_gen, s_ent = source_triple
    t_rec, t_gen, t_ent = target_triple
    results = m1_run(
        cfg_syn,
        nq_records=t_rec, nq_generations=t_gen, nq_entailments=t_ent,
        tqa_records=s_rec, tqa_generations=s_gen, tqa_entailments=s_ent,
    )
    save_cache(results,
               os.path.join(cfg["paths"]["results_dir"], "synthetic_a_m1_results.json"))
    return results


def run_m2_on_pair(cfg, source_triple, target_triple):
    from ds_sgen.conservative import run_conservative_experiment
    cfg_syn = _cfg_with_scratch_results(cfg)
    s_rec, s_gen, s_ent = source_triple
    t_rec, t_gen, t_ent = target_triple
    results = run_conservative_experiment(
        cfg_syn,
        nq_records=t_rec, nq_gen=t_gen, nq_ent=t_ent,
        tqa_records=s_rec, tqa_gen=s_gen, tqa_ent=s_ent,
    )
    save_cache(results,
               os.path.join(cfg["paths"]["results_dir"], "synthetic_a_m2_results.json"))
    return results


def run_m3_on_pair(cfg, source_triple, target_triple, source_emb, target_emb):
    from ds_sgen.importance_weighted import run_experiment as m3_run
    cfg_syn = _cfg_with_scratch_results(cfg)

    cache_dir = cfg["paths"]["cache_dir"]
    tqa_emb_path = os.path.join(cache_dir, "tqa_embeddings.npy")
    nq_emb_path = os.path.join(cache_dir, "nq_embeddings.npy")
    bak_tqa = tqa_emb_path + ".bak_synth_a"
    bak_nq = nq_emb_path + ".bak_synth_a"

    if not os.path.exists(tqa_emb_path):
        raise FileNotFoundError(f"Expected {tqa_emb_path} before backup")
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
        elif os.path.exists(nq_emb_path):
            os.remove(nq_emb_path)

    save_cache(results,
               os.path.join(cfg["paths"]["results_dir"], "synthetic_a_m3_results.json"))
    return results


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--per-cell", type=int, default=80)
    ap.add_argument("--n-splits", type=int, default=None)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--skip-m1", action="store_true")
    ap.add_argument("--skip-m2", action="store_true")
    ap.add_argument("--skip-m3", action="store_true")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.60, 0.65, 0.70, 0.75, 0.80, 0.85])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    if args.n_splits is not None:
        cfg["sgen"]["n_splits"] = args.n_splits

    cache_dir = cfg["paths"]["cache_dir"]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Design A — Synthetic QA covariate-shift experiment")
    logger.info("=" * 60)
    t0 = time.time()

    # Stage 0: generate QA pool
    logger.info("")
    logger.info("Stage 0: Synthetic QA generation")
    if args.skip_generation:
        records = load_cache(get_cache_path(cache_dir, f"{DATASET_NAME}_data"))
        if records is None:
            raise FileNotFoundError("--skip-generation set but no cached records")
        logger.info("  Loaded %d cached records", len(records))
    else:
        records = generate_qa_pool(cfg, dataset_name=DATASET_NAME, per_cell=args.per_cell)
    logger.info("  Pool size: %d records", len(records))

    # Stage 1: generate model answers
    logger.info("")
    logger.info("Stage 1: Generating model answers via GPT-4o-mini")
    gens = generate_and_cache_openai(cfg, DATASET_NAME, records)

    # Stage 2: entailment scoring
    logger.info("")
    logger.info("Stage 2: Entailment scoring via DeBERTa")
    ents = score_and_cache(cfg, DATASET_NAME, records, gens)

    log_pool_diagnostics(records, gens, ents, logger)

    # Stage 3: embeddings
    logger.info("")
    logger.info("Stage 3: MiniLM sentence embeddings")
    emb_path = os.path.join(cache_dir, f"{DATASET_NAME}_embeddings.npy")
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        logger.info("  Loaded cached embeddings: %s", embeddings.shape)
    else:
        questions = [r["question"] for r in records]
        embeddings = compute_embeddings(
            questions,
            cfg["importance_weighted"]["embedding_model"],
            cfg["paths"].get("hf_cache"),
        )
        np.save(emb_path, embeddings)
        logger.info("  Cached embeddings: %s -> %s", embeddings.shape, emb_path)

    merged = _merge_records(records, gens, ents)
    pool_acc = float(np.mean([m["entail_label"] for m in merged]))
    logger.info("")
    logger.info("Pool overall accuracy: %.3f (n=%d)", pool_acc, len(merged))

    # Stage 4: alpha sweep + screening
    sa = cfg.get("synthetic_a", {})
    n_target_per_side = sa.get("n_source", 800)
    fm1_quantile = sa.get("fm1_quantile", 0.00)
    K = sa.get("K", 10)
    partition_strategy = sa.get("partition_strategy", "accuracy_sorted")

    logger.info("")
    logger.info("Stage 4: alpha sweep (fm1_quantile=%.2f, K=%d, n_side=%d, partition=%s)",
                fm1_quantile, K, n_target_per_side, partition_strategy)
    sweep = sweep_alpha_with_screening(
        merged, embeddings,
        alphas=args.alphas,
        K=K,
        n_S=n_target_per_side,
        n_T=n_target_per_side,
        fm1_quantile=fm1_quantile,
        epsilon=cfg["sgen"]["epsilon"],
        classifier_C=cfg["importance_weighted"]["classifier_C"],
        seed=cfg["seed"],
        partition_strategy=partition_strategy,
    )
    save_cache(
        {"sweep": [
            {"alpha": e["alpha"], "n_pass": e["n_pass"], "scorecard": e["scorecard"]}
            for e in sweep
        ]},
        os.path.join(results_dir, "synthetic_a_screening_sweep.json"),
    )
    best = pick_best_alpha(sweep, target_clf_acc=sa.get("target_clf_acc", 0.66))
    logger.info("")
    logger.info("Best alpha = %.2f (n_pass=%d/7, T3=%.3f, T4=%.3f, T6=%.3f)",
                best["alpha"], best["n_pass"],
                best["scorecard"]["gap"],
                best["scorecard"]["acc_clf"],
                best["scorecard"]["quartile_spread"])

    chosen_pair = best["pair"]
    save_cache(chosen_pair,
               os.path.join(cache_dir, "synthetic_a_pair_indices.json"))
    save_cache(best["scorecard"],
               os.path.join(results_dir, "synthetic_a_screening.json"))

    source_idx = chosen_pair["source_idx"]
    target_idx = chosen_pair["target_idx"]
    assert len(set(source_idx) & set(target_idx)) == 0, "source/target overlap"

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

    if not args.skip_m1:
        logger.info("")
        logger.info("Stage 5: Method 1 (SGen-Semi) on synthetic_a pair")
        m1 = run_m1_on_pair(cfg, source_triple, target_triple)
        logger.info("  M1 in-domain validity: %.3f (FDR-E %.3f)",
                    m1["indomain"]["validity_rate"], m1["indomain"]["mean_fdr_e"])
        logger.info("  M1 shifted  validity: %.3f (FDR-E %.3f)",
                    m1["shifted"]["validity_rate"], m1["shifted"]["mean_fdr_e"])

    if not args.skip_m2:
        logger.info("")
        logger.info("Stage 6: Method 2 (Conservative) on synthetic_a pair")
        m2 = run_m2_on_pair(cfg, source_triple, target_triple)
        logger.info("  M2 sweep complete (%d options)", len(m2))

    if not args.skip_m3:
        logger.info("")
        logger.info("Stage 7: Method 3 (DS-SGen) on synthetic_a pair")
        m3 = run_m3_on_pair(cfg, source_triple, target_triple, source_emb, target_emb)
        logger.info("  M3 in-domain validity: %.3f (FDR-E %.3f)",
                    m3["indomain"]["validity_rate"], m3["indomain"]["mean_fdr_e"])
        logger.info("  M3 shifted  validity: %.3f (FDR-E %.3f)",
                    m3["shifted"]["validity_rate"], m3["shifted"]["mean_fdr_e"])
        logger.info("  M3 mean n_eff across splits: %.1f",
                    m3["diagnostics"]["mean_n_eff_across_splits"])

    logger.info("")
    logger.info("Total time: %.1f seconds", time.time() - t0)


if __name__ == "__main__":
    main()
