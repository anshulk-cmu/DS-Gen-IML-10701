"""Screening pre-flight for PopQA head→tail dataset pair.

Runs the 6-test screening battery to determine whether this pair has the
right shift structure for DS-SGen importance reweighting to succeed.

Pipeline:
  Stage 1: Load PopQA, split head/tail by popularity, sample 1000 each.
  Stage 2: Generate GPT-4o-mini responses (greedy + K=5 sampled).
  Stage 3: Score correctness + self-consistency with DeBERTa.
  Stage 4: Embed questions with all-MiniLM-L6-v2.
  Stage 5: Run 6-test screening battery, print scorecard.

Reuses existing ds_sgen modules for generation, entailment, and embedding.
All intermediate results are cached — safe to re-run.

Usage:
    python run_screening.py
    python run_screening.py --config configs/default.yaml
    python run_screening.py --epsilon 0.30
"""

import argparse
import logging
import sys
import time

import numpy as np

from ds_sgen.utils import load_config, set_seed, save_cache
from ds_sgen.screening import load_popqa, run_screening_tests, print_scorecard
from ds_sgen.generate_responses import generate_and_cache_openai
from ds_sgen.entailment_scoring import score_and_cache
from ds_sgen.importance_weighted import compute_embeddings
from ds_sgen.sgen_semi import _merge_records


def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run DS-SGen screening pre-flight on PopQA head→tail"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override epsilon for screening thresholds")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    scfg = cfg["screening"]
    epsilon = args.epsilon if args.epsilon is not None else scfg.get("epsilon") or cfg["sgen"]["epsilon"]

    logger.info("=" * 60)
    logger.info("DS-SGen Screening: PopQA head -> tail")
    logger.info("=" * 60)
    logger.info("  Config: %s", args.config)
    logger.info("  Epsilon: %.2f", epsilon)
    logger.info("  Sample size: %d per domain", scfg["sample_size"])
    logger.info("")

    t_start = time.time()

    # ── Stage 1: Load PopQA ──────────────────────────────────────────────
    logger.info("Stage 1: Loading PopQA (head/tail split)")
    head_records, tail_records = load_popqa(cfg)
    logger.info("  Head: %d questions, Tail: %d questions",
                len(head_records), len(tail_records))
    logger.info("")

    # ── Stage 2: Generate responses ──────────────────────────────────────
    logger.info("Stage 2: Generating GPT-4o-mini responses")
    head_gen = generate_and_cache_openai(cfg, "popqa_head", head_records)
    tail_gen = generate_and_cache_openai(cfg, "popqa_tail", tail_records)
    logger.info("")

    # ── Stage 3: Entailment scoring ──────────────────────────────────────
    logger.info("Stage 3: Entailment scoring (DeBERTa)")
    head_ent = score_and_cache(cfg, "popqa_head", head_records, head_gen)
    tail_ent = score_and_cache(cfg, "popqa_tail", tail_records, tail_gen)
    logger.info("")

    # ── Stage 4: Embeddings ──────────────────────────────────────────────
    logger.info("Stage 4: Computing embeddings")
    model_name = cfg["importance_weighted"]["embedding_model"]
    hf_cache = cfg["paths"].get("hf_cache")
    cache_dir = cfg["paths"]["cache_dir"]

    head_emb_path = f"{cache_dir}/popqa_head_embeddings.npy"
    tail_emb_path = f"{cache_dir}/popqa_tail_embeddings.npy"

    import os
    if os.path.exists(head_emb_path) and os.path.exists(tail_emb_path):
        logger.info("  Loading cached embeddings")
        emb_S = np.load(head_emb_path)
        emb_T = np.load(tail_emb_path)
    else:
        head_questions = [r["question"] for r in head_records]
        tail_questions = [r["question"] for r in tail_records]
        emb_S = compute_embeddings(head_questions, model_name, hf_cache)
        emb_T = compute_embeddings(tail_questions, model_name, hf_cache)
        np.save(head_emb_path, emb_S)
        np.save(tail_emb_path, emb_T)
        logger.info("  Cached embeddings")

    logger.info("  Head embeddings: %s, Tail embeddings: %s",
                emb_S.shape, emb_T.shape)
    logger.info("")

    # ── Stage 5: Screening battery ───────────────────────────────────────
    logger.info("Stage 5: Running screening battery")

    # Extract arrays
    head_merged = _merge_records(head_records, head_gen, head_ent)
    tail_merged = _merge_records(tail_records, tail_gen, tail_ent)

    y_S = np.array([r["entail_label"] for r in head_merged])
    y_T = np.array([r["entail_label"] for r in tail_merged])
    fM_S = np.array([r["fM1"] for r in head_merged])
    fM_T = np.array([r["fM1"] for r in tail_merged])

    results = run_screening_tests(
        y_S=y_S, y_T=y_T,
        fM_S=fM_S, fM_T=fM_T,
        emb_S=emb_S, emb_T=emb_T,
        epsilon=epsilon,
        classifier_C=cfg["importance_weighted"]["classifier_C"],
    )

    # Print scorecard
    print_scorecard(results)

    # Save results
    results_path = f"{cfg['paths']['results_dir']}/screening_popqa_results.json"
    # Remove non-serializable numpy types
    serializable = {k: v for k, v in results.items()}
    save_cache(serializable, results_path)
    logger.info("  Results saved to %s", results_path)

    elapsed = time.time() - t_start
    logger.info("Total screening time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
