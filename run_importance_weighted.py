"""Orchestrator for Method 3: DS-SGen with Importance Reweighting.

Requires cached outputs from Stages 1-3 of run_baseline.py (datasets,
generations, entailment scores). Adds embedding + domain classifier +
weighted SGen-Semi. GPU recommended for embedding (~2 min), rest is CPU.

Usage:
    python run_importance_weighted.py
    python run_importance_weighted.py --config configs/default.yaml
"""

import argparse
import logging
import sys
import time

from ds_sgen.utils import load_config, set_seed, load_cache, get_cache_path
from ds_sgen.importance_weighted import run_experiment, print_importance_weighted_summary


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)


# ── Cache loading (same as run_conservative.py) ──────────────────────────────

def load_cached_stages(cfg: dict):
    """Load cached outputs from Stages 1-3 (data, generations, entailment)."""
    cache_dir = cfg["paths"]["cache_dir"]

    required = {
        "nq_data": "Stage 1 (data loading)",
        "tqa_data": "Stage 1 (data loading)",
        "nq_generations": "Stage 2 (LLM generation)",
        "tqa_generations": "Stage 2 (LLM generation)",
        "nq_entailment": "Stage 3 (entailment scoring)",
        "tqa_entailment": "Stage 3 (entailment scoring)",
    }

    caches = {}
    for name, stage in required.items():
        path = get_cache_path(cache_dir, name)
        data = load_cache(path)
        if data is None:
            raise FileNotFoundError(
                f"Missing cache: {path}\n"
                f"  This comes from {stage}. Run 'python run_baseline.py' first."
            )
        caches[name] = data

    return caches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Method 3: DS-SGen with Importance Reweighting"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("DS-SGen — Method 3: Importance Reweighting")
    logger.info("=" * 60)
    logger.info("  Config: %s", args.config)
    logger.info("  Seed: %d", cfg["seed"])
    logger.info("")

    # Pre-flight: check embedding model availability
    iw_cfg = cfg["importance_weighted"]
    try:
        from sentence_transformers import SentenceTransformer
        _ = SentenceTransformer(
            iw_cfg["embedding_model"],
            cache_folder=cfg["paths"].get("hf_cache"),
        )
        logger.info("  Embedding model loaded: %s", iw_cfg["embedding_model"])
    except Exception as e:
        logger.error("Cannot load embedding model '%s': %s",
                      iw_cfg["embedding_model"], e)
        logger.error("Pre-download with: python -c \"from sentence_transformers "
                      "import SentenceTransformer; SentenceTransformer('%s')\"",
                      iw_cfg["embedding_model"])
        sys.exit(1)

    # Load all cached data from Stages 1-3
    logger.info("")
    logger.info("Loading cached data from Stages 1-3...")
    try:
        caches = load_cached_stages(cfg)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    nq_records = caches["nq_data"]
    tqa_records = caches["tqa_data"]
    nq_gen = caches["nq_generations"]
    tqa_gen = caches["tqa_generations"]
    nq_ent = caches["nq_entailment"]
    tqa_ent = caches["tqa_entailment"]

    logger.info("  NQ:  %d records, %d generations, %d entailments",
                len(nq_records), len(nq_gen), len(nq_ent))
    logger.info("  TQA: %d records, %d generations, %d entailments",
                len(tqa_records), len(tqa_gen), len(tqa_ent))

    # Validate sizes
    for name, a, b in [("NQ gen", nq_records, nq_gen),
                       ("NQ ent", nq_records, nq_ent),
                       ("TQA gen", tqa_records, tqa_gen),
                       ("TQA ent", tqa_records, tqa_ent)]:
        if len(a) != len(b):
            logger.error("%s size mismatch: %d vs %d. Re-run run_baseline.py.",
                        name, len(a), len(b))
            sys.exit(1)

    # Run experiment
    t0 = time.time()

    results = run_experiment(
        cfg,
        nq_records, nq_gen, nq_ent,
        tqa_records, tqa_gen, tqa_ent,
    )

    elapsed = time.time() - t0

    # Print summary
    print_importance_weighted_summary(results)

    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
