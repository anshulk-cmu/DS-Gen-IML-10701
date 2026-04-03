"""Orchestrator for Method 2: Conservative Threshold experiment.

Requires cached outputs from Stages 1-3 of run_baseline.py (datasets,
generations, entailment scores). Runs only the SGen-Semi algorithm with
conservative parameter sweeps — no GPU needed, completes in minutes.

Usage:
    python run_conservative.py                       # default config
    python run_conservative.py --config configs/default.yaml
"""

import argparse
import logging
import sys
import time

from ds_sgen.utils import load_config, set_seed, load_cache, get_cache_path
from ds_sgen.conservative import run_conservative_experiment, print_conservative_summary

# ── Logging setup ───────────────────────────────────────────────────────────

def setup_logging():
    """Configure logging to both stdout and a structured format."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler with clean format
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console)


# ── Cache loading ───────────────────────────────────────────────────────────

def load_cached_stages(cfg: dict):
    """Load cached outputs from Stages 1-3 (data, generations, entailment).

    These must have been produced by a prior run of run_baseline.py.
    Raises FileNotFoundError if any required cache is missing.
    """
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


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Run Method 2: Conservative Threshold experiment"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    logger.info("=" * 60)
    logger.info("DS-SGen — Method 2: Conservative Threshold")
    logger.info("=" * 60)
    logger.info("  Config: %s", args.config)
    logger.info("  Seed: %d", cfg["seed"])
    logger.info("")

    # Load all cached data from Stages 1-3
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
    logger.info("")

    # Validate sizes match
    for name, a, b in [("NQ gen", nq_records, nq_gen),
                       ("NQ ent", nq_records, nq_ent),
                       ("TQA gen", tqa_records, tqa_gen),
                       ("TQA ent", tqa_records, tqa_ent)]:
        if len(a) != len(b):
            logger.error("%s size mismatch: %d records vs %d cached. "
                        "Re-run run_baseline.py.", name, len(a), len(b))
            sys.exit(1)

    # Run conservative experiment
    t0 = time.time()

    results = run_conservative_experiment(
        cfg,
        nq_records, nq_gen, nq_ent,
        tqa_records, tqa_gen, tqa_ent,
    )

    elapsed = time.time() - t0

    # Print summary table
    print_conservative_summary(results)

    logger.info("Total time: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
