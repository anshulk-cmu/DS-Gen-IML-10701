"""Load NQ-Open and TriviaQA, normalize to a common schema, cache as JSON.

Normalized schema per question:
{
    "idx": int,
    "question": str,
    "reference_answer": str,      # single canonical answer
    "all_answers": [str],         # all valid answers (for evaluation flexibility)
    "dataset": "nq" | "tqa"
}
"""

import logging
import os

from datasets import load_dataset

from ds_sgen.utils import get_cache_path, load_cache, save_cache

logger = logging.getLogger(__name__)


def _setup_file_logger(log_dir: str):
    """Add a file handler to the module logger if one doesn't exist yet."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "data_loading.log"))
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def load_nq(cfg: dict) -> list[dict]:
    """Load NQ-Open validation set, normalize, and cache."""
    _setup_file_logger(cfg.get("log_dir", "logs"))
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], "nq_data")
    cached = load_cache(cache_path)
    if cached is not None:
        msg = f"NQ: loaded {len(cached)} questions from cache ({cache_path})"
        logger.info(msg)
        print(f"  {msg}")
        return cached

    logger.info("NQ: downloading from HuggingFace (dataset=%s, split=%s)",
                cfg["data"]["nq_dataset"], cfg["data"]["nq_split"])
    print("  NQ: downloading from HuggingFace...")
    ds = load_dataset(
        cfg["data"]["nq_dataset"],
        split=cfg["data"]["nq_split"],
        cache_dir=cfg["paths"]["hf_cache"],
    )
    logger.info("NQ: downloaded %d raw examples", len(ds))

    records = []
    for i, ex in enumerate(ds):
        records.append({
            "idx": i,
            "question": ex["question"],
            "reference_answer": ex["answer"][0],
            "all_answers": ex["answer"],
            "dataset": "nq",
        })

    save_cache(records, cache_path)
    logger.info("NQ: cached %d questions to %s", len(records), cache_path)
    print(f"  NQ: cached {len(records)} questions")
    return records


def load_tqa(cfg: dict) -> list[dict]:
    """Load TriviaQA (unfiltered.nocontext) validation, downsample, normalize, cache."""
    _setup_file_logger(cfg.get("log_dir", "logs"))
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], "tqa_data")
    cached = load_cache(cache_path)
    if cached is not None:
        msg = f"TQA: loaded {len(cached)} questions from cache ({cache_path})"
        logger.info(msg)
        print(f"  {msg}")
        return cached

    logger.info("TQA: downloading from HuggingFace (dataset=%s, config=%s, split=%s)",
                cfg["data"]["tqa_dataset"], cfg["data"]["tqa_config"], cfg["data"]["tqa_split"])
    print("  TQA: downloading from HuggingFace (nocontext = ~633MB, not 29GB)...")
    ds = load_dataset(
        cfg["data"]["tqa_dataset"],
        cfg["data"]["tqa_config"],
        split=cfg["data"]["tqa_split"],
        cache_dir=cfg["paths"]["hf_cache"],
    )
    logger.info("TQA: downloaded %d raw examples", len(ds))

    # Downsample
    sample_size = cfg["data"]["tqa_sample_size"]
    sample_seed = cfg["data"]["sample_seed"]
    ds = ds.shuffle(seed=sample_seed).select(range(sample_size))
    logger.info("TQA: downsampled from %d to %d (seed=%d)", len(ds) + (11313 - sample_size), sample_size, sample_seed)
    print(f"  TQA: downsampled to {sample_size}")

    records = []
    for i, ex in enumerate(ds):
        answer = ex["answer"]
        primary = answer["value"]
        aliases = answer.get("aliases", [])
        all_ans = list(dict.fromkeys([primary] + aliases))
        records.append({
            "idx": i,
            "question": ex["question"],
            "reference_answer": primary,
            "all_answers": all_ans,
            "dataset": "tqa",
        })

    save_cache(records, cache_path)
    logger.info("TQA: cached %d questions to %s", len(records), cache_path)
    print(f"  TQA: cached {len(records)} questions")
    return records


def load_and_cache_datasets(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load both datasets. Returns (nq_records, tqa_records)."""
    _setup_file_logger(cfg.get("log_dir", "logs"))
    logger.info("=" * 50)
    logger.info("Stage 1: Loading datasets")
    print("Stage 1: Loading datasets")
    nq = load_nq(cfg)
    tqa = load_tqa(cfg)
    logger.info("Stage 1 complete: NQ=%d, TQA=%d", len(nq), len(tqa))
    return nq, tqa
