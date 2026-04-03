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

from datasets import load_dataset

from ds_sgen.utils import get_cache_path, load_cache, save_cache


def load_nq(cfg: dict) -> list[dict]:
    """Load NQ-Open validation set, normalize, and cache."""
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], "nq_data")
    cached = load_cache(cache_path)
    if cached is not None:
        print(f"  NQ: loaded {len(cached)} questions from cache")
        return cached

    print("  NQ: downloading from HuggingFace...")
    ds = load_dataset(
        cfg["data"]["nq_dataset"],
        split=cfg["data"]["nq_split"],
        cache_dir=cfg["paths"]["hf_cache"],
    )

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
    print(f"  NQ: cached {len(records)} questions")
    return records


def load_tqa(cfg: dict) -> list[dict]:
    """Load TriviaQA (unfiltered.nocontext) validation, downsample, normalize, cache."""
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], "tqa_data")
    cached = load_cache(cache_path)
    if cached is not None:
        print(f"  TQA: loaded {len(cached)} questions from cache")
        return cached

    print("  TQA: downloading from HuggingFace (nocontext = ~633MB, not 29GB)...")
    ds = load_dataset(
        cfg["data"]["tqa_dataset"],
        cfg["data"]["tqa_config"],
        split=cfg["data"]["tqa_split"],
        cache_dir=cfg["paths"]["hf_cache"],
    )

    # Downsample to match NQ size
    sample_size = cfg["data"]["tqa_sample_size"]
    sample_seed = cfg["data"]["sample_seed"]
    ds = ds.shuffle(seed=sample_seed).select(range(sample_size))
    print(f"  TQA: downsampled from 11,313 to {sample_size}")

    records = []
    for i, ex in enumerate(ds):
        answer = ex["answer"]
        primary = answer["value"]
        aliases = answer.get("aliases", [])
        # Deduplicate: primary + aliases
        all_ans = list(dict.fromkeys([primary] + aliases))
        records.append({
            "idx": i,
            "question": ex["question"],
            "reference_answer": primary,
            "all_answers": all_ans,
            "dataset": "tqa",
        })

    save_cache(records, cache_path)
    print(f"  TQA: cached {len(records)} questions")
    return records


def load_and_cache_datasets(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Load both datasets. Returns (nq_records, tqa_records)."""
    print("Stage 1: Loading datasets")
    nq = load_nq(cfg)
    tqa = load_tqa(cfg)
    return nq, tqa
