"""Stage 3: Entailment scoring only.

Loads data and generation caches directly — no OpenAI dependency.
Designed to run on GPU via SLURM after Stage 2 completes locally.

Usage:
  python run_entailment.py --config configs/default.yaml
"""

import argparse
import time

from ds_sgen.utils import load_config, set_seed, load_cache, get_cache_path
from ds_sgen.entailment_scoring import score_and_cache


def main():
    parser = argparse.ArgumentParser(description="Run entailment scoring (Stage 3)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["log_dir"] = "logs"
    set_seed(cfg["seed"])

    cache_dir = cfg["paths"]["cache_dir"]

    print("DS-SGen Stage 3: Entailment Scoring")
    print(f"  Config: {args.config}")
    print(f"  Model: {cfg['paths']['entailment_model']}")
    print(f"  Batch size: {cfg['entailment']['batch_size']}")
    print()

    # Load data caches
    print("Loading caches...")
    nq_records = load_cache(get_cache_path(cache_dir, "nq_data"))
    tqa_records = load_cache(get_cache_path(cache_dir, "tqa_data"))
    nq_gen = load_cache(get_cache_path(cache_dir, "nq_generations"))
    tqa_gen = load_cache(get_cache_path(cache_dir, "tqa_generations"))

    for name, data in [("nq_data", nq_records), ("tqa_data", tqa_records),
                        ("nq_generations", nq_gen), ("tqa_generations", tqa_gen)]:
        if data is None:
            print(f"  ERROR: {name} cache not found in {cache_dir}")
            raise FileNotFoundError(f"{name}.json not found in {cache_dir}")
        print(f"  {name}: {len(data)} records")

    if len(nq_records) != len(nq_gen):
        raise ValueError(f"NQ mismatch: {len(nq_records)} data vs {len(nq_gen)} generations")
    if len(tqa_records) != len(tqa_gen):
        raise ValueError(f"TQA mismatch: {len(tqa_records)} data vs {len(tqa_gen)} generations")

    print()
    t0 = time.time()

    print("Scoring NQ...")
    nq_ent = score_and_cache(cfg, "nq", nq_records, nq_gen)
    print()

    print("Scoring TQA...")
    tqa_ent = score_and_cache(cfg, "tqa", tqa_records, tqa_gen)
    print()

    elapsed = time.time() - t0

    # Summary
    nq_correct = sum(1 for e in nq_ent if e["entail_label"] == 1)
    tqa_correct = sum(1 for e in tqa_ent if e["entail_label"] == 1)
    print("=" * 60)
    print(f"Stage 3 complete in {elapsed/60:.1f} min")
    print(f"  NQ:  {nq_correct}/{len(nq_ent)} correct ({nq_correct/len(nq_ent)*100:.1f}%)")
    print(f"  TQA: {tqa_correct}/{len(tqa_ent)} correct ({tqa_correct/len(tqa_ent)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
