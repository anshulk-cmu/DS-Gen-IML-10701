"""Orchestrator for DS-SGen baseline experiment.

Pipeline stages (each is cached — safe to restart after preemption):
  1. Load datasets (NQ-Open validation + TriviaQA unfiltered.nocontext)
  2. Generate responses (greedy + K=5 sampled) with LLaMA-3.1-8B-Instruct
  3. Entailment scoring (correctness + self-consistency) with DeBERTa-v2-xxlarge-mnli
  4. SGen-Semi algorithm (100 random splits, PAC-FDR threshold selection)
  5. Print summary
"""

import argparse
import time

from ds_sgen.utils import load_config, set_seed
from ds_sgen.data_loading import load_and_cache_datasets
from ds_sgen.generate_responses import generate_and_cache
from ds_sgen.entailment_scoring import score_and_cache
from ds_sgen.sgen_semi import run_experiment


def print_summary(results: dict):
    """Print a formatted summary of experiment results."""
    print("\n" + "=" * 70)
    print("DS-SGen BASELINE RESULTS")
    print("=" * 70)

    id_r = results["indomain"]
    sh_r = results["shifted"]
    epsilon = results["config"]["epsilon"]

    print(f"\n  {id_r['label']} (in-domain, calibration):")
    print(f"    Validity rate:   {id_r['validity_rate']:.2%}  (target: >= 98%)")
    print(f"    Mean FDR-E:      {id_r['mean_fdr_e']:.4f} +/- {id_r['std_fdr_e']:.4f}  "
          f"(target: <= {epsilon})")
    print(f"    Mean efficiency: {id_r['mean_efficiency']:.4f} +/- {id_r['std_efficiency']:.4f}")

    print(f"\n  {sh_r['label']} (shifted test):")
    print(f"    Validity rate:   {sh_r['validity_rate']:.2%}  (target: >= 98%)")
    print(f"    Mean FDR-E:      {sh_r['mean_fdr_e']:.4f} +/- {sh_r['std_fdr_e']:.4f}  "
          f"(target: <= {epsilon})")
    print(f"    Mean efficiency: {sh_r['mean_efficiency']:.4f} +/- {sh_r['std_efficiency']:.4f}")

    print("\n" + "=" * 70)

    id_val = id_r["validity_rate"]
    sh_val = sh_r["validity_rate"]
    print(f"\n  {id_r['label']} validity:  {id_val:.2%}")
    print(f"  {sh_r['label']} validity: {sh_val:.2%}")
    if sh_val < id_val - 0.05:
        print(f"  >>> Domain shift detected: {sh_r['label']} validity dropped significantly")
    else:
        print("  >>> No significant domain shift in validity")
    print()


def main():
    parser = argparse.ArgumentParser(description="Run DS-SGen baseline experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "data", "generate", "entailment", "sgen"],
                        help="Run a specific stage only (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    print(f"DS-SGen Baseline Experiment")
    print(f"  Config: {args.config}")
    print(f"  Stage: {args.stage}")
    print(f"  Generator: {cfg['paths']['model']}")
    print(f"  Entailment: {cfg['paths']['entailment_model']}")
    print(f"  Seed: {cfg['seed']}")
    print()

    t0 = time.time()

    # Stage 1: Load datasets
    nq_records, tqa_records = load_and_cache_datasets(cfg)
    print(f"  NQ: {len(nq_records)} questions, TQA: {len(tqa_records)} questions\n")
    if args.stage == "data":
        return

    # Stage 2: Generate responses
    print("Stage 2: Generating responses")
    nq_gen = generate_and_cache(cfg, "nq", nq_records)
    tqa_gen = generate_and_cache(cfg, "tqa", tqa_records)
    print()
    if args.stage == "generate":
        return

    # Stage 3: Entailment scoring
    print("Stage 3: Entailment scoring")
    nq_ent = score_and_cache(cfg, "nq", nq_records, nq_gen)
    tqa_ent = score_and_cache(cfg, "tqa", tqa_records, tqa_gen)
    print()
    if args.stage == "entailment":
        return

    # Stage 4: SGen-Semi algorithm
    results = run_experiment(
        cfg,
        nq_records, nq_gen, nq_ent,
        tqa_records, tqa_gen, tqa_ent,
    )

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    # Stage 5: Summary
    print_summary(results)


if __name__ == "__main__":
    main()
