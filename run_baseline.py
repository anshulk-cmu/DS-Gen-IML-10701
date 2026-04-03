"""Orchestrator for ds-sgen baseline experiments."""

import argparse
import os
from ds_sgen.utils import load_config, set_seed, ensure_dirs
from ds_sgen.data_loading import load_questions, load_cached, save_cached
from ds_sgen.generate_responses import load_generator, generate_responses
from ds_sgen.entailment_scoring import load_entailment_model, compute_entailment_matrix
from ds_sgen.sgen_semi import cluster_responses, compute_sgen_score


def main():
    parser = argparse.ArgumentParser(description="Run ds-sgen baseline")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ensure_dirs(cfg["cache_dir"], cfg["results_dir"])

    print("Configuration loaded.")
    print(f"  Generator: {cfg['generator_model']}")
    print(f"  Entailment: {cfg['entailment_model']}")
    print(f"  Cache dir: {cfg['cache_dir']}")
    print(f"  Results dir: {cfg['results_dir']}")


if __name__ == "__main__":
    main()
