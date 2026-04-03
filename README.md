# DS-Gen-IML-701

Semantic generation with entailment-based scoring for LLM uncertainty quantification.

## Setup

```bash
# Activate environment
conda activate /data/user_data/anshulk/envs/dsgen

# Verify GPU access (preempt partition — we're at the 8-GPU regular limit)
sbatch scripts/check_gpu.sh

# Run baseline experiment
sbatch scripts/run_gpu.sh
```

## Model

**LLaMA-3.1-8B-Instruct** downloaded from [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and stored locally:

```
/data/user_data/anshulk/dsgen/model_cache/Llama-3.1-8B-Instruct/
```

Config points to this local path — no HF download at runtime.

## Project Structure

```
ds-gen-10701/                       (code — /home/anshulk/ds-gen-10701/)
├── configs/default.yaml            # Hyperparameters, paths, model settings
├── scripts/
│   ├── check_gpu.sh                # SLURM GPU sanity check (preempt)
│   └── run_gpu.sh                  # SLURM experiment runner (preempt)
├── ds_sgen/
│   ├── data_loading.py             # Dataset loading and JSON caching
│   ├── generate_responses.py       # LLM response generation
│   ├── entailment_scoring.py       # NLI cross-encoder scoring
│   ├── sgen_semi.py                # SGen-Semi clustering and entropy
│   └── utils.py                    # Seed, config, helpers
├── run_baseline.py                 # Orchestrator
├── logs/                           # SLURM job logs
├── cache/ -> /data/.../cache       # Symlink to heavy cache
├── results/ -> /data/.../results   # Symlink to heavy results
└── environment.yml
```

## Storage Layout

| What | Where |
|------|-------|
| Code, logs, plots, README | `/home/anshulk/ds-gen-10701/` |
| Cache, results, model weights | `/data/user_data/anshulk/dsgen/` |
| Conda env (`dsgen`, Python 3.10) | `/data/user_data/anshulk/envs/dsgen` |
| LLaMA-3.1-8B-Instruct (~15GB) | `/data/user_data/anshulk/dsgen/model_cache/` |

## Environment

Conda env `dsgen` (Python 3.10) with:
- `torch 2.6.0+cu124` (CUDA 12.4, matches Babel driver 575.51)
- `transformers 5.5.0` (LLaMA-3.1 support)
- `datasets`, `sentence-transformers`, `accelerate`
- `scipy`, `numpy`, `scikit-learn`, `matplotlib`

## SLURM Notes

All jobs use `--partition=preempt` since we're at the 8-GPU regular allocation limit. Jobs may be preempted — use caching (`cache/`) to avoid recomputing expensive results.
