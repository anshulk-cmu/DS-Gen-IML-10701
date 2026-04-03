# DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs

**10-701 Introduction to Machine Learning (Spring 2026)**

This project implements and extends the SGen-Semi algorithm from [Lee et al., "Selective Generation for Controllable Language Models" (NeurIPS 2024)](https://arxiv.org/abs/2307.09254). We investigate whether SGen's PAC false discovery rate (FDR-E) guarantees hold under domain shift — when the model is calibrated on one QA distribution (Natural Questions) but deployed on another (TriviaQA).

## Method Overview

SGen-Semi is a selective generation framework: rather than answering every question, the model abstains when it is likely to be wrong. The key guarantee is **PAC FDR-E control** — with probability >= 1 - delta, the fraction of answered questions that are wrong is at most epsilon.

### Pipeline

```
Questions ──> LLaMA-3.1-8B-Instruct ──> Greedy answer + K=5 sampled answers
                                              │
                     ┌────────────────────────┘
                     │
              DeBERTa-v2-xxlarge-mnli (NLI)
                     │
           ┌─────────┴──────────┐
           │                    │
     fM1: mean log-prob    fM2: self-consistency
     (generation confidence)  (semantic agreement)
           │                    │
           └─────────┬──────────┘
                     │
              SGen-Semi Algorithm
              (conformal calibration + PAC-FDR threshold selection)
                     │
              Select: answer if fM1 >= tau1 AND fM2 >= tau2
              Abstain: otherwise
```

### Scoring Functions

- **fM1 (mean log-probability):** Average token-level log-prob of the greedy answer. Higher = model is more confident in its generation.
- **fM2 (self-consistency):** Fraction of bidirectionally entailing pairs among K=5 sampled answers. Higher = sampled answers agree with each other semantically.

### SGen-Semi Algorithm (per split)

1. **Split NQ** into calibration (70%) and in-domain test (30%)
2. **Split calibration** into Z_U (75%, unlabeled) and Z_E (25%, labeled)
3. **Conformal threshold** tau_CP from Z_E: `k = ceil((n+1)(1-epsilon_e))`, take k-th sorted entailment score
4. **Pseudo-label Z_U**: correct if entail_score >= tau_CP
5. **Grid search** over (tau1, tau2) threshold pairs on Z_U with pseudo-labels:
   - Clopper-Pearson upper bound on FDR-E with Bonferroni correction: `delta_adj = (delta - delta_p) / |H|`
   - Select pair with highest efficiency where CP upper bound <= epsilon
6. **Evaluate** learned thresholds on NQ test (in-domain) and full TriviaQA (shifted)
7. **Repeat** 100 times with different random splits

### Key Hypothesis

SGen's FDR-E guarantee holds in-domain (NQ validity rate ~98%) but may **break under domain shift** (TriviaQA validity rate drops), because the calibration distribution no longer matches the deployment distribution.

## Datasets

| Dataset | HuggingFace ID | Config | Split | Size |
|---------|---------------|--------|-------|------|
| NQ-Open | `google-research-datasets/nq_open` | — | validation | 3,610 |
| TriviaQA | `mandarjoshi/trivia_qa` | `unfiltered.nocontext` | validation | 11,313 → 3,610 (downsampled) |

- **NQ-Open**: `question` (str), `answer` (list[str]) — multiple valid answers per question
- **TriviaQA**: `question` (str), `answer.value` (str), `answer.aliases` (list[str])
- Using `unfiltered.nocontext` avoids downloading 29GB of context documents (~633MB instead)

## Models

| Model | Purpose | Size | VRAM (fp16) |
|-------|---------|------|-------------|
| [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Response generation (greedy + sampled) | 8B params | ~16GB |
| [DeBERTa-v2-xxlarge-mnli](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli) | NLI entailment scoring | 1.5B params | ~6GB |

**DeBERTa label order: `{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`**
(Different from cross-encoder/nli-deberta-v3-large which has entailment at index 1)

LLaMA is loaded locally from `/data/user_data/anshulk/dsgen/model_cache/Llama-3.1-8B-Instruct/` — no runtime download. DeBERTa downloads to the same cache on first use.

## Hyperparameters

All parameters validated against the SGen paper and upstream code at [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation).

| Parameter | Value | Source |
|-----------|-------|--------|
| epsilon (FDR-E target) | 0.25 | Paper's main experiments |
| delta (PAC confidence) | 0.02 | Paper experiments (1-delta = 98%) |
| delta_p (pseudo-label failure prob) | 1e-5 | Paper default; subtracted from delta before Bonferroni |
| epsilon_e (conformal error rate) | 0.10 | Paper default |
| K (sampled responses) | 5 | Paper default; Kuhn et al. confirm diminishing returns past 5 |
| cal_frac (calibration fraction) | 0.70 | 70% calibration, 30% in-domain test |
| zu_frac (unlabeled fraction) | 0.75 | 75% Z_U, 25% Z_E within calibration |
| n_splits (random splits) | 100 | Paper standard |
| n_grid (threshold grid points) | 50 | Percentile-based; |H| = 50^2 = 2,500 |

## Project Structure

```
ds-gen-10701/
├── configs/
��   └── default.yaml                # All hyperparameters and paths (single source of truth)
├── ds_sgen/
│   ├── __init__.py
│   ├── utils.py                    # Config loading, seed, atomic JSON caching
│   ├── data_loading.py             # NQ-Open + TriviaQA: download, normalize, cache
│   ├── generate_responses.py       # LLaMA chat-template generation: greedy (fM1) + sampled (K=5)
│   ├── entailment_scoring.py       # DeBERTa NLI: correctness + self-consistency (fM2)
│   └── sgen_semi.py                # SGen-Semi: conformal, Clopper-Pearson, Bonferroni, grid search
├── run_baseline.py                 # Staged orchestrator (--stage data|generate|entailment|sgen|all)
├── scripts/
│   ├── check_gpu.sh                # SLURM GPU sanity check (preempt, A6000)
│   └── run_gpu.sh                  # SLURM full pipeline (preempt, A6000, 48GB mem)
├── logs/                           # SLURM .out/.err files
├── cache/ -> /data/.../cache       # Symlink: cached JSON for each pipeline stage
├── results/ -> /data/.../results   # Symlink: final experiment outputs
└── environment.yml                 # Conda env export
```

## Storage Layout

| What | Where | Size |
|------|-------|------|
| Code, logs, plots, README | `/home/anshulk/ds-gen-10701/` | ~1MB |
| Conda env (`dsgen`, Python 3.10) | `/data/user_data/anshulk/envs/dsgen/` | 5.8GB |
| LLaMA-3.1-8B-Instruct weights | `/data/user_data/anshulk/dsgen/model_cache/` | 30GB |
| Pipeline caches (JSON) | `/data/user_data/anshulk/dsgen/cache/` | ~50MB per dataset |
| Experiment results | `/data/user_data/anshulk/dsgen/results/` | ~5MB |

`cache/` and `results/` in the project root are symlinks to `/data/...` — gitignored so heavy files never enter the repo.

## Running

```bash
# Activate environment
conda activate /data/user_data/anshulk/envs/dsgen

# Full pipeline via SLURM (preempt partition, 1x A6000)
sbatch scripts/run_gpu.sh

# Or run specific stages
python run_baseline.py --stage data         # Download and cache datasets only
python run_baseline.py --stage generate     # Data + LLM generation
python run_baseline.py --stage entailment   # Data + generation + NLI scoring
python run_baseline.py --stage sgen         # Full pipeline including SGen-Semi
python run_baseline.py                      # Same as --stage all
```

Every stage caches its output as JSON. If a SLURM job is preempted, resubmit — it resumes from the last cached checkpoint (saves every 50 questions for generation, every 200 for entailment). Atomic writes via tempfile prevent corrupted caches.

## Runtime Estimates (1x NVIDIA RTX A6000, 48GB)

| Stage | Time | VRAM | Notes |
|-------|------|------|-------|
| Data loading | ~2 min | CPU | TriviaQA nocontext = 633MB download |
| LLM generation (2 x 3,610 questions) | ~1-2 hours | ~16GB | Greedy + K=5 sampled per question |
| Entailment scoring (~185K NLI calls) | ~3-5 min | ~6GB | Batch size 64 |
| SGen-Semi (100 splits) | ~30 sec | CPU | Pure numpy/scipy |
| **Total** | **~1.5-2.5 hours** | | |

## Environment

Conda env `dsgen` (Python 3.10):

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| transformers | 5.5.0 |
| datasets | 4.8.4 |
| sentence-transformers | 5.3.0 |
| accelerate | 1.13.0 |
| scipy | 1.15.3 |
| numpy | 2.2.6 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.8 |

GPU verified: NVIDIA RTX A6000, CUDA 12.4, PyTorch compute test passed.

## SLURM Notes

All jobs use `--partition=preempt` with `--requeue` since we are at the 8-GPU regular allocation limit. The preempt partition provides A6000 GPUs but jobs may be interrupted. The incremental caching system ensures no work is lost on preemption.

## References

- Lee et al., "Selective Generation for Controllable Language Models," NeurIPS 2024 (Spotlight). [Paper](https://arxiv.org/abs/2307.09254) | [Code](https://github.com/ml-postech/selective-generation)
- Kuhn, Gal, Farquhar, "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation," Nature 2024. [Paper](https://www.nature.com/articles/s41586-024-07421-0)
