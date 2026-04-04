# DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs

**10-701 Introduction to Machine Learning (Spring 2026)**
**Team: Anshul Kumar, Justin Luan**

This project implements and extends the SGen-Semi algorithm from [Lee et al., "Selective Generation for Controllable Language Models" (NeurIPS 2024)](https://arxiv.org/abs/2307.09254). We investigate whether SGen's PAC false discovery rate (FDR-E) guarantees hold under domain shift — when the model is calibrated on one QA distribution (Natural Questions) but deployed on another (TriviaQA).

---

## Research Question

Can a selective generation system for LLMs maintain provable PAC guarantees on its false discovery rate even when test queries come from a different domain than the calibration data?

SGen's FDR-E guarantee holds in-domain (NQ validity rate ~98%) but is expected to **break under domain shift** (TriviaQA validity rate drops), because the calibration distribution no longer matches the deployment distribution. This project first demonstrates this failure, then proposes DS-SGen to fix it.

---

## Theoretical Foundations — Paper-by-Paper

This project synthesizes six papers, spanning two independent research tracks that converge at DS-SGen.

### Track 1: Selective Prediction with Statistical Guarantees

#### Paper 1 — SGen (Lee et al., NeurIPS 2024) **[Foundational — our baseline]**

The core paper. SGen introduces **selective generation**: the model either answers or says "I don't know," controlling the **FDR-E** (False Discovery Rate with respect to textual Entailment) with PAC guarantees. The key innovation is using **textual entailment** (not exact match) as the correctness metric — if the model's answer logically implies the reference, it's correct. This resolves the "metric misalignment" problem where exact string matching rejects correct answers with different phrasing.

**SGen-Semi** (the semi-supervised variant we implement) works in four stages:
1. Learn a conformal entailment set on labeled data Z_E to produce pseudo-labels
2. Pseudo-label the larger unlabeled set Z_U using the conformal threshold
3. Decompose FDR-E into three independently-boundable terms: FER (false entailment rate), FNER (false non-entailment rate), NER (non-entailment rate) — Lemma 1
4. Grid search over (tau1, tau2) threshold pairs, using Clopper-Pearson bounds with Bonferroni correction across |H| candidates

The **neuro-selection function** uses two confidence signals jointly: fM1 (mean log-probability — generation confidence) and fM2 (self-consistency — semantic agreement among K=5 sampled answers). Selection rule: answer if fM1 >= tau1 AND fM2 >= tau2.

**Critical limitation:** All guarantees require i.i.d. data. Under domain shift, the binomial tail bounds become invalid because calibration samples are not representative of the test distribution.

#### Paper 2 — Conformal Factuality (Mohri & Hashimoto, Stanford, 2024) **[Alternative approach]**

A complementary approach: instead of binary answer/abstain, this paper **removes uncertain sub-claims** from the output, keeping only reliable parts. The conceptual breakthrough is that every LLM output implicitly defines an uncertainty set through entailment — making conformal prediction tractable for open-ended generation without enumerating the infinite output space.

The back-off function progressively removes claims ranked by confidence (frequency scoring = self-consistency across multiple samples is the best practical scorer). Theorem 4.1 provides coverage: P(backed-off output is correct) >= 1 - alpha, with a tight upper bound of 1 - alpha + 1/(n+1).

**Limitations:** Marginal-only guarantee (fails for hard subgroups), removes ~76% of claims on medical QA to achieve 90% correctness, and assumes i.i.d. data.

#### Paper 3 — Enhanced CP (Cherian, Gibbs, Candes, NeurIPS 2024) **[State-of-the-art]**

Fixes Conformal Factuality's two problems simultaneously using the Candes group's conformal machinery:

- **Conditional boosting** (Proposition 3.1): Learns optimal linear combinations of scoring functions by differentiating through the conformal calibration algorithm. The key insight is that for linear function classes, the quantile regression is a linear program whose solution is locally linear in the scoring weights, enabling gradient-based optimization. Improves claim retention from 24% to 39%.

- **Level-adaptive CP** (Theorem 3.2): Instead of a fixed alpha for all prompts, adapts the guarantee level per-prompt to ensure >= 70% claim retention. Issues honest, calibrated probabilities: among prompts where the system claims 70-80% correctness, the output actually IS correct 70-80% of the time.

**Corollary A.1** provides a covariate shift interpretation: if domain-related features (embedding distance, predicted domain membership) are included in the function class F, the conditional guarantee holds under shifts describable by functions in F — domain-shift robustness "for free" without explicit density estimation. This is our **Approach B** for the full paper.

### Track 2: Conformal Prediction Under Distribution Shift

#### Paper 4 — Weighted CP (Tibshirani, Barber, Candes, Ramdas, NeurIPS 2019) **[Theoretical foundation]**

THE foundational theory for adapting conformal prediction to covariate shift. Introduces **weighted exchangeability** — a generalization where data points can have different importance weights, and the standard quantile argument still works if you use weighted quantiles.

Under covariate shift (P_test(Y|X) = P_cal(Y|X) but P_test(X) != P_cal(X)), each calibration point gets weight w(x_i) = P_test(x_i)/P_cal(x_i). The **classifier trick** converts density estimation into classification: train a classifier to distinguish calibration vs. test inputs, use P(test|x)/(1-P(test|x)) as the weight. Crucially, only the ratio matters (Remark 3) — no normalization constant needed.

**Key result:** Coverage >= 1 - alpha is restored under covariate shift with known weights. The price: reduced effective sample size n_eff = (sum w_i)^2 / (sum w_i^2), which widens intervals.

**Limitations:** Formal guarantee assumes exact weights (no theory for estimated weights), single low-dimensional dataset evaluation, extreme weights can make the method vacuous.

#### Paper 5 — DS-CP (Lin et al., UC Berkeley/Munich RE, arXiv 2025) **[Direct precursor]**

Applies weighted CP to LLMs via a practical three-step pipeline:
1. **Embed** prompts using a sentence transformer (all-MiniLM-L6-v2) — reduces the intractable text space to 384 dimensions
2. **Classify** using XGBoost to distinguish calibration from test embeddings, convert to density ratios
3. **Regularize** by replacing the test-point weight with lambda=1 to prevent degenerate (all-inclusive) prediction sets

Tested across 16 LLMs (1.8B-72B params) and 272 MMLU domain pairs. DS-CP is **adaptive**: it helps most where standard CP fails worst, and barely changes behavior when standard CP already works.

**Key theoretical insight (Theorems 1-2):** The coverage gap depends on **score distribution similarity**, not prompt distribution similarity. Even if prompts look very different, if the model's uncertainty patterns are similar across domains, DS-CP works well.

**Limitations:** Only multiple-choice QA (finite output space), no abstention capability, approximate guarantees with unknowable error terms, not peer-reviewed.

#### Paper 6 — Subpopulation CP (Wang et al., USC, arXiv 2025) **[Alternative shift model]**

Models domain shift as **subpopulation shift**: the test distribution is a different mixture of K known subpopulations. Three algorithms with progressively weaker requirements:
- **Algorithm 1:** Per-test-point domain classifier weights. Guarantee under multicalibration.
- **Algorithm 2:** Averaged domain classifier predictions. Guarantee under multiaccuracy (weaker).
- **Algorithm 3:** No domain knowledge — filters to top-beta similar calibration points, reweights by softmax-normalized embedding similarity. No formal guarantee but empirically matches the oracle.

**Theorem 2.1 (negative result):** Group-conditional CP with an imperfect classifier (accuracy gamma) can drop coverage to gamma - alpha. Even 90% classifier accuracy at alpha=0.1 yields coverage as low as 80%.

**Relevance:** Algorithm 3's filter-and-reweight approach is a practical alternative to density ratio estimation. The Dirichlet-based evaluation methodology (sampling 100 test environments) provides a systematic framework for testing under varied shifts.

---

## How the Papers Connect — The Gap DS-SGen Fills

```
Track 1: Selective Prediction          Track 2: Domain Shift Adaptation
 
SGen (NeurIPS 2024)                    Weighted CP (NeurIPS 2019)
  PAC FDR-E control                       Coverage under covariate shift
  Entailment-based correctness            Likelihood ratio reweighting
  Answer/abstain decisions                Weighted exchangeability
  LIMITATION: i.i.d. only                 LIMITATION: low-dimensional only
       |                                       |
Conformal Factuality (2024)            DS-CP (arXiv 2025)
  Sub-claim removal                       Embedding-based density ratios
  Entailment sets for open-ended          Regularization for LLM prompts
  LIMITATION: marginal only               LIMITATION: multiple-choice only
       |                                       |
Enhanced CP (NeurIPS 2024)             Subpop CP (arXiv 2025)
  Conditional boosting                    Mixture-of-domains model
  Level-adaptive guarantees               Embedding similarity filtering
  Covariate shift in F                    LIMITATION: no open-ended gen
       |                                       |
       +------- DS-SGen (this project) --------+
                     |
            PAC FDR-E control (from SGen)
          + Entailment-based correctness (from SGen)
          + Embedding-based importance weights (from DS-CP)
          + Selective generation / abstention (from SGen)
          + Domain shift robustness (from Weighted CP / DS-CP)
```

**No existing method provides all four simultaneously:**
1. PAC-style finite-sample guarantees (not just marginal coverage)
2. Open-ended text generation with entailment-based correctness
3. Selective prediction with abstention ("I don't know")
4. Robustness to domain shift between calibration and test data

---

## Current Implementation — Vanilla SGen-Semi Baseline

The current codebase implements the SGen-Semi baseline (Method 1: no shift handling). This establishes the motivating failure: the PAC guarantee holds in-domain but breaks under domain shift.

### Pipeline

```
Questions --> LLaMA-3.1-8B-Instruct --> Greedy answer + K=5 sampled answers
                                              |
                     +------------------------+
                     |
              DeBERTa-v2-xxlarge-mnli (NLI)
                     |
           +---------+----------+
           |                    |
     fM1: mean log-prob    fM2: self-consistency
     (generation confidence)  (semantic agreement)
           |                    |
           +---------+----------+
                     |
              SGen-Semi Algorithm
              (conformal calibration + PAC-FDR threshold selection)
                     |
              Select: answer if fM1 >= tau1 AND fM2 >= tau2
              Abstain: otherwise
```

### How Each Code Module Maps to the Papers

| Module | Paper Section | What It Implements |
|--------|-------------|-------------------|
| `data_loading.py` | SGen Sec. 4 (Datasets) | NQ-Open + TriviaQA download, normalization, caching |
| `generate_responses.py` | SGen Sec. 3.1 (fM1) + Sec. 3.2 (Sampling) | Greedy decoding with per-token log-probs (fM1 = mean log-prob) + K=5 temperature sampling for self-consistency |
| `entailment_scoring.py` | SGen Sec. 2 (Entailment) + Sec. 3.2 (fM2) | Unidirectional NLI for correctness (greedy -> reference, argmax == ENTAILMENT) + Bidirectional NLI for self-consistency (all C(K,2) pairs, both directions must entail) |
| `sgen_semi.py` | SGen Algorithm 2 + Theorem 1 | Conformal threshold from Z_E (Eq. 3), pseudo-labeling Z_U, Clopper-Pearson upper bound with Bonferroni correction (delta_adj = (delta - delta_p) / \|H\|), grid search over (tau1, tau2) |
| `run_baseline.py` | SGen Sec. 4 (Experiments) | Staged orchestrator with 100 random splits, evaluates NQ-test (in-domain) vs. TriviaQA (shifted) |
| `conservative.py` | Method 2: Conservative Threshold | Three naive domain-shift fixes: (A) safety factor on tau, (B) reduced epsilon, (C) delta budget allocation |
| `run_conservative.py` | Method 2 orchestrator | Loads cached Stages 1-3, runs conservative parameter sweeps (CPU-only, no GPU needed) |
| `configs/default.yaml` | SGen Table 1 + upstream code | All hyperparameters validated against paper and [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation) |

### Scoring Functions — Paper-to-Code Mapping

**fM1 (mean log-probability):** `generate_responses.py:_extract_logprobs_from_scores()` computes per-token log-probs from the `output_logits` of greedy decoding. The mean across all generated tokens is the fM1 score. Higher = model is more confident. This corresponds to SGen's "conditional probability" scoring function.

**fM2 (self-consistency):** `entailment_scoring.py:score_self_consistency()` generates K=5 samples, runs all K*(K-1) directed NLI pairs through DeBERTa, then counts unordered pairs where BOTH directions have argmax == ENTAILMENT. fM2 = (agreeing pairs) / C(K,2). This implements the "semantic clustering" approach from SGen, similar to Kuhn et al.'s semantic uncertainty (Nature 2024).

**Entailment correctness:** `entailment_scoring.py:score_correctness()` runs NLI(greedy_answer, reference_answer). Both a continuous score (P(entailment) softmax probability, used for conformal thresholding) and a binary label (argmax == ENTAILMENT, used for FDR-E evaluation) are produced.

### SGen-Semi Algorithm — Detailed Code Walkthrough

In `sgen_semi.py:_run_single_split()`, each of the 100 splits executes:

1. **Split NQ** (line 98-104): Random permutation, 70% calibration / 30% test
2. **Split calibration** (line 107-109): 75% Z_U (unlabeled) / 25% Z_E (labeled)
3. **Conformal threshold** (line 112-113): `_compute_conformal_threshold()` sorts Z_E entailment scores, takes the k-th value where k = ceil((n+1)(1-epsilon_e)). This is the split conformal quantile from Vovk (2005).
4. **Pseudo-label** (line 116-117): Z_U entries with entail_score >= tau_CP are labeled "correct"
5. **Grid search** (line 120-151): For each (tau1, tau2) in the 50x50 percentile grid:
   - Count selected examples (fM1 >= tau1 AND fM2 >= tau2)
   - Count failures (selected AND pseudo_label == 0)
   - Clopper-Pearson upper bound via `scipy.stats.beta.ppf(1 - delta_adj, failures+1, selected-failures)`
   - Keep the pair with highest efficiency where the bound <= epsilon
6. **Evaluate** (line 154-177): Apply learned thresholds to NQ-test and full TriviaQA

### Key Hypothesis

SGen's FDR-E guarantee holds in-domain (NQ validity rate ~98%) but **breaks under domain shift** (TriviaQA validity rate drops), because the calibration distribution no longer matches the deployment distribution. This is the fundamental limitation of the i.i.d. assumption.

---

## Datasets

| Dataset | HuggingFace ID | Config | Split | Size |
|---------|---------------|--------|-------|------|
| NQ-Open | `google-research-datasets/nq_open` | -- | validation | 3,610 |
| TriviaQA | `mandarjoshi/trivia_qa` | `unfiltered.nocontext` | validation | 11,313 -> 3,610 (downsampled) |

- **NQ-Open**: `question` (str), `answer` (list[str]) -- multiple valid answers per question. These are genuine Google search queries.
- **TriviaQA**: `question` (str), `answer.value` (str), `answer.aliases` (list[str]). Trivia-style factual questions -- more specific, differently phrased.
- Using `unfiltered.nocontext` avoids downloading 29GB of context documents (~633MB instead)
- **Domain shift characterization:** NQ = how people naturally ask search engines; TriviaQA = how trivia enthusiasts phrase questions. The embedding distributions are measurably different.

## Models

| Model | Purpose | Size | VRAM (fp16) |
|-------|---------|------|-------------|
| [LLaMA-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Response generation (greedy + sampled) | 8B params | ~16GB |
| [DeBERTa-v2-xxlarge-mnli](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli) | NLI entailment scoring | 1.5B params | ~6GB |

**DeBERTa label order: `{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`**
(Different from cross-encoder/nli-deberta-v3-large which has entailment at index 1)

LLaMA is loaded locally from `/data/user_data/anshulk/dsgen/model_cache/Llama-3.1-8B-Instruct/` -- no runtime download. DeBERTa downloads to the same cache on first use.

## Hyperparameters

All parameters validated against the SGen paper and upstream code at [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation).

| Parameter | Value | Source | Role in the Math |
|-----------|-------|--------|-----------------|
| epsilon (FDR-E target) | 0.25 | Paper's main experiments | Upper bound on P(wrong \| answered) |
| delta (PAC confidence) | 0.02 | Paper experiments (1-delta = 98%) | P{FDR-E <= epsilon} >= 1-delta |
| delta_p (pseudo-label failure prob) | 1e-5 | Paper default | Subtracted from delta before Bonferroni |
| epsilon_e (conformal error rate) | 0.10 | Paper default | Controls pseudo-label quality via FER |
| K (sampled responses) | 5 | Paper default; Kuhn et al. confirm diminishing returns past 5 | Number of samples for fM2 self-consistency |
| cal_frac (calibration fraction) | 0.70 | 70% calibration, 30% in-domain test | NQ data split ratio |
| zu_frac (unlabeled fraction) | 0.75 | 75% Z_U, 25% Z_E within calibration | Semi-supervised split ratio |
| n_splits (random splits) | 100 | Paper standard | Repeated experiments for validity estimation |
| n_grid (threshold grid points) | 50 | Percentile-based; \|H\| = 50^2 = 2,500 | Bonferroni correction divides delta by \|H\| |

## Project Structure

```
ds-gen-10701/
├── configs/
│   └── default.yaml                # All hyperparameters and paths (single source of truth)
├── ds_sgen/
│   ├── __init__.py
│   ├── utils.py                    # Config loading, seed, atomic JSON caching
│   ├── data_loading.py             # NQ-Open + TriviaQA: download, normalize, cache
│   ├── generate_responses.py       # LLaMA chat-template generation: greedy (fM1) + sampled (K=5)
│   ├── entailment_scoring.py       # DeBERTa NLI: correctness + self-consistency (fM2)
│   ├── sgen_semi.py                # SGen-Semi: conformal, Clopper-Pearson, Bonferroni, grid search
│   └── conservative.py             # Method 2: Conservative Threshold (3 options for naive shift fix)
├── run_baseline.py                 # Staged orchestrator (--stage data|generate|entailment|sgen|all)
├── run_conservative.py             # Method 2 orchestrator (loads cached Stages 1-3, CPU-only)
├── plot_results.py                 # Visualization: all plots for paper (--stage generation|entailment|baseline|conservative|all)
├── plots/                          # Generated PNG plots (300 DPI)
├── papers/                         # Detailed analysis notes for all 6 foundational papers + project plans
├── scripts/
│   ├── check_gpu.sh                # SLURM GPU sanity check (preempt, A6000)
│   ├── run_gpu.sh                  # SLURM full baseline pipeline (preempt, A6000, 48GB mem)
│   └── run_conservative.sh         # SLURM Method 2: conservative threshold sweep
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

`cache/` and `results/` in the project root are symlinks to `/data/...` -- gitignored so heavy files never enter the repo.

## Running

```bash
# Activate environment
conda activate /data/user_data/anshulk/envs/dsgen

# Full baseline pipeline via SLURM (preempt partition, 1x A6000)
sbatch scripts/run_gpu.sh

# Or run specific stages
python run_baseline.py --stage data         # Download and cache datasets only
python run_baseline.py --stage generate     # Data + LLM generation
python run_baseline.py --stage entailment   # Data + generation + NLI scoring
python run_baseline.py --stage sgen         # Full pipeline including SGen-Semi
python run_baseline.py                      # Same as --stage all

# Method 2: Conservative Threshold (requires cached Stages 1-3 from baseline)
sbatch scripts/run_conservative.sh
# Or run directly (CPU-only, completes in minutes):
python run_conservative.py --config configs/default.yaml
```

Every stage caches its output as JSON. If a SLURM job is preempted, resubmit -- it resumes from the last cached checkpoint (saves every 50 questions for generation, every 200 for entailment). Atomic writes via tempfile prevent corrupted caches.

```bash
# Generate plots (runs incrementally — only plots data that exists)
python plot_results.py                          # all available plots
python plot_results.py --stage generation       # fM1 histograms, answer length, boxplots
python plot_results.py --stage entailment       # entailment scores, fM2, correctness, scatter
python plot_results.py --stage baseline         # FDR-E distribution, efficiency, validity bars
python plot_results.py --stage conservative     # Pareto frontier, summary table
```

Plots are saved to `plots/` as 300 DPI PNGs.

## Runtime Estimates (1x NVIDIA RTX A6000, 48GB)

| Stage | Time | VRAM | Notes |
|-------|------|------|-------|
| Data loading | ~2 min | CPU | TriviaQA nocontext = 633MB download |
| LLM generation (2 x 3,610 questions) | ~48 hours | ~16GB | Greedy + K=5 sampled per question; ~24h per dataset |
| Entailment scoring (~185K NLI calls) | ~3-5 min | ~6GB | Batch size 64 |
| SGen-Semi (100 splits) | ~30 sec | CPU | Pure numpy/scipy |
| **Total** | **~48-50 hours** | | |

## Pipeline Status (Updated 2026-04-04)

Current SLURM job: `6951565` on `babel-w9-20` (preempt, 7-day wall time)

| Stage | NQ (3,610 Qs) | TQA (3,610 Qs) | Status |
|-------|---------------|----------------|--------|
| 1. Data loading | 3,610 cached | 3,610 cached | **Complete** |
| 2. LLM generation | 3,610 cached | 2,550/3,610 (71%) | **In progress** |
| 3. Entailment scoring | Not started | Not started | Blocked on Stage 2 |
| 4. SGen-Semi | Not started | Not started | Blocked on Stage 3 |

### Early Results — Generation Confidence (fM1) by Domain

| Metric | NQ (in-domain) | TQA (shifted) |
|--------|----------------|---------------|
| Mean fM1 (log-prob) | -0.2261 +/- 0.1371 | -0.1814 +/- 0.1389 |
| Median fM1 | -0.1985 | -0.1445 |
| Range | [-0.887, -0.001] | [-0.896, -0.000] |
| Mean answer length | 156 chars | 113 chars |
| Samples per question | K=5 | K=5 |

**Observation:** TQA answers are *more* confident (higher fM1) than NQ, and shorter. This is interesting — higher generation confidence does not necessarily mean higher correctness under domain shift. Whether this confidence is well-calibrated will be tested in Stage 3 (entailment) and Stage 4 (SGen-Semi validity rate).

### Job History

| Job ID | Partition | Duration | Outcome |
|--------|-----------|----------|---------|
| 6942461 | preempt | 5 sec | GPU check passed (A6000, CUDA 12.4) |
| 6943087 | preempt | ~2 min | Failed (config/import issues, fixed) |
| 6943094 | preempt | ~4 hours | NQ complete, TQA 2350/3610 — **killed: 48h time limit** |
| 6951565 | preempt | running | Resumed from TQA 2350, 7-day time limit |

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

All jobs use `--partition=preempt` with `--requeue` and a 7-day wall time, since we are at the 8-GPU regular allocation limit. The preempt partition provides A6000 GPUs but jobs may be interrupted. The incremental caching system ensures no work is lost on preemption — generation saves every 50 questions, entailment every 200.

---

## Method 2: Conservative Threshold (Implemented)

Implemented in [conservative.py](ds_sgen/conservative.py) and run via [run_conservative.py](run_conservative.py). Three options for naive domain-shift handling, each swept over multiple parameter values:

- **Option A — Safety Factor on Thresholds:** After grid search finds (tau1, tau2), inflate: `tau1 += log(gamma)` (fM1 is log-scale), `tau2 *= gamma` (fM2 in [0,1]). Swept over gamma = {1.0, 1.2, 1.5, 2.0}.
- **Option B — Reduced Epsilon:** Grid search uses `eps_eff = epsilon/k`, but validity is evaluated against the original epsilon for fair comparison. Swept over k = {1.0, 1.5, 2.0, 3.0, 4.0}.
- **Option C — Delta Budget Allocation:** Reserve a fraction of delta for shift uncertainty: `delta_cp = delta - delta_p - delta_s`. Smaller delta_adj widens Clopper-Pearson bounds. Swept over frac = {0.0, 0.25, 0.50, 0.75}.

Expected result: restores TQA validity but at severe efficiency cost (too many "I don't know" responses). Requires cached Stages 1-3 from baseline; runs on CPU only in minutes.

---

## Planned Extension — Method 3: DS-SGen with Importance Reweighting (Proposed)

The core contribution, combining SGen's PAC machinery with DS-CP's density ratio pipeline:

1. **Embed** all NQ and TriviaQA prompts using sentence-transformers (all-MiniLM-L6-v2, 384-dim)
2. **Train domain classifier** (logistic regression on embeddings) to distinguish NQ from TriviaQA
3. **Compute importance weights** w(x_i) = P(TQA|x_i) / (1 - P(TQA|x_i)), clip at 95th percentile
4. **Weighted conformal pseudo-labeling** — replace uniform empirical distribution with weighted distribution
5. **Weighted binomial bounds** — use effective sample size n_eff = (sum w)^2 / (sum w^2) in Clopper-Pearson
6. **Domain-aware selection** — add cosine similarity to NQ centroid as third signal alongside fM1 and fM2
7. **Threshold optimization** — maximize efficiency subject to weighted PAC constraint

**Theoretical basis (informal):** Under covariate shift, SGen's FDR-E decomposition (Lemma 1) is purely algebraic and distribution-agnostic. Each component (FER, FNER, NER) can be re-bounded using importance-weighted Hoeffding-type bounds instead of uniform binomial bounds. The union bound yields: P{FDR-E <= epsilon} >= 1 - delta - O(Delta_w), where Delta_w is the weight estimation error.

## References

- Lee et al., "Selective Generation for Controllable Language Models," NeurIPS 2024 (Spotlight). [Paper](https://arxiv.org/abs/2307.09254) | [Code](https://github.com/ml-postech/selective-generation)
- Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees," 2024. [Paper](https://arxiv.org/abs/2402.10978) | [Code](https://github.com/tatsu-lab/conformal-factual-lm)
- Cherian, Gibbs, Candes, "Large Language Model Validity via Enhanced Conformal Prediction Methods," NeurIPS 2024. [Paper](https://arxiv.org/abs/2406.09714) | [Code](https://github.com/jjcherian/conformal-safety)
- Tibshirani, Barber, Candes, Ramdas, "Conformal Prediction Under Covariate Shift," NeurIPS 2019. [Paper](https://arxiv.org/abs/1904.06019)
- Lin et al., "Domain-Shift-Aware Conformal Prediction for Large Language Models," arXiv 2025. [Paper](https://arxiv.org/abs/2510.05566)
- Wang et al., "Conformal Prediction Adaptive to Unknown Subpopulation Shifts," arXiv 2025. [Paper](https://arxiv.org/abs/2506.05583)
- Kuhn, Gal, Farquhar, "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation," Nature 2024. [Paper](https://www.nature.com/articles/s41586-024-07421-0)
