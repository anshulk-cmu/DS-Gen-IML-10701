# DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs

**10-701 Introduction to Machine Learning (Spring 2026)**
**Team: Anshul Kumar, Justin Luan**

This project implements and extends the SGen-Semi algorithm from [Lee et al., "Selective Generation for Controllable Language Models" (NeurIPS 2024)](https://arxiv.org/abs/2307.09254). We investigate whether SGen's PAC false discovery rate (FDR-E) guarantees hold under domain shift — when the model is calibrated on one QA distribution (Natural Questions) but deployed on another (TriviaQA).

---

## Research Question

Can a selective generation system for LLMs maintain provable PAC guarantees on its false discovery rate even when test queries come from a different domain than the calibration data?

SGen's FDR-E guarantee holds in-domain (TQA validity rate = 100%) but **breaks under domain shift** (NQ validity rate = 12.4%), because the calibration distribution no longer matches the deployment distribution. This project demonstrates this failure, then proposes DS-SGen to fix it.

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

## Current Implementation — Three Methods

The codebase implements three methods for selective generation under domain shift:

- **Method 1 (Vanilla SGen-Semi):** Baseline with no shift handling. Demonstrates the PAC guarantee breaking under domain shift.
- **Method 2 (Conservative Threshold):** Three naive domain-shift fixes (safety factor, reduced epsilon, delta budget). Shows that ad-hoc conservatism cannot restore guarantees without severe efficiency loss.
- **Method 3 (DS-SGen with Importance Reweighting):** The core contribution. Uses embedding-based density ratio estimation to reweight calibration samples, attempting to restore PAC guarantees under domain shift. Results show the method correctly detects infeasibility but cannot fix concept shift.
- **Epsilon Sweep:** Runs all three methods at epsilon = {0.25, 0.30, 0.35, 0.40} to characterize validity-epsilon tradeoffs across methods.

**Generator:** GPT-4o-mini via OpenAI API.

### Pipeline

```
Questions --> GPT-4o-mini (OpenAI API) --> Greedy answer + K=5 sampled answers
                                                       |
                          +----------------------------+
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
| `generate_responses.py` | SGen Sec. 3.1 (fM1) + Sec. 3.2 (Sampling) | GPT-4o-mini via OpenAI API: greedy with logprobs (fM1 = mean log-prob) + K=5 temperature sampling for self-consistency |
| `entailment_scoring.py` | SGen Sec. 2 (Entailment) + Sec. 3.2 (fM2) | Unidirectional NLI for correctness (greedy -> reference, argmax == ENTAILMENT) + Bidirectional NLI for self-consistency (all C(K,2) pairs, both directions must entail) |
| `sgen_semi.py` | SGen Algorithm 2 + Theorem 1 | Conformal threshold from Z_E (Eq. 3), pseudo-labeling Z_U, Clopper-Pearson upper bound with Bonferroni correction (delta_adj = (delta - delta_p) / \|H\|), grid search over (tau1, tau2) |
| `run_baseline.py` | SGen Sec. 4 (Experiments) | Staged orchestrator with 500 random splits, evaluates NQ-test (in-domain) vs. TriviaQA (shifted) |
| `conservative.py` | Method 2: Conservative Threshold | Three naive domain-shift fixes: (A) safety factor on tau, (B) reduced epsilon, (C) delta budget allocation |
| `run_conservative.py` | Method 2 orchestrator | Loads cached Stages 1-3, runs conservative parameter sweeps (CPU-only, no GPU needed) |
| `importance_weighted.py` | DS-CP + Weighted CP | Embedding, domain classifier, density ratio weights, weighted conformal threshold, weighted Clopper-Pearson bounds |
| `run_importance_weighted.py` | Method 3 orchestrator | Loads cached Stages 1-3, computes embeddings (GPU), runs weighted SGen-Semi |
| `run_epsilon_sweep.py` | All methods comparison | Runs Methods 1-3 at epsilon = {0.25, 0.30, 0.35, 0.40}, produces headline figure data |
| `configs/default.yaml` | SGen Table 1 + upstream code | All hyperparameters validated against paper and [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation) |

### Scoring Functions — Paper-to-Code Mapping

**fM1 (mean log-probability):** `generate_responses.py` extracts per-token log-probs from the OpenAI API response (`logprobs=True` on greedy calls). The mean across all generated tokens is the fM1 score. Higher = model is more confident. This corresponds to SGen's "conditional probability" scoring function.

**fM2 (self-consistency):** `entailment_scoring.py:score_self_consistency()` generates K=5 samples, runs all K*(K-1) directed NLI pairs through DeBERTa, then counts unordered pairs where BOTH directions have argmax == ENTAILMENT. fM2 = (agreeing pairs) / C(K,2). This implements the "semantic clustering" approach from SGen, similar to Kuhn et al.'s semantic uncertainty (Nature 2024).

**Entailment correctness:** `entailment_scoring.py:score_correctness()` runs NLI(greedy_answer, reference_answer). Both a continuous score (P(entailment) softmax probability, used for conformal thresholding) and a binary label (argmax == ENTAILMENT, used for FDR-E evaluation) are produced.

### SGen-Semi Algorithm — Detailed Code Walkthrough

In `sgen_semi.py:_run_single_split()`, each of the 500 splits executes:

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

### Key Findings

1. **SGen's guarantee breaks under domain shift:** TQA validity = 100% but NQ validity = 12.4% — the i.i.d. assumption fails when calibration and test distributions differ.
2. **Conservative adjustments help minimally:** Method 2's best variant improves NQ validity from 12.4% to 22.0% — insufficient for deployment.
3. **Importance reweighting detects infeasibility but cannot fix concept shift:** Method 3 achieves 68.8% NQ validity, but entirely through abstention (344/500 vacuous splits). 0/156 non-vacuous splits are valid.
4. **TQA → NQ is a domain change, not a domain shift:** 91.7% classifier accuracy and 27.7pp accuracy gap prove the domains are fundamentally different, not shifted versions of each other. No calibration-time method can fix this.
5. **Validity decreases with higher epsilon** (counterintuitively) because higher epsilon removes the abstention floor, exposing the systematic failure of TQA-calibrated thresholds on NQ.

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

| Model | Purpose | Notes |
|-------|---------|-------|
| GPT-4o-mini (OpenAI API) | Response generation (greedy + sampled) | Logprobs supported; ~$0.05 per session (~$0.25 total across 5 sessions) |
| [DeBERTa-v2-xxlarge-mnli](https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli) | NLI entailment scoring | 1.5B params, ~6GB VRAM |

**DeBERTa label order: `{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`**
(Different from cross-encoder/nli-deberta-v3-large which has entailment at index 1)

GPT-4o-mini exposes logprobs for the fM1 confidence score via the API. Generation makes two calls per question (greedy+logprobs, n=5 sampled), with results cached incrementally.

## Hyperparameters

All parameters validated against the SGen paper and upstream code at [ml-postech/selective-generation](https://github.com/ml-postech/selective-generation).

| Parameter | Value | Source | Role in the Math |
|-----------|-------|--------|-----------------|
| epsilon (FDR-E target) | 0.25 | Paper's main experiments | Upper bound on P(wrong \| answered) |
| delta (PAC confidence) | 0.02 | Paper experiments (1-delta = 98%) | P{FDR-E <= epsilon} >= 1-delta |
| delta_p (pseudo-label failure prob) | 1e-5 | Paper default | Subtracted from delta before Bonferroni |
| epsilon_e (conformal error rate) | 0.05 | Stricter than paper (0.10) for better pseudo-labels | Controls pseudo-label quality via FER |
| K (sampled responses) | 5 | Paper default; Kuhn et al. confirm diminishing returns past 5 | Number of samples for fM2 self-consistency |
| cal_frac (calibration fraction) | 0.70 | 70% calibration, 30% in-domain test | Data split ratio |
| zu_frac (unlabeled fraction) | 0.75 | 75% Z_U, 25% Z_E within calibration | Semi-supervised split ratio |
| n_splits (random splits) | 500 | SE ≈ 0.021 at p=0.35 for tight method comparison | Repeated experiments for validity estimation |
| n_grid (threshold grid points) | 20 | Percentile-based; \|H\|=20 for fm1_only mode | Bonferroni correction divides delta by \|H\| |
| cal_dataset | "tqa" | Higher correctness expected | Calibrate on TQA, test on NQ |
| selection_mode | "fm1_only" | 1D threshold; \|H\|=20 not 400 | Less Bonferroni penalty |
| embedding_model | all-MiniLM-L6-v2 | DS-CP (Lin et al., 2025) | 384-dim sentence embeddings for density ratio |
| classifier_C | 1.0 | Default regularization | Logistic regression for domain classification |
| weight_clip_percentile | 95 | Standard practice | Clip extreme weights to prevent vacuous bounds |

## Project Structure

```
ds-gen-10701/
├── configs/
│   └── default.yaml                # All hyperparameters and paths (single source of truth)
├── ds_sgen/
│   ├── __init__.py
│   ├── utils.py                    # Config loading, seed, atomic JSON caching
│   ├── data_loading.py             # NQ-Open + TriviaQA: download, normalize, cache
│   ├── generate_responses.py       # GPT-4o-mini via OpenAI API: greedy (fM1) + sampled (K=5)
│   ├── entailment_scoring.py       # DeBERTa NLI: correctness + self-consistency (fM2)
│   ├── sgen_semi.py                # SGen-Semi: conformal, Clopper-Pearson, Bonferroni, grid search
│   ├── conservative.py             # Method 2: Conservative Threshold (3 options for naive shift fix)
│   └── importance_weighted.py      # Method 3: DS-SGen (embeddings, domain classifier, weighted CP)
├── run_baseline.py                 # Staged orchestrator: OpenAI generation + local entailment + SGen
├── run_conservative.py             # Method 2 orchestrator (loads cached Stages 1-3, CPU-only)
├── run_importance_weighted.py      # Method 3 orchestrator (loads cached Stages 1-3, GPU for embeddings)
├── run_epsilon_sweep.py            # Epsilon sweep: all 3 methods at eps={0.25, 0.30, 0.35, 0.40}
├── plot_results.py                 # Visualization: all plots for paper (--stage generation|...|method3|epsilon_sweep)
├── plots/                          # Generated PNG plots (300 DPI)
├── papers/                         # Detailed analysis notes for all 6 foundational papers + project plans
├── docs/                           # Per-method analysis documents
│   ├── method1_baseline_analysis.md
│   ├── method2_conservative_analysis.md
│   └── method3_importance_weighted_analysis.md
├── scripts/
│   ├── run_entailment.sh           # SLURM entailment scoring (general, 1 GPU, 2-day limit)
│   └── run_method1.sh              # SLURM full Method 1: entailment + SGen + plots
├── logs/                           # SLURM .out/.err files
├── cache/ -> /data/.../cache       # Symlink: cached JSON for each pipeline stage
├── results/ -> /data/.../results   # Symlink: final experiment outputs
└── environment.yml                 # Conda env export
```

## Storage Layout

| What | Where | Size (actual) |
|------|-------|---------------|
| Code, logs, plots, README | `/home/anshulk/ds-gen-10701/` | ~4 MB |
| Pipeline caches (JSON) | `/data/user_data/anshulk/dsgen/cache/` | ~60 MB total (nq_data 0.8MB, tqa_data 4.2MB, nq_gen 18.9MB, tqa_gen 29.5MB, nq_ent 2.1MB, tqa_ent 4.2MB) |
| Experiment results | `/data/user_data/anshulk/dsgen/results/` | All methods complete (baseline, conservative, importance_weighted, epsilon_sweep) |
| DeBERTa model cache | `/data/user_data/anshulk/dsgen/model_cache/` | ~6GB |

`cache/` and `results/` in the project root are symlinks to `/data/...` -- gitignored so heavy files never enter the repo. OpenAI API key stored in `.env` (gitignored).

## Running

```bash
# Set OpenAI API key (or use .env file)
export OPENAI_API_KEY=sk-...

# Full pipeline (stages can be run individually):
python run_baseline.py --config configs/default.yaml                # all stages
python run_baseline.py --config configs/default.yaml --stage data   # Stage 1 only
python run_baseline.py --config configs/default.yaml --stage generate  # Stage 2 (needs API key)
python run_baseline.py --config configs/default.yaml --stage sgen   # Stage 4 (needs Stages 1-3)

# Entailment scoring via SLURM (Stage 3, needs GPU):
sbatch scripts/run_entailment.sh

# Method 2: Conservative Threshold (requires cached Stages 1-3)
python run_conservative.py --config configs/default.yaml

# Method 3: DS-SGen + Epsilon Sweep (requires cached Stages 1-3 + GPU for embeddings)
python run_importance_weighted.py --config configs/default.yaml
python run_epsilon_sweep.py --config configs/default.yaml
```

Every stage caches its output as JSON with incremental saves (every 50 questions). If interrupted, rerun and it resumes from the last checkpoint. Entailment scoring requires a GPU (DeBERTa) and can run via SLURM.

```bash
# Generate plots (runs incrementally — only plots data that exists)
python plot_results.py                          # all available plots
python plot_results.py --stage generation       # fM1 histograms, answer length, boxplots
python plot_results.py --stage entailment       # entailment scores, fM2, correctness, scatter
python plot_results.py --stage baseline         # FDR-E distribution, efficiency, validity bars
python plot_results.py --stage conservative     # Pareto frontier, summary table
python plot_results.py --stage method3          # Weight analysis, FDR-E comparison
python plot_results.py --stage epsilon_sweep    # Validity vs epsilon (headline figure)
```

Plots are saved to `plots/` as 300 DPI PNGs.

## Runtime (Actual, Validated)

| Stage | Actual Time | Notes |
|-------|-------------|-------|
| Data loading | ~2 sec | TriviaQA nocontext cached after first 633MB download |
| NQ generation (3,610 Qs) | **109.3 min** | GPT-4o-mini, 0.6 q/s, 232K prompt + 510K completion tokens |
| TQA generation (3,610 Qs) | **~18 hrs wall clock** | 5 sessions due to 10K RPD rate limit; ~18 min effective in final session |
| NQ entailment scoring | **4.8 min** | DeBERTa-v2-xxlarge-mnli, 12.6 q/s, 75,810 NLI pairs, L40S GPU |
| TQA entailment scoring | **9.0 min** | 13.4 q/s, 151,620 NLI pairs, L40S GPU |
| SGen-Semi (500 splits) | **6 sec** | CPU, pure numpy/scipy |
| Method 2: Conservative (500 splits x 4 options) | ~10 min | CPU-only |
| Method 3: Embedding + classifier + weighted SGen | **12.2 sec** | all-MiniLM-L6-v2 + logistic regression, CPU |
| Epsilon sweep (3 methods x 4 epsilons x 500 splits) | **28.4 sec** | CPU, reuses pre-computed weights |
| **Total (Stages 1-3, first run)** | **~14 min compute + ~18 hrs API wait** | Dominated by OpenAI daily rate limit |
| **Total (from cache, Methods 1-3 + sweep)** | **~12 min** | With cached Stages 1-3 |

## Pipeline Status (Updated 2026-04-06)

| Stage | NQ (3,610 Qs) | TQA (3,610 Qs) | Status |
|-------|---------------|----------------|--------|
| 1. Data loading | 3,610 cached (792 KB) | 3,610 cached (2.1 MB) | **Complete** |
| 2. GPT-4o-mini generation | 3,610 (19 MB) | 3,610 (~15 MB) | **Complete** |
| 3. Entailment scoring | 3,610 (2.1 MB, 4.8 min) | 3,610 (~2.1 MB, 9.0 min) | **Complete** |
| 4. Method 1: SGen-Semi (500 splits) | NQ validity: 12.4% | TQA validity: 100% | **Complete** |
| 5. Method 2: Conservative (500 splits x 4 options) | Best NQ validity: 22.0% | TQA validity: 100% | **Complete** |
| 6. Method 3: DS-SGen (importance reweighting, 500 splits) | NQ validity: 68.8% (vacuous) | TQA validity: 100% | **Complete** |
| 7. Epsilon sweep (3 methods x 4 epsilons x 500 splits) | No method reaches 98% | All TQA: 100% | **Complete** |

### Stages 1-3: Validated Results Summary

**Generation (GPT-4o-mini, completed 2026-04-06):**

|                | NQ (3,610)  | TQA (3,610) |
|----------------|-------------|-------------|
| Mean fM1       | -0.0882     | -0.0577     |
| Median fM1     | -0.0718     | -0.0393     |
| Min / Max fM1  | -0.4019 / -0.0000 | -0.5524 / -0.0000 |
| Mean answer length | 102 chars  | 79 chars    |
| Samples per Q  | 5           | 5           |

**Entailment (DeBERTa-v2-xxlarge-mnli on L40S, completed 2026-04-06):**

|                      | NQ (3,610) | TQA (3,610) |
|----------------------|-----------|-------------|
| **Correctness rate** | **43.1%** (1,556/3,610) | **70.8%** (2,557/3,610) |
| Mean entail_score    | 0.3646    | 0.5511      |
| Mean fM2             | 0.5419    | 0.7271      |
| fM1↔correctness (r) | 0.3185    | 0.3402      |
| fM2↔correctness (r) | 0.3483    | 0.3596      |

**Key observations:**
- TQA has 66% higher correctness than NQ (70.8% vs 43.1%) — the core domain shift
- TQA has higher confidence (fM1 mean -0.058 vs -0.088) and higher self-consistency (fM2 mean 0.73 vs 0.54)
- Feature-correctness correlations are moderate (r ≈ 0.32-0.36) — sufficient for threshold selection but not perfect
- TQA-calibrated thresholds applied to NQ are too lenient, causing the PAC guarantee to break

**Method 1 Results (SGen-Semi Baseline, 500 splits, epsilon=0.25, delta=0.02):**

|                    | TQA (in-domain) | NQ (shifted) |
|--------------------|----------------|-------------|
| **Validity rate**  | **100.00%**    | **12.4%**   |
| Mean FDR-E         | 0.1472 ± 0.0186 | 0.3015 ± 0.1176 |
| Mean efficiency    | 0.4078 ± 0.0831 | 0.2287 ± 0.1090 |

The PAC guarantee holds perfectly in-domain (100% validity on TQA) but breaks under domain shift (12.4% validity on NQ — only 62 out of 500 splits satisfy FDR-E ≤ 0.25, all via abstention). This is the motivating failure that Methods 2 and 3 aim to address.

**Example answers across categories:**

| Category | Question | Answer | fM1 | Correct? |
|----------|----------|--------|-----|----------|
| True positive (high conf, correct) | "Who played Dr Kimble in The Fugitive?" | "Harrison Ford" | -0.0001 | Yes (entail_p=0.81) |
| True negative (low conf, wrong) | "to aru kagaku no railgun s episode 3" | "Misaka Mikoto investigates..." | -0.4019 | No (entail_p=0.01) |
| False positive (high conf, wrong) | "who is the coach for the ottawa senators" | "DJ Smith" (outdated) | -0.0071 | No (entail_p=0.00) |
| False negative (low conf, correct) | "time setting of game of thrones" | "medieval-like world" | -0.3344 | Yes (entail_p=0.50) |
| Consistent but wrong (high fM2, wrong) | "Which club won Scottish league cup..." | "Celtic" (answer: East Fife) | -0.0172 | No (fM2=1.0) |
| Correct but inconsistent (low fM2, correct) | "What does DSM-IV define as..." | "Voyeuristic Disorder" | -0.1636 | Yes (fM2=0.0) |

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

GPU tested: NVIDIA L40S (46 GB VRAM), CUDA 12.4, PyTorch 2.6.0. DeBERTa-v2-xxlarge-mnli uses 3.1 GB VRAM at batch=64 with float16.

## SLURM Notes

Entailment scoring runs on `--partition=general` with 1 GPU, `--time=2-00:00:00`. Actual runtime for full entailment (10,830 questions) was **13.8 minutes** on an NVIDIA L40S (46 GB VRAM, 3.1 GB used). The incremental caching system ensures no work is lost on preemption — generation saves every 50 questions, entailment every 200.

Generation was run locally using `nohup python run_baseline.py --stage generate`, resuming across sessions due to OpenAI's 10K requests/day rate limit. Method 1 (Stage 4) runs on CPU in ~6 seconds.

---

## Method 2: Conservative Threshold (Implemented)

Implemented in [conservative.py](ds_sgen/conservative.py) and run via [run_conservative.py](run_conservative.py). Three options for naive domain-shift handling, each swept over multiple parameter values:

- **Option A — Safety Factor on Thresholds:** After grid search finds (tau1, tau2), inflate: `tau1 += log(gamma)` (fM1 is log-scale), `tau2 *= gamma` (fM2 in [0,1]). Swept over gamma = {1.0, 1.2, 1.5, 2.0}.
- **Option B — Reduced Epsilon:** Grid search uses `eps_eff = epsilon/k`, but validity is evaluated against the original epsilon for fair comparison. Swept over k = {1.0, 1.5, 2.0, 3.0, 4.0}.
- **Option C — Delta Budget Allocation:** Reserve a fraction of delta for shift uncertainty: `delta_cp = delta - delta_p - delta_s`. Smaller delta_adj widens Clopper-Pearson bounds. Swept over frac = {0.0, 0.25, 0.50, 0.75}.

### Results (500 splits, epsilon=0.25, delta=0.02)

**Option A (Safety Factor):** gamma=1.0 is identical to Method 1 (NQ validity=12.4%). Any gamma > 1.0 causes all thresholds to be rejected → 0% efficiency, 100% validity (trivially, by selecting nothing).

**Option B (Reduced Epsilon):** Same collapse — eps/1.5 or lower makes bounds too strict, all splits abstain (100% validity, 0% efficiency).

**Option C (Delta Budget):** Gradual improvement. Best at frac=0.75: NQ validity=22.0% (vs 12.4% for M1), efficiency=18.5%, FDR-E=0.2604. Options A and B are too binary (either no effect or total abstention); Option C provides a smooth tradeoff.

**Conclusion:** Conservative methods cannot meaningfully restore the PAC guarantee. The best variant improves NQ validity from 12.4% to 22.0% — a 9.6pp gain, but still far below the 98% target.

---

## Method 3: DS-SGen with Importance Reweighting (Implemented)

The core contribution, combining SGen's PAC machinery with DS-CP's density ratio pipeline. Implemented in `ds_sgen/importance_weighted.py`.

### Pipeline

```
TQA questions  ──┐                    ┌──  NQ questions
                  │                    │
              all-MiniLM-L6-v2 (384-dim sentence embeddings)
                  │                    │
                  └─── Logistic Regression ───┘
                       (domain classifier: TQA=0, NQ=1)
                              │
                    P(NQ|x) for each TQA sample
                              │
                    w(x) = P(NQ|x) / (1 - P(NQ|x))
                    (density ratio = importance weight)
                              │
                    Clip at 95th percentile, normalize to sum=n
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
  Weighted conformal threshold          Weighted Clopper-Pearson
  (pseudo-labeling on Z_E)              (PAC bound with n_eff)
            │                                   │
            └─────────────────┬─────────────────┘
                              │
                    SGen-Semi grid search
                    (same as Method 1, but with weighted bounds)
                              │
                    Select: answer if fM1 >= tau1
                    Abstain: otherwise
```

### Key Modifications vs. Method 1

1. **Weighted conformal threshold:** `_weighted_conformal_threshold()` computes the epsilon_e-th weighted quantile of correct entailment scores on Z_E, instead of the uniform quantile.

2. **Weighted Clopper-Pearson bound:** `_weighted_clopper_pearson_upper()` scales the observed failure rate to the effective sample size n_eff = (sum w)^2 / (sum w^2), giving wider bounds when weights are non-uniform. Guard: returns 1.0 (vacuous) if n_eff < 5.

3. **Weight indexing through splits:** When permuting calibration data, weights are permuted with the same index array so each data point retains its importance weight through all splits.

### Results (500 splits, epsilon=0.25, delta=0.02)

**Diagnostics:**
- Domain classifier 5-fold CV accuracy: **91.7%** (±0.8%) — TQA and NQ are nearly separable
- Effective sample size: n_eff = **1,112.5 / 3,610 (30.8%)** — 69% information loss from reweighting
- Weights: min=0.041, median=0.332, max=5.692 (raw max before clipping: 32.7)

**Headline numbers:**

| Method | NQ Validity | NQ FDR-E | NQ Efficiency |
|--------|-------------|----------|---------------|
| M1 (Vanilla SGen) | 12.4% | 0.3015 | 22.9% |
| M2 (Conservative) | 22.0% | 0.2604 | 18.5% |
| **M3 (DS-SGen)** | **68.8%** | **0.1065** | **8.1%** |
| *Target* | *≥ 98%* | *≤ 0.25* | *maximize* |

**Critical finding: the 68.8% validity is entirely vacuous.** Of 500 splits, 344 (68.8%) find no valid threshold and abstain entirely (selecting nothing = FDR-E = 0 = "valid"). The remaining 156 splits find thresholds, but **0/156 are valid on NQ** (mean FDR-E = 0.3413, minimum = 0.2550). The "improvement" over M1 comes from more abstention (wider weighted bounds due to n_eff collapse), not from better-calibrated thresholds.

### Epsilon Sweep Results

| Epsilon | M1 NQ Valid | M2 NQ Valid | M3 NQ Valid | M3 NQ Eff |
|---------|-------------|-------------|-------------|-----------|
| 0.25 | 12.4% | 22.0% | 68.8% | 8.1% |
| 0.30 | 0.0% | 0.0% | 11.0% | 46.1% |
| 0.35 | 0.0% | 0.0% | 0.2% | 80.0% |
| 0.40 | 0.0% | 0.0% | 0.0% | 98.2% |

**No method achieves ≥ 98% validity at any tested epsilon.** Validity *decreases* with higher epsilon (counterintuitively) because higher epsilon makes thresholds easier to find, which removes the abstention-based validity floor. All TQA-calibrated thresholds systematically fail on NQ at every epsilon level due to concept shift.

### Key Finding: Domain Shift vs. Domain Change

The results reveal that TQA → NQ is a **domain change** (both P(X) and P(Y|X) differ), not a **domain shift** (only P(X) differs). DS-SGen's importance reweighting correctly addresses covariate shift but cannot fix concept shift — the 27.7pp accuracy gap (70.8% TQA vs 43.1% NQ) represents a fundamental change in the model's knowledge, not just a change in question distribution.

**Evidence:**
1. **91.7% classifier accuracy** — domains are nearly separable (not overlapping subsets)
2. **Identical non-vacuous FDR-E** — M1: 0.3442, M3: 0.3413 (reweighting doesn't improve actual error rates)
3. **0/156 valid non-vacuous splits** — correcting P(X) doesn't change P(Y|X)
4. **Minimum FDR-E = 0.2550** — concept shift imposes a hard floor above ε=0.25

**When DS-SGen would work:** For genuine domain shifts where P(Y|X) is approximately stable — e.g., a clinical QA system deployed in a cardiac clinic (same medical knowledge, narrower question focus, classifier accuracy ~70%, accuracy gap < 10pp). Our TQA → NQ experiment establishes the boundary between tractable shifts and intractable domain changes.

**Theoretical basis:** Under covariate shift, SGen's FDR-E decomposition (Lemma 1) is purely algebraic and distribution-agnostic. Each component (FER, FNER, NER) can be re-bounded using importance-weighted Hoeffding-type bounds instead of uniform binomial bounds. The effective sample size n_eff accounts for information loss from non-uniform weighting. See WR-CP (Xu et al., ICLR 2025) for the formal decomposition into covariate + concept shift terms.

## References

- Lee et al., "Selective Generation for Controllable Language Models," NeurIPS 2024 (Spotlight). [Paper](https://arxiv.org/abs/2307.09254) | [Code](https://github.com/ml-postech/selective-generation)
- Mohri & Hashimoto, "Language Models with Conformal Factuality Guarantees," 2024. [Paper](https://arxiv.org/abs/2402.10978) | [Code](https://github.com/tatsu-lab/conformal-factual-lm)
- Cherian, Gibbs, Candes, "Large Language Model Validity via Enhanced Conformal Prediction Methods," NeurIPS 2024. [Paper](https://arxiv.org/abs/2406.09714) | [Code](https://github.com/jjcherian/conformal-safety)
- Tibshirani, Barber, Candes, Ramdas, "Conformal Prediction Under Covariate Shift," NeurIPS 2019. [Paper](https://arxiv.org/abs/1904.06019)
- Lin et al., "Domain-Shift-Aware Conformal Prediction for Large Language Models," arXiv 2025. [Paper](https://arxiv.org/abs/2510.05566)
- Wang et al., "Conformal Prediction Adaptive to Unknown Subpopulation Shifts," arXiv 2025. [Paper](https://arxiv.org/abs/2506.05583)
- Kuhn, Gal, Farquhar, "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation," Nature 2024. [Paper](https://www.nature.com/articles/s41586-024-07421-0)
