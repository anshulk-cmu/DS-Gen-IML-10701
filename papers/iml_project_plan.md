# DS-SGen PoC: Scoping the NeurIPS Proposal into a Perfect 10-701 Term Project

## Team: Anshul Kumar, Justin Luan

---

## 1. Understanding the Constraints — What 10-701 Requires vs. What We Have

The full DS-SGen proposal targets NeurIPS with 5 datasets, 5+ models, two theoretical contributions, and hundreds of domain-shift pairs. That is a summer-long effort. The 10-701 project needs to be a **convincing, self-contained proof-of-concept** that demonstrates the core idea works, while perfectly hitting every rubric criterion. Let us walk through the exact mapping.

### 1.1 Deliverables and Deadlines

| Deliverable | Due Date | Length | Weight | Status |
|---|---|---|---|---|
| Proposal | Fri, March 13 | 1–2 pages | 2% | Write immediately |
| Mentor Meeting #1 | By Fri, March 20 | 15–30 min | — | Schedule after proposal |
| Check-in | Wed, April 8 | 4 pages | 2% | Baseline results ready |
| Mentor Meeting #2 | By Wed, April 8 | 15–30 min | — | Show initial DS-SGen results |
| Video Showcase | Mon, April 20 | 4 minutes | 4% | Polish key results |
| Final Report | Wed, April 22 | 8 pages | 12% | Everything complete |

### 1.2 Rubric Mapping — How We Score 100/100

| Criterion | Points | What They Want | Our Strategy |
|---|---|---|---|
| **Completeness** | 20 | All required sections present | Strict checklist adherence (Section 5 below) |
| **Literature Review** | 10 | Synthesis, connections, trends — not just listing | 6 papers with a narrative arc showing the gap (Section 3) |
| **Technical Soundness** | 30 | Reproducible detail, proper ML practices, val/test split | Full algorithm pseudocode, proper held-out tuning (Section 4) |
| **Implementation Correctness** | 20 | Code matches writeup, clean code, meaningful names | Modular codebase, docstrings, matches equations exactly |
| **Clarity** | 10 | Clear, well-organized writing | LaTeX template, logical flow, labeled figures |
| **Formatting** | 5 | 8-page limit, correct template | ArXiv template, references on extra pages |
| **Performance** | 5 | Reasonable results, bonus for exceptional | DS-SGen should clearly beat baseline under shift |

---

## 2. The PoC Scoping — What We Keep, What We Defer

The key principle: **one dataset, two models, three methods, one clean story.** We want depth on the core idea, not breadth across every possible extension.

### 2.1 What We KEEP for the PoC (Core Scope)

**One primary dataset: Natural Questions → TriviaQA domain shift.**
- Calibrate on Natural Questions (NQ), test on TriviaQA. This is the same base dataset SGen used (NQ), so we can directly compare to their reported results under i.i.d., then show what breaks under shift and how DS-SGen fixes it. Both are open-ended QA with known ground truth. Both are freely available, well-documented, and widely used in the hallucination detection literature.
- Split NQ into train (for LLM few-shot if needed), calibration (for conformal/SGen), and validation (for hyperparameter tuning). TriviaQA is the held-out test domain — never touched during development.
- This creates a real, meaningful domain shift: NQ consists of genuine Google search queries (how people naturally ask things), while TriviaQA consists of trivia-style questions (more specific, factual, differently phrased).

**One LLM: LLaMA-3.1-8B-Instruct.**
- LLaMA-3.1-8B fits on a single A6000 GPU (48GB VRAM, ~16GB for fp16 inference) and generates high-quality open-ended answers. We have white-box access for token log-probabilities.
- We do NOT need GPT-4 or 70B models for the PoC. The research question is about the *statistical framework*, not the base model quality.
- A second model (e.g., Mistral-7B) may be added if compute allows, but is not required for the core story.

**Three methods (≥1 baseline + ≥1 proposed, we do three for rigor):**

**Method 1 — Vanilla SGen-Semi (Baseline 1: No shift handling).**
Implement the original SGen-Semi algorithm from Lee et al. (2024) as faithfully as possible. This uses standard (unweighted) conformal prediction for pseudo-labeling, standard binomial bounds for PAC guarantees, and a simple selection function (threshold on self-consistency score). This is our "what happens if you ignore domain shift" baseline. We expect the PAC guarantee to fail when calibrated on NQ and tested on TriviaQA — this is the motivating failure we want to demonstrate.

**Method 2 — Conservative Threshold (Baseline 2: Naive shift handling).**
Three options for making SGen-Semi more conservative without explicit domain-shift modeling:
  - **Option A — Safety Factor:** After grid search finds (tau1, tau2), inflate thresholds: tau1 += log(gamma) (fM1 is log-scale), tau2 *= gamma (fM2 in [0,1]). Swept over gamma = {1.0, 1.2, 1.5, 2.0}.
  - **Option B — Reduced Epsilon:** Use epsilon_eff = epsilon/k in the grid search constraint, but evaluate validity against the original epsilon. Swept over k = {1.0, 1.5, 2.0, 3.0, 4.0}.
  - **Option C — Delta Budget Allocation:** Reserve a fraction of delta for potential shift: delta_cp = delta - delta_p - delta_s. Smaller delta_adj widens Clopper-Pearson bounds. Swept over frac = {0.0, 0.25, 0.50, 0.75}.
This is what a practitioner would do without our method — "just be more cautious." We expect this to restore validity but at a severe cost to efficiency (the model says "I don't know" too often).

**Method 3 — DS-SGen with Importance Reweighting (Our Proposed Method).**
The core contribution. Embed prompts using a sentence transformer (all-MiniLM-L6-v2), train a domain classifier (logistic regression on embeddings) to distinguish NQ from TriviaQA prompts, convert classifier probabilities to importance weights, clip extreme weights, and plug these into weighted versions of SGen's binomial bounds. The selection function adds domain similarity as a third signal. This should restore validity (PAC guarantee holds) while maintaining high efficiency (the model answers most questions it can).

### 2.2 What We DEFER to the Full Paper (Summer)

| Deferred Element | Reason |
|---|---|
| Approach B (Conditional Conformal from Cherian et al.) | Requires more complex implementation and theoretical work |
| MMLU multi-domain experiments (272 pairs) | Scale is for the full paper |
| MedLFQA, TruthfulQA, CoQA datasets | Breadth is for the full paper |
| Formal theorem with tight bounds | PoC demonstrates the idea works empirically; formal proof is summer work |
| 70B and GPT-4 models | Compute constraints |
| Neuro-selection function with learned weights | Simple threshold is sufficient for PoC |
| Subpopulation shift comparison | A different shift model, defer |
| Level-adaptive guarantees | Extension from Cherian et al., defer |

### 2.3 Validation: Is This Scope Right for 10-701?

Let us check against the example proposal (machine translation with 3 methods from scratch). Our project has: 3 distinct methods (vanilla SGen, conservative threshold, DS-SGen), each implemented by us, using a contemporary research paper component (SGen from NeurIPS 2024, DS-CP from arXiv 2025), evaluated on a held-out test set with clearly defined metrics. This matches the expected scope exactly — arguably more sophisticated given the theoretical grounding.

---

## 3. Literature Review Plan — 6 Papers, One Narrative Arc

The rubric says "minimum 4 papers, at least 2 from last 3 years" and emphasizes **synthesis** over listing. We use 6 papers arranged in a narrative that tells the story of why DS-SGen is the natural next step. Every paper connects to the next, and we explicitly state what gap each leaves open.

### The Narrative Arc

**Act 1: Conformal Prediction Foundations**

*Paper 1 — Vovk et al. (2005), "Algorithmic Learning in a Random World."* The foundational textbook that established conformal prediction. We cite this for the core exchangeability-based coverage guarantee. Takeaway: CP gives distribution-free coverage, but requires exchangeability. Connection to next: what if exchangeability breaks?

*Paper 2 — Tibshirani et al. (NeurIPS 2019), "Conformal Prediction Under Covariate Shift."* Introduces weighted exchangeability and shows that reweighting by the density ratio restores CP guarantees under covariate shift. Takeaway: the theoretical solution exists, but requires knowing the likelihood ratio. Connection to next: how do you estimate this for LLMs?

**Act 2: Conformal Prediction Meets LLMs**

*Paper 3 — Lee et al. (NeurIPS 2024), "Selective Generation for Controllable Language Models" (SGen).* Our foundational paper. Introduces entailment-based correctness, selective prediction (answer/abstain), and PAC guarantees for LLM generation. Takeaway: the selective generation framework works beautifully under i.i.d. Connection to next: but it assumes i.i.d., which fails in practice.

*Paper 4 — Mohri & Hashimoto (ICML 2024), "Language Models with Conformal Factuality Guarantees."* Shows that entailment sets make CP tractable for open-ended text, enabling factuality guarantees by removing uncertain sub-claims. Takeaway: a complementary approach to SGen (graduated back-off vs. binary abstention), also i.i.d. only. Connection to next: both SGen and this paper break under domain shift.

**Act 3: Addressing Domain Shift for LLMs**

*Paper 5 — Lin et al. (arXiv 2025), "Domain-Shift-Aware Conformal Prediction for LLMs" (DS-CP).* Applies weighted CP to LLMs by embedding prompts, training a domain classifier, and reweighting. Tested on MMLU across 272 domain pairs. Takeaway: the density ratio estimation pipeline works for LLMs in embedding space. Limitation: only handles multiple-choice QA, no abstention, approximate guarantees.

*Paper 6 — Cherian, Gibbs, Candès (NeurIPS 2024), "LLM Validity via Enhanced CP Methods."* Introduces conditional boosting and level-adaptive CP. Shows that conditional conformal prediction can handle certain covariate shifts if domain-related features are in the function class F. Takeaway: an alternative to density ratio estimation for handling shift. Limitation: requires i.i.d. for the base guarantee; the covariate shift extension is limited to shifts within F.

**The Synthesis (This Is What Gets Full Marks):**

These six papers form two parallel tracks — selective prediction (SGen, Conformal Factuality) and domain-shift adaptation (Tibshirani, DS-CP, Enhanced CP) — that have developed independently. No paper combines all four desiderata: PAC guarantees + open-ended generation + abstention + domain-shift robustness. DS-SGen is the first framework to bridge these tracks, inheriting SGen's entailment-based selective generation and DS-CP's embedding-based importance reweighting into a unified pipeline with provable guarantees under shift.

---

## 4. Technical Plan — Step-by-Step Implementation

### 4.1 Data Pipeline

**Step 1: Download and preprocess datasets.**
- Natural Questions: use the simplified open-domain version (NQ-Open). ~3,610 question-answer pairs. Source: Google's official release or HuggingFace (`natural_questions`).
- TriviaQA: use the unfiltered validation set. ~3,842 question-answer pairs. Source: HuggingFace (`trivia_qa`).
- Split NQ into: Calibration (70%), Validation (15%), Held-in Test (15%). TriviaQA is entirely held-out test (the shifted domain).

**Step 2: Generate LLM responses.**
- For each question in NQ and TriviaQA, generate a greedy answer (with per-token log-probs for fM1) plus K=5 sampled responses from LLaMA-3.1-8B-Instruct using temperature=0.7 and chat template.
- Store: question, greedy answer, mean log-prob, per-token log-probs, 5 sampled answers.
- This is the most compute-intensive step (~1-2 hours on A6000). Pre-generate and cache everything with incremental saves every 50 questions.

**Step 3: Compute confidence scores.**
- Self-consistency score (fM2): fraction of the K=5 sampled pairs that bidirectionally entail (measured by pairwise NLI using DeBERTa-v2-xxlarge-mnli, label order: {0:CONTRADICTION, 1:NEUTRAL, 2:ENTAILMENT}).
- Token log-probability score (fM1): average log-probability of greedy answer tokens.
- Both scores are computed once and cached.

**Step 4: Compute entailment labels.**
- For each (question, greedy_answer, reference_answer) triple, use DeBERTa-v2-xxlarge-mnli to determine if the greedy answer entails the reference (unidirectional argmax).
- Label: 1 if argmax == ENTAILMENT, 0 otherwise.
- Also store continuous P(entailment) for conformal thresholding.

### 4.2 Method 1 — Vanilla SGen-Semi (Baseline) — **IMPLEMENTED** in `sgen_semi.py`

Implements Algorithm 2 from Lee et al. (2024) with 100 random calibration splits:

1. Split NQ into 70% calibration / 30% in-domain test.
2. Split calibration into 75% Z_U (unlabeled) / 25% Z_E (labeled).
3. Compute conformal threshold from Z_E entailment scores: tau_CP = sorted[ceil((n+1)(1-epsilon_e)) - 1].
4. Pseudo-label Z_U: correct if entail_score >= tau_CP.
5. Grid search over 50x50 percentile grid of (tau1, tau2) pairs. For each: count selected (fM1 >= tau1 AND fM2 >= tau2), count failures, compute Clopper-Pearson upper bound with Bonferroni correction (delta_adj = (delta - delta_p) / |H|). Keep highest-efficiency pair where bound <= epsilon.
6. Evaluate on NQ-test (in-domain, should work) and full TriviaQA (shifted, should fail).

### 4.3 Method 2 — Conservative Threshold (Baseline) — **IMPLEMENTED** in `conservative.py`

Same SGen-Semi split logic with three conservative overrides, each swept over multiple values:
- **Option A (Safety Factor):** After grid search, inflate: tau1 += log(gamma), tau2 *= gamma. Sweep gamma = {1.0, 1.2, 1.5, 2.0}.
- **Option B (Reduced Epsilon):** Grid search uses eps_eff = epsilon/k. Evaluate against original epsilon. Sweep k = {1.0, 1.5, 2.0, 3.0, 4.0}.
- **Option C (Delta Budget):** Reserve frac of (delta - delta_p) for shift: delta_cp = delta - delta_p - delta_s. Sweep frac = {0.0, 0.25, 0.50, 0.75}.

Evaluate on TriviaQA. Expected: restores validity but at severe efficiency cost.

### 4.4 Method 3 — DS-SGen (Proposed)

Our novel method, combining SGen with importance reweighting:

1. **Embed all prompts.** Use sentence-transformers (`all-MiniLM-L6-v2`) to embed every question from NQ-cal and a pool of TriviaQA questions into 384-dimensional vectors. This is the dimension reduction step from DS-CP.

2. **Train domain classifier.** Fit logistic regression (or XGBoost) on embeddings, where label=0 for NQ and label=1 for TriviaQA. Use the validation set to tune regularization.

3. **Compute importance weights.** For each calibration sample x_i from NQ, compute: ŵ(x_i) = p̂(x_i) / (1 − p̂(x_i)), where p̂ is the classifier's predicted probability of being TriviaQA. Clip weights at the 95th percentile. Normalize so weights sum to n (the calibration set size).

4. **Weighted conformal pseudo-labeling.** Replace the uniform empirical distribution in conformal prediction with the weighted distribution. The quantile computation uses weights: q̂_α = inf{q : Σ_i ŵ_i · 1{s_i ≤ q} / Σ_i ŵ_i ≥ 1−α}. This produces pseudo-labels calibrated to the target domain.

5. **Weighted binomial bounds.** Replace the standard binomial tail bound with a weighted version. The effective sample size is n_eff = (Σ ŵ_i)² / (Σ ŵ_i²). Use n_eff in place of n in the binomial CDF inversion.

6. **Domain-aware selection.** Add a third confidence signal: d(x) = cosine_similarity(embedding(x), centroid(NQ-cal embeddings)). Lower similarity → less trustworthy → more likely to abstain. Combine with self-consistency and log-probability scores via a simple linear combination, with weights tuned on the validation set.

7. **Threshold optimization.** Find τ* that maximizes efficiency subject to the weighted PAC constraint.

### 4.5 Evaluation Metrics

| Metric | What It Measures | How We Compute It |
|---|---|---|
| **Empirical FDR-E** | Fraction of answered questions that are wrong (using entailment) | Count wrong answers / count answered |
| **PAC Validity** | Does the guarantee hold? Run 100 random splits, check if FDR-E ≤ ε in ≥ (1−δ)% of splits | Binary: pass/fail across splits |
| **Selection Efficiency** | Fraction of questions the model answers (doesn't say IDK) | Count answered / count total |
| **Coverage Gap** | Difference between target coverage and actual coverage | |target − actual| across splits |
| **Effective Sample Size** | How much of the calibration data is "useful" after reweighting | n_eff = (Σw)² / (Σw²) |

### 4.6 Key Experiments

**Experiment 1: The Motivating Failure (Why this matters).**
Run vanilla SGen calibrated on NQ. Evaluate on NQ-test (in-domain) and TriviaQA (shifted). Show that the PAC guarantee holds on NQ but fails on TriviaQA. This is a histogram plot: distribution of empirical FDR-E across 100 random splits, with the ε threshold marked. On NQ, the histogram should be mostly below ε. On TriviaQA, it should spill over.

**Experiment 2: The Fix Works (DS-SGen restores validity).**
Run DS-SGen calibrated on NQ, tested on TriviaQA. Show the PAC guarantee is restored — the histogram shifts back below ε. Compare with the conservative threshold baseline to show DS-SGen achieves this without sacrificing as much efficiency.

**Experiment 3: Efficiency Comparison (Bar chart).**
For all three methods at the same ε and δ, compare selection efficiency on TriviaQA. Expected result: Vanilla SGen has high efficiency but invalid guarantee. Conservative threshold has valid guarantee but low efficiency. DS-SGen has valid guarantee AND high efficiency.

**Experiment 4: Weight Analysis (Understanding the mechanism).**
Visualize the importance weights: which NQ questions are weighted highest (most "similar" to TriviaQA)? What does the effective sample size look like? t-SNE plot of NQ and TriviaQA embeddings, colored by weight magnitude. This provides interpretability.

**Experiment 5: Sensitivity Analysis (Robustness).**
Vary: (a) weight clipping percentile (90th, 95th, 99th, no clipping), (b) embedding model (MiniLM vs. BGE), (c) calibration set size (25%, 50%, 75%, 100% of NQ-cal). Report FDR-E validity rate and efficiency for each. This shows the method is not fragile.

---

## 5. Work Division — Who Does What

### Member Responsibilities

| Task | Owner | Support | Deadline |
|---|---|---|---|
| **Data pipeline** (download NQ, TriviaQA, preprocess, splits) | **Anshul** | Justin | March 17 |
| **LLM inference** (generate responses, cache log-probs) | **Anshul** | Justin | March 20 |
| **Entailment scoring** (DeBERTa setup, compute labels) | **Anshul** | Justin | March 22 |
| **Method 1: Vanilla SGen** (implement from paper) | **Anshul** | Justin | March 28 |
| **Method 2: Conservative Threshold** (implement) | **Anshul** | Justin | March 30 |
| **Method 3: DS-SGen** (embeddings, classifier, weights, weighted bounds) | **Anshul** | Justin | April 5 |
| **Experiments 1–3** (main results, plots) | **Justin** | Anshul | April 8 |
| **Check-in writeup** (4 pages) | **Anshul** | Justin | April 8 |
| **Experiments 4–5** (analysis, sensitivity) | **Justin** | Anshul | April 14 |
| **Final report writing** (8 pages) | **Anshul** | Justin | April 20 |
| **Video showcase** (4 min) | **Justin** | Anshul | April 20 |
| **Code cleanup and documentation** | **Both** | — | April 22 |

### Parallel Workstreams

The project has three natural parallelizable tracks:

- **Track A (Data + Infrastructure):** Anshul leads. Download datasets, set up HuggingFace pipelines, generate LLM responses, compute entailment labels. This is the foundation everything else depends on — must be done first.
- **Track B (Methods Implementation):** Anshul leads. Implement all three methods in a modular Python codebase with shared utilities.
- **Track C (Evaluation + Visualization):** Justin leads once methods are ready. Run all experiments, generate plots, perform sensitivity analysis.

---

## 6. Timeline with Milestones

### Phase 1: Foundation (March 3–13) — PROPOSAL

- [x] Finalize 3 methods and scope
- [x] Identify 6 papers for literature review (SGen, Weighted CP, DS-CP, Conformal Factuality, Enhanced CP, Subpop CP)
- [x] Write 1–2 page proposal in LaTeX (arXiv template)
- [x] Submit proposal on Gradescope by March 13

### Phase 2: Infrastructure (March 14–22) — DATA + BASELINES

- [x] Download and preprocess NQ and TriviaQA
- [x] Set up LLaMA-3.1-8B-Instruct inference pipeline (HuggingFace Transformers, chat template)
- [ ] Generate and cache all LLM responses (K=5 sampled + greedy with log-probs) — **IN PROGRESS: NQ 2050/3610 done (job 6943094 running)**
- [x] Set up DeBERTa-v2-xxlarge-mnli entailment pipeline
- [ ] Compute all entailment labels and confidence scores — **BLOCKED: waiting on generation**
- [x] Mentor Meeting #1 (by March 20) — present proposal, get approval
- [x] Begin implementing Method 1 (Vanilla SGen)

### Phase 3: Core Implementation (March 23–April 5) — ALL METHODS

- [x] Complete Method 1 (Vanilla SGen-Semi) — code done, awaiting cached data to produce results
- [x] Complete Method 2 (Conservative Threshold) — 3 options implemented (safety factor, reduced epsilon, delta budget)
- [ ] Implement embedding pipeline (sentence-transformers) — Method 3
- [ ] Implement domain classifier + importance weight estimation — Method 3
- [ ] Implement weighted conformal prediction — Method 3
- [ ] Complete Method 3 (DS-SGen) — **NOT STARTED**
- [ ] Run preliminary experiments to verify DS-SGen works

### Phase 4: Check-in (April 6–8) — CHECKPOINT

- [ ] Run Experiments 1–3 (even if preliminary) — **BLOCKED: baseline still generating**
- [ ] Create skeleton plots/tables for remaining experiments
- [ ] Write 4-page check-in report
- [ ] Include: Problem/Dataset, 2+ reviewed papers, Methods description, preliminary results, current progress checklist, GitHub link
- [ ] Submit check-in on Gradescope by April 8
- [ ] Mentor Meeting #2 (by April 8)

### Phase 5: Analysis and Polish (April 9–18) — FULL RESULTS

- [ ] Run all experiments with final hyperparameters
- [ ] Run Experiments 4–5 (weight analysis, sensitivity)
- [ ] Generate all publication-quality figures
- [ ] Compute confidence intervals (bootstrap over 100 splits)

### Phase 6: Deliverables (April 19–22) — SUBMIT EVERYTHING

- [ ] Write 8-page final report
- [ ] Record 4-minute video showcase
- [ ] Clean, document, and submit all code
- [ ] Write individual contribution statements
- [ ] Submit video by April 20
- [ ] Submit report + code by April 22

---

## 7. Repository Structure

```
ds-gen-10701/
├── README.md                       # Project overview, setup, hyperparameters, runtime estimates
├── environment.yml                 # Conda env export (Python 3.10, PyTorch 2.6, transformers 5.5)
├── configs/
│   └── default.yaml                # All hyperparameters and paths (single source of truth)
├── ds_sgen/
│   ├── __init__.py
│   ├── utils.py                    # Config loading, seed, atomic JSON caching
│   ├── data_loading.py             # NQ-Open + TriviaQA: download, normalize, cache
│   ├── generate_responses.py       # LLaMA-3.1-8B-Instruct: greedy (fM1) + K=5 sampled
│   ├── entailment_scoring.py       # DeBERTa-v2-xxlarge-mnli: correctness + self-consistency (fM2)
│   ├── sgen_semi.py                # Method 1: SGen-Semi baseline (conformal + PAC-FDR)
│   └── conservative.py             # Method 2: Conservative Threshold (3 options)
├── run_baseline.py                 # Method 1 orchestrator (--stage data|generate|entailment|sgen|all)
├── run_conservative.py             # Method 2 orchestrator (loads cached Stages 1-3)
├── papers/                         # Detailed analysis notes for all 6 papers + project plans
├── scripts/
│   ├── check_gpu.sh                # SLURM GPU sanity check
│   ├── run_gpu.sh                  # SLURM baseline pipeline (A6000, 48GB mem)
│   └── run_conservative.sh         # SLURM Method 2 sweep
├── logs/                           # SLURM .out/.err files (gitignored)
├── cache/ -> /data/.../cache       # Symlink: cached JSON for each pipeline stage
├── results/ -> /data/.../results   # Symlink: final experiment outputs
├── report/                         # (to be created) LaTeX report
└── figures/                        # (to be created) Generated plots
```

---

## 8. Risk Mitigation for the PoC

| Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|
| LLaMA inference too slow | Medium | High | Pre-generate on A6000 cluster; cache with incremental saves every 50 questions; SLURM preempt + requeue | **Active** — generation ~1.5 hrs running, ~57% NQ done |
| SGen is hard to reimplement | Medium | High | Start early; the paper has clear algorithms; use their released code as reference | **Resolved** — Methods 1+2 fully implemented |
| Importance weights are all ~1 (shift too mild) | Low | Medium | NQ→TriviaQA has meaningful distributional difference; verify with t-SNE early | Pending (Method 3) |
| Weighted bounds are vacuous | Low | Medium | Weight clipping prevents this; check effective sample size early | Pending (Method 3) |
| Compute budget exceeded | Medium | Medium | 8B model fits in 16GB VRAM; cache everything; SLURM preempt partition | **Managed** — using preempt queue |
| Entailment model fails on TriviaQA | Low | Medium | DeBERTa-v2-xxlarge-mnli is domain-general; test early | Pending |
| SLURM preemption loses work | Medium | Medium | Atomic JSON writes via tempfile; incremental caching; --requeue flag | **Mitigated** — implemented in utils.py |
| Transformers API changes | Low | Medium | Pin versions; handle BatchEncoding returns from apply_chat_template | **Hit and fixed** — first run crashed on apply_chat_template return type |

---

## 9. What Makes This a 100/100 Project

**Completeness (20/20):** Every required section is present. We have a clear problem statement, well-sourced dataset, 6-paper synthesis literature review, 3 fully described methods, 5 experiments with labeled figures, discussion/analysis, and code availability.

**Literature Review (10/10):** We do not just list papers. We tell a story — conformal prediction was born assuming exchangeability (Vovk), the theory was extended to covariate shift (Tibshirani), it was made practical for LLMs (SGen, Conformal Factuality), domain shift was addressed for multiple-choice only (DS-CP), and enhanced methods partially handle shift (Cherian). Our DS-SGen is the natural culmination. Every paper connects to the next. We identify limitations in each that the next addresses.

**Technical Soundness (30/30):** Every method is described with sufficient detail to reproduce. We give the exact algorithm steps, the exact formulas for importance weights, the exact metrics. Hyperparameter tuning is done on a held-out validation split (NQ-val), never on the test set. We report confidence intervals from 100 bootstrap splits.

**Implementation Correctness (20/20):** Modular codebase with each method as a class inheriting from a common interface. Meaningful variable names (`importance_weights`, not `w`). Docstrings on every function. A `run_all.sh` script that reproduces every result.

**Clarity (10/10):** The writeup follows a logical arc: motivate the problem → show the failure → present the fix → validate it works → analyze why. Every figure is labeled and referenced in text.

**Formatting (5/5):** ArXiv template, within 8 pages, references on extra pages.

**Performance (5/5):** We expect DS-SGen to clearly outperform both baselines — restoring PAC validity (like conservative threshold) while maintaining high efficiency (like vanilla SGen). This is a "best of both worlds" result that should score full marks.

---

## 10. From PoC to Full Paper — The Summer Bridge

| PoC (10-701, April) | Full Paper Extension (Summer) |
|---|---|
| 1 domain pair (NQ → TriviaQA) | 5 datasets, 272+ domain pairs |
| 1–2 models (LLaMA-3B, Mistral-7B) | 5+ models including GPT-4o, 70B |
| Empirical demonstration of validity | Formal theorem with tight bounds |
| Importance reweighting only | + Conditional conformal comparison |
| Simple linear selection function | Neuro-selection with learned weights |
| 8-page report | Full NeurIPS paper |

The PoC answers: "Does the core idea work at all?" The full paper answers: "How well does it work, when, and why?" The PoC is the proof that the summer investment is justified.