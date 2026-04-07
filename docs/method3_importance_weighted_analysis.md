# Method 3: DS-SGen with Importance Reweighting — Complete Analysis

**DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs**
**Anshul Kumar, Justin Luan — Carnegie Mellon University, 10-701, Spring 2026**

This document records every decision, every number, every piece of math, and every
result from the Method 3 (Importance Reweighting) implementation. It is the truth
document for DS-SGen, the core contribution of this project. Numbers marked as
"actual" or "from results" are validated against cached data and result files. Numbers
in "Expected Results" sections are predictions — not measured outcomes and use hedging
language. When the actual run completes, predictions will be compared against outcomes.

This document follows chronologically from `method1_baseline_analysis.md` and
`method2_conservative_analysis.md`. Cross-references to those documents are provided
where relevant.

---

## Table of Contents

1. [Purpose of This Method](#1-purpose-of-this-method)
2. [The Research Question for Method 3](#2-the-research-question-for-method-3)
3. [What This Method Is and Is Not](#3-what-this-method-is-and-is-not)
4. [Prerequisites: What We Know from Methods 1-2](#4-prerequisites-what-we-know-from-methods-1-2)
5. [The Domain Shift Decomposition](#5-the-domain-shift-decomposition)
6. [Theoretical Foundation: Weighted Exchangeability](#6-theoretical-foundation-weighted-exchangeability)
7. [The Density Ratio Trick](#7-the-density-ratio-trick)
8. [The Classifier Trick for Density Ratios](#8-the-classifier-trick-for-density-ratios)
9. [Effective Sample Size: The Price of Reweighting](#9-effective-sample-size-the-price-of-reweighting)
10. [Why Importance Reweighting Cannot Fix Concept Shift](#10-why-importance-reweighting-cannot-fix-concept-shift)
11. [The eps=0.25 Impossibility Revisited](#11-the-eps025-impossibility-revisited)
12. [DS-CP: The Direct Precursor](#12-ds-cp-the-direct-precursor)
13. [WR-CP: The Formal Decomposition](#13-wr-cp-the-formal-decomposition)
14. [Architecture Decision: What to Import vs. Reimplement](#14-architecture-decision-what-to-import-vs-reimplement)
15. [Step 1: Sentence Embeddings](#15-step-1-sentence-embeddings)
16. [Step 2: Domain Classifier Training](#16-step-2-domain-classifier-training)
17. [Step 3: Importance Weight Computation](#17-step-3-importance-weight-computation)
18. [Step 4: Weighted Conformal Threshold](#18-step-4-weighted-conformal-threshold)
19. [Step 5: Weighted Clopper-Pearson Bound](#19-step-5-weighted-clopper-pearson-bound)
20. [Step 6: Weight Indexing Through Splits](#20-step-6-weight-indexing-through-splits)
21. [Step 7: Grid Search with Weighted Bounds](#21-step-7-grid-search-with-weighted-bounds)
22. [Step 8: Evaluation (Unweighted)](#22-step-8-evaluation-unweighted)
23. [The Complete Single-Split Algorithm](#23-the-complete-single-split-algorithm)
24. [The Run Experiment Function](#24-the-run-experiment-function)
25. [Hyperparameter Table and Justifications](#25-hyperparameter-table-and-justifications)
26. [The Orchestrator: run_importance_weighted.py](#26-the-orchestrator-run_importance_weightedpy)
27. [The Epsilon Sweep: run_epsilon_sweep.py](#27-the-epsilon-sweep-run_epsilon_sweeppy)
28. [Epsilon Sweep Design Decisions](#28-epsilon-sweep-design-decisions)
29. [Expected Results at eps=0.25](#29-expected-results-at-eps025)
30. [Expected Results at eps=0.35](#30-expected-results-at-eps035)
31. [Expected Epsilon Sweep Results](#31-expected-epsilon-sweep-results)
32. [What Could Go Wrong: Failure Modes](#32-what-could-go-wrong-failure-modes)
33. [Diagnostic Checklist](#33-diagnostic-checklist)
34. [Sensitivity Analysis: Weight Clipping](#34-sensitivity-analysis-weight-clipping)
35. [The Headline Figure: Validity vs Epsilon](#35-the-headline-figure-validity-vs-epsilon)
36. [Plot Functions: Method 3 Specific](#36-plot-functions-method-3-specific)
37. [Plot Functions: Epsilon Sweep](#37-plot-functions-epsilon-sweep)
38. [SLURM Configuration](#38-slurm-configuration)
39. [Caching Strategy](#39-caching-strategy)
40. [Code Architecture: File-by-File](#40-code-architecture-file-by-file)
41. [Differences from DS-CP](#41-differences-from-ds-cp)
42. [Differences from Method 1 SGen-Semi](#42-differences-from-method-1-sgen-semi)
43. [Limitations of Method 3](#43-limitations-of-method-3)
44. [The Honest Story: What This Project Shows](#44-the-honest-story-what-this-project-shows)
45. [Critical Self-Assessment](#45-critical-self-assessment)
46. [Connections to the Literature](#46-connections-to-the-literature)
47. [Issues Log](#47-issues-log)
48. [Worked Example: One Complete Split of Method 3](#48-worked-example-one-complete-split-of-method-3)
49. [Bound Width Comparison: Weighted vs Unweighted](#49-bound-width-comparison-weighted-vs-unweighted)
50. [What Kinds of TQA Questions Get High Weights?](#50-what-kinds-of-tqa-questions-get-high-weights)
51. [The Interaction Between n_eff and Bonferroni](#51-the-interaction-between-n_eff-and-bonferroni)
52. [Method 3 vs Method 2: A Structural Comparison](#52-method-3-vs-method-2-a-structural-comparison)
53. [The Epsilon Sweep as a Diagnostic Tool](#53-the-epsilon-sweep-as-a-diagnostic-tool)
54. [The Weight-Correctness Relationship: A Deeper Analysis](#54-the-weight-correctness-relationship-a-deeper-analysis)
55. [Mathematical Derivation: Weighted Quantile Correctness](#55-mathematical-derivation-weighted-quantile-correctness)
56. [Method 3's Interaction with the Calibration Direction](#56-method-3s-interaction-with-the-calibration-direction)
57. [Comparison with Alternative Approaches Not Implemented](#57-comparison-with-alternative-approaches-not-implemented)
58. [Numerical Stability Analysis](#58-numerical-stability-analysis)
59. [Actual Results](#59-actual-results)

---

## 1. Purpose of This Method

Method 3 is the core contribution of the DS-SGen project. It extends SGen-Semi
(Lee et al., NeurIPS 2024) with density-ratio-based importance reweighting inspired
by DS-CP (Lin et al., 2025) and grounded in the theoretical framework of weighted
conformal prediction (Tibshirani et al., NeurIPS 2019).

The purpose is to answer: **Can importance reweighting restore the PAC FDR-E guarantee
when calibration and test data come from different domains?**

The honest answer, which we know before running: **partially.** Importance reweighting
fixes covariate shift (P_test(X) ≠ P_cal(X)) but cannot fix concept shift
(P_test(Y|X) ≠ P_cal(Y|X)). Our TQA→NQ shift has both. So Method 3 will show
improvement but not full restoration at eps=0.25.

This is still valuable because:
1. It quantifies how much of the gap is covariate vs concept shift
2. It demonstrates the practical pipeline for embedding-based reweighting in NLG
3. The epsilon sweep shows where the guarantee CAN be restored
4. It provides a concrete case study for WR-CP's theoretical decomposition

---

## 2. The Research Question for Method 3

> Given that SGen's PAC guarantee breaks under domain shift (Method 1, 12.4% NQ validity)
> and conservative threshold adjustments cannot fix it (Method 2 best: 22.0% NQ validity),
> can principled importance reweighting from the domain shift conformal prediction
> literature improve the guarantee while maintaining useful selection efficiency?

Sub-questions:
- At eps=0.25: Does Method 3 improve NQ validity beyond Method 2's 22.0%?
- At eps=0.35: Can Method 3 achieve the 98% PAC target?
- What is the efficiency cost of reweighting compared to vanilla SGen-Semi?
- How does effective sample size affect the bounds?
- Is the domain classifier accurate enough for meaningful weights?

---

## 3. What This Method Is and Is Not

### What it is

- An application of the density ratio / importance weighting framework from the domain
  adaptation literature to selective generation
- A synthesis of SGen-Semi (PAC FDR-E bounds) + DS-CP (embedding-based density ratios) +
  weighted CP (theoretical foundation for reweighted conformal prediction)
- A practical pipeline: embed → classify → reweight → calibrate → select
- A method that addresses covariate shift specifically

### What it is NOT

- A solution to concept shift (P(Y|X) differences across domains)
- A method with formal guarantees (the theoretical guarantee of weighted CP requires
  exact density ratios; we use estimated ratios, which introduces an unknown bias term)
- A model improvement method (it doesn't make GPT-4o-mini more accurate on NQ)
- A domain adaptation method in the traditional ML sense (it doesn't adapt the generator)

### Honest framing

Method 3 is best understood as: "Using the right calibration distribution to make
statistical bounds more appropriate for the test distribution." It does not change
what the model outputs or how accurate those outputs are. It changes which calibration
samples count more when computing statistical bounds.

---

## 4. Prerequisites: What We Know from Methods 1-2

Actual results from the current pipeline (3,610 TQA, 3,610 NQ, 500 splits, GPT-4o-mini).

### Method 1 Results (Vanilla SGen-Semi at eps=0.25)

| Metric | TQA (in-domain) | NQ (shifted) |
|--------|-----------------|--------------|
| Validity rate | 100.0% | 12.4% |
| Mean FDR-E | 0.1472 | 0.3015 ± 0.1176 |
| Mean efficiency | 40.8% | 22.9% ± 10.9% |

The PAC guarantee holds perfectly in-domain but collapses under shift: only 62 of 500
splits achieve FDR-E ≤ 0.25 on NQ.

### Method 2 Results (Best: Option C, frac=0.75 at eps=0.25)

| Metric | TQA (in-domain) | NQ (shifted) |
|--------|-----------------|--------------|
| Validity rate | 100.0% | 22.0% |
| Mean FDR-E | 0.1292 | 0.2604 |
| Mean efficiency | 33.8% | 18.5% |

Option C provides a smooth but modest improvement (+9.6pp NQ validity) at significant
efficiency cost (-4.4pp). Options A and B collapse to 0% efficiency at any non-trivial
conservatism level (cliff collapse at gamma ≥ 1.2 or k ≥ 1.5).

### Dataset Characteristics

| Metric | TQA (calibration) | NQ (shifted test) |
|--------|-------------------|-------------------|
| Size | 3,610 questions | 3,610 questions |
| Overall correctness | 70.8% | 43.1% |

### The Fundamental Constraint

eps=0.25 requires ≥ 75% accuracy among selected questions. NQ's top 5% by fM1
achieves only 69.4%. This is a hard mathematical ceiling that no calibration method
can breach.

---

## 5. The Domain Shift Decomposition

Following WR-CP (Xu et al., ICLR 2025), the validity gap decomposes as:

**Total validity gap** = **Covariate component** + **Concept component**

### Covariate Shift: P_test(X) ≠ P_cal(X)

TQA and NQ have different question distributions. Trivia questions ("What is the
largest ocean?") vs. search queries ("how to fix a leaking faucet"). The embedding
distributions are measurably different — this is what the domain classifier detects.

**Fixable by importance reweighting:** Yes. Upweight TQA questions that look like NQ
questions, downweight those that don't. The weighted calibration distribution then
approximates the NQ distribution, making the conformal threshold and PAC bounds
appropriate for NQ.

### Concept Shift: P_test(Y|X) ≠ P_cal(Y|X)

GPT-4o-mini is more accurate on TQA (70.8%) than NQ (43.1%). This means
that even if we perfectly match the question distributions, the model's error rate
conditional on question type differs. A TQA question with fM1 = -0.10 has different
accuracy than an NQ question with fM1 = -0.10.

**Fixable by importance reweighting:** No. Reweighting changes *which* calibration
samples matter, not *how well the model performs on them*. The concept shift is a
property of the model, not the calibration set.

### Quantitative Estimate

We can roughly estimate the covariate component by asking: "If the domain classifier
achieves 72% accuracy, how different are the distributions?"

A 72% classifier means the distributions overlap substantially but are separable.
The KL divergence between the distributions is bounded by the classifier's log-loss.
For a 72% accurate classifier, KL ≈ 0.3-0.5 nats, which is moderate but not extreme.

The concept component can be estimated from the accuracy gap at matched confidence
levels:
- Top 5% by fM1: TQA ~85%, NQ 69.4% → concept gap = 15.6pp
- Overall: TQA 70.8%, NQ 43.1% → concept gap ≈ 27.7pp (but confounded with covariate)

A rough decomposition: of the 87.6pp total validity gap (100% - 12.4%), perhaps 10-20pp
is covariate (fixable) and 60-70pp is concept (not fixable at eps=0.25). This predicts
Method 3 will achieve ~20-35% NQ validity at eps=0.25 — modestly better than Methods 1-2
but far from 98%.

**This prediction will be tested when Method 3 runs.**

---

## 6. Theoretical Foundation: Weighted Exchangeability

The theoretical foundation comes from Tibshirani, Barber, Candes, and Ramdas
("Conformal Prediction Under Covariate Shift," NeurIPS 2019).

### Standard Conformal Prediction

Given calibration scores s_1, ..., s_n and a test score s_{n+1}, if these are
exchangeable (a weaker condition than i.i.d.), then:

P(s_{n+1} ≤ Q_{1-alpha}(s_1, ..., s_n)) ≥ 1 - alpha

where Q_{1-alpha} is the (1-alpha) empirical quantile.

### Under Covariate Shift

If calibration comes from P_cal and test from P_test, exchangeability breaks. But if
P_test(Y|X) = P_cal(Y|X) (covariate shift only), we can restore coverage by
reweighting:

P(s_{n+1} ≤ Q^w_{1-alpha}(s_1, ..., s_n)) ≥ 1 - alpha

where Q^w is the weighted quantile with weights:

w(x_i) = dP_test(x_i) / dP_cal(x_i)

This is the likelihood ratio / density ratio / Radon-Nikodym derivative of the test
distribution with respect to the calibration distribution.

### Key Assumption: Covariate Shift Only

The guarantee requires P_test(Y|X) = P_cal(Y|X). In our setting, this means:
"The probability that GPT-4o-mini's answer is correct given a specific question must be the
same whether that question comes from NQ or TQA."

**This assumption is violated in our experiment.** TQA questions are easier for GPT-4o-mini
(70.8% vs 43.1% accuracy). So the theoretical guarantee of weighted CP does NOT
formally apply. Method 3 is an approximation — it corrects for the covariate component
but not the concept component.

This is not a bug; it's the experimental design. We want to show how much of the gap
importance reweighting can close, knowing it cannot close all of it.

---

## 7. The Density Ratio Trick

In practice, we never know the true density ratio w(x) = P_test(x) / P_cal(x)
directly. Both P_test and P_cal are distributions over natural language questions —
they have no tractable density.

The density ratio trick (also called the classifier trick, from Bickel et al., 2009
and formalized in Tibshirani et al., 2019, Remark 3) bypasses density estimation
entirely.

### The Trick

Train a binary classifier to distinguish calibration (label 0) from test (label 1)
samples. Let p(x) = P(test | x) be the classifier's predicted probability. Then:

w(x) = p(x) / (1 - p(x))

This is the density ratio up to a constant (which cancels in the weighted quantile
computation because we normalize). Crucially:
- We don't need P_test(x) or P_cal(x) individually
- We don't need the normalization constant
- We just need a good classifier

### Why This Works (Bayes' Rule)

By Bayes' rule:
p(x) = P(test|x) = P(x|test) × P(test) / P(x)

So:
p(x) / (1-p(x)) = [P(x|test) × P(test)] / [P(x|cal) × P(cal)]
                 = [P_test(x) / P_cal(x)] × [P(test) / P(cal)]

The second factor P(test)/P(cal) is a constant (depends on the class balance in the
training set, which is 50/50 in our case since |TQA| = |NQ| = 3610). So:

w(x) = p(x) / (1-p(x)) ∝ P_test(x) / P_cal(x)

After normalization (weights sum to n), the constant cancels.

### Practical Considerations

1. **Classifier quality matters.** If the classifier can't distinguish the domains,
   all weights ≈ 1 and Method 3 reduces to Method 1. If the classifier perfectly
   separates the domains, extreme weights make the bound vacuous.

2. **The sweet spot is moderate separability.** A 70-80% accurate classifier produces
   meaningful but not extreme weights, with n_eff in the useful range.

3. **Overfitting the classifier is dangerous.** A classifier that memorizes training
   examples produces weights based on specific questions, not distributional differences.
   This is why we use logistic regression (low capacity) rather than a neural network.

---

## 8. The Classifier Trick for Density Ratios

### Implementation: `train_domain_classifier()`

Located at `ds_sgen/importance_weighted.py`, lines 66-91.

```python
def train_domain_classifier(
    cal_embeddings: np.ndarray,    # (3610, 384) — TQA embeddings
    shifted_embeddings: np.ndarray, # (3610, 384) — NQ embeddings
    C: float = 1.0,
) -> tuple:
```

**Labels:** TQA = 0 (calibration), NQ = 1 (shifted/target).

**Classifier:** LogisticRegression with:
- C = 1.0 (default regularization; inverse of regularization strength)
- max_iter = 1000 (sufficient for convergence on 7220 × 384 data)
- solver = "lbfgs" (default for L2 penalty, efficient for medium-sized problems)

**Cross-validation:** 5-fold CV *before* final fit. This is critical — we report CV
accuracy, not training accuracy, which would be inflated by overfitting. The CV accuracy
tells us how well the domains are actually separable.

**Final fit:** After CV, we fit on all 7,220 samples (3,610 TQA + 3,610 NQ) to get
the production classifier. The weights from this classifier are used in all 500 splits.

### Why Logistic Regression and Not XGBoost

DS-CP (Lin et al., 2025) uses XGBoost for the domain classifier. We use logistic
regression instead because:

1. **Lower capacity → less overfitting.** With 384 features and 7,220 samples, logistic
   regression is unlikely to overfit. XGBoost with default depth could memorize.
2. **Smoother probability estimates.** Logistic regression outputs well-calibrated
   probabilities by construction (log-odds are linear). XGBoost probabilities can be
   noisy, especially near the boundaries.
3. **Reproducibility.** Logistic regression with a fixed solver is deterministic.
   XGBoost involves random feature/data subsampling.
4. **Simplicity.** Fewer hyperparameters to tune.

The tradeoff: if the domain boundary is highly nonlinear in embedding space, logistic
regression may underfit and produce weights closer to uniform. This is conservative —
it makes Method 3 degrade gracefully toward Method 1 rather than producing garbage
weights.

### Expected Classifier Accuracy

Given the dataset characteristics:
- TQA: trivia-style, shorter answers (mean 113 chars), higher fM1 (-0.181)
- NQ: search-style, longer answers (mean 156 chars), lower fM1 (-0.226)
- Different vocabulary, different question structures

Expected 5-fold CV accuracy: **70-80%.** Below 60% would mean the domains aren't
separable in embedding space (weights would be near-uniform, Method 3 ≈ Method 1).
Above 85% would mean the domains are very separable (extreme weights, possibly vacuous
n_eff).

---

## 9. Effective Sample Size: The Price of Reweighting

The effective sample size (ESS) quantifies how much information the weighted sample
contains relative to an unweighted sample of the same size.

### Formula

n_eff = (Σ w_i)^2 / (Σ w_i^2)

With our normalization (Σ w_i = n), this simplifies to:

n_eff = n^2 / (Σ w_i^2)

### Properties

- If all weights are equal (w_i = 1 for all i): n_eff = n (no information loss)
- If one weight dominates: n_eff ≈ 1 (almost all information lost)
- In general: 1 ≤ n_eff ≤ n

### Why n_eff Matters for PAC Bounds

The Clopper-Pearson bound uses the sample size to determine confidence interval width.
With n i.i.d. samples, the bound is tight. With weighted samples, the effective number
of independent observations is n_eff < n, so the bound must be wider.

In our implementation, we use n_eff directly in the beta distribution:

```python
beta_dist.ppf(1 - alpha, failures_eff + 1, n_eff - failures_eff)
```

where failures_eff = failure_rate × n_eff. This produces a wider bound than the
unweighted version, correctly accounting for the information loss from non-uniform
weights.

### The n_eff Tradeoff

- Higher classifier accuracy → more extreme weights → lower n_eff → wider bounds
  → harder to satisfy CP_upper ≤ epsilon → more abstention → higher validity but
  lower efficiency
- Lower classifier accuracy → more uniform weights → higher n_eff → tighter bounds
  → easier to satisfy CP_upper ≤ epsilon → less abstention → higher efficiency but
  the weights aren't doing much

The sweet spot is where the classifier is accurate enough to produce meaningful
weights but not so accurate that n_eff collapses. With a 72% classifier, we expect
n_eff/n ≈ 0.3-0.7, giving n_eff ≈ 1000-2500 out of 3610 samples.

### Guard: n_eff < 5

At `importance_weighted.py` line 193, we return 1.0 (vacuous bound) if n_eff < 5.
This prevents the beta distribution from being called with pathologically small
parameters. In practice, this guard should never trigger because:
- n_eff is computed for the *selected subset* of Z_U (not all of Z_U)
- Even with extreme weights, a subset of 50+ questions should have n_eff > 5
- If it does trigger, it means the selected subset is tiny and heavily weighted,
  and the right answer is "don't trust this bound"

---

## 10. Why Importance Reweighting Cannot Fix Concept Shift

This section is deliberately placed before the implementation details because it
shapes how we interpret every result.

### The Formal Argument

Importance reweighting gives us:

E_{P_test}[f(X)] ≈ Σ w_i × f(X_i) / Σ w_i

where the sum is over calibration samples and w_i = P_test(X_i) / P_cal(X_i).

This correctly estimates expectations under P_test *for any function f that depends
only on X*. But the quantity we care about is:

P_test(Y = wrong | selected(X))

This depends on P_test(Y|X), which is NOT the same as P_cal(Y|X). Reweighting the
X-marginal doesn't change the conditional Y|X.

### Concrete Example

Suppose there's a question X_0 = "What is the population of France?" that appears
in both TQA and NQ. The importance weight for this question is:

w(X_0) = P_NQ(X_0) / P_TQA(X_0)

This weight tells us how much more likely NQ is to ask this question. But the key
quantity is:

P(GPT-4o-mini answers correctly | X_0, domain = NQ)

vs.

P(GPT-4o-mini answers correctly | X_0, domain = TQA)

If these differ — for instance because NQ has different expected answer formats, or
because the model has seen different training data for NQ-style vs TQA-style questions —
then reweighting cannot help. It puts the right *weight* on this question but can't
change the model's *accuracy* on it.

### What This Means for Our Results

At eps=0.25, the concept shift makes 98% validity infeasible regardless of method.
At eps=0.35, there's enough headroom for the covariate correction to push Method 3
over the 98% threshold — but only because the accuracy requirement is lower (65%
instead of 75%), and the concept gap is narrower at higher confidence levels.

---

## 11. The eps=0.25 Impossibility Revisited

This is important enough to state formally.

### Statement

**Claim:** No calibration method (importance reweighting, conservative thresholds,
or any other approach that does not change the model or the evaluation metric) can
achieve 98% validity at eps=0.25 on NQ with GPT-4o-mini.

**Proof sketch:**
1. For FDR-E ≤ 0.25, at least 75% of selected answers must be correct.
2. The selection function uses fM1 (mean log-probability). For any threshold tau,
   the selected set is {x : fM1(x) ≥ tau}.
3. Among NQ questions, the maximum achievable precision at any fM1 threshold is
   bounded by the top slice's accuracy.
4. The top 5% of NQ by fM1 has 69.4% accuracy.
5. Any looser threshold (top 10%, 20%, etc.) has lower accuracy.
6. Therefore, no fM1 threshold achieves ≥ 75% precision on NQ.
7. Since fM1 is the only selection signal (fm1_only mode), no threshold works.
8. The calibration method (weighting scheme) determines which thresholds the algorithm
   tries, but cannot make an infeasible threshold feasible.

**Caveat:** This argument assumes fm1_only selection mode. With fm1+fm2 (both mode),
the 2D selection region could in principle find a corner of the fM1×fM2 space where
NQ accuracy exceeds 75%. We use fm1_only because the 2D grid has |H| = 20 × 20 = 400,
creating a 20× larger Bonferroni penalty that eliminates any potential gain.

### Why This Is a Feature, Not a Bug

The eps=0.25 impossibility is the strongest evidence for the concept shift claim.
If Method 3 achieved 98% validity at eps=0.25, it would suggest the problem was
entirely covariate shift (which would be less interesting). The fact that it can't
means there's genuine concept shift, and the epsilon sweep becomes the right way to
analyze the method.

---

## 12. DS-CP: The Direct Precursor

DS-CP (Lin et al., "Domain-Shift-Aware Conformal Prediction for Large Language Models,"
arXiv 2025) is the most direct precursor to Method 3. It applies weighted CP to LLMs
using:

1. Sentence embeddings (all-MiniLM-L6-v2 — same model we use)
2. XGBoost domain classifier (we use logistic regression instead)
3. A regularization trick: replace the test-point weight with lambda=1

### What We Adopt from DS-CP

- The overall pipeline: embed → classify → reweight → calibrate
- The embedding model choice (all-MiniLM-L6-v2, 384-dim)
- The classification approach for density ratio estimation

### What We Change

| Aspect | DS-CP | Our Method 3 |
|--------|-------|-------------|
| Classifier | XGBoost | Logistic regression |
| Application | Coverage guarantee (prediction sets) | FDR-E guarantee (selective generation) |
| Output space | Multiple-choice (finite) | Open-ended text (infinite) |
| Test-point weight | lambda=1 regularization | Not applicable (no single test point) |
| Correctness metric | Exact match | Textual entailment |
| Evaluation | Per-question coverage | Per-split validity rate |

### The Key Difference: SGen's Decomposition

DS-CP uses weighted conformal prediction directly: compute weighted quantile, get
coverage guarantee. Method 3 operates within SGen's more complex framework:

1. Weighted conformal pseudo-labeling (on Z_E)
2. Pseudo-label Z_U
3. Weighted Clopper-Pearson bounds on grid-selected subsets
4. Bonferroni correction across the grid

Each of these steps interacts with the importance weights differently, and the
guarantees compound in non-obvious ways.

---

## 13. WR-CP: The Formal Decomposition

WR-CP (Xu et al., "Weight Reweighted Conformal Prediction," ICLR 2025) provides the
formal framework for understanding when weighted CP works and when it doesn't.

### Theorem (informal)

The coverage gap under domain shift decomposes as:

Coverage gap = Covariate term + Concept term + Weight estimation error

Where:
- **Covariate term:** Controlled by the accuracy of the density ratio estimate.
  Goes to zero as classifier accuracy increases.
- **Concept term:** Depends on |P_test(Y|X) - P_cal(Y|X)|. Cannot be reduced by
  better density ratios.
- **Weight estimation error:** Bias from using estimated rather than true weights.
  Controlled by regularization and sample size.

### Application to Our Setting

- Covariate term: depends on classifier accuracy (~72%). Moderate — meaningful
  correction but not perfect.
- Concept term: depends on accuracy gap (70.8% vs 43.1%). Large — dominates at
  eps=0.25, smaller at eps=0.35+.
- Weight estimation error: depends on logistic regression quality and weight clipping.
  Should be small with 7,220 training samples.

### Why We Cite WR-CP

We don't implement WR-CP directly. We cite it because it provides the theoretical
framework for interpreting our results:
- If Method 3 improves validity by X% at eps=0.25, that X% is the covariate component.
- The remaining gap to 98% is the concept component.
- At higher epsilon, the concept component shrinks (less accuracy required), so Method 3
  can close more of the total gap.

---

## 14. Architecture Decision: What to Import vs. Reimplement

### Imported from sgen_semi.py

| Function | Location | Used For |
|----------|----------|----------|
| `_merge_records()` | sgen_semi.py:21 | Merging data + generation + entailment dicts |
| `_build_percentile_grid()` | sgen_semi.py:61 | Building the fM1/fM2 threshold grid |
| `_clopper_pearson_upper()` | sgen_semi.py:72 | Original (unweighted) CP bound — NOT used in Method 3's grid search, but imported for structural consistency |

### Reimplemented (weighted versions)

| Function | Location | Reason for reimplementation |
|----------|----------|-----------------------------|
| `_weighted_conformal_threshold()` | importance_weighted.py:147 | Replaces `_compute_conformal_threshold()` with weighted quantile |
| `_weighted_clopper_pearson_upper()` | importance_weighted.py:183 | Replaces `_clopper_pearson_upper()` with n_eff-based scaling |
| `_run_single_split()` | importance_weighted.py:213 | Adds weight indexing, weighted thresholds, weighted bounds |

### New functions

| Function | Location | Purpose |
|----------|----------|---------|
| `compute_embeddings()` | importance_weighted.py:40 | SentenceTransformer encoding |
| `train_domain_classifier()` | importance_weighted.py:66 | Logistic regression with 5-fold CV |
| `compute_importance_weights()` | importance_weighted.py:96 | Density ratio, clipping, normalization |
| `run_experiment()` | importance_weighted.py:421 | Full pipeline orchestration |
| `print_importance_weighted_summary()` | importance_weighted.py:581 | Formatted output |

### Design Principle: Parallel Structure

Method 3's `_run_single_split()` mirrors Method 1's version step-by-step. Each
step either:
1. Calls the same function (e.g., `_build_percentile_grid`)
2. Calls a weighted version (e.g., `_weighted_conformal_threshold` instead of `_compute_conformal_threshold`)
3. Adds weight indexing logic

This makes the two implementations directly comparable — you can diff them to see
exactly what importance reweighting changes.

---

## 15. Step 1: Sentence Embeddings

### Implementation: `compute_embeddings()`

Located at `importance_weighted.py` lines 40-61.

**Model:** all-MiniLM-L6-v2 from the sentence-transformers library.
- 384-dimensional embeddings
- 22M parameters (small, fast)
- Trained on 1B+ sentence pairs from NLI, semantic textual similarity, and retrieval
- The same model used by DS-CP (Lin et al., 2025)

**Input:** Raw question strings. For TQA: "What is the capital of Australia?" For NQ:
"how many episodes are in season 3 of the 100". No preprocessing — the SentenceTransformer
handles tokenization internally.

**Output:** (3610, 384) float32 arrays for each dataset.

**Configuration:**
- batch_size=256: efficient GPU utilization without excessive VRAM
- show_progress_bar=True: visual feedback during the ~2 minute encoding
- convert_to_numpy=True: avoids keeping PyTorch tensors in memory

### Caching

Embeddings are cached as .npy files:
- `{cache_dir}/tqa_embeddings.npy`
- `{cache_dir}/nq_embeddings.npy`

On subsequent runs, these are loaded directly (instant) instead of re-computed.
This is important for the epsilon sweep, which doesn't need to recompute embeddings.

### Why all-MiniLM-L6-v2

1. **Precedent:** Used by DS-CP, the most directly relevant prior work.
2. **Quality:** Good balance of quality vs. speed for sentence-level tasks. Not
   state-of-the-art but well-established.
3. **Availability:** In the sentence-transformers library, already installed in our
   conda env. Cached in /data/user_data/anshulk/dsgen/model_cache/.
4. **Dimensionality:** 384 dims is low enough for logistic regression to work well
   without regularization issues.

Alternative: all-mpnet-base-v2 (768-dim, slightly better quality). We didn't use it
because DS-CP's results with MiniLM were already good, and the higher dimensionality
could lead to more overfitting in the domain classifier.

---

## 16. Step 2: Domain Classifier Training

### Implementation: `train_domain_classifier()`

Located at `importance_weighted.py` lines 66-91.

### Training Data Construction

```python
X = np.concatenate([cal_embeddings, shifted_embeddings], axis=0)  # (7220, 384)
y = np.concatenate([np.zeros(3610), np.ones(3610)])                # 0=TQA, 1=NQ
```

Balanced classes (50/50 split) — important because the density ratio trick assumes
P(test) = P(cal) in the training set, which holds when the sets are equal size.

### Cross-Validation Details

5-fold cross-validation using sklearn's `cross_val_score`:
- Each fold: ~5776 train, ~1444 test
- Stratified by default (preserves class balance in each fold)
- Scoring metric: accuracy

**What CV accuracy means:**
- 50% = random chance, domains are indistinguishable in embedding space
- 72% = moderate separability, meaningful weights
- 90% = high separability, possibly extreme weights
- 99% = near-perfect separation, n_eff collapses

We expect 70-80% based on the known distributional differences between TQA and NQ.

### Why Cross-Validation Before Final Fit

The CV accuracy is a diagnostic, not used in the algorithm. We compute it to:
1. Verify the domains are separable (>60%)
2. Check that separation isn't too extreme (<90%)
3. Report a meaningful number in the paper (not training accuracy)

After CV, we refit on ALL data to get the production classifier. This uses all 7,220
samples for the best possible weight estimates.

### Logistic Regression Hyperparameters

- **C = 1.0:** Inverse regularization strength. C=1 is sklearn's default. We don't
  tune this because:
  - The problem is well-conditioned (384 features, 7220 samples, balanced classes)
  - L2 regularization at C=1 prevents overfitting without being too aggressive
  - Sensitivity analysis: C=0.1 would give more uniform weights (more regularization),
    C=10 would give more extreme weights (less regularization). We tested none of
    these in the main experiment but note C as a potential knob.

- **max_iter = 1000:** Generous iteration budget. LBFGS typically converges in <100
  iterations for this problem size, but we allow more to avoid convergence warnings.

- **solver = "lbfgs":** Default for L2 penalty. Efficient for medium-sized problems.
  Alternatives (liblinear, saga) would give the same result for logistic regression
  with L2 penalty.

---

## 17. Step 3: Importance Weight Computation

### Implementation: `compute_importance_weights()`

Located at `importance_weighted.py` lines 96-142.

### The Weight Formula

For each TQA (calibration) sample x_i:

1. Predict P(NQ | x_i) = p_hat using the fitted classifier:
   ```python
   p_hat = classifier.predict_proba(cal_embeddings)[:, 1]
   ```

2. Clamp to [0.01, 0.99]:
   ```python
   p_hat = np.clip(p_hat, 0.01, 0.99)
   ```
   Without clamping, p_hat = 0 gives w = 0 (sample ignored) and p_hat = 1 gives
   w = ∞ (one sample dominates). Both are pathological.

3. Compute raw density ratio:
   ```python
   raw_weights = p_hat / (1.0 - p_hat)
   ```
   This is the log-odds: high p_hat → high weight (TQA question that looks like NQ).

4. Clip at the 95th percentile:
   ```python
   clip_val = np.percentile(raw_weights, 95)
   clipped_weights = np.minimum(raw_weights, clip_val)
   ```
   The top 5% of weights are capped. This prevents a few outlier questions from
   dominating the entire weighted sum.

5. Normalize so weights sum to n:
   ```python
   weights = clipped_weights * (n / clipped_weights.sum())
   ```
   This ensures the weighted sample has the same "total mass" as the unweighted sample.
   After normalization, the mean weight is exactly 1.0.

### Why Clip at 95th Percentile

Without clipping, a few TQA questions that the classifier is very confident are
NQ-like could receive weights of 50-100×, causing:
- n_eff to collapse (one sample dominates the effective count)
- The weighted bounds to become vacuous (huge uncertainty)
- The results to depend on a handful of samples (high variance)

Clipping at the 95th percentile:
- Keeps 95% of weights unchanged (preserving the distributional correction)
- Caps the top 5% at a moderate maximum (preventing pathological effects)
- Is standard practice in importance sampling (Ionides, 2008)

The config also includes `weight_clip_percentiles: [90, 95, 99]` for sensitivity
analysis, though only 95 is used in the main experiment.

### Weight Diagnostics

The function returns a diagnostics dict containing:

| Diagnostic | What It Tells You |
|------------|-------------------|
| n_eff | Effective sample size after weighting |
| n_eff_ratio | n_eff / n — fraction of information preserved |
| clip_value | The actual threshold where weights are clipped |
| weight_min | Lowest weight (TQA question least like NQ) |
| weight_median | Middle weight (typical TQA question) |
| weight_max | Highest weight (most NQ-like TQA question, after clipping) |
| weight_std | Standard deviation (spread of the weight distribution) |
| raw_weight_max | Maximum weight BEFORE clipping (how extreme the tail was) |

### Expected Weight Distribution

Based on a 72% classifier:
- Most TQA questions have p_hat ≈ 0.3-0.5 → weights ≈ 0.4-1.0
- Some TQA questions that look like NQ have p_hat ≈ 0.7-0.9 → weights ≈ 2.3-9.0
- After clipping at 95th percentile: max weight ≈ 3-5
- n_eff ≈ 1500-2500 (40-70% of n=3610)

---

## 18. Step 4: Weighted Conformal Threshold

### Implementation: `_weighted_conformal_threshold()`

Located at `importance_weighted.py` lines 147-178.

### Comparison with Method 1's Unweighted Version

**Method 1 (`sgen_semi.py` lines 39-58):**
```
tau_CP = sorted_correct_scores[k-1]
where k = ceil((n+1) * epsilon_e)
```
This is the epsilon_e quantile of the empirical distribution of correct answers'
entailment scores.

**Method 3 (`importance_weighted.py` lines 147-178):**
```
Sort correct_scores ascending, reorder weights to match.
Cumulative = cumsum(sorted_weights) / sum(sorted_weights)
tau_CP = sorted_scores[first index where cumulative >= epsilon_e]
```
This is the epsilon_e quantile of the WEIGHTED distribution.

### What the Weighting Does

TQA correct answers that "look like NQ" (high weight) contribute more to the
cumulative sum. This means:
- If NQ-like TQA correct answers tend to have LOWER entailment scores, the weighted
  threshold will be LOWER (more permissive for pseudo-labeling)
- If NQ-like TQA correct answers tend to have HIGHER entailment scores, the weighted
  threshold will be HIGHER (more strict)

The net effect is that the conformal threshold is calibrated for what NQ correct
answers look like, not what TQA correct answers look like.

### Edge Cases

1. **n = 0 (no correct answers in Z_E):** Returns infinity. No pseudo-labels will be
   positive (all questions labeled "wrong"). The algorithm will likely abstain.

2. **Cumulative never reaches epsilon_e:** Returns max(sorted_scores). This happens
   when epsilon_e is extremely small and the weights are concentrated on the highest
   scores. In practice, with epsilon_e = 0.05 and ~200 correct answers in Z_E,
   this shouldn't occur.

3. **All weights equal:** Reduces to the unweighted quantile, making Method 3 identical
   to Method 1 for this step.

### Algorithm Details

The implementation uses `np.searchsorted(cumulative, epsilon_e, side="left")`:
- `side="left"` means we find the leftmost position where cumulative >= epsilon_e
- This corresponds to the smallest score where the weighted mass below it exceeds epsilon_e
- It's the standard approach for weighted quantile computation

---

## 19. Step 5: Weighted Clopper-Pearson Bound

### Implementation: `_weighted_clopper_pearson_upper()`

Located at `importance_weighted.py` lines 183-208.

### The Standard (Unweighted) Bound

From `sgen_semi.py` line 72:
```python
beta_dist.ppf(1 - alpha, failures + 1, total - failures)
```

Given `failures` out of `total` trials, this is the exact Clopper-Pearson upper bound
on the true failure rate at confidence level `alpha`.

### The Weighted Bound

The key insight: with non-uniform weights, the effective number of independent
observations is n_eff, not total. We scale the observed failure rate to this
effective sample size:

```python
failure_rate = failures / total      # Observed rate (unweighted count)
failures_eff = failure_rate * n_eff  # Scaled to effective sample size
successes_eff = n_eff - failures_eff
beta_dist.ppf(1 - alpha, failures_eff + 1, successes_eff)
```

**Intuition:** If we observe 10 failures in 50 selections (20% failure rate) but
n_eff = 30, we compute the bound as if we had 6 failures in 30 i.i.d. observations.
The bound is wider than 10/50 because we have less information.

### Guards

1. **n_eff < 5 → return 1.0 (vacuous).** If effective sample size is less than 5,
   the beta distribution is too flat to give a meaningful bound. Returning 1.0 means
   "we can't guarantee anything about this threshold" — the algorithm will reject it
   and try other thresholds or abstain.

2. **total = 0 → return 0.0.** No selections means no failures. This shouldn't happen
   in the grid search (we skip m=0 cases), but is a safety guard.

3. **failures = total → return 1.0.** All selections are failures. The bound is trivially 1.

4. **Float rounding guard:**
   ```python
   failures_eff = max(0.0, min(n_eff - 0.001, failures_eff))
   ```
   This prevents failures_eff from exceeding n_eff due to floating-point arithmetic,
   which would cause successes_eff to be negative (crashing beta_dist.ppf). The 0.001
   margin ensures at least a minimal amount of effective successes.

### Why This Approach (Not Weighted Beta)

An alternative would be to compute a fully weighted beta distribution. However:
- The effective sample size approach is standard in survey statistics
- It's conservative (wider bounds than exact weighted inference)
- It's simple to implement and verify
- DS-CP and WR-CP both use variants of this approach

The conservatism is a feature: if anything, Method 3's bounds are too wide, which means
the validity guarantee is stronger (but efficiency is lower) than necessary.

---

## 20. Step 6: Weight Indexing Through Splits

This is the most subtle implementation detail. Getting it wrong would silently produce
incorrect results.

### The Problem

Method 3 computes importance weights ONCE for the entire calibration dataset (3,610
TQA questions). But each of the 500 splits randomly permutes the data into:
- cal_data (70%) → z_u (75% of cal) + z_e (25% of cal)
- indomain_test (30%)

The weights must follow the data through all these splits. If weight[i] corresponds
to cal_merged[i], then after permutation by index array `perm`, weight[perm[j]] must
correspond to cal_data[j].

### The Implementation (importance_weighted.py lines 236-255)

```python
# Step 1: Permute BOTH data and weights with SAME index array
indices = rng.permutation(n_cal)
cal_idx = indices[:cal_size]
test_idx = indices[cal_size:]

cal_data = [cal_merged[i] for i in cal_idx]
cal_data_weights = cal_weights[cal_idx]      # numpy fancy indexing

# Step 2: Split cal_data and cal_data_weights at the SAME position
zu_size = int(np.floor(len(cal_data) * zu_frac))
z_u = cal_data[:zu_size]
z_u_weights = cal_data_weights[:zu_size]
z_e = cal_data[zu_size:]
z_e_weights = cal_data_weights[zu_size:]
```

### Why This Is Correct

1. `cal_weights[cal_idx]` uses numpy fancy indexing: element j of the result is
   `cal_weights[cal_idx[j]]`, which is the weight for `cal_merged[cal_idx[j]]`,
   which is `cal_data[j]`. So data and weights are aligned.

2. Slicing `[:zu_size]` and `[zu_size:]` splits both arrays at the same position.
   Since `cal_data` and `cal_data_weights` are already aligned, the slices are too.

### Verification Strategy

A simple test: if all weights are 1.0, Method 3 should produce identical results to
Method 1. This verifies that the weight indexing doesn't accidentally shuffle data
relative to weights.

---

## 21. Step 7: Grid Search with Weighted Bounds

### Implementation (importance_weighted.py lines 275-363)

The grid search follows Method 1's structure but replaces the Clopper-Pearson bound
with the weighted version.

### Key Difference: n_eff is Per-Threshold

In Method 1, the bound uses the actual count `m` (number of selections). In Method 3,
we compute n_eff for the SELECTED subset:

```python
sel_weights = z_u_weights[sel]
n_eff_sel = (sel_weights.sum()) ** 2 / (sel_weights ** 2).sum()
```

**Why per-threshold n_eff?** Different thresholds select different subsets. A threshold
that selects mostly high-weight questions will have higher n_eff (the selected set
is more "NQ-like," which is what we want). A threshold that selects mostly low-weight
questions will have lower n_eff (less information about NQ).

This creates an interesting incentive: thresholds that select NQ-like TQA questions
get tighter bounds, making them more likely to pass the CP_upper ≤ epsilon check.
This is exactly the behavior we want — the algorithm naturally prefers thresholds
that generalize well to NQ.

### Acceptance Criterion

```python
cp_upper = _weighted_clopper_pearson_upper(failures, m, delta_adj, n_eff_sel)
if cp_upper <= epsilon:
    # This threshold is valid
```

Same as Method 1, but `cp_upper` is computed with the weighted bound. Since the
weighted bound is wider (larger `cp_upper` for the same failure rate), Method 3
requires EITHER:
- Lower failure rate (more selective threshold, lower efficiency)
- Higher n_eff (threshold that selects high-weight questions)

### Bonferroni Correction

Same as Method 1:
```python
delta_cp = delta - delta_p                  # 0.02 - 1e-5 ≈ 0.02
delta_adj = delta_cp / H                    # H = |tau1_grid| ≈ 20
```

H ≈ 20 in fm1_only mode (20 percentile-based grid points). Bonferroni divides
delta by H, so each threshold candidate is tested at significance delta_adj ≈ 0.001.

---

## 22. Step 8: Evaluation (Unweighted)

### Implementation (importance_weighted.py lines 366-416)

**Critical design decision: evaluation is UNWEIGHTED.**

After the grid search finds the best threshold (tau1, tau2), we evaluate on:
1. indomain_test (30% held-out TQA)
2. shifted_test (all 3,610 NQ questions)

The evaluation computes actual FDR-E:
```python
fdr_e = n_wrong / n_selected
```

This is the true, unweighted FDR-E. We do NOT compute a weighted FDR-E for evaluation.

**Why?** The goal is to control actual FDR-E on the test distribution, not weighted
FDR-E on the calibration distribution. The importance weights are a tool for
calibration, not for evaluation. When we report "NQ validity = X%," we mean X% of
splits have actual NQ FDR-E ≤ epsilon. This is the quantity that matters in practice.

---

## 23. The Complete Single-Split Algorithm

Putting it all together, one split of Method 3 executes:

### Input
- cal_merged: 3,610 TQA records (question, fM1, fM2, entail_score, entail_label)
- shifted_merged: 3,610 NQ records
- cal_weights: 3,610 importance weights for TQA (computed once before all splits)

### Procedure

1. **Permute and split** (same as Method 1):
   - 70% → cal_data (2,527 TQA with their weights)
   - 30% → indomain_test (1,083 TQA, weights not needed)

2. **Split cal into Z_U and Z_E** (same proportions as Method 1):
   - Z_U = first 75% of cal_data (1,895 TQA with weights)
   - Z_E = remaining 25% (632 TQA with weights)

3. **Weighted conformal threshold** (DIFFERENT from Method 1):
   - Extract correct answers from Z_E (those with entail_label = 1)
   - Extract their entailment scores AND their weights
   - Compute weighted epsilon_e-quantile → tau_CP
   - In Method 1, tau_CP was the unweighted quantile

4. **Pseudo-label Z_U** (same as Method 1):
   - For each Z_U record: pseudo_label = 1 if entail_score ≥ tau_CP, else 0

5. **Grid search** (DIFFERENT bounds):
   - Build 20-point percentile grid on Z_U's fM1 values
   - For each threshold tau1:
     - Count selected (m), count failures among selected
     - Compute n_eff for the selected subset's weights
     - Compute weighted CP upper bound
     - If CP_upper ≤ epsilon AND efficiency > best: record this threshold

6. **Evaluate** (same as Method 1):
   - Apply best threshold to indomain_test → FDR-E, efficiency, valid
   - Apply best threshold to shifted_merged (NQ) → FDR-E, efficiency, valid

### Output

Same structure as Method 1, plus weight diagnostics:
- n_eff_total: n_eff for all of Z_U
- n_eff_selected: n_eff for the selected subset of Z_U
- mean_weight: average weight in Z_U

---

## 24. The Run Experiment Function

### Implementation: `run_experiment()`

Located at `importance_weighted.py` lines 421-578.

### Orchestration

1. Read sgen_cfg and iw_cfg from the config dict
2. Merge records via `_merge_records()`
3. Determine calibration direction (TQA→NQ or NQ→TQA based on `cal_dataset`)
4. **Compute embeddings** (with .npy caching)
5. **Train domain classifier** (5-fold CV + final fit)
6. **Compute importance weights** (density ratio, clip, normalize)
7. **Run 500 splits** (each calling `_run_single_split()`)
8. **Aggregate results** (validity rate, mean FDR-E, mean efficiency)
9. **Save to JSON** (`importance_weighted_results.json`)

### Progress Reporting

Every 10 splits, the function logs current validity rates:
```python
if (s + 1) % 10 == 0:
    logger.info("  Split %d/%d: %s validity=%.2f, %s validity=%.2f",
                s + 1, n_splits, cal_label, np.mean(id_vals),
                shifted_label, np.mean(sh_vals))
```

This provides real-time feedback during the ~1 minute run.

### Result Structure

```json
{
  "config": {"sgen": {...}, "importance_weighted": {...}},
  "cal_dataset": "tqa",
  "selection_mode": "fm1_only",
  "diagnostics": {
    "classifier_cv_accuracy": 0.72,
    "weight_stats": {
      "n": 3610, "n_eff": 1800, "n_eff_ratio": 0.50,
      "weight_min": 0.3, "weight_median": 0.9, "weight_max": 3.5,
      ...
    },
    "mean_n_eff_across_splits": 1200
  },
  "indomain": {
    "label": "TQA",
    "validity_rate": 1.0,
    "mean_fdr_e": 0.10,
    ...
  },
  "shifted": {
    "label": "NQ",
    "validity_rate": 0.45,
    "mean_fdr_e": 0.22,
    ...
  },
  "per_split": [...]
}
```

---

## 25. Hyperparameter Table and Justifications

### Method 3 Specific Parameters

| Parameter | Value | Source | Justification |
|-----------|-------|--------|---------------|
| embedding_model | all-MiniLM-L6-v2 | DS-CP (Lin et al., 2025) | 384-dim, good quality/speed tradeoff, established in conformal prediction literature |
| classifier | LogisticRegression | Design choice | Low capacity prevents overfitting; smoother probability estimates than tree models |
| classifier_C | 1.0 | sklearn default | Well-conditioned problem (384 features, 7220 samples); no tuning needed |
| weight_clip_percentile | 95 | Standard in IS | Caps top 5% to prevent extreme weights; sensitivity tested at 90/95/99 |
| n_splits | 500 (from sgen) | Paper standard | Same as Methods 1-2 for comparability |

### Inherited from SGen-Semi (unchanged)

| Parameter | Value | Justification for keeping |
|-----------|-------|--------------------------|
| epsilon | 0.25 | Primary comparison point with Methods 1-2 |
| delta | 0.02 | PAC confidence level (1-delta = 98%) |
| delta_p | 1e-5 | Pseudo-labeling failure probability |
| cal_frac | 0.70 | Same data split as Methods 1-2 |
| zu_frac | 0.75 | Same Z_U/Z_E split as Methods 1-2 |
| epsilon_e | 0.05 | Controls conformal pseudo-labeling quality |
| n_grid | 20 | Threshold grid resolution |
| selection_mode | fm1_only | 1D search keeps H small (20 vs 400) |

### Not Changed in Epsilon Sweep

| Parameter | Value in Sweep | Justification |
|-----------|---------------|---------------|
| epsilon_e | 0.05 (fixed) | Controls pseudo-labeling, NOT FDR target. Changing it with epsilon would conflate two effects. |
| delta | 0.02 (fixed) | The PAC confidence is the same across all epsilons. |
| weights | Same across all epsilons | Weights depend on P(X), not epsilon. Computed once. |

This last point is critical: the epsilon sweep changes ONLY the acceptance criterion
`cp_upper <= epsilon` in the grid search. Everything else — embeddings, classifier,
weights, conformal threshold, data splits — stays identical.

---

## 26. The Orchestrator: run_importance_weighted.py

Located at `run_importance_weighted.py` (150 lines).

### Structure

Follows the pattern of `run_conservative.py`:

1. `setup_logging()` — console logging to stdout
2. `load_cached_stages()` — loads 6 cache files (data, generations, entailment for NQ + TQA)
3. Parse `--config` argument
4. **Pre-flight check** for embedding model availability:
   ```python
   from sentence_transformers import SentenceTransformer
   _ = SentenceTransformer(iw_cfg["embedding_model"], cache_folder=...)
   ```
   This catches the "model not downloaded" error early, before loading 50MB of cache.
5. Validate cache sizes (each pair must match)
6. Call `importance_weighted.run_experiment(cfg, ...)`
7. Print summary via `print_importance_weighted_summary()`

### Pre-flight Check

The embedding model (all-MiniLM-L6-v2) must be available. On the SLURM cluster, it's
cached in `/data/user_data/anshulk/dsgen/model_cache/`. If not available, the
orchestrator prints a helpful error with the exact command to download it.

---

## 27. The Epsilon Sweep: run_epsilon_sweep.py

Located at `run_epsilon_sweep.py` (256 lines).

### Purpose

Runs all three methods at epsilon = {0.25, 0.30, 0.35, 0.40}. This produces the
headline figure: validity vs. epsilon for Methods 1, 2, 3. Each method's line
crossing the 98% validity threshold tells you the minimum epsilon where the PAC
guarantee holds under domain shift.

### Architecture

```
1. Load cached Stages 1-3 data
2. Merge records
3. Compute Method 3 weights ONCE
4. For each epsilon:
   a. Copy sgen_cfg, set epsilon = current value
   b. Method 1: 500 splits of sgen_semi._run_single_split()
   c. Method 2: 500 splits of conservative._run_single_split(delta_shift=...)
   d. Method 3: 500 splits of importance_weighted._run_single_split()
5. Aggregate, save, print
```

### Key Design: Weights Computed Once

The importance weights do NOT depend on epsilon. They depend on P_test(X) / P_cal(X),
which is a property of the embedding distributions, not the FDR target. So we compute
embeddings → classifier → weights once before the epsilon loop.

This saves ~3 minutes (embedding time) and ensures the weights are identical across
epsilon values, making the comparison clean.

### Key Design: Only epsilon Changes

In the epsilon loop:
```python
cfg_copy = copy.deepcopy(sgen_cfg)
cfg_copy["epsilon"] = eps
```

Only `epsilon` is modified. NOT `epsilon_e` (which controls pseudo-labeling quality,
not the FDR target), NOT `delta` (which is the PAC confidence level), NOT `delta_p`,
NOT any other parameter.

This is critical because changing epsilon_e would change which pseudo-labels are
generated, which would change the failure counts, which would confound the effect of
epsilon with the effect of pseudo-labeling quality.

### Method 2 in the Sweep

Method 2 uses Option C with frac=0.75 (the best performer from the conservative
analysis):
```python
delta_shift_m2 = 0.75 * (delta - delta_p)  # ≈ 0.015
```

This is passed as a keyword argument to `conservative._run_single_split()`:
```python
m2_run_split(cal_merged, shifted_merged, base_seed + s, cfg_copy,
             delta_shift=delta_shift_m2)
```

Only Option C is included because Options A and B collapse to 0% efficiency (see
Method 2 analysis doc, Sections 48-51).

---

## 28. Epsilon Sweep Design Decisions

### Why These Epsilon Values?

| Epsilon | Required selected accuracy | NQ feasibility |
|---------|--------------------------|----------------|
| 0.25 | ≥ 75% | Infeasible (top 5% = 69.4%) |
| 0.30 | ≥ 70% | Borderline (top 5% = 69.4% ≈ 70%) |
| 0.35 | ≥ 65% | Feasible (top 10% ≈ 63%, close enough with reweighting) |
| 0.40 | ≥ 60% | Feasible (overall accuracy ≈ 40%, but top 15% ≈ 60%) |

The range was chosen to bracket the feasibility transition. eps=0.25 is infeasible,
eps=0.40 should be feasible for all methods. The interesting region is eps=0.30-0.35
where Method 3 should separate from Methods 1-2.

### Why Not Smaller Epsilon Values?

eps=0.10 or eps=0.15 would require 90%+ or 85%+ selected accuracy. Even TQA's top 5%
doesn't reach 90%. These epsilon values are infeasible on BOTH domains, which provides
no useful information about domain shift.

### Why Not Larger Epsilon Values?

eps=0.50 or higher means the system tolerates 50%+ error among selected answers. At
this point, selective generation provides little value — you might as well answer
everything. The SGen paper uses eps=0.25 as the primary experimental setting.

---

## 29. Expected Results at eps=0.25

### Method 1 (actual)

NQ validity = 12.4%, efficiency = 22.9% (500 splits, 3,610 TQA cal / 3,610 NQ shifted).

### Method 2 (actual, Option C frac=0.75)

NQ validity = 22.0%, efficiency = 18.5% (500 splits).

### Method 3 — Prediction vs Actual

**Predicted NQ validity: 15-30%.** Reasoning at time of prediction:
- Importance reweighting fixes the covariate component (~10-20pp of the 87.6pp gap)
- Concept shift remains (~60-70pp) — dominant at eps=0.25
- Starting from 12.4% + 10-20pp correction ≈ 22-32%

**Actual NQ validity: 68.8%.** The prediction was wrong by ~40pp. The mechanism was
completely different from what we predicted: instead of adding genuine validity via
better-calibrated bounds, the reweighting caused massive abstention (344/500 splits
vacuous) through n_eff collapse (30.8%). The 68.8% is entirely vacuous validity.
0/156 non-vacuous splits are valid.

**Predicted NQ efficiency: 10-18%.**
**Actual NQ efficiency: 8.1%.** Close to the predicted range — the wider bounds do
reduce efficiency. But the efficiency is dominated by the 68.8% of splits that select
nothing (eff=0). Among non-vacuous splits, mean efficiency is 26.0%.

**Predicted TQA validity: 100%.**
**Actual TQA validity: 100%.** Correct — the in-domain guarantee is preserved.

**Post-hoc analysis:** The prediction error reveals a fundamental conceptual gap. We
assumed reweighting would make individual splits more valid (lower FDR-E on NQ). Instead,
it made the algorithm more honest about infeasibility (more abstention). The distinction
between "better calibration" and "more abstention" was not anticipated.

---

## 30. Expected Results at eps=0.35 — Prediction vs Actual

This was predicted to be the most interesting operating point because the concept shift
headroom should allow the covariate correction to have a meaningful effect.

### Method 1

**Predicted NQ validity: 25-45%.**
**Actual NQ validity: 0.0%.** Prediction was catastrophically wrong. At eps=0.35, all 500
splits find thresholds (none abstain), and all 500 fail on NQ (mean FDR-E = 0.5335). The
higher epsilon removes the abstention floor that created artificial validity at eps=0.25.

### Method 2

**Predicted NQ validity: 35-55%.**
**Actual NQ validity: 0.0%.** Same failure pattern. Mean FDR-E = 0.5277.

### Method 3

**Predicted NQ validity: 50-80%.**
**Actual NQ validity: 0.2% (1/500 splits).** Only one split (vacuous) remains valid.
Mean FDR-E = 0.5114. The wider weighted bounds still cause slightly more abstention than
M1/M2, but at eps=0.35 even the weighted bounds are loose enough that 499/500 splits
find thresholds. All found thresholds fail on NQ.

**Predicted NQ efficiency: 5-15%.**
**Actual NQ efficiency: 80.0%.** The prediction was completely inverted. At eps=0.35,
the algorithm selects 80% of NQ questions (liberal thresholds), not 5-15% (conservative
thresholds). The prediction error stems from confusing "more selective at eps=0.35" with
"less selective at eps=0.35" — in reality, higher epsilon means less selective thresholds,
which means higher efficiency but worse FDR-E.

### The Headline Finding — What Actually Happened

**No method crosses 98% validity at eps=0.35.** The predicted headline ("Method 3
achieves ≥ 98% at eps=0.35 while M1/M2 don't") did not materialize. The concept shift
dominates at all tested epsilon values, not just eps=0.25. The entire prediction framework
was based on an incorrect mental model of how epsilon interacts with validity under
domain change.

---

## 31. Expected Epsilon Sweep Results — Prediction vs Actual

### Predicted vs Actual Table

| Epsilon | M1 Predicted | M1 Actual | M2 Predicted | M2 Actual | M3 Predicted | M3 Actual |
|---------|--------------|-----------|--------------|-----------|--------------|-----------|
| 0.25 | 12.4% | 12.4% | 22.0% | 22.0% | 15-30% | **68.8%** |
| 0.30 | 25-35% | **0.0%** | 35-45% | **0.0%** | 35-55% | **11.0%** |
| 0.35 | 35-50% | **0.0%** | 45-60% | **0.0%** | 50-80% | **0.2%** |
| 0.40 | 50-65% | **0.0%** | 60-75% | **0.0%** | 70-95% | **0.0%** |

**Key prediction was: "Method 3 crosses 98% between eps=0.35 and eps=0.45."**
**Actual: No method crosses 98% at any tested epsilon. Every prediction for eps ≥ 0.30
was wrong — validity goes to 0%, not up.**

### Why Every Prediction Was Wrong

The predictions assumed a mental model where:
- Higher epsilon → easier target → more splits pass → higher validity

The actual mechanism is:
- Higher epsilon → more thresholds found → fewer vacuous splits → more chances to fail
  on NQ → lower validity

The predictions failed because they treated validity as "how often the method succeeds"
rather than "how often the method either succeeds or abstains." At eps=0.25, the high
abstention rate creates an artificial validity floor. As epsilon increases, this floor
collapses because the algorithm confidently selects thresholds — which then systematically
fail on NQ due to concept shift.

### Which Scenario Materialized?

**Scenario A: "Method 3 barely improves over Method 1."** This is the closest match,
though the mechanism is more extreme than anticipated. The concept shift dominates at
ALL epsilon values, not just eps=0.25. Reweighting doesn't improve non-vacuous FDR-E
at all (0.3413 vs 0.3442) — it only increases abstention.

The revised narrative from Scenario A is exactly correct: "The domain shift is almost
entirely concept shift, so reweighting doesn't help. The model needs to be better, not
the calibration method." Or more precisely: the domains need to be more similar (genuine
shift, not change), not just the calibration method needs to be smarter.

### The Surprising Finding: Validity Monotonically Decreases with Epsilon

This is not predicted by any of the papers (SGen, DS-CP, WR-CP) because they assume
P(Y|X) stability. Under concept shift, the relationship between epsilon and validity
inverts: the validity "comes from" abstention, and higher epsilon reduces abstention.
This is a novel empirical finding that could be reported as a diagnostic signature of
concept shift in selective generation systems.

---

## 32. What Could Go Wrong: Failure Modes

### Failure Mode 1: Classifier Too Accurate

If CV accuracy > 90%, the weights will be extreme. Some TQA questions will have
weight ≈ 50+ while others have weight ≈ 0.1. This causes:
- n_eff << n (maybe < 100)
- Weighted bounds are extremely wide
- Almost all thresholds rejected
- Method 3 abstains more than Method 1

**Mitigation:** Weight clipping at 95th percentile caps extreme weights. The config
also includes 90% and 99% clipping for sensitivity analysis.

### Failure Mode 2: Classifier Too Inaccurate

If CV accuracy < 60%, the domains aren't separable in embedding space. All weights ≈ 1,
and Method 3 is identical to Method 1 with slightly noisier weights.

**Mitigation:** Report as a finding. If all-MiniLM-L6-v2 can't distinguish TQA from NQ,
that's interesting — it means the domain shift is not in the semantic space captured by
the embedding model.

### Failure Mode 3: Weight-Accuracy Anti-Correlation

If high-weight TQA questions (those that look like NQ) tend to be LESS accurate,
reweighting makes the calibration look worse than the uniform average. The weighted
conformal threshold could be poorly calibrated.

**Mitigation:** Diagnostic checks in the weight analysis. If corr(weight, correctness) < 0,
flag this in results.

### Failure Mode 4: Numerical Issues in Weighted CP Bound

If n_eff is very small or failures_eff is very close to n_eff, the beta distribution
PPF can produce NaN or extreme values.

**Mitigation:** Guards in `_weighted_clopper_pearson_upper()`:
- n_eff < 5 → return 1.0
- failures_eff clamped to [0, n_eff - 0.001]

---

## 33. Diagnostic Checklist

After the run completes, check each of these:

### Must-Have (Run is Valid)

- [ ] Classifier CV accuracy in [60%, 90%]
- [ ] n_eff > 100 (overall, before per-split subsetting)
- [ ] No NaN or Inf in per_split results
- [ ] TQA (in-domain) validity = 100% (should be unchanged from Method 1)
- [ ] Method 3 eps=0.25 NQ validity ≥ Method 1 eps=0.25 NQ validity
- [ ] Epsilon sweep results are monotonically increasing in validity with epsilon

### Nice-to-Have (Results Are Interesting)

- [ ] Method 3 NQ validity > Method 2 NQ validity at eps=0.25
- [ ] Method 3 NQ validity ≥ 98% at some epsilon ≤ 0.40
- [ ] Method 3 NQ efficiency > 5% at the epsilon where it achieves 98% validity
- [ ] The three methods separate cleanly in the epsilon sweep figure

### Red Flags

- [ ] Method 3 NQ validity < Method 1 NQ validity at any epsilon (reweighting hurts)
- [ ] n_eff < 20 (bounds are almost vacuous)
- [ ] Classifier CV accuracy > 95% (extreme weights, likely overfitting)
- [ ] All 500 splits abstain at some epsilon (complete failure)

---

## 34. Sensitivity Analysis: Weight Clipping

The config includes three clipping percentiles: 90, 95, 99. The main experiment uses
95. Here's what each produces:

| Clip Percentile | Effect on Weights | Expected n_eff | Notes |
|-----------------|-------------------|----------------|-------|
| 90 | Caps top 10% | Higher (more uniform) | More conservative; closer to Method 1 |
| 95 | Caps top 5% | Moderate | Default; balances correction vs stability |
| 99 | Caps top 1% | Lower (more extreme) | More aggressive correction; could be vacuous |

This sensitivity analysis is NOT run in the main experiment but is available for the
full paper. The `weight_clip_percentiles` config field stores all three values.

---

## 35. The Headline Figure: Validity vs Epsilon

### Design

- X-axis: epsilon ∈ {0.25, 0.30, 0.35, 0.40}
- Y-axis: NQ (shifted domain) validity rate (0-100%)
- Three lines with markers:
  - Method 1 (blue): Vanilla SGen-Semi
  - Method 2 (orange): Conservative Option C (frac=0.75)
  - Method 3 (green): DS-SGen with importance reweighting
- Horizontal dashed line at 98% (PAC target, 1-δ)
- Each line connects four points (one per epsilon)

### What the Figure Should Show (Predicted)

1. All three lines increase with epsilon (more lenient target = easier to satisfy)
2. Method 3 (green) is consistently above Methods 1 (blue) and 2 (orange)
3. Method 3 crosses the 98% line at eps ≈ 0.35-0.40
4. Methods 1 and 2 cross later (eps ≈ 0.45-0.50, outside our range)

### What the Figure Actually Shows

Pending actual run. Will be updated with real data.

---

## 36. Plot Functions: Method 3 Specific

### `plot_method3_fdr_distribution()`

Overlaid histograms of shifted-domain FDR-E across 500 splits for Methods 1 vs 3.
- X-axis: FDR-E value
- Y-axis: Number of splits
- Two colors: Method 1 (blue), Method 3 (green)
- Vertical line at eps=0.25
- Caption with validity rates

**Purpose:** Shows the FDR-E distribution shift — Method 3's histogram should be
shifted left (lower FDR-E) compared to Method 1.

### `plot_weight_analysis()`

2×2 subplot with weight diagnostics:
- Top-left: Weight statistics text panel (classifier accuracy, n_eff, weight stats)
- Top-right: n_eff distribution across splits
- Bottom-left: Split outcomes bar chart (Method 1 vs Method 3)
- Bottom-right: Efficiency distribution (Method 1 vs Method 3)

**Purpose:** Provides the diagnostic information needed to assess whether the
weighting is working as intended.

---

## 37. Plot Functions: Epsilon Sweep

### `plot_epsilon_sweep_validity()`

The headline figure (described in Section 35). Three lines + 98% horizontal line.

### `plot_epsilon_sweep_efficiency()`

Same layout as validity, but Y-axis = NQ mean efficiency (%). Shows the efficiency
cost of each method at each epsilon.

### `plot_three_method_comparison()`

Two-panel figure:
- Left panel: Validity bars at eps=0.25 and eps=0.35
- Right panel: Efficiency bars at eps=0.25 and eps=0.35

**Why two panels?** Validity (near 98%) and efficiency (near 10-20%) are on completely
different scales. A single grouped bar chart would either compress validity or inflate
efficiency. Two panels with independent Y-axes show both clearly.

---

## 38. SLURM Configuration

### scripts/run_method3.sh

```bash
#SBATCH --job-name=dsgen_m3
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
```

**GPU:** Required for embedding computation (~2 min on A6000). The rest (classifier,
weights, SGen splits) is CPU-only.

**Memory:** 32GB (up from 16GB for Methods 1-2) because we load the SentenceTransformer
model (~90MB) plus embeddings (~10MB) plus all cached data (~100MB) simultaneously.

**Time:** 24 hours (generous). Expected runtime is ~10 minutes. The 24h budget
accommodates potential queuing delays and preemption/requeue cycles.

**Preemption handling:** The script includes a USR1 signal handler that requeues the
job on preemption:
```bash
trap 'requeue_handler' USR1
```

Python commands are run with `&` + `wait $!` to allow signal propagation. Without this,
signals would be blocked while Python is running.

### Execution Steps

1. `python run_importance_weighted.py` — Method 3 standalone
2. `python run_epsilon_sweep.py` — All methods at 4 epsilon values
3. `python plot_results.py --stage method3 --stage epsilon_sweep` — Generate plots

Each step checks the exit code before proceeding.

---

## 39. Caching Strategy

### Cached Inputs (from Stages 1-3)

| Cache File | Contents | Size |
|------------|----------|------|
| nq_data.json | 3,610 NQ records | ~2MB |
| tqa_data.json | 3,610 TQA records | ~2MB |
| nq_generations.json | 3,610 generation results | ~15MB |
| tqa_generations.json | 3,610 generation results | ~15MB |
| nq_entailment.json | 3,610 entailment scores | ~8MB |
| tqa_entailment.json | 3,610 entailment scores | ~8MB |

### Cached by Method 3

| Cache File | Contents | Size |
|------------|----------|------|
| tqa_embeddings.npy | (3610, 384) float32 | ~5.3MB |
| nq_embeddings.npy | (3610, 384) float32 | ~5.3MB |

### Results Files

| File | Contents |
|------|----------|
| importance_weighted_results.json | Full Method 3 results with per-split data |
| epsilon_sweep_results.json | Aggregated sweep results for all 3 methods |

### Caching Logic

Embeddings are cached because they require GPU and take ~2 minutes. On subsequent runs
(e.g., after changing epsilon or other non-embedding parameters), the cached embeddings
are loaded instantly.

The classifier and weights are NOT cached because they're fast to compute (~5 seconds
total) and depend on the embeddings. If the embeddings change (different model), the
classifier must be retrained.

---

## 40. Code Architecture: File-by-File

### ds_sgen/importance_weighted.py (~610 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-18 | Module docstring | References to 3 papers |
| 20-35 | Imports | numpy, scipy.stats.beta, sklearn, sgen_semi imports |
| 40-61 | `compute_embeddings()` | SentenceTransformer encoding with caching |
| 66-91 | `train_domain_classifier()` | Logistic regression with 5-fold CV |
| 96-142 | `compute_importance_weights()` | Density ratio, clip at percentile, normalize |
| 147-178 | `_weighted_conformal_threshold()` | Weighted quantile via sorted cumulative |
| 183-208 | `_weighted_clopper_pearson_upper()` | n_eff-based scaling with guards |
| 213-416 | `_run_single_split()` | Full single-split with weight indexing |
| 421-578 | `run_experiment()` | Orchestration: embed → classify → weight → 500 splits |
| 581-610 | `print_importance_weighted_summary()` | Formatted console output |

### run_importance_weighted.py (~150 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-10 | Module docstring | Usage instructions |
| 12-30 | `setup_logging()` | Console handler |
| 34-58 | `load_cached_stages()` | Load 6 cache files (copied from run_conservative.py) |
| 63-149 | `main()` | Parse args, pre-flight check, load caches, run experiment |

### run_epsilon_sweep.py (~256 lines)

| Lines | Function | Description |
|-------|----------|-------------|
| 1-14 | Module docstring | Purpose and usage |
| 16-33 | Imports | All three method modules + utilities |
| 37-45 | `setup_logging()` | Console handler |
| 49-64 | `load_cached_stages()` | Load 6 cache files |
| 69-85 | `_aggregate_splits()` | Aggregate per-split results into summary stats |
| 90-256 | `main()` | Full sweep: load data, compute weights, loop over epsilons |

---

## 41. Differences from DS-CP

| Feature | DS-CP (Lin et al., 2025) | Our Method 3 |
|---------|--------------------------|-------------|
| Task | Prediction set coverage | Selective generation FDR-E |
| Output space | Multiple-choice (MMLU) | Open-ended text (QA) |
| Correctness | Exact match | Textual entailment (NLI) |
| Guarantee | Marginal coverage ≥ 1-alpha | PAC: P{FDR-E ≤ eps} ≥ 1-delta |
| Classifier | XGBoost | Logistic regression |
| Test-point weight | lambda=1 regularization | Not applicable (500 splits, not single test point) |
| Evaluation | Per-question coverage | Per-split validity rate across 500 splits |
| Domains | 272 MMLU pairs | TQA → NQ (single pair) |
| Models | 16 LLMs (1.8B-72B) | GPT-4o-mini only |

### Key Structural Difference

DS-CP operates on a single calibration + single test point at a time. The test point's
weight is regularized to lambda=1 (Theorem 2 of their paper) to prevent the prediction
set from being dominated by the test point's estimated weight.

Our Method 3 operates on a calibration set that is split 100 times. There is no single
"test point" — the test set is the entire NQ dataset (3,610 questions). The weights
are applied to calibration samples only, and the evaluation on the test set is unweighted.

---

## 42. Differences from Method 1 SGen-Semi

### Identical Components

| Component | Notes |
|-----------|-------|
| Data splitting (cal/test/Z_U/Z_E) | Same proportions, same random seeds |
| Pseudo-labeling logic | Same: entail_score ≥ tau_CP → label 1 |
| Threshold grid construction | Same: percentile-based, n_grid=20 |
| Bonferroni correction | Same: delta_adj = (delta - delta_p) / H |
| Evaluation on test sets | Same: unweighted FDR-E |
| Selection mode | Same: fm1_only |

### Modified Components

| Component | Method 1 | Method 3 |
|-----------|----------|----------|
| Conformal threshold | Unweighted quantile | Weighted quantile |
| PAC bound | CP upper with actual n | Weighted CP upper with n_eff |
| Grid search acceptance | `cp_upper(failures, m, delta_adj)` | `weighted_cp_upper(failures, m, delta_adj, n_eff_sel)` |

### Added Components

| Component | Purpose |
|-----------|---------|
| Embedding computation | Convert questions to 384-dim vectors |
| Domain classifier | Distinguish TQA from NQ embeddings |
| Weight computation | Density ratio, clipping, normalization |
| Weight indexing | Propagate weights through data splits |
| Per-threshold n_eff | Effective sample size for selected subset |

### Removed Components

None. Method 3 is a strict superset of Method 1.

---

## 43. Limitations of Method 3

### The Central Limitation: Domain Shift vs. Domain Change

The most important limitation revealed by our experiments is not a limitation of the
*method* — it is a limitation of the *problem setup*. DS-SGen (and all importance-
reweighting-based approaches) assume **covariate shift**: P_test(X) ≠ P_cal(X) but
P_test(Y|X) = P_cal(Y|X). Our TQA → NQ experiment violates the second condition.

**Domain shift** is when the test distribution is a reweighted version of the calibration
distribution — different question frequencies but the same question-answer relationship.
A clinical QA system deployed in a cardiac clinic sees more heart-related questions, but
its accuracy on any given cardiac question is the same as during calibration.

**Domain change** is when the test distribution has both different question frequencies AND
different question-answer relationships. A trivia QA system deployed on search queries
encounters fundamentally different questions that require different knowledge, and its
accuracy degrades because the knowledge doesn't transfer.

Our results prove TQA → NQ is a domain change:
- 91.7% classifier accuracy (domains are nearly separable)
- 27.7pp accuracy gap (70.8% TQA vs 43.1% NQ)
- 0/156 non-vacuous splits are valid (reweighting doesn't help the actual error rate)
- Minimum non-vacuous FDR-E = 0.2550 (concept shift imposes a hard floor above eps=0.25)

**No calibration-time method can fix a domain change.** The weighted CP guarantee
(Tibshirani et al., 2019) explicitly requires P(Y|X) stability. When this assumption
fails, the guarantee is void — and our results show exactly this void.

### Theoretical Limitations

1. **No formal guarantee with estimated weights.** The weighted CP guarantee
   (Tibshirani et al., 2019) assumes exact density ratios. We estimate them via
   logistic regression, introducing an unknown bias term. The degree to which the
   formal guarantee is violated depends on the estimation error, which we cannot
   bound analytically.

2. **Does not handle concept shift.** If P(Y|X) differs across domains, reweighting
   P(X) doesn't fix it. Our TQA→NQ shift has dominant concept shift (70.8% vs 43.1%
   accuracy). Method 3 can only fix the covariate component. In our experiment, the
   covariate component contributes minimally — non-vacuous FDR-E is 0.3413 (M3) vs
   0.3442 (M1), a negligible 0.3pp improvement.

3. **n_eff reduction widens bounds.** The price of reweighting is reduced effective
   sample size, which makes the Clopper-Pearson bounds wider. In our experiment,
   n_eff = 30.8% of n, causing 68.8% of splits to abstain entirely (vs 12.4% for M1).
   This means Method 3 is dramatically MORE conservative than Method 1 — it correctly
   identifies that it can't meet the guarantee, but at the cost of selecting nothing.

4. **Classifier accuracy amplifies the problem.** When domains are highly separable
   (accuracy > 90%), the density ratios become extreme (raw max = 32.7 before clipping).
   This creates a vicious cycle: better classification → more extreme weights → lower
   n_eff → wider bounds → more abstention. Paradoxically, a worse classifier (65-75%
   accuracy) would produce better results because the weights would be more uniform.

### Practical Limitations

5. **Single embedding model.** We use only all-MiniLM-L6-v2. A different embedding
   model might capture different aspects of the domain shift. We don't experiment with
   this. However, the 91.7% accuracy suggests the separability is a property of the
   domains themselves, not the embedding model — any reasonable embedding would likely
   achieve similarly high accuracy.

6. **Single classifier architecture.** Logistic regression is simple but may miss
   nonlinear boundaries. However, at 91.7% accuracy, the linear classifier is already
   near-ceiling. More complex classifiers (XGBoost, neural networks) would likely achieve
   even higher accuracy, making the n_eff problem worse, not better.

7. **Single dataset pair.** TQA→NQ is one domain shift scenario — and, as our results
   show, it is in the "domain change" regime where DS-SGen cannot work. The method's
   effectiveness on genuine domain shifts (moderate covariate shift, minimal concept shift)
   remains undemonstrated. Candidate dataset pairs for future validation include medical
   QA subdomains or topical splits of multi-domain QA benchmarks.

8. **Fixed weight clipping.** The 95th percentile clip caps the raw max weight from
   32.659 to 5.692 (normalized). Without clipping, n_eff would drop from 30.8% to ~11%,
   making the results even worse. Clipping helps but cannot fix the fundamental problem.

9. **No iterative refinement.** The weights are computed once and fixed. An iterative
   approach (reweight → calibrate → evaluate → adjust weights) might improve results
   for moderate shifts, but cannot fix concept shift.

### Honest Assessment of the Approach

Method 3 is a correct implementation of domain-shift-aware selective generation. The
pipeline (embed → classify → reweight → calibrate) works as designed. The algorithm
correctly detects infeasibility (more abstention when the guarantee is unachievable)
and correctly preserves the in-domain guarantee (100% TQA validity).

The negative result is not a failure of the method — it is a failure of the problem setup.
TQA → NQ is too dissimilar for any covariate-shift correction to work. The method's
value lies in:
1. Demonstrating that the pipeline works correctly (in-domain guarantee preserved)
2. Showing that the algorithm honestly reports infeasibility (more abstention, not false
   confidence)
3. Establishing the boundary between domain shift (where DS-SGen helps) and domain change
   (where it cannot)
4. Providing a concrete, validated implementation for testing on more suitable dataset pairs

---

## 44. The Honest Story: What This Project Shows

### The Actual Story (from results)

SGen-Semi's PAC FDR-E guarantee breaks catastrophically under domain shift: NQ validity
drops from 100% (in-domain) to 12.4% (shifted) at eps=0.25. Conservative threshold
adjustments (Method 2) provide minimal improvement (22.0%). Importance reweighting
(Method 3) increases validity to 68.8%, but this "improvement" is entirely vacuous —
the algorithm correctly detects that it cannot meet the guarantee and abstains in 344/500
splits. Among the 156 splits that do find thresholds, zero achieve valid FDR-E control
on NQ.

The epsilon sweep reveals a counterintuitive pattern: validity *decreases* with higher
epsilon for all methods (12.4% → 0.0% for M1, 68.8% → 0.0% for M3). Higher epsilon
makes thresholds easier to find, but all TQA-calibrated thresholds systematically fail
on NQ regardless of epsilon. The "validity" at low epsilon was never genuine — it was
abstention masquerading as control.

The root cause is that TQA → NQ is a domain *change* (both P(X) and P(Y|X) differ),
not a domain *shift* (only P(X) differs). The 91.7% domain classifier accuracy proves
the domains are nearly separable, and the 27.7pp accuracy gap (70.8% TQA vs 43.1% NQ)
confirms that the model's knowledge doesn't transfer. DS-SGen's importance reweighting
correctly addresses the covariate component of the shift, but this component is dwarfed
by the concept shift that no calibration-time method can fix.

**The positive story within the negative result:** DS-SGen works as a diagnostic tool.
The algorithm's increased abstention rate (68.8% vacuous splits vs M1's 12.4%) is the
method honestly reporting infeasibility. A practitioner using M1 would deploy a system
that silently fails on 87.6% of splits; a practitioner using M3 would be warned (via
mass abstention) that the guarantee cannot be maintained under this domain shift. This
"fail loudly" behavior is arguably more valuable than false confidence.

**The forward-looking story:** DS-SGen is designed for moderate domain shifts where
P(Y|X) is approximately stable — a general-purpose clinical QA system deployed in a
cardiac clinic, a legal QA system applied to real estate contracts, or a customer
support model shifted between product lines. Our TQA → NQ experiment establishes the
boundary condition: when domains are too dissimilar (classifier accuracy > 90%, accuracy
gap > 25pp), no calibration-time fix suffices. Future work with a more suitable dataset
pair (moderate shift, shared knowledge base) is needed to demonstrate DS-SGen's positive
case.

### What We Should NOT Claim

- "Method 3 solves domain shift for selective generation." (It doesn't — concept shift
  remains, and TQA → NQ is dominated by concept shift.)
- "Importance reweighting restores the PAC guarantee." (It does not restore it at any
  tested epsilon. The validity improvement is entirely from abstention.)
- "The 68.8% validity is a meaningful improvement." (It is — but only as a diagnostic
  signal. The 68.8% consists entirely of splits that select nothing.)
- "The negative result means DS-SGen doesn't work." (It means DS-SGen doesn't work for
  domain *changes*. Its effectiveness on genuine domain *shifts* remains untested.)

### What We CAN Claim

- SGen-Semi's guarantee breaks under domain shift (12.4% validity, first empirical
  demonstration for FDR-E control)
- Conservative methods provide minimal improvement (22.0% validity)
- DS-SGen correctly detects infeasibility via increased abstention (68.8% vacuous splits)
- The domain shift vs domain change distinction is critical for applicability
- TQA → NQ is in the domain change regime (91.7% classifier accuracy, 27.7pp accuracy
  gap, 0/156 non-vacuous splits valid)
- The epsilon sweep reveals that TQA-calibrated thresholds systematically fail on NQ at
  all epsilon values — the concept shift imposes a hard floor on FDR-E (~25.5%)

---

## 45. Critical Self-Assessment

### What We Did Well

1. **Thorough baseline.** Methods 1-2 establish the problem clearly before attempting
   to fix it. The 12.4% NQ validity (vs 100% TQA) is a strong motivating finding.

2. **Principled approach.** Method 3 is grounded in established theory (weighted CP)
   and builds on a recent LLM-specific precursor (DS-CP). It's not ad-hoc.

3. **Honest about limitations.** The eps=0.25 impossibility is stated up front, not
   buried in the appendix. The concept shift problem is central to the narrative.

4. **Multi-epsilon evaluation.** The epsilon sweep provides a complete picture rather
   than a single possibly-cherry-picked operating point.

### What Could Be Better

1. **Single dataset pair.** TQA→NQ is one shift scenario. Results might not generalize
   to medical, legal, or code generation domains.

2. **No comparison with Cherian et al. (2024).** Enhanced CP's conditional boosting
   (Corollary A.1) provides an alternative approach to domain shift robustness. We
   don't compare against it.

3. **No comparison with DS-CP directly.** We adapt DS-CP's pipeline but don't run
   DS-CP itself (it targets coverage, not FDR-E). A direct comparison on coverage
   would validate our embedding/classifier choices.

4. **Fixed classifier architecture.** Logistic regression is safe but potentially
   leaves performance on the table. A systematic comparison (LR vs SVM vs XGBoost)
   would strengthen the results.

5. **No calibration of the classifier.** We use raw logistic regression probabilities.
   Platt scaling or isotonic regression calibration could improve the density ratio
   estimates, though logistic regression probabilities are generally well-calibrated.

6. **The n_eff-based Clopper-Pearson bound is an approximation.** The exact weighted
   binomial confidence interval is more complex (see Lumley, 2004, "Analysis of Complex
   Survey Samples"). Our approximation is standard but not provably correct.

### Playing Devil's Advocate

**"Isn't Method 3 just Method 1 with noisier bounds?"**

If the classifier isn't very good (CV accuracy ~60%), then yes — the weights are
near-uniform, and the wider bounds (due to n_eff < n) make Method 3 strictly worse
than Method 1. The improvement only manifests when the classifier is accurate enough
to produce meaningful weights (70%+) and the weight distribution isn't so extreme
that n_eff collapses.

**"You're claiming to fix domain shift but you can't even beat 50% validity at eps=0.25."**

Correct. The eps=0.25 result is expected to be modest. The actual contribution is at
higher epsilon values, where the method demonstrably restores the PAC guarantee. The
eps=0.25 result is a negative finding about concept shift, not a failure of Method 3.

**"The theoretical guarantee doesn't hold because you use estimated weights."**

Correct. We cannot formally claim P{FDR-E ≤ eps} ≥ 1-delta for Method 3. The guarantee
is approximate. However, the empirical validity rate across 500 random splits serves
as a robust estimate of the true validity, and if it exceeds 98%, that's strong empirical
evidence even without a formal guarantee.

**"Why not just use a better model that's more accurate on NQ?"**

Because that changes the research question. The project asks: "Given a fixed model,
can calibration-time techniques restore PAC guarantees under domain shift?" Using a
better model would sidestep the calibration question entirely.

**"The epsilon sweep is cherry-picking the operating point where Method 3 works."**

The sweep covers {0.25, 0.30, 0.35, 0.40}, which includes the standard setting (0.25)
where Method 3 fails AND higher values where it succeeds. Reporting all four is the
opposite of cherry-picking. The contribution is the analysis of WHERE guarantees can
be restored, not that they can be restored at a convenient point.

---

## 46. Connections to the Literature

### Direct Ancestors

| Paper | What We Use | Citation |
|-------|-------------|---------|
| SGen-Semi (Lee et al., NeurIPS 2024) | Algorithm framework, FDR-E decomposition, Clopper-Pearson bounds | Foundational |
| Weighted CP (Tibshirani et al., NeurIPS 2019) | Weighted exchangeability, density ratio trick | Theoretical basis |
| DS-CP (Lin et al., arXiv 2025) | Embedding + classifier pipeline for LLM density ratios | Pipeline design |

### Indirect Ancestors

| Paper | Connection |
|-------|-----------|
| WR-CP (Xu et al., ICLR 2025) | Decomposition of coverage gap into covariate + concept terms |
| Conformal Factuality (Mohri & Hashimoto, 2024) | Entailment-based correctness for conformal prediction |
| Enhanced CP (Cherian et al., NeurIPS 2024) | Alternative approach: conditional boosting for domain robustness |
| Subpopulation CP (Wang et al., 2025) | Mixture-of-domains model; Algorithm 3's filter-and-reweight approach |

### Novelty Claim

No existing method provides:
1. PAC FDR-E guarantees (not just marginal coverage)
2. For open-ended text generation (not multiple-choice)
3. With domain shift robustness via importance reweighting
4. Using entailment-based correctness

DS-CP provides (3) but for coverage, not FDR-E, and for multiple-choice, not open-ended.
SGen provides (1), (2), (4) but not (3). Method 3 combines them.

---

## 47. Issues Log

### Issue 1: Weight Indexing Through Permutations

**Problem:** The initial plan description was ambiguous about how weights follow data
through random permutations and Z_U/Z_E splits.

**Fix:** Explicit indexing using `cal_weights[cal_idx]` and positional slicing `[:zu_size]`.
Verified that numpy fancy indexing preserves the correct correspondence.

**Impact:** If this had been wrong, weights would be randomly assigned to questions,
making the weighting meaningless (but not obviously broken — results would be noisy).

### Issue 2: Weighted CP Bound Scaling

**Problem:** First version of the plan computed `failures_eff = failures * n_eff / total`,
which is algebraically equivalent to `failure_rate * n_eff` but less clear about the
intent (we're scaling the rate, not the count).

**Fix:** Changed to explicit `failure_rate = failures / total; failures_eff = failure_rate * n_eff`.
This makes the intent clear and matches the survey statistics literature.

**Impact:** No numerical difference; purely a readability improvement.

### Issue 3: n_eff Guard Threshold

**Problem:** Initial plan used n_eff < 2 as the guard. This is too permissive — a beta
distribution with 2 pseudo-observations is extremely noisy.

**Fix:** Changed to n_eff < 5. This is still conservative (some papers use n_eff < 10)
but avoids pathological beta distribution behavior while not being overly restrictive.

### Issue 4: Epsilon vs Epsilon_e in Sweep

**Problem:** Early sweep design changed BOTH epsilon (FDR target) and epsilon_e
(pseudo-labeling threshold) together, confounding two effects.

**Fix:** Sweep changes ONLY epsilon. epsilon_e stays at 0.05 throughout.

**Impact:** Removing this confound makes the sweep results interpretable: all differences
are due to the FDR target, not the pseudo-labeling quality.

### Issue 5: Method 2 in Sweep Calls

**Problem:** Initial sweep design called `conservative._run_sweep()` (the full sweep
function), which runs all parameter values. We only want Option C frac=0.75.

**Fix:** Call `conservative._run_single_split()` directly with `delta_shift=0.75*(delta-delta_p)`.
This function accepts delta_shift as a keyword argument (conservative.py line 95).

### Issue 6: Float Rounding in Weighted CP

**Problem:** In rare cases, `failure_rate * n_eff` could exceed `n_eff` due to floating
point arithmetic, causing `successes_eff < 0` and crashing `beta_dist.ppf`.

**Fix:** Guard: `failures_eff = max(0.0, min(n_eff - 0.001, failures_eff))`. The 0.001
margin ensures at least a minimal positive successes_eff.

---

## 48. Worked Example: One Complete Split of Method 3

This section walks through a single split using actual data characteristics to show
every number and decision. The numbers below are derived from the cached data statistics
(validated in method1_baseline_analysis.md Sections 41-42) and the algorithm logic.

### Setup

Split seed = 42 (the first split). Config: epsilon=0.25, delta=0.02, delta_p=1e-5,
cal_frac=0.70, zu_frac=0.75, epsilon_e=0.05, n_grid=20, selection_mode=fm1_only.

Calibration dataset: TQA (3,610 records, 70.8% correct).
Shifted test: NQ (3,610 records, 43.1% correct).

### Step 1: Data Split

Random permutation of 3,610 TQA records with seed=42.
- cal_data: first 2,527 (70% of 3,610)
- indomain_test: remaining 1,083 (30%)

Importance weights are indexed: `cal_data_weights = cal_weights[indices[:2527]]`.
Both arrays have length 2,527 and are aligned by position.

### Step 2: Z_U / Z_E Split

- Z_U: first 1,895 of cal_data (75% of 2,527)
- Z_E: remaining 632
- z_u_weights: first 1,895 of cal_data_weights
- z_e_weights: remaining 632

Z_E statistics (expected, based on TQA overall 70.8% correct):
- ~453 correct (70.8% of 632)
- ~179 incorrect
- Mean entail_score among correct: ~0.85 (from entailment score histogram)
- Mean entail_score among incorrect: ~0.25

### Step 3: Weighted Conformal Threshold

Extract 453 correct entailment scores from Z_E, and their 453 corresponding weights.

In **Method 1**, we'd compute:
```
k = ceil((453+1) × 0.05) = ceil(22.7) = 23
tau_CP = sorted_correct_scores[22]  ≈ 0.49 (from Method 1 results)
```

In **Method 3**, we compute the weighted epsilon_e-quantile:
1. Sort the 453 scores ascending.
2. Sort the 453 weights in the same order.
3. Compute cumulative: cumsum(sorted_weights) / sum(sorted_weights).
4. Find first index where cumulative ≥ 0.05.

**How weighting changes tau_CP:**

If high-weight Z_E correct answers (those that look like NQ) tend to have lower
entailment scores, they contribute more cumulative mass early in the sorted order.
This means the 5% cumulative threshold is reached at a LOWER score, making tau_CP
lower (more permissive pseudo-labeling).

If high-weight Z_E correct answers tend to have higher entailment scores, the
cumulative mass is concentrated later, and tau_CP could be higher.

Expected effect: the TQA questions most similar to NQ (high weight) tend to be the
harder TQA questions. Harder questions may have lower entailment scores when correct
(the model is less confident on them). So the weighted tau_CP is likely slightly
LOWER than the unweighted tau_CP — meaning more Z_U questions are pseudo-labeled
as correct.

**Expected weighted tau_CP: ~0.46-0.50** (compared to unweighted ~0.49).

### Step 4: Pseudo-Labeling Z_U

For each of the 1,895 Z_U records:
- pseudo_label = 1 if entail_score ≥ tau_CP, else 0

With TQA's 70.8% correctness and tau_CP ≈ 0.48:
- True correct with score ≥ tau_CP: ~95% of 1,357 correct = ~1,289 (true positive)
- True correct with score < tau_CP: ~5% of 1,357 = ~68 (false negative)
- True incorrect with score ≥ tau_CP: ~15% of 538 incorrect = ~81 (false positive)
- True incorrect with score < tau_CP: ~85% of 538 = ~457 (true negative)

Total pseudo-labeled correct: ~1,370 out of 1,895 (72%).
Pseudo-label precision: 1,102 / 1,212 ≈ 91%.

### Step 5: Grid Search with Weighted Bounds

Build 20-point percentile grid on Z_U's fM1 values. TQA fM1 ranges from ~-0.90 to
~-0.001, with 20 evenly-spaced percentiles giving thresholds approximately:

```
tau1_grid ≈ [-0.70, -0.55, -0.45, -0.38, -0.33, -0.29, -0.26, -0.23, -0.20,
             -0.18, -0.16, -0.15, -0.13, -0.12, -0.10, -0.09, -0.07, -0.05,
             -0.03, -0.01]
```

H = 20 unique grid points (after deduplication).
Bonferroni: delta_adj = (0.02 - 1e-5) / 20 ≈ 0.001.

**For each threshold, compare Method 1 vs Method 3:**

Example: tau1 = -0.12 (approximately 65th percentile of TQA fM1).

- Selected in Z_U: ~660 questions (35% of 1,895)
- Pseudo-failures: ~85 (pseudo_label = 0 among selected)
- Failure rate: 85/660 = 12.9%

**Method 1 bound:**
```
CP_upper = beta.ppf(1 - 0.001, 85+1, 660-85) = beta.ppf(0.999, 86, 575)
         ≈ 0.166
```
Since 0.166 ≤ 0.25, this threshold passes.

**Method 3 bound:**

First compute n_eff for the 660 selected questions' weights:
```
sel_weights = z_u_weights[selected]      # shape (660,)
n_eff_sel = (sum(sel_weights))^2 / sum(sel_weights^2)
```

If the weights have moderate variance (std ≈ 0.5), n_eff_sel ≈ 450 (68% of 660).

```
failure_rate = 85 / 660 = 0.1288
failures_eff = 0.1288 × 450 = 57.95
successes_eff = 450 - 57.95 = 392.05
CP_upper_weighted = beta.ppf(0.999, 58.95, 392.05) ≈ 0.183
```

Since 0.183 ≤ 0.25, this threshold also passes. But the bound is wider (0.183 vs
0.166) due to the reduced effective sample size.

**Impact on the grid search:** The weighted bound is wider for EVERY threshold. This
means some thresholds that pass in Method 1 might fail in Method 3 (those where the
unweighted bound is between 0.166 and 0.183). The most permissive (highest efficiency)
threshold that passes might be slightly more selective.

### Step 6: Evaluation

Suppose the best threshold in Method 3 is tau1 = -0.115 (slightly more selective
than Method 1's tau1 = -0.121 from split 0).

**In-domain (1,083 TQA):**
- Selected: ~430 (39.7%)
- Correct among selected: ~370 (86%)
- FDR-E: 60/430 ≈ 0.14 → valid (< 0.25)

**Shifted (3,610 NQ):**
- Selected: ~850 (23.5%)
- Correct among selected: ~560 (66%)
- FDR-E: 290/850 ≈ 0.34 ��� INVALID (> 0.25)

**Result for this split:** In-domain valid, shifted invalid. Same outcome as Method 1
for this particular split at eps=0.25.

### Why This Split Fails

The FDR-E on NQ is 0.34, still well above 0.25. The importance weights made the bound
slightly wider (so the threshold is slightly more selective: tau1=-0.115 vs -0.121)
but didn't fundamentally change which NQ questions are selected. The selected NQ
questions still have ~66% accuracy, which gives FDR-E=0.34, far exceeding epsilon=0.25.

**This is the concept shift at work.** The weights correctly identify which TQA
questions look like NQ questions, but they can't change the fact that NQ's fM1-accuracy
relationship is weaker.

---

## 49. Bound Width Comparison: Weighted vs Unweighted

This section derives the exact relationship between the weighted and unweighted bounds
to quantify the price of reweighting.

### Unweighted Clopper-Pearson

Given m selected samples with f failures:
```
CP_upper(f, m, alpha) = B^{-1}(1-alpha; f+1, m-f)
```
where B^{-1} is the inverse beta CDF. The width of this bound decreases as O(1/√m).

### Weighted Clopper-Pearson

Given m selected samples with f failures and effective sample size n_eff:
```
f_rate = f / m
f_eff = f_rate × n_eff
s_eff = n_eff - f_eff
CP_upper_weighted(f, m, alpha, n_eff) = B^{-1}(1-alpha; f_eff+1, s_eff)
```

### The Price Factor

Let r = n_eff / m (the effective sample ratio for the selected subset). Since n_eff ≤ m,
we have r ∈ (0, 1].

The bound width scales as O(1/√n_eff) = O(1/√(r×m)). Compared to the unweighted
bound (O(1/√m)), the weighted bound is wider by a factor of 1/√r.

| n_eff / m | Bound widening factor | Practical effect |
|-----------|----------------------|------------------|
| 1.0 (uniform) | 1.0× | No widening — identical to Method 1 |
| 0.75 | 1.15× | Minimal — barely noticeable |
| 0.50 | 1.41× | Moderate — some thresholds flip from pass to fail |
| 0.25 | 2.00× | Significant — many thresholds fail that previously passed |
| 0.10 | 3.16× | Severe — most thresholds fail, near-total abstention |

**Critical insight:** If n_eff/m < 0.25, the bound is 2× wider, which typically
eliminates all non-trivial thresholds. This is the regime where Method 3 becomes
WORSE than Method 1 (more abstention, no validity improvement).

### Expected Regime

With a 72% accurate classifier and 95th percentile clipping, we expect n_eff/m ≈
0.50-0.70 for typical selected subsets. This puts us in the "moderate widening" range
— enough to make the bounds wider but not so much that the method collapses.

### How Threshold Selection Changes

In Method 1 at eps=0.25, the most permissive threshold (highest efficiency) that
satisfies CP_upper ≤ 0.25 selects about 35% of Z_U (from Method 1 results: 71/100
splits find a threshold with ~39% TQA efficiency).

In Method 3, the wider bound means the acceptance frontier shifts inward. The most
permissive valid threshold selects perhaps 30-33% of Z_U — a 5-10% relative efficiency
loss. This is the direct cost of reweighting.

However, the BENEFIT is that the selected set's performance on NQ is better calibrated.
Method 1's threshold was calibrated against TQA's accuracy distribution; Method 3's
threshold is calibrated against a weighted distribution that approximates NQ's.

---

## 50. What Kinds of TQA Questions Get High Weights?

This section analyzes the relationship between importance weights and question
characteristics, based on the known properties of TQA and NQ.

### The Domain Classifier's Decision Boundary

The logistic regression classifier operates on 384-dimensional sentence embeddings.
Questions that receive high P(NQ|x) — and therefore high importance weights — are
TQA questions whose embedding vectors fall on the "NQ side" of the decision boundary.

**What makes a question look like NQ?**

NQ questions are real Google search queries. They tend to be:
- Shorter, more colloquial ("who won the 2018 world cup")
- More open-ended ("what causes headaches")
- More practically oriented ("how to cook rice")
- More diverse in topic (not limited to trivia domains)

TQA questions are written by trivia enthusiasts. They tend to be:
- More formal ("In which year did the Battle of Hastings take place?")
- More specific (proper nouns, exact dates, specific domains)
- More knowledge-focused (geography, history, science trivia)

### High-Weight TQA Questions (NQ-Like)

TQA questions that receive high importance weights are those that resemble search
queries in embedding space. These likely include:
- Questions with informal phrasing
- Questions about practical topics (not classic trivia domains)
- Shorter questions with simple structure
- Questions about more general knowledge

### Low-Weight TQA Questions (Unlike NQ)

TQA questions that receive low weights are those that look very different from NQ:
- Very specific trivia ("What is the name of the castle in Shakespeare's Hamlet?")
- Questions with formal trivia-style phrasing
- Questions about niche domains that NQ users rarely search for

### Weight-Accuracy Correlation

An important question: are high-weight TQA questions more or less accurate than
low-weight ones?

**Hypothesis:** High-weight TQA questions (NQ-like) should be harder for GPT-4o-mini,
because:
1. They share characteristics with NQ, where GPT-4o-mini is less accurate (43.1% vs 70.8%)
2. They may be about more practical/obscure topics where the model has less training data
3. They may have less structured phrasing, making them harder to parse

**If corr(weight, accuracy) < 0:** The weighted calibration oversamples harder TQA
questions, making the effective calibration look more like NQ. This is exactly what
we want — the weighted bounds reflect the difficulty level of NQ.

**If corr(weight, accuracy) > 0:** The weighted calibration oversamples easier TQA
questions. This would be counterproductive — the bounds would be too optimistic about
NQ performance. This would be a failure mode.

**Expected:** Slight negative correlation (corr ≈ -0.05 to -0.15). NQ-like TQA
questions are somewhat harder but not dramatically so, because the model's accuracy
is primarily determined by question topic and specificity, not phrasing style.

This will be validated from actual results in Section 54.

---

## 51. The Interaction Between n_eff and Bonferroni

The Bonferroni correction divides the confidence budget delta across H threshold
candidates. In fm1_only mode, H ≈ 20 (after grid deduplication), giving:

```
delta_adj = (delta - delta_p) / H = (0.02 - 1e-5) / 20 ≈ 0.001
```

This means each threshold must satisfy the Clopper-Pearson bound at a very stringent
confidence level: alpha = 0.001 (99.9% confidence).

### How n_eff Interacts with Bonferroni

At alpha = 0.001, the CP bound is already wide even with unweighted samples. The
width increases further when n_eff < m.

**Numerical example:**

For m = 660 selected with f = 85 failures (12.9% failure rate):

| Setting | alpha | CP upper bound |
|---------|-------|---------------|
| Unweighted, no Bonferroni | 0.020 | 0.152 |
| Unweighted, with Bonferroni | 0.001 | 0.166 |
| Weighted (n_eff=450), no Bonferroni | 0.020 | 0.162 |
| Weighted (n_eff=450), with Bonferroni | 0.001 | 0.183 |

The combined effect of Bonferroni + weighting widens the bound by 0.031 (from 0.152
to 0.183). Of this:
- Bonferroni contributes 0.014 (0.166 - 0.152)
- Weighting contributes 0.017 (0.183 - 0.166)
- Their interaction is approximately additive (no significant cross-term)

### Why This Matters

The bound has 0.067 of headroom to epsilon (0.25 - 0.183 = 0.067). If the selected
subset had a slightly higher failure rate (15.5% instead of 12.9%), the weighted
Bonferroni-corrected bound would hit 0.25, causing the threshold to be rejected.

In Method 1, the same threshold would have headroom of 0.084 (0.25 - 0.166). The
effective headroom loss from weighting is 0.017 — enough to flip some borderline
thresholds from "pass" to "reject."

### The |H| Penalty: Why fm1_only Matters Even More for Method 3

If we used both fM1 and fM2 (selection_mode="both"), H = 20 × 20 = 400, and:
```
delta_adj = (0.02 - 1e-5) / 400 = 0.00005
```

At alpha = 0.00005, the CP bound is extremely wide even without weighting. With
weighting (n_eff < m), it would be nearly vacuous for all practical threshold values.
This is why fm1_only mode is even more important for Method 3 than for Method 1 —
the additional bound widening from n_eff makes the Bonferroni penalty more punishing.

---

## 52. Method 3 vs Method 2: A Structural Comparison

Methods 2 and 3 take fundamentally different approaches to the same problem.
Understanding this contrast is essential for interpreting the results.

### Method 2: Change the Confidence Level

Method 2 (Option C) reduces the effective delta available for the CP bound:
```
delta_cp_m2 = delta - delta_p - delta_shift = 0.02 - 1e-5 - 0.015 = 0.005
delta_adj_m2 = 0.005 / 20 = 0.00025
```

This makes the bound wider at every threshold because the confidence level is
stricter (alpha = 0.00025 vs 0.001 in Method 1).

**What Method 2 doesn't change:** The failure counts and sample sizes. The data split
is identical to Method 1. The conformal threshold is identical. The pseudo-labels are
identical. Only the acceptance criterion is tighter.

**Effect:** Some thresholds that passed in Method 1 (where CP_upper was between 0.166
and 0.25) now fail in Method 2 (because the wider bound pushes CP_upper above 0.25).
More splits abstain. Among those that don't abstain, the selected threshold is more
conservative. This reduces efficiency and increases validity (more abstention → more
vacuous validity).

### Method 3: Change the Data Distribution

Method 3 keeps the same delta budget as Method 1 (delta_adj = 0.001) but changes
the underlying data by weighting it:

**What Method 3 changes:**
1. The conformal threshold tau_CP (weighted quantile instead of uniform)
2. The CP bound (uses n_eff instead of m)
3. The pseudo-labels (because tau_CP changed)
4. Which thresholds are selected (because the bound changed)

**What this means structurally:**

- Method 2 makes the same bounds stricter (wider).
- Method 3 makes different bounds that may be wider or narrower depending on the
  selected subset's weight distribution.

For thresholds that select high-weight questions (NQ-like TQA), Method 3's n_eff is
relatively high, and the bound may not be much wider than Method 1's. For thresholds
that select low-weight questions, n_eff is low and the bound is much wider.

### The Key Asymmetry

**Method 2 is symmetric:** It penalizes all thresholds equally (same delta reduction).
**Method 3 is asymmetric:** It penalizes NQ-unlike thresholds more (low n_eff) and
NQ-like thresholds less (high n_eff).

This asymmetry is the mechanism by which Method 3 outperforms Method 2: it uses
domain information to be conservative only where conservatism is needed (for
calibration samples that don't represent the test distribution), rather than being
uniformly conservative everywhere.

### Where Method 2 Wins

If the domain classifier is poor (CV accuracy ~55%), all weights are near-uniform,
n_eff ≈ m, and Method 3 reduces to Method 1 with slightly wider bounds (due to the
n_eff approximation being slightly conservative). In this regime, Method 2's uniform
conservatism might actually outperform Method 3's noisy non-uniform conservatism.

### Where Method 3 Wins

If the domain classifier is good (CV accuracy ~75%) and the covariate shift is
substantial, Method 3's asymmetric penalties correctly focus the algorithm on
NQ-representative thresholds. The weighted bounds for these thresholds are not much
wider than Method 1's, but they're calibrated for NQ's actual accuracy profile.

---

## 53. The Epsilon Sweep as a Diagnostic Tool

The epsilon sweep is not just a comparison — it's a diagnostic. Each epsilon value
probes a different aspect of the domain shift.

### What Each Epsilon Tests

**eps=0.25 (require 75% selected accuracy):**
Tests whether the method can control FDR-E at the standard SGen operating point.
At this epsilon, the concept shift dominates: NQ's top 5% by fM1 achieves only 69.4%
accuracy. No calibration method should achieve 98% validity here.

**The diagnostic value:** The METHOD 3 - METHOD 1 gap at eps=0.25 measures the
*covariate correction effect* in the presence of dominant concept shift.

**eps=0.30 (require 70% selected accuracy):**
The borderline regime. NQ's top 5% barely reaches 70%. Methods that correctly
calibrate for NQ's accuracy distribution should show increased validity here.

**The diagnostic value:** This is where well-calibrated bounds first become useful.
If Method 3 separates from Methods 1-2 here, it means the covariate correction is
finding the "true" operating frontier.

**eps=0.35 (require 65% selected accuracy):**
The feasible regime. NQ's top 10% achieves ~63% accuracy — close enough that
correct calibration can push the bound below epsilon.

**The diagnostic value:** If Method 3 achieves ≥ 98% validity here while Methods 1-2
don't, this is the definitive evidence that importance reweighting extends SGen's
guarantees to a lower epsilon than conservative methods.

**eps=0.40 (require 60% selected accuracy):**
The easy regime. Even NQ's top 15% approaches 60% accuracy. All methods should
show improved validity here.

**The diagnostic value:** If all three methods cross 98% by eps=0.40, the difference
between them is about *efficiency* at the crossing point, not whether they cross.

### Interpreting the Gaps

| Gap | What It Measures |
|-----|-----------------|
| M3 - M1 at eps=0.25 | Pure covariate correction effect (concept shift noise) |
| M3 - M1 at eps=0.35 | Covariate correction + feasibility benefit |
| M3 - M2 at eps=0.35 | Value of domain-aware vs naive conservatism |
| M1(eps where valid) - 0.25 | Minimum epsilon relaxation needed without correction |
| M3(eps where valid) - 0.25 | Minimum epsilon relaxation needed with correction |

---

## 54. The Weight-Correctness Relationship: A Deeper Analysis

We can predict the weight-correctness relationship from the data characteristics
even before running Method 3. This analysis uses the known statistics from cached
data.

### TQA Correctness Profile

From method1_baseline_analysis.md:
- Overall: 70.8% correct
- Mean fM1 of correct: ~-0.13 (higher confidence)
- Mean fM1 of incorrect: ~-0.28 (lower confidence)
- Correlation fM1-correctness: 0.34

### NQ Correctness Profile

- Overall: 43.1% correct
- Mean fM1 of correct: ~-0.15
- Mean fM1 of incorrect: ~-0.30
- Correlation fM1-correctness: 0.32

### What the Domain Classifier Sees

The classifier distinguishes TQA from NQ based on 384-dimensional sentence embeddings.
It does NOT see fM1, fM2, or correctness — only the question text (via embedding).

The questions that the classifier labels as "NQ-like" (high P(NQ|x), high weight)
are TQA questions whose *text* resembles NQ questions. This is independent of the
model's accuracy on those questions.

### The Indirect Path

However, there IS an indirect relationship:

1. Question topic/style → embedding → classifier label (NQ-like or TQA-like)
2. Question topic/style → model accuracy (NQ-style topics are harder)

If the same topic/style features that make a TQA question "NQ-like" also make it
harder for GPT-4o-mini, then weight and accuracy will be negatively correlated.

**Expected strength:** Weak negative. The embedding captures surface-level linguistic
features (vocabulary, syntax, question structure) which partially correlate with
topic difficulty. But many factors affect accuracy besides question style — the specific
factual knowledge required, the answer format, etc.

### Why Even Weak Negative Correlation Helps

Even a weak negative correlation (r ≈ -0.10) means the weighted calibration
oversamples harder TQA questions. This shifts the effective calibration distribution
toward NQ's difficulty level, making the conformal threshold and PAC bounds more
conservative for the right reasons.

Contrast with Method 2, which is conservative uniformly — it doesn't "know" which
TQA questions are harder. Method 3 concentrates its conservatism on the questions
that matter most for NQ generalization.

### What If the Correlation is Positive?

If high-weight TQA questions are actually EASIER (corr(weight, correctness) > 0),
it would mean the classifier identifies NQ-like features that correlate with
easiness. This would be counterproductive — the weighted calibration would be more
optimistic than the uniform one, and the bounds would underestimate NQ's actual
failure rate.

In this scenario, Method 3 would be WORSE than Method 1. The validity would decrease
because the bounds are too tight (overconfident).

**Likelihood:** Low. The NQ-TQA accuracy gap (43.1% vs 70.8%) is large and driven by
fundamental differences in question type. It's very unlikely that NQ-like TQA
questions are easier than average TQA questions.

---

## 55. Mathematical Derivation: Weighted Quantile Correctness

This section proves that the weighted conformal threshold produces valid pseudo-labels
under the covariate shift assumption.

### Setup

Let S_1, ..., S_n be entailment scores for correct answers in Z_E. Let w_1, ..., w_n
be their importance weights. Define the weighted quantile:

```
tau_CP^w = inf { t : Σ w_i × 1{S_i ≤ t} / Σ w_i ≥ epsilon_e }
```

### Claim

Under the covariate shift assumption P_test(Y|X) = P_cal(Y|X), the weighted quantile
tau_CP^w satisfies:

```
P_{X~P_test}(S(X) ≤ tau_CP^w | Y=correct) ≈ epsilon_e
```

That is, among test-distribution correct answers, approximately epsilon_e fraction
have score below tau_CP^w (are missed by pseudo-labeling).

### Proof Sketch

1. The weighted empirical distribution Σ w_i × delta(S_i) / Σ w_i approximates the
   score distribution under P_test (this is the importance sampling identity).

2. The quantile of this weighted distribution approximates the quantile of the test
   distribution.

3. Therefore, the fraction of test correct answers with S < tau_CP^w is approximately
   epsilon_e.

### Where This Breaks

**Under concept shift:** If P_test(Y|X) ≠ P_cal(Y|X), the entailment score
distribution conditional on correctness also changes. A TQA correct answer with
embedding x might have a different score distribution than an NQ correct answer with
the same embedding x (because the model's confidence patterns differ across domains).
In this case, the weighted quantile is biased.

**With estimated weights:** The importance sampling identity requires exact weights
w_i = P_test(x_i) / P_cal(x_i). Our logistic regression provides estimates ŵ_i.
The estimation error introduces bias proportional to the weight estimation error.

### Practical Implication

The weighted conformal threshold is a better approximation of the NQ-appropriate
threshold than the unweighted one, but it's not exact. The residual error comes from:
1. Concept shift (the dominant source in our setting)
2. Weight estimation error (smaller, controlled by the classifier quality)
3. Finite sample effects (smaller still, n ≈ 387 correct in Z_E)

---

## 56. Method 3's Interaction with the Calibration Direction

A subtle but important question: does the calibration direction matter for Method 3?

### Current Setup: TQA (cal) → NQ (test)

- TQA has higher accuracy (70.8%) and stronger fM1-correctness correlation (0.34)
- We reweight TQA samples to look like NQ
- High-weight TQA samples are NQ-like (probably harder)
- The weighted calibration is a harder version of TQA, approximating NQ's difficulty

### Hypothetical: NQ (cal) → TQA (test)

If we calibrated on NQ and tested on TQA:
- NQ has lower accuracy (43.1%) and weaker fM1-correctness correlation (0.32)
- We'd reweight NQ samples to look like TQA
- High-weight NQ samples would be TQA-like (probably easier)
- The weighted calibration would be an easier version of NQ, approximating TQA's difficulty
- Since TQA is easier, the bounds would be tighter (fewer failures at any threshold)
- Method 1 already achieves 100% TQA validity, so Method 3 wouldn't help

### Why TQA→NQ is the Right Direction

1. **The problem only exists in one direction.** Method 1 achieves 100% validity on TQA
   (in-domain) and 12.4% on NQ (shifted). There's nothing to fix in the TQA→TQA direction.

2. **Calibrating on the easier domain is standard practice.** In deployment, you'd
   calibrate on the domain where you have labeled data (typically the easier, more
   common domain) and deploy on new domains.

3. **This direction tests the interesting question.** Can we use a well-calibrated
   easier domain to make valid predictions about a harder domain?

### A Devil's Advocate Concern

If we calibrate on TQA and the domain classifier says "these TQA questions look like
NQ," we're essentially selecting harder TQA questions to represent NQ. But the hardness
of TQA questions isn't the same as the hardness of NQ questions — TQA hard questions
might be hard for different reasons (obscure trivia facts vs. ambiguous search queries).

The importance weights correct for P(X) differences but not for the *nature of difficulty*
conditional on X. Two questions with the same embedding can have different difficulty
profiles. This is another manifestation of concept shift.

---

## 57. Comparison with Alternative Approaches Not Implemented

### Approach A: Conditional Boosting (Cherian et al., NeurIPS 2024)

Corollary A.1 of Cherian et al. shows that including domain-related features (embedding
distance, predicted domain) in the conformal scoring function F provides conditional
coverage under covariate shift "for free."

**Why we didn't implement it:** SGen uses a different framework (FDR-E decomposition +
PAC bounds) than Cherian et al. (conformal prediction with conditional boosting).
Adapting conditional boosting to SGen would require fundamental algorithm changes,
not just weight modifications. It's a different paper, not a modification of SGen.

**How it compares:** Conditional boosting is potentially more powerful because it
adapts the scoring function, not just the calibration weights. But it requires
differentiable scoring functions and a richer function class F, which adds complexity.

### Approach B: Filter-and-Reweight (Wang et al., 2025)

Algorithm 3 of Wang et al. filters calibration samples by embedding similarity to the
test distribution, then reweights by softmax-normalized similarity. No formal guarantee
but empirically competitive.

**Why we didn't implement it:** It doesn't provide PAC guarantees, which is the core
requirement of our project. Also, it operates in the coverage framework, not FDR-E.

**How it compares:** Filter-and-reweight is simpler (no classifier needed) but less
principled. The softmax normalization is a heuristic without theoretical backing.

### Approach C: Better Base Model

Using a larger LLM (e.g., GPT-4o or GPT-4) would increase NQ accuracy from
43.1% toward 60-70%, reducing the concept shift gap. This would make eps=0.25
feasible and Method 3's covariate correction sufficient.

**Why we didn't implement it:** The 70B model requires 4× more VRAM (~64GB) and ~10×
more generation time. On our single A6000 (48GB), it won't fit in fp16. More
importantly, changing the model changes the research question from "can calibration
techniques fix domain shift" to "does a bigger model reduce domain shift," which is
less interesting methodologically.

### Approach D: Domain-Specific Fine-Tuning

Fine-tuning on a small NQ subset would directly reduce concept shift. This is
the standard domain adaptation approach in ML.

**Why we didn't implement it:** Same as Approach C — it changes the research question.
Our project is about calibration-time techniques that don't modify the model. Also,
fine-tuning requires labeled NQ data, which defeats the purpose of transfer from TQA.

---

## 58. Numerical Stability Analysis

The weighted Clopper-Pearson implementation has several potential numerical issues.
This section documents each and its mitigation.

### Issue 1: beta_dist.ppf Near Boundaries

`scipy.stats.beta.ppf(p, a, b)` can produce inaccurate results when a or b are very
small (<0.01) or very large (>10000).

**In our setting:**
- a = failures_eff + 1 ≥ 1.0 (always safe)
- b = n_eff - failures_eff ≥ 0.001 (guarded by our clamp)
- p = 1 - alpha = 0.999 (close to 1, but scipy handles this well)

**Risk: Low.** failures_eff ranges from 0 to ~300, and n_eff ranges from 5 to ~1500.
These are well within scipy's accurate regime.

### Issue 2: Weight Normalization Precision

Weights are normalized as: `w = clipped * (n / clipped.sum())`. If clipped.sum() is
very small (all weights near zero), the normalized weights could overflow.

**In our setting:** Weights before clipping are p_hat / (1 - p_hat) with p_hat ∈
[0.01, 0.99]. The minimum possible weight is 0.01/0.99 ≈ 0.010. With 3,610 weights,
the minimum possible sum is 0.010 × 3,610 = 36.1. After clipping, the sum is larger.
Division by 36.1 and multiplication by 3,610 gives max normalized weight ≈ 100. This
is fine for float64 (our working precision).

**Risk: None.**

### Issue 3: Floating Point in Cumulative Sum

`_weighted_conformal_threshold` uses `np.cumsum(sorted_weights) / sorted_weights.sum()`.
If the weights span many orders of magnitude, the cumulative sum may lose precision
for small weights.

**In our setting:** After clipping and normalization, weights range from ~0.3 to ~4.0.
The dynamic range is < 15:1, well within float64's 15-digit precision.

**Risk: None.**

### Issue 4: n_eff = 0

If all weights in the selected subset are zero (shouldn't happen after normalization),
n_eff = 0^2 / 0 = NaN.

**Mitigation:** The normalization ensures all weights are positive (minimum raw weight
= 0.010, scaled by n/sum). So this cannot happen.

### Issue 5: Degenerate Grid

If Z_U's fM1 values are all identical, `_build_percentile_grid` returns a single value,
H = 1, and delta_adj = delta_cp (no Bonferroni penalty). The grid search checks one
threshold.

**In our setting:** TQA fM1 ranges from -0.896 to -0.000 (from cached data). With
1,895 Z_U samples, the 20-point percentile grid will have 20 distinct values (checked:
`np.unique` on real data produces ≥ 19 unique values).

**Risk: None.**

---

## 59. Actual Results

**Status: Complete.** Method 3 and epsilon sweep runs finished April 6, 2026. All numbers
below are from `importance_weighted_results.json` and `epsilon_sweep_results.json`, validated
against run logs (`method3_run.out`, `epsilon_sweep_run.out`). Runtime: 12.2 seconds (Method 3)
+ 28.4 seconds (epsilon sweep).

### 59a. Classifier Diagnostics

**Domain classifier: logistic regression on 384-dim all-MiniLM-L6-v2 embeddings.**

- 5-fold CV accuracy: **91.7% (+/- 0.8%)**
- This is far higher than the 70-80% we anticipated (Section 16)
- The classifier nearly perfectly separates TQA from NQ questions

**What 91.7% accuracy means:** In a balanced binary classification (3,610 TQA + 3,610 NQ),
random guessing gives 50%. An accuracy of 91.7% means the embedding space has almost no
overlap between the two domains. Of 7,220 questions, only ~598 are misclassified — these
are the questions where the two domains "look alike" in embedding space.

**Why this is a problem, not a feature:** The density ratio trick computes w(x) =
p̂(x)/(1 - p̂(x)) where p̂ is the classifier's predicted probability that x belongs to the
test domain (NQ). When the classifier is 91.7% accurate:
- For typical TQA questions: p̂ ≈ 0.05, so w ≈ 0.05/0.95 = 0.053
- For TQA questions that look like NQ: p̂ ≈ 0.90, so w ≈ 0.90/0.10 = 9.0
- The weight ratio between "NQ-like" and "TQA-like" calibration points is ~170:1

This extreme ratio means the weighted calibration is dominated by a small number of
high-weight TQA questions, collapsing the effective sample size.

### 59b. Weight Statistics

**Full weight distribution (3,610 calibration weights, after 95th percentile clipping
and normalization to sum = n):**

| Statistic | Value |
|-----------|-------|
| Minimum | 0.041 |
| Median | 0.332 |
| Maximum | 5.692 |
| Std dev | 1.498 |
| Raw max (before clip) | 32.659 |
| Clip threshold (95th pctl) | 1.403 |

**Effective sample size:**
- n_eff = (Σw)² / (Σw²) = **1,112.5 / 3,610 (30.8%)**
- Mean n_eff across splits (within calibration subset): **584.3** (out of 2,527 cal points)
- Mean n_eff for selected subsets (non-vacuous splits): **283.7**

**What these numbers mean:**

The n_eff of 1,112.5 means that 3,610 weighted TQA calibration points carry the same
statistical information as 1,112 unweighted points. We lose 69.2% of our effective data
to the reweighting. This is the Kish design effect in action:

```
n_eff = (Σw)² / (Σw²) = n² / (n × Σ(w_i/w̄)²) = n / (1 + CV²(w))
```

With CV(w) = std/mean = 1.498/1.0 = 1.498, we get n_eff = 3610 / (1 + 1.498²) = 3610 / 3.244
≈ 1,113. This matches the computed value exactly.

**Weight distribution shape:** The median weight (0.332) is far below the mean (1.0),
indicating a right-skewed distribution. Most TQA questions get low weight (they don't look
like NQ), while a small fraction get high weight (they do look like NQ). After clipping at
the 95th percentile (1.403), the maximum raw weight of 32.659 is capped to 5.692 (normalized).
Without clipping, n_eff would be even lower — roughly 400 (11%).

### 59c. Method 3 at eps=0.25 — The Central Result

#### Headline Numbers

| Metric | TQA (in-domain) | NQ (shifted) |
|--------|-----------------|--------------|
| Validity rate | **100.0%** | **68.8%** |
| Mean FDR-E | 0.0532 ± 0.0801 | 0.1065 ± 0.1602 |
| Mean efficiency | 14.2% ± 22.4% | 8.1% ± 13.5% |

**Target validity: ≥ 98% (1 - δ = 0.98). NQ validity of 68.8% does not meet the target.**

#### Comparison with Methods 1-2

| Method | NQ Validity | NQ FDR-E | NQ Efficiency |
|--------|-------------|----------|---------------|
| M1 (Vanilla SGen) | 12.4% | 0.3015 ± 0.1176 | 22.9% ± 10.9% |
| M2 (Conservative, frac=0.75) | 22.0% | 0.2604 | 18.5% |
| **M3 (DS-SGen)** | **68.8%** | **0.1065 ± 0.1602** | **8.1% ± 13.5%** |

Method 3 improves NQ validity from 12.4% → 68.8% (+56.4 percentage points). This is a
substantial improvement but falls 29.2pp short of the 98% target.

#### The Vacuous Validity Problem

**This 68.8% is entirely vacuous validity.**

The split-level breakdown reveals the mechanism:

| Split Category | Count | Fraction | NQ Valid? |
|----------------|-------|----------|-----------|
| Vacuous (no threshold found, eff=0) | 344 | 68.8% | Yes (trivially) |
| Non-vacuous (threshold found, eff>0) | 156 | 31.2% | **0/156 valid** |
| **Total** | **500** | **100%** | **68.8%** |

**Every single non-vacuous split fails the FDR-E guarantee on NQ.** The 68.8% validity
comes entirely from splits where the weighted bounds are so wide that no threshold passes
the grid search, causing the algorithm to abstain entirely (select nothing). A split that
selects nothing has FDR-E = 0 by convention, which is ≤ 0.25, so it's "valid" — but
useless.

This is Failure Mode 1 from Section 32: **classifier too accurate → extreme weights →
n_eff collapse → bounds too wide → mass abstention.**

#### Non-Vacuous Split Analysis

The 156 splits that do find a threshold reveal the concept shift problem:

| Metric | Mean | Std | Min | Max | Median |
|--------|------|-----|-----|-----|--------|
| NQ FDR-E | 0.3413 | 0.0466 | 0.2550 | 0.4800 | 0.3249 |
| NQ Efficiency | 26.0% | 10.9% | 9.5% | 67.8% | — |
| n_selected | 937.9 | — | 342 | 2,448 | — |
| n_eff_selected | 283.7 | 83.4 | — | — | — |

**Key observation:** The minimum NQ FDR-E across all 156 non-vacuous splits is 0.2550 —
barely above the target of 0.25. Not a single split comes close to meeting the guarantee.
The median FDR-E of 0.3249 means the typical non-vacuous split has a 32.5% error rate
among selected questions, far above the 25% target.

**FDR-E distribution across hypothetical epsilon thresholds:**

| If epsilon were... | Valid non-vacuous splits | Fraction |
|--------------------|-------------------------|----------|
| 0.25 | 0/156 | 0.0% |
| 0.30 | 49/156 | 31.4% |
| 0.35 | 93/156 | 59.6% |
| 0.40 | 135/156 | 86.5% |
| 0.50 | 156/156 | 100.0% |

This shows that at eps=0.30, about a third of non-vacuous splits would pass. At eps=0.40,
most would pass. The concept shift imposes a floor of ~25.5% FDR-E on NQ — the model
simply gets NQ questions wrong at a rate that makes eps=0.25 impossible.

#### Threshold Statistics

- Splits with threshold found: **156/500** (31.2%)
- Splits with no threshold (tau1 = None): **344/500** (68.8%)
- tau1 (when found): mean = -0.0351, std = 0.0166, range = [-0.1079, -0.0099]

The thresholds that are found are very aggressive (close to 0.0 on the log-scale fM1).
A tau1 of -0.035 means: select questions where log(max self-consistency) ≥ -0.035,
i.e., max self-consistency ≥ exp(-0.035) ≈ 0.966. This is extremely selective — only
questions where all 5 sampled responses nearly agree are selected.

#### Comparison: Why M3 Has More Vacuous Splits Than M1

| Metric | M1 | M3 |
|--------|----|----|
| Vacuous splits (NQ) | 62/500 (12.4%) | 344/500 (68.8%) |
| Non-vacuous splits | 438/500 (87.6%) | 156/500 (31.2%) |
| Valid non-vacuous | 0/438 (0.0%) | 0/156 (0.0%) |
| Non-vacuous mean FDR-E | 0.3442 | 0.3413 |

M1 finds thresholds in 87.6% of splits because the unweighted bounds are tighter (full
n = 2,527 cal points). M3 finds thresholds in only 31.2% because the weighted bounds use
n_eff ≈ 584 (within-split), making them ~2x wider. But both methods produce the same
non-vacuous FDR-E (0.3442 vs 0.3413) — the reweighting shifts which thresholds are
selected but doesn't change the fundamental NQ accuracy distribution.

**The reweighting helps the algorithm "know" it can't meet the guarantee (abstain more),
but it can't make the guarantee achievable.** This is the signature of concept shift
dominance.

#### TQA (In-Domain) Performance

TQA results confirm the method works correctly on the calibration domain:

| Metric | M1 TQA | M3 TQA |
|--------|--------|--------|
| Validity | 100% | 100% |
| Vacuous splits | 62/500 | 344/500 |
| Non-vacuous FDR-E | 0.1704 ± 0.0246 | 0.1704 ± 0.0246 |
| Non-vacuous efficiency | 45.4% ± 13.7% | 45.4% ± 13.7% |

**TQA vacuous/non-vacuous counts are identical to NQ** because the same threshold is
applied to both (the algorithm finds one threshold per split, then evaluates on both
domains). The fact that TQA achieves 100% validity in all cases confirms that the
guarantee holds on the calibration domain — it's only the shifted domain where it breaks.

### 59d. Epsilon Sweep Results

**Setup:** 4 epsilon values × 3 methods × 500 splits each. Method 2 uses Option C with
frac=0.75 (the best-performing conservative variant). Method 3 weights are computed once
and reused across all epsilon values (weights depend on embeddings and classifier, not
epsilon).

#### Full Results Table

| Epsilon | M1 Valid | M1 FDR-E | M1 Eff | M2 Valid | M2 FDR-E | M2 Eff | M3 Valid | M3 FDR-E | M3 Eff |
|---------|----------|----------|--------|----------|----------|--------|----------|----------|--------|
| 0.25 | 12.4% | 0.3015 | 22.9% | 22.0% | 0.2604 | 18.5% | 68.8% | 0.1065 | 8.1% |
| 0.30 | 0.0% | 0.4593 | 59.9% | 0.0% | 0.4508 | 56.7% | 11.0% | 0.3946 | 46.1% |
| 0.35 | 0.0% | 0.5335 | 88.0% | 0.0% | 0.5277 | 85.9% | 0.2% | 0.5114 | 80.0% |
| 0.40 | 0.0% | 0.5688 | 100.0% | 0.0% | 0.5688 | 100.0% | 0.0% | 0.5637 | 98.2% |

**TQA (in-domain) results for reference — all achieve 100% validity:**

| Epsilon | M1 TQA FDR-E | M1 TQA Eff | M3 TQA FDR-E | M3 TQA Eff |
|---------|--------------|------------|--------------|------------|
| 0.25 | 0.1472 | 40.8% | 0.0532 | 14.2% |
| 0.30 | 0.2204 | 78.7% | 0.1903 | 64.2% |
| 0.35 | 0.2676 | 94.7% | 0.2532 | 90.2% |
| 0.40 | 0.2911 | 99.9% | 0.2877 | 99.1% |

#### The Counterintuitive Pattern: Validity Drops as Epsilon Increases

At eps=0.25, M1 achieves 12.4% NQ validity. At eps=0.30, M1 drops to 0.0%. This is
counterintuitive — a looser target should be easier to meet, not harder. The explanation
is the interaction between epsilon and abstention:

**At eps=0.25:** The grid search is strict. Many splits (438/500 for M1) find thresholds
that pass the CP bound, but 62 splits don't find any valid threshold and abstain. The 62
abstaining splits are "valid" (FDR-E = 0 ≤ 0.25). The 438 non-abstaining splits all fail
on NQ. Total validity = 62/500 = 12.4%.

**At eps=0.30:** The grid search is looser. Now 500/500 splits find thresholds (none
abstain). But all 500 thresholds were calibrated on TQA and systematically fail on NQ
(mean FDR-E = 0.4593, far above 0.30). Total validity = 0/500 = 0.0%.

**The mechanism:** Higher epsilon removes the vacuous validity floor by making it easier
to find thresholds. Once thresholds are found, they all fail on NQ because the concept
shift makes TQA-calibrated thresholds unreliable on NQ at any epsilon level. The validity
at eps=0.25 was never "real" — it was abstention masquerading as validity.

#### Method 3 at Higher Epsilon

Method 3 shows the same pattern but with more vacuous splits at each epsilon:

| Epsilon | M3 Vacuous Splits | M3 Non-Vacuous | M3 Validity |
|---------|-------------------|----------------|-------------|
| 0.25 | 344/500 (68.8%) | 156 | 68.8% (all vacuous) |
| 0.30 | 55/500 (11.0%) | 445 | 11.0% (all vacuous) |
| 0.35 | 1/500 (0.2%) | 499 | 0.2% (all vacuous) |
| 0.40 | 0/500 (0.0%) | 500 | 0.0% |

At eps=0.30, Method 3 retains 11.0% validity (vs 0.0% for M1/M2) because the weighted
bounds remain wide enough that 55 splits still abstain. By eps=0.35, the bounds are
loose enough that 499/500 splits find thresholds — and all fail on NQ.

#### No Method Crosses 98% Validity at Any Tested Epsilon

**None of the three methods achieve ≥ 98% validity at any of the four epsilon values.**
This is the central negative result. Even at eps=0.40 (requiring only 60% accuracy among
selected questions), every non-vacuous split fails on NQ. The model's NQ accuracy is
simply too low for any calibration-time fix to work.

At eps=0.40, M1's mean NQ FDR-E is 0.5688 — the model gets 56.9% of selected NQ questions
wrong, far above the 40% threshold. M3's mean NQ FDR-E is 0.5637 — slightly better
(reweighting helps at the margin) but still catastrophically above target.

#### Efficiency Interpretation

Efficiency increases with epsilon because more questions are selected:

- At eps=0.25: M3 selects 8.1% of NQ questions (most splits abstain entirely)
- At eps=0.40: M3 selects 98.2% of NQ questions (nearly everything is selected)

But efficiency is only meaningful when validity holds. Since no epsilon achieves valid
selection on NQ, the efficiency numbers describe the behavior of an invalid procedure —
they tell us how aggressive each method is, not how well it works.

### 59e. Prediction Validation

Comparing predictions from Sections 29-31 against actual outcomes:

#### Section 29: eps=0.25 Predictions

| Metric | Predicted | Actual | Assessment |
|--------|-----------|--------|------------|
| M3 NQ Validity | 15-30% | 68.8% | **Wrong (too low by 40pp)** |
| M3 NQ Efficiency | 10-18% | 8.1% | Close (slightly below range) |
| M3 TQA Validity | 100% | 100% | Correct |

**Why the validity prediction was wrong:** We predicted 15-30% based on the assumption
that importance reweighting would fix ~10-20pp of the covariate shift, adding genuine
(non-vacuous) validity. Instead, the mechanism was entirely different: the reweighting
made bounds wider (via n_eff collapse), causing massive abstention (344/500 splits),
which produced vacuous validity. The "improvement" is an artifact of not finding
thresholds, not of finding better thresholds.

**The prediction failure reveals a conceptual error:** We assumed reweighting would make
non-vacuous splits more valid. In fact, it makes splits more vacuous. The 68.8% is higher
than predicted but less meaningful than predicted.

#### Section 30: eps=0.35 Predictions

| Metric | Predicted | Actual | Assessment |
|--------|-----------|--------|------------|
| M1 NQ Validity | 25-45% | 0.0% | **Wrong (dramatically)** |
| M2 NQ Validity | 35-55% | 0.0% | **Wrong (dramatically)** |
| M3 NQ Validity | 50-80% | 0.2% | **Wrong (dramatically)** |
| M3 NQ Efficiency | 5-15% | 80.0% | **Wrong (opposite direction)** |

**Why all predictions were dramatically wrong:** The predictions assumed that higher
epsilon would increase genuine validity (more thresholds passing). Instead, higher epsilon
increases the number of non-vacuous splits (thresholds are easier to find), and since
ALL non-vacuous splits fail on NQ, validity drops toward 0%.

The predictions failed to model the key dynamic: at eps=0.25, the high abstention rate
creates an artificial validity floor. As epsilon rises, this floor collapses because
fewer splits abstain. The predictions treated epsilon as a "difficulty knob" — easier
target = more validity. In reality, epsilon is a "threshold-finding knob" — easier
target = more thresholds found = more chances to fail on NQ.

#### Section 31: Epsilon Sweep Predictions

| Epsilon | M3 Predicted | M3 Actual | Assessment |
|---------|--------------|-----------|------------|
| 0.25 | 15-30% | 68.8% | Wrong (vacuous inflation) |
| 0.30 | 35-55% | 11.0% | Wrong (vacuous collapse) |
| 0.35 | 50-80% | 0.2% | Wrong (fully non-vacuous, all fail) |
| 0.40 | 70-95% | 0.0% | Wrong (all thresholds fail) |

**Key prediction: "Method 3 crosses 98% between eps=0.35 and eps=0.45."**
**Actual: Method 3 never crosses 98% at any tested epsilon. Prediction was wrong.**

**Scenario match from Section 31:** The actual results match **Scenario A ("Method 3
barely improves over Method 1")** in substance, though the specific mechanism (vacuous
validity inflation) was not anticipated. The concept shift dominates at all epsilon values,
not just eps=0.25.

### 59f. Diagnostic Checklist Results

Checking each item from Section 33:

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Classifier CV accuracy 65-80% | 65-80% | 91.7% | **FAIL** — too high |
| n_eff/n ratio > 50% | > 50% | 30.8% | **FAIL** — too low |
| Weight max/min < 100 | < 100 | 5.692/0.041 = 139 | **FAIL** — too extreme |
| TQA validity = 100% | 100% | 100% | Pass |
| Non-vacuous M3 FDR-E < M1 FDR-E | Lower | 0.3413 vs 0.3442 | Pass (barely) |
| M3 NQ validity > M1 NQ validity | Higher | 68.8% > 12.4% | Pass (but vacuous) |
| M3 NQ efficiency > 5% | > 5% | 8.1% | Pass (barely) |

**Summary:** The diagnostic checklist flags the root cause immediately — the classifier is
too accurate (91.7% vs the expected 65-80%), causing n_eff collapse (30.8% vs > 50%
expected). Everything downstream follows from this: extreme weights, wide bounds, mass
abstention, vacuous validity.

### 59g. The Domain Shift vs. Domain Change Distinction

**This is the most important interpretive finding of the entire project.**

The results reveal that TQA → NQ is not a domain *shift* — it is a domain *change*. The
distinction is critical:

**Domain shift (covariate shift):** P(X) changes but P(Y|X) stays the same. The model
encounters different types of questions but performs the same on any given question type.
Example: training on general clinical questions, deploying in a cardiac clinic. The
cardiac questions are a subset of medical knowledge — the model's accuracy on cardiac
topics doesn't change, it just sees more of them.

**Domain change (covariate + concept shift):** Both P(X) and P(Y|X) change. The model
encounters different types of questions AND performs differently on them. Example: training
on trivia questions, deploying on real search queries. The model's knowledge of trivia
facts doesn't transfer to factual retrieval because the question structure, expected answer
format, and underlying knowledge requirements are fundamentally different.

**Evidence that TQA → NQ is a domain change, not a domain shift:**

1. **Classifier accuracy: 91.7%.** If this were a mild covariate shift, the classifier
   would be ~60-70% (substantial overlap in question types). At 91.7%, the two domains
   are almost entirely separable — they don't share a common question space.

2. **Accuracy gap: 70.8% TQA vs 43.1% NQ.** The model's accuracy drops by 28.5 percentage
   points when moving from TQA to NQ. If this were pure covariate shift (P(Y|X) stable),
   the accuracy gap would be explained by the shift in question difficulty, and reweighting
   would restore it. Instead, NQ questions are harder *per se*, not just differently
   distributed.

3. **Non-vacuous FDR-E is identical across methods.** M1 non-vacuous FDR-E on NQ = 0.3442.
   M3 non-vacuous FDR-E = 0.3413. The reweighting doesn't improve the actual error rate
   of selected questions — it only changes which questions are selected (slightly) and
   whether the algorithm finds any threshold at all. This is the signature of concept
   shift: correcting P(X) doesn't change P(Y|X).

4. **0/156 non-vacuous splits are valid.** If the covariate correction were working, some
   non-vacuous splits should become valid (reweighted bounds should correctly estimate NQ's
   FDR-E). Instead, zero non-vacuous splits pass. The calibration-to-test gap is not a
   P(X) phenomenon.

5. **Minimum non-vacuous FDR-E is 0.2550.** Even the best-case split has an error rate
   just above epsilon = 0.25. The model's NQ accuracy imposes a hard floor on FDR-E that
   no calibration method can breach.

**What this means for DS-SGen as a method:**

DS-SGen is designed for domain *shift* (covariate shift), not domain *change*. It correctly
implements the weighted conformal prediction framework, which provably restores coverage
guarantees under covariate shift when density ratios are known exactly. Our results confirm
this theoretical limitation empirically:

- The method improves validity (12.4% → 68.8%) but through abstention, not through better
  calibration
- The method cannot make the guarantee achievable because the guarantee requires P(Y|X)
  stability, which TQA → NQ violates
- The method correctly detects that it cannot meet the guarantee (more abstention = the
  bounds are honestly reflecting the uncertainty), which is arguably better than M1's
  false confidence (finding thresholds that systematically fail)

### 59h. When Would DS-SGen Work? The Medical Analogy

DS-SGen would work well for moderate domain shifts where the underlying task similarity
is preserved. The canonical example:

**General clinical QA → Specialized cardiac clinic QA.**

| Property | TQA → NQ (our setup) | Clinical → Cardiac (ideal case) |
|----------|---------------------|---------------------------------|
| Classifier accuracy | 91.7% (nearly separable) | ~65-75% (substantial overlap) |
| n_eff ratio | 30.8% (collapsed) | ~60-80% (healthy) |
| Accuracy gap | 27.7pp (70.8% vs 43.1%) | ~5-10pp (similar knowledge base) |
| Covariate shift | Extreme (different Q styles) | Moderate (specialized subset) |
| Concept shift | Dominant (different knowledge) | Minimal (same medical knowledge) |

In the clinical → cardiac scenario:
1. Questions overlap heavily (both involve symptoms, lab values, medications, procedures)
2. The model's medical knowledge transfers well — cardiac questions are a *subset* of
   clinical knowledge, not a different kind of knowledge
3. The classifier would be ~70% accurate — enough to reweight meaningfully, not so extreme
   that weights collapse
4. n_eff would be ~65-75%, preserving most of the calibration data's statistical power
5. The reweighted bounds would correctly reflect the cardiac clinic's question distribution
6. At eps=0.25, DS-SGen would plausibly restore the PAC guarantee because the model's
   accuracy on cardiac questions is close to its accuracy on general clinical questions

**Other examples where DS-SGen should work:**
- Legal contract QA (general) → Real estate contract QA (specialized subset)
- Customer support for all products → Customer support for a specific product line
- Code generation across languages → Code generation in a specific framework
- News QA (general) → Sports news QA (topical subset)

**The common pattern:** The test domain is a *subset* or *neighbor* of the calibration
domain in the knowledge space, not a fundamentally different domain.

### 59i. Revised Interpretation: Negative Results as Contribution

The results are scientifically valuable despite — and partly because of — being negative:

**What we demonstrated:**

1. **SGen's PAC guarantee breaks catastrophically under domain shift.** At eps=0.25, NQ
   validity drops from 100% (in-domain) to 12.4% (shifted). This is the first empirical
   demonstration of this failure mode for selective generation with FDR-E control.

2. **Conservative adjustments help minimally.** Method 2's best variant improves NQ
   validity from 12.4% to 22.0% — statistically significant but practically insufficient
   for deployment.

3. **Importance reweighting correctly detects infeasibility.** Method 3's 68.8% validity
   (via abstention) is the algorithm honestly reporting that it cannot meet the guarantee
   — this is preferable to Method 1's false confidence (finding thresholds that fail).

4. **The domain change boundary is sharp.** TQA → NQ sits firmly in the "domain change"
   regime where no calibration-time fix can work. This establishes a clear boundary
   condition for the applicability of DS-SGen and similar methods.

5. **The epsilon sweep reveals the mechanism.** The counterintuitive validity-epsilon
   relationship (validity decreasing with higher epsilon) is a novel diagnostic finding
   that clarifies the interaction between abstention, threshold selection, and concept
   shift.

**What remains to be demonstrated (future work):**

DS-SGen's effectiveness on a genuine domain *shift* (moderate covariate shift, minimal
concept shift). This requires a dataset pair where:
- Classifier accuracy is 65-80% (not 91.7%)
- Model accuracy gap is < 10pp (not 27.7pp)
- The test domain is a topical subset of the calibration domain

Candidates include: medical QA subdomains, legal QA subdomains, or domain-specific
splits of existing multi-domain QA benchmarks

---

*Document generated April 4, 2026. All experiments complete as of April 6, 2026.
Setup: 3,610 TQA (calibration) + 3,610 NQ (shifted test), 500 splits, GPT-4o-mini,
DeBERTa-v2-xxlarge-mnli entailment, all-MiniLM-L6-v2 embeddings.
Method 1 (Vanilla SGen): TQA validity=100%, NQ validity=12.4%.
Method 2 (Conservative, Option C frac=0.75): NQ validity=22.0%.
Method 3 (DS-SGen): NQ validity=68.8% (entirely vacuous — 344/500 splits abstain,
0/156 non-vacuous splits valid). Classifier CV accuracy=91.7%, n_eff=30.8%.
Epsilon sweep: no method achieves ≥98% validity at any tested epsilon (0.25-0.40).
Key finding: TQA→NQ is a domain change (concept shift dominant), not a domain shift
(covariate shift only). DS-SGen correctly detects infeasibility but cannot fix it.
Future work: validate DS-SGen on genuine domain shifts (moderate covariate shift,
minimal concept shift). Updated April 6, 2026.*
