# Method 2: Conservative Threshold — Complete Analysis

**DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs**
**Anshul Kumar, Justin Luan — Carnegie Mellon University, 10-701, Spring 2026**

This document records every decision, every number, every piece of math, and every
result from the Method 2 (Conservative Threshold) implementation. It is the truth
document for the naive domain-shift fix. Numbers marked as "actual" or "from config"
are validated against code and configuration files. Numbers in the "Expected Results"
and "Worked Example" sections (Sections 32-37) are predictions or illustrative — not
measured outcomes and use hedging language ("probably", "approximately", "should").
The full pipeline completed on April 6, 2026. All results reflect the current pipeline
(3,610 TQA, 500 splits, GPT-4o-mini).

This document follows chronologically from `method1_baseline_analysis.md` and assumes
familiarity with all concepts defined there. Cross-references to Method 1 sections are
provided where relevant.

---

## Table of Contents

1. [Purpose of This Method](#1-purpose-of-this-method)
2. [The Research Question, Revisited](#2-the-research-question-revisited)
3. [What This Method Is and Is Not](#3-what-this-method-is-and-is-not)
4. [Prerequisites: Method 1 Cached Data](#4-prerequisites-method-1-cached-data)
5. [The Domain Shift Problem: Why Method 1 Breaks](#5-the-domain-shift-problem-why-method-1-breaks)
6. [The Naive Fix Idea](#6-the-naive-fix-idea)
7. [Why Three Options, Not One](#7-why-three-options-not-one)
8. [Self-Contained Architecture: Why Reimplement](#8-self-contained-architecture-why-reimplement)
9. [The Shared Foundation: SGen-Semi Split Logic](#9-the-shared-foundation-sgen-semi-split-logic)
10. [Option A: Safety Factor on Thresholds](#10-option-a-safety-factor-on-thresholds)
11. [Option A: Mathematical Derivation](#11-option-a-mathematical-derivation)
12. [Option A: Code Walkthrough](#12-option-a-code-walkthrough)
13. [Option A: Sweep Parameters and Rationale](#13-option-a-sweep-parameters-and-rationale)
14. [Option A: Expected Behavior](#14-option-a-expected-behavior)
15. [Option B: Reduced Epsilon in Grid Search](#15-option-b-reduced-epsilon-in-grid-search)
16. [Option B: Mathematical Derivation](#16-option-b-mathematical-derivation)
17. [Option B: Code Walkthrough](#17-option-b-code-walkthrough)
18. [Option B: Sweep Parameters and Rationale](#18-option-b-sweep-parameters-and-rationale)
19. [Option B: Expected Behavior](#19-option-b-expected-behavior)
20. [Option C: Delta Budget Allocation](#20-option-c-delta-budget-allocation)
21. [Option C: Mathematical Derivation](#21-option-c-mathematical-derivation)
22. [Option C: Code Walkthrough](#22-option-c-code-walkthrough)
23. [Option C: Sweep Parameters and Rationale](#23-option-c-sweep-parameters-and-rationale)
24. [Option C: Expected Behavior](#24-option-c-expected-behavior)
25. [The delta_cp <= 0 Edge Case](#25-the-delta_cp--0-edge-case)
26. [Why Evaluation Always Uses Original Epsilon](#26-why-evaluation-always-uses-original-epsilon)
27. [The Sweep Runner: Aggregation Logic](#27-the-sweep-runner-aggregation-logic)
28. [The Orchestrator: run_conservative.py](#28-the-orchestrator-run_conservativepy)
29. [Cache Loading and Validation](#29-cache-loading-and-validation)
30. [The Cache Naming Bug (Fixed)](#30-the-cache-naming-bug-fixed)
31. [Configuration: Full Hyperparameter Table](#31-configuration-full-hyperparameter-table)
32. [Worked Example: Option A with gamma=1.5](#32-worked-example-option-a-with-gamma15)
33. [Worked Example: Option B with k=2.0](#33-worked-example-option-b-with-k20)
34. [Worked Example: Option C with frac=0.50](#34-worked-example-option-c-with-frac050)
35. [Expected Results: Option A](#35-expected-results-option-a)
36. [Expected Results: Option B](#36-expected-results-option-b)
37. [Expected Results: Option C](#37-expected-results-option-c)
38. [Comparing the Three Options](#38-comparing-the-three-options)
39. [The Validity-Efficiency Tradeoff](#39-the-validity-efficiency-tradeoff)
40. [Why This Method Cannot Fully Solve Domain Shift](#40-why-this-method-cannot-fully-solve-domain-shift)
41. [The Print Summary: Output Format](#41-the-print-summary-output-format)
42. [Results Saving: What Gets Persisted](#42-results-saving-what-gets-persisted)
43. [Running Method 2](#43-running-method-2)
44. [Runtime Estimates](#44-runtime-estimates)
45. [Current Status](#45-current-status)
46. [What Method 2 Already Tells Us (in Theory)](#46-what-method-2-already-tells-us-in-theory)
47. [Connection to Method 3: What Method 2 Motivates](#47-connection-to-method-3-what-method-2-motivates)
48. [Actual Results: Clean Run (SLURM Job 6952802)](#48-actual-results-clean-run-slurm-job-6952802)
49. [Actual Results: Option A — Safety Factor](#49-actual-results-option-a--safety-factor-on-thresholds)
50. [Actual Results: Option B — Reduced Epsilon](#50-actual-results-option-b--reduced-epsilon-in-grid-search)
51. [Actual Results: Option C — Delta Budget Allocation](#51-actual-results-option-c--delta-budget-allocation)
52. [Cross-Option Comparison](#52-cross-option-comparison)
53. [Answering Appendix G Questions](#53-answering-appendix-g-questions)
54. [Issues and Fixes Applied to Method 2](#54-issues-and-fixes-applied-to-method-2)
55. [Implications for Method 3](#55-implications-for-method-3)

---

## 1. Purpose of This Method

Method 2 is the **naive fix** for the domain shift problem exposed by Method 1. It is
not the final solution — that is Method 3 (DS-SGen with importance reweighting). Method
2 serves three purposes in the paper's narrative:

**Purpose 1: Show that "being more conservative" helps validity.** By tightening the
SGen-Semi thresholds in various ways, we can recover some of the PAC validity that
domain shift destroys. This confirms that the validity failure in Method 1 is indeed
about the thresholds being too permissive for the shifted domain, not about some
fundamental breakdown in the entailment scoring or pseudo-labeling.

**Purpose 2: Show that naive conservatism is too expensive.** Every conservative option
improves validity at the cost of efficiency (the fraction of questions answered). At the
conservatism levels needed to fully restore 98% validity on TQA, the model probably
answers very few questions — possibly approaching 0%. A system that says "I don't know"
to everything is trivially valid but useless.

**Purpose 3: Motivate principled domain adaptation.** The gap between "high validity but
low efficiency" (Method 2) and "high validity AND high efficiency" (goal of Method 3)
justifies the complexity of importance reweighting. If naive conservatism worked well
enough, there would be no need for Method 3.

In short: Method 2 is the "easy wrong answer" that proves the problem requires a
principled solution.

---

## 2. The Research Question, Revisited

From Method 1 (Section 2), the core question is:

> Can a selective generation system for LLMs maintain provable PAC guarantees on its
> false discovery rate even when test queries come from a different domain than the
> calibration data?

Method 1 establishes: **No, vanilla SGen cannot.**

Method 2 asks the follow-up: **Can we fix this by simply being more conservative?**

The expected answer is: **Partially.** More conservatism improves validity, but the cost
in efficiency makes this impractical. This motivates Method 3's principled approach.

The specific experimental questions for Method 2 are:

1. How much does each conservative option improve TQA validity compared to Method 1?
2. How much efficiency is lost for each level of conservatism?
3. Is there a sweet spot where validity is restored without destroying efficiency?
4. Which conservative option provides the best validity-efficiency tradeoff?
5. Does the answer depend on the severity of domain shift?

We expect (but have not yet verified) that questions 1-2 will have clear answers, while
questions 3-5 will motivate Method 3 by showing that no conservative option achieves
the "best of both worlds."

---

## 3. What This Method Is and Is Not

### What it is

- Three independent conservative modifications to the SGen-Semi algorithm
- A parameter sweep over multiple conservatism levels for each modification
- A quantitative demonstration of the validity-efficiency tradeoff under domain shift
- The motivating failure for Method 3 (DS-SGen)
- A CPU-only computation that reuses all cached data from Method 1

### What it is not

- It is not a principled solution to domain shift. The modifications are ad hoc.
- It does not use any information about the test domain distribution.
- It does not estimate density ratios, train domain classifiers, or use embeddings.
- It does not modify the pseudo-labeling step (conformal threshold is unchanged).
- It does not change the scoring functions fM1 or fM2. The model's answers and
  confidence scores are identical to Method 1. Only the selection thresholds change.
- It is not computationally expensive. The entire Method 2 sweep should complete in
  minutes on a single CPU core.

---

## 4. Prerequisites: Method 1 Cached Data

Method 2 requires all four cached outputs from Method 1's Stages 1-3:

| Cache file | Contents | Source stage | Expected size |
|-----------|---------|-------------|--------------|
| `nq_data.json` | 3,610 NQ records (question, reference answer, idx, dataset) | Stage 1 | ~0.8 MB |
| `tqa_data.json` | 3,610 TQA records (same fields) | Stage 1 | ~2.1 MB |
| `nq_generations.json` | 3,610 generation records (greedy answer, K=5 samples, mean_logprob) | Stage 2 | ~6.3 MB (actual, from cache) |
| `tqa_generations.json` | 3,610 generation records (same fields) | Stage 2 | ~6-7 MB (estimate) |
| `nq_entailment.json` | 3,610 entailment records (fM2, entail_score, entail_label) | Stage 3 | ~1-2 MB (estimate) |
| `tqa_entailment.json` | 3,610 entailment records (same fields) | Stage 3 | ~1-2 MB (estimate) |

**Current status (actual, as of April 4, 2026):** The cache directory contains:
- `nq_data.json` — 810,547 bytes (complete, 3,610 records)
- `tqa_data.json` — 2,075,312 bytes (complete, 3,610 records)
- `nq_generations.json` — 8,208,951 bytes (complete, 3,610 records)
- `tqa_generations.json` — ~4.5 MB (partial, 2,550/3,610 records — in progress)
- `nq_entailment.json` — does not yet exist
- `tqa_entailment.json` — does not yet exist

Method 2 cannot run until all six cache files exist. Job 6951565 is currently running
on the preempt partition (7-day wall time) to complete TQA generation and then proceed
through entailment scoring.

**Important:** The cache file names use **singular** "entailment", not plural
"entailments". This was the source of a bug discussed in Section 30.

---

## 5. The Domain Shift Problem: Why Method 1 Breaks

This section recaps the theory from Method 1 Section 29, focused on what Method 2
needs to address.

### The i.i.d. assumption

SGen-Semi's PAC guarantee rests on the assumption that calibration data and test data
are drawn independently from the same distribution. Formally:

```
(X_1, Y_1), ..., (X_n, Y_n) ~ P    (calibration)
(X_{n+1}, Y_{n+1}) ~ P              (test)
```

All data points come from the same joint distribution P. Under this assumption, the
empirical error rate on the calibration set is an unbiased estimator of the true error
rate, and the Clopper-Pearson bound provides a valid upper confidence bound.

### What happens under covariate shift

When test data comes from TQA instead of NQ, we have:

```
Calibration: X ~ P_NQ(X)
Test:        X ~ P_TQA(X)
```

The conditional distributions P(Y|X) (i.e., the LLM's behavior given a question) might
be similar, but the marginal distributions over questions are different. NQ questions
are informal Google searches; TQA questions are trivia-style factual questions.

The consequence: thresholds calibrated on NQ data reflect the error distribution of
NQ-style questions. If TQA questions are harder (or easier, or differently distributed
in the (fM1, fM2) space), the calibrated thresholds may select too many (or too few)
answers, and the FDR-E guarantee breaks.

### The specific failure mode

Consider the SGen-Semi selection rule:

```
Select if fM1(x) >= tau1 AND fM2(x) >= tau2
```

If NQ and TQA have different distributions of (fM1, fM2) conditional on correctness:
- NQ questions the model gets wrong might have low fM1 (the model is uncertain).
  The threshold tau1 successfully filters them out.
- TQA questions the model gets wrong might have HIGHER fM1 (the model is confidently
  wrong about trivia questions). The same threshold tau1 fails to filter them.

This asymmetry — where the model's confidence signals have different calibration
properties across domains — is exactly what domain shift means in our context.

### What Method 2 does about it

Method 2's approach is simple: if the thresholds are too permissive for the shifted
domain, make them stricter. This is the equivalent of adding a "safety margin" to an
engineering design — you don't know exactly how bad things can get, so you build in
extra headroom.

The problem, of course, is that you don't know how much headroom you need. Too little,
and the guarantee still fails. Too much, and you reject everything. Method 2 sweeps
across multiple conservatism levels to map this tradeoff.

---

## 6. The Naive Fix Idea

The core insight behind Method 2 is that there are exactly three places in the SGen-Semi
algorithm where you can inject conservatism:

1. **The selection thresholds (tau1, tau2):** After finding the best thresholds via grid
   search, inflate them. Higher thresholds mean fewer answers pass selection, which
   reduces the chance of selecting wrong answers. This is Option A.

2. **The epsilon constraint in grid search:** Instead of accepting threshold pairs where
   `CP_upper <= epsilon`, require `CP_upper <= epsilon/k` for some k > 1. This forces
   the algorithm to find thresholds that would satisfy a *stricter* FDR-E target on the
   calibration data. When applied to shifted test data, the extra strictness might
   absorb the shift. This is Option B.

3. **The delta budget:** The Bonferroni-corrected confidence level depends on
   `delta_cp = delta - delta_p`. By reserving a fraction of delta_cp for "potential
   domain shift" (reducing the available budget), the Clopper-Pearson bounds become
   wider, requiring more evidence to accept a threshold pair. This is Option C.

Each option attacks a different part of the PAC machinery, and they have different
characteristics. Option A is a post-hoc adjustment (doesn't change the search). Option B
changes the search criterion. Option C changes the statistical confidence level.

---

## 7. Why Three Options, Not One

A natural question: why implement three separate conservative options instead of one
combined approach?

### Experimental design rationale

By isolating each conservative mechanism, we can measure:
- Which mechanism has the largest effect on validity?
- Which mechanism is most efficient (best validity improvement per unit of efficiency loss)?
- Do the mechanisms have complementary or redundant effects?

If we combined them into a single knob, we could not answer these questions.

### Coverage of the design space

Each option operates on a different part of the algorithm:

| Option | What it modifies | When it acts | Mathematical effect |
|--------|-----------------|-------------|-------------------|
| A | Selection thresholds | After grid search | Shifts decision boundary |
| B | Grid search constraint | During grid search | Restricts candidate set |
| C | Confidence level | During grid search | Widens confidence intervals |

This means they are largely orthogonal. In principle, you could combine all three (use
reduced epsilon AND reduced delta AND inflate thresholds), though our implementation
tests them independently.

### Paper narrative value

In the paper, the three options provide a natural structure: "We tried three intuitive
approaches, each with a clear mechanism, and none of them satisfactorily solves the
problem. This motivates our principled approach in Section 5 (Method 3)."

---

## 8. Self-Contained Architecture: Why Reimplement

The conservative module (`ds_sgen/conservative.py`, 479 lines) reimplements the entire
SGen-Semi split logic instead of importing and modifying `ds_sgen/sgen_semi.py`. This
was a deliberate architectural decision.

### Why not inherit or monkey-patch

The SGen-Semi split logic in `sgen_semi.py` is a single function (`_run_single_split`)
with approximately 60 lines of core logic. The three conservative options each inject
changes at different points in this logic:

- Option A injects after line 186 (threshold inflation after grid search)
- Option B modifies the comparison at line 146 (`cp_upper <= epsilon_effective`)
- Option C modifies the delta computation at line 129 (`delta_cp = delta - delta_p - delta_shift`)

To support all three options through the original function, we would need either:
1. Add three optional parameters with if-else branches (clutters the baseline code)
2. Create a subclass with method overrides (Python's class system adds complexity for
   a fundamentally procedural algorithm)
3. Use function callbacks for the three injection points (over-engineered)

All of these are worse than simply copying the 60 lines and adding the modifications
inline. The resulting code is self-contained, readable, and independently testable.

### The cost of duplication

The duplicated code is the helper functions: `_merge_records`, `_compute_conformal_threshold`,
`_build_percentile_grid`, `_clopper_pearson_upper`. These are pure mathematical functions
with no business logic — they compute the same thing regardless of context. If a bug
were found in one copy, it would need to be fixed in both.

This cost is acceptable because:
1. The functions are small (5-10 lines each) and mathematically well-defined
2. The formulas come directly from the paper and are unlikely to change
3. The project is a research codebase, not a production library
4. Self-containment makes it possible to verify Method 2 without reading Method 1 code

### File structure

```
ds_sgen/conservative.py   (479 lines)
├── Imports and logger setup
├── _merge_records()                    — copied from sgen_semi.py
├── _compute_conformal_threshold()      — copied from sgen_semi.py
├── _build_percentile_grid()            — copied from sgen_semi.py
├── _clopper_pearson_upper()            — copied from sgen_semi.py
├── _run_single_split()                 — modified: three conservative knobs
├── _run_sweep()                        — new: runs n_splits with fixed overrides
├── run_conservative_experiment()        — new: orchestrates all three option sweeps
└── print_conservative_summary()         — new: formatted output tables
```

---

## 9. The Shared Foundation: SGen-Semi Split Logic

Before examining the three conservative options, we trace the shared split logic that
they all modify. This mirrors Method 1 Section 14, but focuses on the specific code in
`conservative.py` rather than `sgen_semi.py`.

### Function signature

```python
def _run_single_split(
    nq_merged: list[dict],
    tqa_merged: list[dict],
    split_seed: int,
    sgen_cfg: dict,
    *,                              # keyword-only after this
    epsilon_effective: float | None = None,  # Option B
    delta_shift: float = 0.0,               # Option C
    tau_safety_factor: float = 1.0,          # Option A
) -> dict:
```

The three conservative parameters are keyword-only with defaults that reduce to the
vanilla Method 1 behavior:
- `epsilon_effective=None` → uses `sgen_cfg["epsilon"]` (no reduction)
- `delta_shift=0.0` → no delta budget reserved (standard Bonferroni)
- `tau_safety_factor=1.0` → no threshold inflation (log(1.0) = 0, 1.0 * tau2 = tau2)

This means calling `_run_single_split(nq, tqa, seed, cfg)` with no keyword arguments
produces **exactly** the same result as Method 1's `_run_single_split()`.

### Step-by-step trace (conservative.py lines 103-233)

**Step 1: Parameter extraction (lines 103-113)**

```python
epsilon = sgen_cfg["epsilon"]           # 0.25
if epsilon_effective is None:
    epsilon_effective = epsilon          # default: no reduction
delta = sgen_cfg["delta"]              # 0.02
delta_p = sgen_cfg["delta_p"]          # 1e-5
cal_frac = sgen_cfg["cal_frac"]        # 0.70
zu_frac = sgen_cfg["zu_frac"]          # 0.75
epsilon_e = sgen_cfg["epsilon_e"]      # 0.10
n_grid = sgen_cfg["n_grid"]            # 50
```

Note the distinction: `epsilon` is always 0.25 (the original target). `epsilon_effective`
may be smaller than epsilon (Option B). This distinction is critical — see Section 26.

**Step 2: NQ split into calibration and test (lines 115-124)**

```python
indices = rng.permutation(n_nq)
cal_size = int(np.floor(n_nq * cal_frac))   # floor(3610 * 0.70) = 2527
cal_idx = indices[:cal_size]
test_idx = indices[cal_size:]
```

Identical to Method 1. The same seeds produce the same splits, ensuring that for a
given split_seed, Method 1 and Method 2 operate on the same calibration/test partition.
This is essential for fair comparison — any difference in results is due to the
conservative modifications, not different data splits.

With 3,610 NQ records:
- `cal_size = floor(3610 * 0.70) = 2527`
- `test_size = 3610 - 2527 = 1083`

**Step 3: Calibration split into Z_U and Z_E (lines 126-129)**

```python
zu_size = int(np.floor(len(cal_data) * zu_frac))  # floor(2527 * 0.75) = 1895
z_u = cal_data[:zu_size]                           # unlabeled (pseudo-labeled)
z_e = cal_data[zu_size:]                           # labeled (used for conformal)
```

With 2,527 calibration records:
- `zu_size = floor(2527 * 0.75) = 1895`
- `ze_size = 2527 - 1895 = 632`

**Step 4: Conformal pseudo-labeling (lines 131-137)**

```python
ze_scores = np.array([r["entail_score"] for r in z_e])
tau_cp = _compute_conformal_threshold(ze_scores, epsilon_e)

for r in z_u:
    r["pseudo_label"] = 1 if r["entail_score"] >= tau_cp else 0
```

This step is **not modified by any conservative option**. The conformal threshold is
computed the same way as Method 1. The pseudo-labels are identical. This is deliberate:
the conservative options operate downstream of pseudo-labeling, on the threshold
selection and evaluation steps.

Conformal threshold formula (see Method 1 Section 16 for derivation):
```
k = ceil((632 + 1) * (1 - 0.10)) = ceil(633 * 0.90) = ceil(569.7) = 570
tau_cp = sorted_ze_scores[569]
```

**Step 5: Grid search (lines 139-187)**

This is where Options B and C inject their modifications. The grid construction is
identical to Method 1:

```python
tau1_grid = _build_percentile_grid(zu_fM1, n_grid)  # up to 50 unique values
tau2_grid = _build_percentile_grid(zu_fM2, n_grid)  # up to 50 unique values
H = len(tau1_grid) * len(tau2_grid)                  # up to 2500
```

The delta computation is where Option C acts:

```python
delta_cp = delta - delta_p - delta_shift   # Option C: reduces budget
delta_adj = delta_cp / H                   # Bonferroni correction
```

And the acceptance criterion is where Option B acts:

```python
if cp_upper <= epsilon_effective:  # Option B: tighter constraint
```

**Step 6: Threshold inflation (lines 189-192)**

This is where Option A acts, after the grid search completes:

```python
if best_tau1 is not None and tau_safety_factor != 1.0:
    best_tau1 = best_tau1 + np.log(tau_safety_factor)
    best_tau2 = min(best_tau2 * tau_safety_factor, 1.0)
```

**Step 7: Evaluation (lines 194-217)**

Evaluation uses the original epsilon, not epsilon_effective:

```python
valid = fdr_e <= epsilon   # always original epsilon
```

This is the same as Method 1. See Section 26 for why.

---

## 10. Option A: Safety Factor on Thresholds

### The idea

After the grid search finds the best (tau1, tau2) pair, inflate both thresholds by a
safety factor gamma. Higher thresholds mean fewer answers are selected, which should
reduce FDR-E.

### The mechanism

```
tau1_new = tau1_old + log(gamma)    (fM1 is log-scale)
tau2_new = min(tau2_old * gamma, 1.0)  (fM2 is in [0, 1])
```

The asymmetry in how gamma is applied reflects the different scales of fM1 and fM2:

**fM1 (mean log-probability):** This is a log-scale quantity. fM1 values are typically
negative (log-probabilities are ≤ 0). A higher fM1 means higher confidence. Adding
`log(gamma)` to tau1 raises the threshold on a log scale, which is equivalent to
requiring the raw probability to be gamma times higher:

```
fM1 >= tau1 + log(gamma)
exp(fM1) >= exp(tau1) * gamma
P(answer) >= P_threshold * gamma
```

So gamma = 1.5 means "require 1.5x the probability threshold." gamma = 2.0 means
"require 2x the probability threshold."

Note: log here is the natural logarithm (np.log). For gamma = 1.2, log(1.2) ≈ 0.182.
For gamma = 1.5, log(1.5) ≈ 0.405. For gamma = 2.0, log(2.0) ≈ 0.693.

**fM2 (self-consistency score):** This is in [0, 1], where 1 means all K=5 sampled
answers agree with the greedy answer. Multiplying tau2 by gamma raises the consistency
threshold proportionally. The `min(..., 1.0)` cap ensures the threshold cannot exceed 1.

For example, if tau2 = 0.6 and gamma = 1.5, then tau2_new = min(0.6 * 1.5, 1.0) =
min(0.9, 1.0) = 0.9. This requires 90% consistency instead of 60%.

If tau2 = 0.8 and gamma = 1.5, then tau2_new = min(0.8 * 1.5, 1.0) = min(1.2, 1.0) =
1.0. This requires 100% consistency (all samples must agree), which is extremely strict.

---

## 11. Option A: Mathematical Derivation

### Setup

Let the grid search produce optimal thresholds (tau1*, tau2*) that satisfy the PAC
constraint on the calibration data:

```
CP_upper(failures(tau1*, tau2*), selected(tau1*, tau2*), delta_adj) <= epsilon
```

where CP_upper is the Clopper-Pearson upper bound.

### Effect of the safety factor

The inflated thresholds define a **subset** of the original selected set:

```
S_original = {x : fM1(x) >= tau1* AND fM2(x) >= tau2*}
S_inflated = {x : fM1(x) >= tau1* + log(gamma) AND fM2(x) >= min(tau2* * gamma, 1.0)}
```

Since gamma >= 1, both thresholds are raised (or equal for gamma = 1), so:

```
S_inflated ⊆ S_original
```

Every answer selected under the inflated thresholds was also selected under the original
thresholds. This is a pure restriction — we never add new answers, only remove marginal
ones.

### Effect on FDR-E

Let's decompose the selected answers into correct (C) and wrong (W):

```
S_original = C_orig ∪ W_orig
S_inflated = C_new  ∪ W_new
```

where C_new ⊆ C_orig and W_new ⊆ W_orig (since S_inflated ⊆ S_original).

FDR-E of the inflated selection:

```
FDR-E_new = |W_new| / |S_inflated| = |W_new| / (|C_new| + |W_new|)
```

Whether FDR-E_new < FDR-E_orig depends on which answers are removed. If the threshold
inflation preferentially removes wrong answers (because wrong answers tend to have lower
confidence), then FDR-E improves. If it equally removes correct and wrong answers, the
ratio stays the same. If it preferentially removes correct answers (unlikely but possible
if correct answers happen to be near the threshold while wrong answers are far above it),
FDR-E could actually worsen.

### The key assumption

Option A works well when:

```
P(wrong | near threshold) > P(wrong | far above threshold)
```

That is, answers near the selection boundary are more likely to be wrong than answers
well above it. This is a reasonable assumption for well-calibrated confidence scores —
the whole point of using confidence for selection is that lower confidence correlates
with higher error probability.

Under domain shift, this assumption may partially break (the confidence-correctness
correlation might change), but it is unlikely to fully reverse. So threshold inflation
should almost always improve validity, though possibly not by enough.

---

## 12. Option A: Code Walkthrough

The Option A modification is just two lines in `conservative.py`, at lines 190-192:

```python
# ── Option A: inflate thresholds by safety factor ──
if best_tau1 is not None and tau_safety_factor != 1.0:
    best_tau1 = best_tau1 + np.log(tau_safety_factor)  # fM1 is log-scale
    best_tau2 = min(best_tau2 * tau_safety_factor, 1.0)  # fM2 in [0,1]
```

Key observations:

1. **Guard condition:** `best_tau1 is not None` — if the grid search found no valid
   threshold pair (all candidates failed the CP bound), we skip inflation. There's
   nothing to inflate. The system already abstains on everything.

2. **Guard condition:** `tau_safety_factor != 1.0` — skip the computation when gamma=1.0
   (no-op case). This is technically unnecessary (adding 0 and multiplying by 1 would
   produce the same result) but avoids floating-point noise from log(1.0) which should
   be exactly 0.0 in IEEE 754 but is good practice to guard.

3. **In-place modification:** `best_tau1` and `best_tau2` are overwritten. The original
   pre-inflation values are not preserved in the return dict. If we needed to compare
   pre- vs post-inflation thresholds, we would need to save both. Currently we don't.

4. **Timing:** This runs after the grid search but before evaluation. The evaluation
   function receives the inflated thresholds and applies them to both NQ-test and TQA.

### The sweep in run_conservative_experiment

```python
option_a = {}
for gamma in safety_factors:
    label = f"gamma={gamma:.1f}"
    option_a[str(gamma)] = _run_sweep(
        nq_merged, tqa_merged, sgen_cfg, base_seed, n_splits,
        tau_safety_factor=gamma,
        label=label,
    )
```

For each gamma value, a complete sweep of n_splits=100 calibration splits is run.
Each call to `_run_sweep` invokes `_run_single_split` 100 times with the same gamma
but different split seeds (base_seed + 0 through base_seed + 99).

---

## 13. Option A: Sweep Parameters and Rationale

From `configs/default.yaml`:

```yaml
safety_factors: [1.0, 1.2, 1.5, 2.0]
```

| gamma | log(gamma) | Effect on tau1 | Effect on tau2 (if tau2=0.6) | Interpretation |
|-------|-----------|----------------|----------------------------|---------------|
| 1.0 | 0.000 | No change | No change (0.6) | Baseline (same as Method 1) |
| 1.2 | 0.182 | +0.182 | ×1.2 (0.72) | Mild conservatism |
| 1.5 | 0.405 | +0.405 | ×1.5 (0.90) | Moderate conservatism |
| 2.0 | 0.693 | +0.693 | ×2.0 (1.0, capped) | Aggressive conservatism |

### Why these specific values

- **gamma = 1.0:** Control group. Produces identical results to Method 1. Included to
  verify that the conservative pipeline with all knobs at their defaults exactly matches
  the baseline. If it doesn't, there's a bug.

- **gamma = 1.2:** A modest safety margin. In engineering, 20% safety factors are
  common for well-understood failure modes. This tests whether a small push is enough.

- **gamma = 1.5:** A substantial margin. This requires 50% more probability mass (for
  fM1) and 50% more consistency (for fM2). For tau2, this often pushes the threshold
  into the 0.8-1.0 range, which is quite strict.

- **gamma = 2.0:** An aggressive margin. Doubling the probability threshold and capping
  consistency at 1.0 will probably eliminate most selections. This tests the "maximum
  conservatism" regime — if even gamma=2.0 doesn't fully restore validity, then Option A
  alone is insufficient.

### What about gamma > 2.0?

Values above 2.0 were not included because:
1. gamma = 2.0 already caps tau2 at 1.0 for most realistic tau2 values
2. For tau1, log(2.0) = 0.693 is already a large shift on the log-prob scale
3. Larger gamma values would likely produce 0% efficiency (total abstention), which
   provides no useful information

### What about gamma between 1.0 and 1.2?

Finer granularity could be useful for finding the exact tradeoff curve, but with only
500 splits, the statistical resolution is limited. The difference between gamma = 1.0
and gamma = 1.1 might not be distinguishable from random noise.

---

## 14. Option A: Expected Behavior

**Disclaimer:** No results exist yet. The following predictions are based on theoretical
analysis and reasoning about the algorithm's structure. Actual numbers may differ
substantially.

### On NQ-test (in-domain)

Method 1 should already achieve high validity on NQ-test (probably ~98%, matching the
PAC guarantee). Inflating thresholds should:

- **Validity:** Stay at or above 98% for all gamma values. The inflated thresholds are
  even more conservative than needed, so the in-domain guarantee should hold with room
  to spare.

- **Efficiency:** Decrease monotonically with gamma. At gamma = 1.0, efficiency should
  match Method 1 (probably 30-50%, depending on the data). At gamma = 2.0, efficiency
  is probably 5-15%.

- **FDR-E:** Decrease with gamma. Since fewer and more-confident answers are selected,
  the error rate among selected answers should drop.

### On TQA (shifted domain)

This is where the interesting behavior should emerge:

- **Validity at gamma = 1.0:** Should match Method 1, which we expect to be below 98%
  (the domain shift failure).

- **Validity at gamma = 1.2:** Probably improves somewhat. A 20% safety margin might
  absorb some of the shift.

- **Validity at gamma = 1.5:** Probably improves further. The question is whether it
  reaches the 98% target.

- **Validity at gamma = 2.0:** Probably near 100% validity, but at very low efficiency.

- **Efficiency:** Drops faster on TQA than on NQ, because the TQA score distributions
  might differ from NQ (the thresholds were calibrated on NQ, so they cut the TQA
  distribution at different percentiles).

### The expected tradeoff curve

If we plot validity vs. efficiency across gamma values, we expect:

```
                    NQ                                TQA
validity  98%  ----●────●────●────●        validity  98% --------- ------●
               |   1.0  1.2  1.5  2.0               |              ●
               |                                     |         ●
               └──────────────────── eff             | ●
          100%     50%  35%  20%  5%           0%    └──────────────────── eff
                                                         50%  30%  15%  3%
```

The NQ curve stays flat at ~98% validity while efficiency drops. The TQA curve rises
toward 98% validity as gamma increases, but efficiency drops even faster. The question
is whether TQA can reach 98% validity at any non-trivial efficiency.

---

## 15. Option B: Reduced Epsilon in Grid Search

### The idea

Instead of accepting threshold pairs where the Clopper-Pearson upper bound is at most
epsilon (0.25), require it to be at most epsilon/k for some divisor k > 1. This forces
the grid search to find thresholds that would satisfy a much stricter FDR-E target on
the calibration data.

When applied to the test set, the evaluation still uses the original epsilon (0.25).
So a threshold pair calibrated to achieve FDR-E ≤ 0.125 (epsilon/2) on calibration
data has "headroom" — even if domain shift degrades performance, the actual FDR-E might
still stay below 0.25.

### The mechanism

In the grid search, the acceptance criterion changes from:

```
CP_upper(failures, selected, delta_adj) <= epsilon
```

to:

```
CP_upper(failures, selected, delta_adj) <= epsilon_effective
```

where `epsilon_effective = epsilon / k`.

For k = 2, epsilon_effective = 0.125. This means the calibration data must show that
at most 12.5% of selected answers are wrong (with confidence 1 - delta_adj), even
though we only need 25% on the test set.

The "slack" of 12.5 percentage points is the margin available to absorb domain shift.

---

## 16. Option B: Mathematical Derivation

### The PAC bound under shift

SGen's PAC guarantee says: if the calibration FDR-E is controlled at level epsilon,
then with probability 1 - delta, the test FDR-E is also controlled.

Under domain shift, the test FDR-E can exceed the calibration FDR-E by some amount
Delta_shift. Informally:

```
FDR-E_test ≤ FDR-E_cal + Delta_shift
```

(This is not a formal bound — Delta_shift depends on the nature and magnitude of the
shift. But it captures the intuition.)

If we calibrate to epsilon_effective = epsilon - Delta_shift, then:

```
FDR-E_test ≤ epsilon_effective + Delta_shift = epsilon
```

The problem: we don't know Delta_shift. By sweeping k, we implicitly try different
guesses for how much slack is needed:

| k | epsilon_effective | Implicit Delta_shift budget |
|---|------------------|---------------------------|
| 1.0 | 0.250 | 0.000 (no slack) |
| 1.5 | 0.167 | 0.083 |
| 2.0 | 0.125 | 0.125 |
| 3.0 | 0.083 | 0.167 |
| 4.0 | 0.063 | 0.188 |

At k = 4.0, the implicit assumption is that domain shift can add up to 18.8 percentage
points of additional error. This is a large margin.

### How this changes the grid search

The grid search iterates over all (tau1, tau2) pairs and for each computes:

```
m = number selected in Z_U
f = number of selected pseudo-labeled as wrong
cp_upper = ClopperPearson_upper(f, m, delta_adj)
accept if cp_upper <= epsilon_effective
```

A smaller epsilon_effective means fewer (tau1, tau2) pairs pass the acceptance criterion.
The surviving pairs select fewer answers (higher thresholds) with lower empirical error
rates. Among the surviving pairs, the one with the highest efficiency (most selections)
is chosen.

### Interaction with Bonferroni correction

The Bonferroni correction is unchanged: delta_adj = (delta - delta_p) / H. The
Clopper-Pearson bound uses the same confidence level. Only the acceptance threshold
changes. This means Option B and Option C have independent effects — Option B tightens
the FDR-E target while Option C tightens the confidence level.

---

## 17. Option B: Code Walkthrough

The Option B modification is a single line, at `conservative.py` line 182:

```python
# ── Option B: use epsilon_effective (possibly < epsilon) ──
if cp_upper <= epsilon_effective:
```

Compared to Method 1's `sgen_semi.py` line 146:

```python
if cp_upper <= epsilon:
```

The variable name change from `epsilon` to `epsilon_effective` is the entire modification.
The value of `epsilon_effective` is set at the top of the function:

```python
epsilon = sgen_cfg["epsilon"]           # always 0.25
if epsilon_effective is None:
    epsilon_effective = epsilon          # default: no reduction
```

### The sweep in run_conservative_experiment

```python
option_b = {}
for k in epsilon_divisors:
    eps_eff = epsilon / k
    label = f"eps_div={k:.1f} (eps_eff={eps_eff:.3f})"
    option_b[str(k)] = _run_sweep(
        nq_merged, tqa_merged, sgen_cfg, base_seed, n_splits,
        epsilon_effective=eps_eff,
        label=label,
    )
```

Note: the key in the results dictionary is the string representation of k (the divisor),
not epsilon_effective. So results["option_b"]["2.0"] contains results for epsilon/2 = 0.125.

---

## 18. Option B: Sweep Parameters and Rationale

From `configs/default.yaml`:

```yaml
epsilon_divisors: [1.0, 1.5, 2.0, 3.0, 4.0]
```

| k | epsilon_effective | Slack | Description |
|---|------------------|-------|------------|
| 1.0 | 0.250 | 0.000 | Baseline (same as Method 1) |
| 1.5 | 0.167 | 0.083 | Mild reduction |
| 2.0 | 0.125 | 0.125 | Halved epsilon |
| 3.0 | 0.083 | 0.167 | Aggressive reduction |
| 4.0 | 0.063 | 0.188 | Very aggressive reduction |

### Why these specific values

- **k = 1.0:** Control group, same as Method 1.

- **k = 1.5:** Tests whether a modest 33% reduction in epsilon is enough.

- **k = 2.0:** A natural choice — halving the epsilon target. This is perhaps the most
  interpretable: "calibrate as if your target is 12.5% error even though you'll be
  judged at 25%."

- **k = 3.0:** Quite aggressive. An epsilon_effective of 0.083 means the calibration
  data must show fewer than 8.3% errors among selected answers. This will severely
  restrict the number of valid threshold pairs.

- **k = 4.0:** Near the practical limit. At epsilon_effective = 0.063, the grid search
  will probably find very few (if any) valid threshold pairs, leading to near-total
  abstention. This tests the boundary of what Option B can achieve.

### Why not k = 10 or k = 100?

At very large k, epsilon_effective approaches 0, meaning the grid search requires
essentially zero errors among selected answers. This is equivalent to total abstention
(select nothing, guarantee trivially satisfied). Including such extreme values would
not provide useful information beyond k = 4.0.

---

## 19. Option B: Expected Behavior

**Disclaimer:** No results exist yet. Predictions only.

### On NQ-test (in-domain)

- **Validity:** Should remain at or above 98% for all k values. Tighter calibration
  means even more conservative thresholds, so in-domain validity only improves.

- **Efficiency:** Drops sharply with k. At k = 1.0, matches Method 1. At k = 2.0,
  probably 15-30% efficiency. At k = 4.0, possibly near 0%.

- **FDR-E:** Decreases with k, because the selected set is smaller and was calibrated
  to a tighter target.

### On TQA (shifted domain)

- **Validity at k = 1.0:** Same as Method 1 (expected below 98%).

- **Validity at k = 2.0:** Probably substantially improved. The 12.5 percentage points
  of slack might absorb much of the NQ-to-TQA shift.

- **Validity at k = 4.0:** Probably near 100%, but at very low efficiency.

### Comparison with Option A

Option B should behave similarly to Option A in terms of the overall tradeoff, but the
mechanism is different:

- Option A operates after the grid search, so the grid search finds the same "best"
  thresholds and then inflates them. The inflation is uniform across all scores.

- Option B changes what the grid search considers "best." The grid search finds
  inherently stricter thresholds. The restriction is non-uniform — it depends on the
  actual distribution of failures in the calibration data.

We expect Option B to be slightly more efficient than Option A at equivalent validity
levels, because Option B lets the grid search optimize within the tighter constraint
space, while Option A makes a blind post-hoc adjustment. But this is a hypothesis,
not a proven claim.

---

## 20. Option C: Delta Budget Allocation

### The idea

The delta parameter in the PAC guarantee controls the confidence level: P{FDR-E ≤ ε} ≥
1 - δ. In the algorithm, delta is split into two budgets:

```
delta_p = 1e-5     (for pseudo-labeling failure)
delta_cp = delta - delta_p    (for Bonferroni-corrected Clopper-Pearson bounds)
```

Option C introduces a third allocation:

```
delta_shift = frac * (delta - delta_p)
delta_cp = delta - delta_p - delta_shift
```

The idea is that some of the confidence budget should be "reserved" for potential domain
shift. A smaller delta_cp means a smaller Bonferroni-corrected alpha for each hypothesis
test, which means the Clopper-Pearson bounds are wider, which means the algorithm needs
stronger evidence to accept a threshold pair.

### The mechanism

The Clopper-Pearson upper bound is a function of alpha:

```
CP_upper(f, m, alpha) = B^{-1}(1 - alpha; f + 1, m - f)
```

where B^{-1} is the beta distribution inverse CDF. As alpha decreases, CP_upper
increases (the bound becomes wider/more conservative).

The Bonferroni-corrected alpha for each threshold pair is:

```
alpha_j = delta_cp / H
```

where H = |tau1_grid| × |tau2_grid| ≈ 2500.

So the chain of effects is:

```
larger frac
  → larger delta_shift
  → smaller delta_cp
  → smaller delta_adj = delta_cp / H
  → larger CP_upper for each (tau1, tau2) pair
  → fewer pairs pass the CP_upper <= epsilon check
  → stricter thresholds selected (or total abstention)
```

---

## 21. Option C: Mathematical Derivation

### The Bonferroni correction

The SGen-Semi algorithm tests H candidate threshold pairs simultaneously. Without
correction, the probability of any single pair producing a false acceptance is delta_cp.
But with H tests, the family-wise error rate (probability that at least one false
acceptance occurs) can be as high as H × delta_cp.

Bonferroni correction ensures the family-wise error rate stays below delta_cp by testing
each individual pair at level delta_adj = delta_cp / H.

### Effect of reducing delta_cp

With delta = 0.02 and delta_p = 1e-5:

```
Standard:   delta_cp = 0.02 - 0.00001 = 0.01999
            delta_adj = 0.01999 / H
```

With frac = 0.25:

```
delta_shift = 0.25 * 0.01999 = 0.004998
delta_cp = 0.01999 - 0.004998 = 0.014993
delta_adj = 0.014993 / H
```

With frac = 0.50:

```
delta_shift = 0.50 * 0.01999 = 0.009995
delta_cp = 0.01999 - 0.009995 = 0.009995
delta_adj = 0.009995 / H
```

With frac = 0.75:

```
delta_shift = 0.75 * 0.01999 = 0.014993
delta_cp = 0.01999 - 0.014993 = 0.004998
delta_adj = 0.004998 / H
```

### Numerical example with H = 2500

| frac | delta_cp | delta_adj = delta_cp / 2500 |
|------|---------|----------------------------|
| 0.00 | 0.01999 | 7.996 × 10^-6 |
| 0.25 | 0.01499 | 5.997 × 10^-6 |
| 0.50 | 0.01000 | 3.998 × 10^-6 |
| 0.75 | 0.00500 | 1.999 × 10^-6 |

These are all extremely small numbers. The question is how much the Clopper-Pearson
bound changes in this range. Let's compute for a concrete example.

### Concrete example: CP_upper at different alpha values

Suppose we have m = 200 selected answers with f = 40 failures (20% empirical error rate).

```
CP_upper(40, 200, alpha = 7.996e-6) = beta.ppf(1 - 7.996e-6, 41, 160)
CP_upper(40, 200, alpha = 5.997e-6) = beta.ppf(1 - 5.997e-6, 41, 160)
CP_upper(40, 200, alpha = 3.998e-6) = beta.ppf(1 - 3.998e-6, 41, 160)
CP_upper(40, 200, alpha = 1.999e-6) = beta.ppf(1 - 1.999e-6, 41, 160)
```

Without computing these numerically (we don't have access to scipy in this analysis),
the differences should be small. The beta distribution inverse CDF at these extreme
quantiles (1 - alpha where alpha ~ 10^-6) is in the deep tail. Moving from alpha =
8e-6 to alpha = 2e-6 (a 4x reduction) probably changes CP_upper by a few percentage
points at most.

This suggests Option C might have a **weaker effect** than Options A or B, at least
within the sweep range tested. The delta budget is already heavily Bonferroni-divided
(by up to 2500), so further reducing it has diminishing marginal impact.

This is a testable prediction — if results show that Option C barely affects validity
or efficiency, this analysis explains why.

---

## 22. Option C: Code Walkthrough

The Option C modification is at `conservative.py` line 149:

```python
# ── Option C: reduced delta budget ──
delta_cp = delta - delta_p - delta_shift
```

Compared to Method 1's `sgen_semi.py` line 129:

```python
delta_cp = delta - delta_p
```

The only change is subtracting `delta_shift`, which defaults to 0.0.

The immediately following edge-case handling is unique to Method 2 (lines 150-164):

```python
if delta_cp <= 0:
    logger.warning("delta_cp=%.6f <= 0 (delta_shift=%.4f too large). "
                    "No valid thresholds possible — abstaining on all.",
                    delta_cp, delta_shift)
    abstain = {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
               "n_selected": 0, "n_total": 0}
    return {
        "split_seed": split_seed, "cal_size": len(cal_data),
        "zu_size": len(z_u), "ze_size": len(z_e), "tau_cp": tau_cp,
        "tau1": None, "tau2": None, "grid_size_H": H,
        "epsilon_effective": epsilon_effective,
        "delta_shift": delta_shift, "tau_safety_factor": tau_safety_factor,
        "nq_test": {**abstain, "n_total": len(nq_test)},
        "tqa": {**abstain, "n_total": len(tqa_merged)},
    }
```

This edge case is discussed in detail in Section 25.

### The sweep in run_conservative_experiment

```python
option_c = {}
for frac in delta_shift_fracs:
    ds = frac * (delta - delta_p)
    label = f"frac={frac:.2f} (delta_s={ds:.6f})"
    option_c[str(frac)] = _run_sweep(
        nq_merged, tqa_merged, sgen_cfg, base_seed, n_splits,
        delta_shift=ds,
        label=label,
    )
```

Note: the computed `delta_shift` value (not the fraction) is passed to `_run_sweep`.
The fraction is only used for labeling. The actual subtraction `delta - delta_p - delta_shift`
happens inside `_run_single_split`.

---

## 23. Option C: Sweep Parameters and Rationale

From `configs/default.yaml`:

```yaml
delta_shift_fracs: [0.0, 0.25, 0.50, 0.75]
```

With delta = 0.02 and delta_p = 1e-5, the available budget is delta - delta_p = 0.01999.

| frac | delta_shift | delta_cp remaining | Fraction of budget consumed |
|------|------------|-------------------|---------------------------|
| 0.00 | 0.000000 | 0.01999 | 0% |
| 0.25 | 0.004998 | 0.01499 | 25% |
| 0.50 | 0.009995 | 0.00999 | 50% |
| 0.75 | 0.014993 | 0.00500 | 75% |

### Why these specific values

- **frac = 0.00:** Control group, same as Method 1.

- **frac = 0.25:** Reserve a quarter of the delta budget. Modest conservatism.

- **frac = 0.50:** Reserve half. This represents an equal split between "calibration
  confidence" and "shift tolerance."

- **frac = 0.75:** Reserve three quarters. Most of the confidence budget goes to shift
  tolerance, leaving very little for the Bonferroni correction. This is quite extreme.

### Why not frac = 1.0?

At frac = 1.0:

```
delta_shift = 1.0 * (0.02 - 1e-5) = 0.01999
delta_cp = 0.02 - 1e-5 - 0.01999 = 1e-5
delta_adj = 1e-5 / 2500 = 4e-9
```

This would make the Clopper-Pearson bounds extremely wide. In practice, delta_adj ≈ 4e-9
would require essentially zero failures to accept any threshold pair. It was not included
because it is almost certainly equivalent to total abstention, providing no information
beyond frac = 0.75.

### Why not frac > 1.0?

At frac > 1.0, delta_cp becomes negative, triggering the edge case handler (total
abstention). See Section 25.

---

## 24. Option C: Expected Behavior

**Disclaimer:** No results exist yet. Predictions only.

### Theoretical analysis

As derived in Section 21, the change in delta_adj across the sweep range is modest:
from ~8e-6 (frac = 0.0) to ~2e-6 (frac = 0.75). The Clopper-Pearson bound is already
operating in the extreme tail of the beta distribution at these values, where the
sensitivity to alpha changes is relatively low.

### Prediction: Option C has the weakest effect

Among the three options, Option C is probably the least effective per unit of
conservatism. The reason is that the Bonferroni correction already divides delta_cp
by H ≈ 2500, producing extremely small per-hypothesis confidence levels. Further
reducing delta_cp by a factor of 2-4 has a logarithmic (not linear) effect on the
bound.

In contrast:
- Option A directly shifts the decision boundary, which has a linear effect on selection
- Option B directly restricts the FDR-E target, which has a linear effect on the
  acceptance criterion

### Expected results

- **Validity:** Probably shows modest improvement over Method 1, but less improvement
  than Options A or B at equivalent "conservatism levels."

- **Efficiency:** Probably shows modest decrease, again less dramatic than A or B.

- **Overall:** Option C might be nearly indistinguishable from Method 1 across the
  tested range, making it a useful "control" comparison. If domain shift can only be
  addressed through the delta budget (and not through epsilon or thresholds), then the
  effect is inherently small.

This is one of the most interesting empirical questions for Method 2: does Option C
have a measurable effect at all?

---

## 25. The delta_cp <= 0 Edge Case

### When it triggers

The edge case `delta_cp <= 0` triggers when:

```
delta - delta_p - delta_shift <= 0
delta_shift >= delta - delta_p
frac * (delta - delta_p) >= delta - delta_p
frac >= 1.0
```

With our configuration (frac ∈ {0.0, 0.25, 0.50, 0.75}), this edge case **never
triggers**. The maximum delta_shift is:

```
0.75 * (0.02 - 1e-5) = 0.014993
delta_cp = 0.02 - 1e-5 - 0.014993 = 0.004998 > 0
```

### Why the guard exists

The guard exists for safety — if someone changes the configuration to include frac >= 1.0,
or if delta and delta_p are set to unusual values, the code should not crash with a
divide-by-zero or produce nonsensical negative alpha values.

### What happens when it triggers

When delta_cp <= 0:
1. A warning is logged
2. The function returns immediately with an "abstain on everything" result
3. Both NQ-test and TQA results show 0 selected, 0% efficiency, 0% FDR-E, and
   valid = True (because selecting nothing trivially satisfies any FDR-E bound)

### The abstain dictionary

```python
abstain = {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
           "n_selected": 0, "n_total": 0}
```

There is a subtle issue here: the `n_total` in the abstain template is 0, but it is
overridden in the return statement:

```python
"nq_test": {**abstain, "n_total": len(nq_test)},
"tqa": {**abstain, "n_total": len(tqa_merged)},
```

The `{**abstain, "n_total": len(nq_test)}` syntax creates a new dict by unpacking
`abstain` and then overriding the `n_total` key. So the final `n_total` values are
correct (1083 for NQ-test, 3610 for TQA).

### Correctness of valid = True for abstention

Is it correct to report valid = True when nothing is selected? Yes, technically.
FDR-E is defined as:

```
FDR-E = (wrong selected) / (total selected)
```

When total selected = 0, FDR-E is undefined (0/0). The convention is to define FDR-E = 0
when nothing is selected, which satisfies 0 ≤ epsilon for any epsilon > 0. This is
standard in the selective prediction literature — a system that always abstains is
trivially valid.

---

## 26. Why Evaluation Always Uses Original Epsilon

This is one of the most important design decisions in Method 2, warranting its own
section.

### The principle

When Option B reduces epsilon to epsilon_effective = epsilon/k during grid search, the
evaluation step still checks:

```python
valid = fdr_e <= epsilon   # always original epsilon (0.25)
```

NOT:

```python
valid = fdr_e <= epsilon_effective   # WRONG — would be unfairly strict
```

### Why this matters

The purpose of the evaluation is to answer: **"Does this method achieve the PAC
guarantee that the user actually cares about?"** The user's target is epsilon = 0.25.
They want to know whether FDR-E ≤ 25% with at least 98% probability.

If we evaluated against epsilon_effective, we would be asking a different question:
"Does this method achieve a *stricter* guarantee?" That's interesting but misleading
when comparing methods:

- Method 1 evaluated at epsilon = 0.25: validity = 80% (hypothetical)
- Method 2 (k=2) evaluated at epsilon = 0.125: validity = 60%
- Conclusion: "Method 2 is worse"

This conclusion would be wrong! Method 2 with k=2 might have validity = 95% when
evaluated at the original epsilon = 0.25, which is much better than Method 1's 80%.

### The analogy

Imagine testing two bridges. Bridge A is rated for 10 tons. Bridge B is internally
designed for 20 tons but the specification is 10 tons. You should test both at 10 tons
(the specification), not test Bridge B at 20 tons. Bridge B's internal over-engineering
is a means, not the metric.

### Where this is implemented

In `conservative.py` line 212:

```python
valid = fdr_e <= epsilon   # always original epsilon
```

The variable `epsilon` (set at line 103 from `sgen_cfg["epsilon"]`) is always 0.25,
regardless of what `epsilon_effective` is. The evaluation function `_evaluate` captures
`epsilon` from the enclosing scope (it's a closure defined inside `_run_single_split`).

This applies to all three options. Even though only Option B changes epsilon_effective,
the evaluation is uniformly against the original epsilon.

---

## 27. The Sweep Runner: Aggregation Logic

### The _run_sweep function

```python
def _run_sweep(
    nq_merged, tqa_merged, sgen_cfg, base_seed, n_splits,
    *, epsilon_effective=None, delta_shift=0.0, tau_safety_factor=1.0,
    label="",
) -> dict:
```

This function orchestrates a complete sweep of n_splits calibration/test splits for a
fixed set of conservative parameters. It:

1. Runs `_run_single_split` n_splits times with seeds base_seed + 0 through
   base_seed + (n_splits - 1)
2. Collects per-split results
3. Computes aggregate statistics
4. Logs a one-line summary

### Aggregation formulas

For both NQ and TQA, the following aggregates are computed:

```python
validity_rate = mean(valid_1, valid_2, ..., valid_100)    # fraction of valid splits
mean_fdr_e    = mean(fdr_e_1, fdr_e_2, ..., fdr_e_100)   # average FDR-E
std_fdr_e     = std(fdr_e_1, fdr_e_2, ..., fdr_e_100)    # std dev of FDR-E
mean_efficiency = mean(eff_1, eff_2, ..., eff_100)        # average selection rate
std_efficiency  = std(eff_1, eff_2, ..., eff_100)         # std dev of selection rate
```

### Interpretation of validity_rate

The validity rate is the fraction of 500 random splits where FDR-E ≤ epsilon on the
test set. Under the PAC guarantee, this should be at least 1 - delta = 0.98 (490/500).

**Important statistical note:** With 500 splits, the granularity of the validity
rate is 0.2%. A validity rate of 0.976 could be 488/500, which is two splits below the
target. Whether this constitutes a "failure" of the PAC guarantee is debatable — it
could be sampling noise. Statistical tests (e.g., exact binomial test) would be needed
to determine if a validity rate below 98% is significantly below target.

For Method 2's purposes, the relative comparison matters more than the absolute number.
If Method 1 has validity_rate = 0.85 on TQA and Method 2 with gamma = 1.5 has
validity_rate = 0.96, the improvement is meaningful regardless of whether 0.96 is
"close enough" to 0.98.

### The per_split data

Each sweep stores the full per-split results list, which includes:
- split_seed
- cal_size, zu_size, ze_size
- tau_cp, tau1, tau2
- grid_size_H
- epsilon_effective, delta_shift, tau_safety_factor
- nq_test dict (fdr_e, efficiency, valid, n_selected, n_total)
- tqa dict (same fields)

This data is available for post-hoc analysis (e.g., examining the distribution of
thresholds across splits, or correlating threshold values with validity).

---

## 28. The Orchestrator: run_conservative.py

### Purpose

`run_conservative.py` is the entry point for Method 2. It:
1. Sets up logging
2. Loads configuration
3. Loads all cached data from Method 1's Stages 1-3
4. Validates that cache sizes are consistent
5. Calls `run_conservative_experiment()`
6. Prints the summary table
7. Reports total execution time

### Design decision: separate script

Method 2 has its own script (`run_conservative.py`) rather than being a flag on
`run_baseline.py` because:

1. Method 2 requires no GPU — it only does numpy/scipy computation on cached data
2. Method 1 takes hours (LLM generation, NLI scoring). Method 2 takes minutes.
3. Separating them allows running Method 2 without repeating the expensive GPU stages
4. Different SLURM resource requirements (CPU-only vs GPU)

### Logging vs printing

The orchestrator uses Python's `logging` module (not `print()`) for all output except
the final summary table. This was a style choice for the conservative module:

- `run_baseline.py` uses `print()` directly (simpler, written first)
- `run_conservative.py` uses `logging.getLogger()` (more structured, written second)

The summary table uses `print()` because it produces formatted output that should always
be visible (not affected by log level settings).

---

## 29. Cache Loading and Validation

### The load_cached_stages function

```python
def load_cached_stages(cfg: dict):
    required = {
        "nq_data": "Stage 1 (data loading)",
        "tqa_data": "Stage 1 (data loading)",
        "nq_generations": "Stage 2 (LLM generation)",
        "tqa_generations": "Stage 2 (LLM generation)",
        "nq_entailment": "Stage 3 (entailment scoring)",
        "tqa_entailment": "Stage 3 (entailment scoring)",
    }

    caches = {}
    for name, stage in required.items():
        path = get_cache_path(cache_dir, name)
        data = load_cache(path)
        if data is None:
            raise FileNotFoundError(
                f"Missing cache: {path}\n"
                f"  This comes from {stage}. Run 'python run_baseline.py' first."
            )
        caches[name] = data
    return caches
```

### Size validation

After loading, the main function validates that all arrays have matching lengths:

```python
for name, a, b in [("NQ gen", nq_records, nq_gen),
                   ("NQ ent", nq_records, nq_ent),
                   ("TQA gen", tqa_records, tqa_gen),
                   ("TQA ent", tqa_records, tqa_ent)]:
    if len(a) != len(b):
        logger.error("%s size mismatch: %d records vs %d cached. "
                    "Run run_baseline.py to regenerate.", name, len(a), len(b))
        sys.exit(1)
```

This catches cases where the pipeline was interrupted mid-stage (e.g., 3000 out of 3610
generations were completed). The `_merge_records` function in conservative.py uses `zip`,
which would silently truncate to the shorter list. The size check makes this failure
explicit.

### Expected sizes (when pipeline completes)

| Cache | Expected length |
|-------|----------------|
| nq_data | 3,610 |
| tqa_data | 3,610 |
| nq_generations | 3,610 |
| tqa_generations | 3,610 |
| nq_entailment | 3,610 |
| tqa_entailment | 3,610 |

All six should be 3,610. If any differ, the pipeline needs to be run again.

---

## 30. The Cache Naming Bug (Fixed)

### The bug

The entailment scoring module (`ds_sgen/entailment_scoring.py`) saves its output as:

```python
save_cache(results, get_cache_path(cache_dir, f"{dataset}_entailment"))
```

This produces files named `nq_entailment.json` and `tqa_entailment.json` (singular).

The original version of `run_conservative.py` looked for:

```python
"nq_entailments": "Stage 3 (entailment scoring)",
"tqa_entailments": "Stage 3 (entailment scoring)",
```

Note the plural "entailments" vs the actual singular "entailment". This would produce a
FileNotFoundError when running Method 2, because `get_cache_path(cache_dir, "nq_entailments")`
looks for `nq_entailments.json`, which doesn't exist.

### The fix

Changed lines 49-51 of `run_conservative.py` from:

```python
"nq_entailments": "Stage 3 (entailment scoring)",
"tqa_entailments": "Stage 3 (entailment scoring)",
```

to:

```python
"nq_entailment": "Stage 3 (entailment scoring)",
"tqa_entailment": "Stage 3 (entailment scoring)",
```

And correspondingly updated the cache access lines (101-102) from:

```python
nq_ent = caches["nq_entailments"]
tqa_ent = caches["tqa_entailments"]
```

to:

```python
nq_ent = caches["nq_entailment"]
tqa_ent = caches["tqa_entailment"]
```

### How the bug was found

During a comprehensive code audit that traced all cache save/load paths. The bug would
not have surfaced until the entailment scoring stage completed and someone actually tried
to run `run_conservative.py`.

### Naming convention

The project uses singular nouns for cache names:
- `nq_data` (not `nq_dataset`)
- `nq_generations` (plural because it's a list of generation records, but the word
  "generations" is the natural plural of the concept, not a naming inconsistency)
- `nq_entailment` (singular — "entailment scoring results")

The inconsistency between "generations" (plural) and "entailment" (singular) is the
root cause. A more consistent naming convention would have prevented the bug. But since
both files are written and the bug is fixed, there is no reason to rename them now.

---

## 31. Configuration: Full Hyperparameter Table

### Inherited from Method 1 (SGen-Semi)

These parameters are used identically in Method 2:

| Parameter | Config path | Value | Source | Purpose |
|-----------|------------|-------|--------|---------|
| epsilon | sgen.epsilon | 0.25 | SGen paper Table 1 | Target FDR-E level |
| delta | sgen.delta | 0.02 | SGen paper | PAC confidence (1-delta = 0.98) |
| delta_p | sgen.delta_p | 1e-5 | SGen paper | Pseudo-labeling failure budget |
| cal_frac | sgen.cal_frac | 0.70 | SGen paper | Fraction of NQ for calibration |
| zu_frac | sgen.zu_frac | 0.75 | SGen paper | Fraction of calibration for Z_U |
| epsilon_e | sgen.epsilon_e | 0.10 | SGen paper | Conformal pseudo-label error rate |
| n_splits | sgen.n_splits | 100 | SGen paper | Random calibration/test splits |
| n_grid | sgen.n_grid | 50 | SGen paper | Grid points per score dimension |
| seed | seed | 42 | Convention | Base random seed |

### Method 2 specific

| Parameter | Config path | Value | Purpose |
|-----------|------------|-------|---------|
| safety_factors | conservative.safety_factors | [1.0, 1.2, 1.5, 2.0] | Option A: threshold inflation factors |
| epsilon_divisors | conservative.epsilon_divisors | [1.0, 1.5, 2.0, 3.0, 4.0] | Option B: epsilon reduction divisors |
| delta_shift_fracs | conservative.delta_shift_fracs | [0.0, 0.25, 0.50, 0.75] | Option C: delta budget fractions |
| n_splits | conservative.n_splits | null | Inherits sgen.n_splits (100) |

### Total number of sweep configurations

- Option A: 4 gamma values × 500 splits = 2,000 split evaluations
- Option B: 5 k values × 500 splits = 2,500 split evaluations
- Option C: 4 frac values × 500 splits = 2,000 split evaluations
- **Total: 6,500 split evaluations**

Each split evaluation involves one grid search over up to 2,500 (tau1, tau2) pairs and
two test-set evaluations (NQ-test and TQA). The grid search is the bottleneck — it is
O(H × n_zu) where H ≈ 2500 and n_zu ≈ 1895. So each split processes approximately
2500 × 1895 ≈ 4.7 million comparisons.

Total: 1,300 × 4.7M ≈ 6.1 billion comparisons. This sounds large but is pure numpy
vectorized operations — no GPU needed, probably minutes on a modern CPU.

---

## 32. Worked Example: Option A with gamma = 1.5

**Note:** This example uses hypothetical numbers for fM1, fM2, and correctness labels
to illustrate the mechanism. The actual distributions are unknown until the pipeline
completes.

### Setup

Assume the grid search (with no conservative modifications) finds:
- best_tau1 = -2.5 (mean log-prob threshold)
- best_tau2 = 0.60 (self-consistency threshold)

These are the same thresholds that Method 1 would find for this particular split.

### Threshold inflation

```
gamma = 1.5
tau1_new = -2.5 + log(1.5) = -2.5 + 0.405 = -2.095
tau2_new = min(0.60 * 1.5, 1.0) = min(0.90, 1.0) = 0.90
```

### Effect on selection

The selection rule changes from:
```
fM1 >= -2.5  AND  fM2 >= 0.60
```
to:
```
fM1 >= -2.095  AND  fM2 >= 0.90
```

Both thresholds are raised. Any answer that was previously rejected is still rejected.
Some answers that were previously accepted are now rejected (those with fM1 between
-2.5 and -2.095, or fM2 between 0.60 and 0.90).

### Hypothetical NQ-test evaluation

Suppose NQ-test has 1,083 questions. Under the original thresholds:
- 450 selected (41.5% efficiency)
- 100 wrong (FDR-E = 100/450 = 22.2%)
- valid = True (22.2% < 25%)

Under inflated thresholds:
- 280 selected (25.9% efficiency) — 170 questions dropped
- 50 wrong (FDR-E = 50/280 = 17.9%)
- valid = True (17.9% < 25%)

The inflated thresholds reduced efficiency by ~15 percentage points but also reduced
FDR-E by ~4 percentage points. Validity was already True and remains True.

### Hypothetical TQA evaluation

Suppose TQA has 3,610 questions. Under the original thresholds:
- 1,200 selected (33.2% efficiency)
- 400 wrong (FDR-E = 400/1200 = 33.3%)
- valid = False (33.3% > 25%) — **domain shift failure**

Under inflated thresholds:
- 700 selected (19.4% efficiency) — 500 questions dropped
- 160 wrong (FDR-E = 160/700 = 22.9%)
- valid = True (22.9% < 25%) — **restored!**

In this hypothetical scenario, gamma = 1.5 was enough to restore TQA validity, at the
cost of reducing efficiency from 33.2% to 19.4%.

### Warning about this example

These numbers are entirely hypothetical. The actual distributions of fM1, fM2, and
correctness labels will determine whether gamma = 1.5 is enough. It is equally possible
that gamma = 1.5 is insufficient (TQA validity remains below 98%) or that gamma = 1.2
is already sufficient (making 1.5 unnecessarily conservative).

---

## 33. Worked Example: Option B with k = 2.0

### Setup

Same hypothetical data as Section 32. The grid search now uses epsilon_effective =
0.25 / 2.0 = 0.125 instead of 0.25.

### Effect on grid search

The acceptance criterion changes from:
```
CP_upper(failures, selected, delta_adj) <= 0.25
```
to:
```
CP_upper(failures, selected, delta_adj) <= 0.125
```

This is much stricter. Threshold pairs that had 20% empirical failure rate (which would
pass the 0.25 check) now need 12.5% or less.

### Hypothetical result

Suppose the grid search with epsilon_effective = 0.125 finds:
- best_tau1 = -1.8 (higher than Method 1's -2.5 — more demanding)
- best_tau2 = 0.75 (higher than Method 1's 0.60 — more demanding)

Note that these thresholds are different from Method 1's thresholds. The grid search
itself found different optimal values because the acceptance criterion changed.

### Hypothetical NQ-test evaluation (against original epsilon = 0.25)

Under the stricter thresholds:
- 220 selected (20.3% efficiency)
- 35 wrong (FDR-E = 35/220 = 15.9%)
- valid = True (15.9% < 25%)

Lower efficiency than both Method 1 and Option A (gamma = 1.5), because the grid search
was constrained to find thresholds with very low calibration error.

### Hypothetical TQA evaluation (against original epsilon = 0.25)

Under the stricter thresholds:
- 550 selected (15.2% efficiency)
- 100 wrong (FDR-E = 100/550 = 18.2%)
- valid = True (18.2% < 25%)

### Comparison with Option A

In this hypothetical example, both options restore TQA validity, but with different
efficiency-FDR tradeoffs:

| Method | NQ Efficiency | NQ FDR-E | TQA Efficiency | TQA FDR-E |
|--------|--------------|----------|---------------|----------|
| Method 1 | 41.5% | 22.2% | 33.2% | 33.3% (FAIL) |
| Option A (gamma=1.5) | 25.9% | 17.9% | 19.4% | 22.9% |
| Option B (k=2.0) | 20.3% | 15.9% | 15.2% | 18.2% |

Option A has higher efficiency (selects more answers) while Option B has lower FDR-E
(is more accurate). This is because Option A makes a uniform threshold shift, while
Option B changes what the grid search optimizes for.

**Again: these numbers are hypothetical.** The actual comparison may differ.

---

## 34. Worked Example: Option C with frac = 0.50

### Setup

Same hypothetical data. Now we reserve half the delta budget for shift:

```
delta_shift = 0.50 * (0.02 - 1e-5) = 0.009995
delta_cp = 0.02 - 1e-5 - 0.009995 = 0.009995
delta_adj = 0.009995 / 2500 = 3.998e-6
```

Compare to baseline:
```
delta_cp = 0.02 - 1e-5 = 0.01999
delta_adj = 0.01999 / 2500 = 7.996e-6
```

### Effect on Clopper-Pearson bound

The alpha for the CP bound has halved: from ~8e-6 to ~4e-6. Let's trace the effect
for a specific threshold pair with 200 selected and 40 failures:

```
Baseline:  CP_upper(40, 200, 7.996e-6) = beta.ppf(1 - 7.996e-6, 41, 160)
Option C:  CP_upper(40, 200, 3.998e-6) = beta.ppf(1 - 3.998e-6, 41, 160)
```

Both values are in the extreme right tail of the Beta(41, 160) distribution. The
difference between the 99.9992% quantile and the 99.9996% quantile of this distribution
is probably very small — perhaps 0.01-0.02 in FDR-E terms.

### Prediction

The grid search with Option C probably finds **very similar** thresholds to Method 1,
because the change in delta_adj has minimal effect on which (tau1, tau2) pairs pass the
CP bound check. The thresholds might be identical or differ by one grid step.

### Hypothetical result

- Thresholds: tau1 = -2.5 (same as Method 1), tau2 = 0.62 (one grid step higher)
- NQ Efficiency: 40.1% (barely lower than 41.5%)
- NQ FDR-E: 21.8% (barely lower than 22.2%)
- TQA Efficiency: 32.0%
- TQA FDR-E: 32.5% (barely lower than 33.3%)
- TQA valid: False (32.5% > 25%) — **domain shift failure NOT fixed**

This hypothetical illustrates why we predict Option C is the weakest of the three
options.

---

## 35. Expected Results: Option A

**Disclaimer:** These are predictions. No actual results exist.

### Validity-efficiency tradeoff table (predicted)

| gamma | NQ Validity | NQ Efficiency | TQA Validity | TQA Efficiency |
|-------|-----------|-------------|------------|--------------|
| 1.0 | ~98% | ~35-50% | ~80-90% | ~25-40% |
| 1.2 | ~98-100% | ~25-40% | ~85-95% | ~18-30% |
| 1.5 | ~99-100% | ~15-30% | ~90-98% | ~10-20% |
| 2.0 | ~100% | ~5-15% | ~95-100% | ~2-10% |

### Key predictions

1. **gamma = 1.0 exactly matches Method 1.** This is verifiable and serves as a
   consistency check.

2. **NQ validity stays ≥ 98% for all gamma.** The in-domain guarantee only gets
   stronger with conservatism.

3. **TQA validity improves monotonically with gamma.** More conservatism → fewer
   wrong answers selected.

4. **Efficiency decreases monotonically with gamma.** This is mathematically guaranteed
   since inflated thresholds define a subset of the original selection.

5. **The "crossover" gamma (where TQA validity first hits ~98%) is probably between 1.2
   and 2.0.** This is the most informative finding — it tells us how much conservatism
   is needed to compensate for the NQ→TQA shift.

---

## 36. Expected Results: Option B

**Disclaimer:** These are predictions. No actual results exist.

### Validity-efficiency tradeoff table (predicted)

| k | epsilon_eff | NQ Validity | NQ Eff | TQA Validity | TQA Eff |
|---|-----------|-----------|--------|------------|---------|
| 1.0 | 0.250 | ~98% | ~35-50% | ~80-90% | ~25-40% |
| 1.5 | 0.167 | ~99% | ~25-40% | ~85-95% | ~15-30% |
| 2.0 | 0.125 | ~99-100% | ~15-30% | ~90-98% | ~10-20% |
| 3.0 | 0.083 | ~100% | ~5-15% | ~95-100% | ~3-10% |
| 4.0 | 0.063 | ~100% | ~0-5% | ~98-100% | ~0-3% |

### Key predictions

1. **k = 1.0 exactly matches Method 1.** Same consistency check.

2. **Option B probably achieves slightly higher efficiency than Option A at the same
   validity level.** Because Option B lets the grid search optimize within the tighter
   constraint space rather than making a blind adjustment.

3. **At k = 4.0, there may be splits where the grid search finds no valid threshold
   pair.** Epsilon_effective = 0.063 is very strict; if the empirical error rate on Z_U
   is above this for all threshold pairs, the system abstains entirely.

4. **The validity-efficiency tradeoff is probably smoother than Option A's**, because
   epsilon divisors produce a more gradual progression than log-scale threshold shifts.

---

## 37. Expected Results: Option C

**Disclaimer:** These are predictions. No actual results exist.

### Validity-efficiency tradeoff table (predicted)

| frac | delta_cp | NQ Validity | NQ Eff | TQA Validity | TQA Eff |
|------|---------|-----------|--------|------------|---------|
| 0.00 | 0.01999 | ~98% | ~35-50% | ~80-90% | ~25-40% |
| 0.25 | 0.01499 | ~98% | ~34-49% | ~81-91% | ~24-39% |
| 0.50 | 0.00999 | ~98-99% | ~33-48% | ~82-92% | ~23-38% |
| 0.75 | 0.00500 | ~98-99% | ~31-46% | ~83-93% | ~22-37% |

### Key predictions

1. **Option C has the smallest effect.** The efficiency and validity changes are
   probably within statistical noise of the Method 1 baseline.

2. **The direction of the effect is correct** (more conservatism → slightly better
   validity, slightly lower efficiency), but the magnitude is probably too small to
   be practically useful.

3. **This negative result is itself valuable.** It shows that simply adjusting the
   confidence budget is not sufficient to address domain shift. The problem is not
   about confidence levels — it's about the distribution mismatch between calibration
   and test data, which no amount of delta-budget reallocation can fix.

---

## 38. Comparing the Three Options

### Theoretical comparison

| Dimension | Option A | Option B | Option C |
|-----------|---------|---------|---------|
| What it modifies | Post-hoc thresholds | Grid search criterion | Confidence level |
| Effect type | Direct, linear | Indirect, optimization-based | Indirect, logarithmic |
| Expected strength | Medium-strong | Strong | Weak |
| Interpretability | Very intuitive | Moderately intuitive | Least intuitive |
| Control group | gamma = 1.0 | k = 1.0 | frac = 0.0 |
| Number of sweep points | 4 | 5 | 4 |

### Which option should "win"?

We predict Option B provides the best validity-efficiency tradeoff because:

1. It operates at the right level of abstraction — changing the optimization target
   rather than post-hoc adjustments or indirect confidence adjustments.

2. The grid search can find thresholds that are specifically optimized for the stricter
   constraint, rather than inflating already-optimized thresholds (Option A) or hoping
   that wider confidence intervals will make the same thresholds more conservative
   (Option C).

3. The sweep range for Option B covers a wider range of effective conservatism levels
   (epsilon from 0.25 to 0.063) compared to Option A (where gamma = 2.0 is already
   quite extreme) or Option C (which has minimal effect).

**However**, this prediction could be wrong. If the grid search landscape has many
near-optimal solutions, Option A's post-hoc adjustment might be nearly as good as
Option B's constraint-based approach.

### Why we test all three

Even if Option B "wins" on the headline numbers, the paper benefits from showing:
- Option A: the simplest fix, most interpretable, serves as a sanity check
- Option B: the most effective fix, but still fundamentally limited
- Option C: shows that the problem is NOT about confidence levels
- All three: motivates the principled approach of Method 3

---

## 39. The Validity-Efficiency Tradeoff

### The fundamental limit of conservative approaches

All three conservative options face the same fundamental tradeoff: **more conservatism
means higher validity but lower efficiency**. There is no way to improve both
simultaneously without additional information about the domain shift.

This is because all three options make the selection rule stricter without changing the
underlying confidence scores. They cannot make the model more accurate on the shifted
domain — they can only choose to answer fewer questions.

### The Pareto frontier

If we plot all 13 configurations (4 + 5 + 4 sweep points) on a validity vs. efficiency
graph for TQA, they should approximately trace a Pareto frontier: each point that
improves validity does so at the cost of efficiency, and no point dominates another on
both dimensions.

Points above the Pareto frontier (better validity AND better efficiency) are impossible
with conservative methods alone. Method 3 (DS-SGen) aims to shift the entire frontier
upward by using domain information to make smarter selections.

### The "abstention barrier"

At some level of conservatism, the system begins to abstain on everything. For Option A,
this happens when the inflated thresholds exceed the maximum fM1 and fM2 values in the
test set. For Option B, this happens when epsilon_effective is too low for any threshold
pair to pass the CP bound. For Option C, this (barely) doesn't happen in the tested
range.

Beyond the abstention barrier, all additional conservatism has no effect (you can't
select fewer than 0 answers). The system trivially achieves FDR-E = 0 and efficiency = 0.

### What Method 3 does differently

Method 3 (DS-SGen) aims to **reweight** the calibration data to account for the
domain shift, rather than blindly increasing conservatism. By estimating importance
weights w(x) = P_test(x) / P_cal(x), the thresholds can be calibrated to the test
domain's distribution of (fM1, fM2) values. This should allow high validity without
sacrificing efficiency — the thresholds are "correct" for the test domain, not just
"extra strict."

The gap between Method 2's best validity-efficiency point and Method 3's result is the
measure of how much principled domain adaptation helps compared to brute-force
conservatism.

---

## 40. Why This Method Cannot Fully Solve Domain Shift

### The core limitation

Conservative approaches treat domain shift as a black-box degradation: "things might
get worse by an unknown amount, so add headroom." They do not model HOW or WHY things
get worse. This leads to several limitations:

**Limitation 1: Unknown conservatism level.** There is no principled way to choose gamma,
k, or frac. The right level depends on the magnitude and nature of the domain shift,
which is exactly what we don't know. The sweep approach (trying multiple levels) is
empirical, not principled.

**Limitation 2: Uniform conservatism.** All three options apply the same level of
conservatism to all questions. But domain shift might affect some questions more than
others. A question about "who won the 2022 World Cup" might have similar fM1/fM2
distributions in NQ and TQA, while a question about "what is the capital of Uzbekistan"
might behave very differently. Uniform conservatism is unnecessarily strict for some
questions and possibly insufficient for others.

**Limitation 3: No test-domain information.** The conservative options use only the
calibration data (NQ) to set thresholds. They do not use any information from the test
domain (TQA), even in an unlabeled form. This is a strong self-imposed handicap.

**Limitation 4: No formal guarantee.** Unlike SGen's original PAC bound, the conservative
modifications do not come with a formal guarantee of the form "if the shift is at most
X, then FDR-E ≤ epsilon." The improved validity is empirical — it works on this dataset
with this shift, but we cannot prove it works in general.

### What a principled solution looks like

Method 3 addresses all four limitations:
1. Importance weights are estimated from data, not guessed
2. Weights are per-sample, so conservatism is adaptive
3. Unlabeled test-domain data is used (for weight estimation)
4. A formal PAC bound with shift-dependent error term is derived

---

## 41. The Print Summary: Output Format

### The print_conservative_summary function

This function prints a formatted comparison table after all sweeps complete. It is
called from `run_conservative.py` line 132:

```python
print_conservative_summary(results)
```

### Output structure

The output has three sections (one per option), each with:
- A header identifying the option
- A description of the mechanism
- A column-aligned table with one row per sweep parameter value

### Column definitions

```
Setting     — descriptive label for the sweep parameter value
NQ Valid    — validity rate on NQ-test (fraction of 500 splits with FDR-E ≤ 0.25)
NQ FDR     — mean FDR-E across 500 splits on NQ-test
NQ Eff     — mean efficiency (fraction of questions answered) on NQ-test
TQA Valid   — validity rate on TQA
TQA FDR    — mean FDR-E across 500 splits on TQA
TQA Eff    — mean efficiency on TQA
```

### Formatting details

- Validity rates are displayed as percentages with 1 decimal place (e.g., "98.0%")
- FDR-E and efficiency are displayed as 4-decimal-place floats (e.g., "0.2234")
- Column widths are fixed at 8-9 characters for alignment
- A separator line of dashes separates the header from the data rows

### The footer key

```
Key: Valid = P(FDR-E <= 0.25) across 500 splits |
     FDR-E = mean empirical error | Eff = selection rate
```

This key uses the hardcoded value 0.25 (epsilon). If epsilon were changed in the config,
this print statement would be wrong. This is a minor maintainability issue — the
function does not receive epsilon as a parameter. In `print_conservative_summary` at
line 453:

```python
print(f"  {label:<32} | {nq['validity_rate']:>7.1%} ...")
```

The `.1%` format multiplies by 100 and appends %. So a validity_rate of 0.98 displays
as "98.0%". A mean_fdr_e of 0.22 displays as "0.2200" (`.4f` format).

### Observation: epsilon hardcoded in summary

The `print_conservative_summary` function at line 453 computes epsilon_effective from
the key:

```python
k = float(key)
eps_eff = 0.25 / k
```

The 0.25 is hardcoded, not read from results. If the config epsilon were changed, this
would display wrong labels. This is acceptable for a research codebase where epsilon =
0.25 is fixed, but worth noting.

---

## 42. Results Saving: What Gets Persisted

### The save_data structure

```python
save_data = {
    "config": {
        "sgen": sgen_cfg,
        "conservative": cons_cfg,
        "seed": cfg["seed"],
    },
    "option_a": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                 for k, v in option_a.items()},
    "option_b": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                 for k, v in option_b.items()},
    "option_c": {k: {kk: vv for kk, vv in v.items() if kk != "per_split"}
                 for k, v in option_c.items()},
}
```

### Per-split data is stripped

The dict comprehension `{kk: vv for kk, vv in v.items() if kk != "per_split"}`
removes the `per_split` list from each sweep result before saving. This is because:

1. Each per_split list has 100 entries, each with ~15 fields
2. With 13 sweep configurations, that's 1,300 entries
3. The resulting JSON would be several MB, mostly repetitive data
4. The aggregate statistics (mean, std, validity rate) capture what's needed for
   the paper's figures and tables

The per_split data is still available in memory during the `print_conservative_summary`
call and could be saved separately if needed for detailed post-hoc analysis.

### The saved summary

What IS saved (for each option, each parameter value):
- `nq.validity_rate` — fraction of 500 splits where FDR-E ≤ 0.25 on NQ-test
- `nq.mean_fdr_e` — average FDR-E across splits
- `nq.std_fdr_e` — standard deviation of FDR-E
- `nq.mean_efficiency` — average selection rate
- `nq.std_efficiency` — standard deviation of selection rate
- Same five fields for `tqa`

Total: 10 numbers per sweep configuration, 130 numbers total.

### Save path

```python
results_path = f"{cfg['paths']['results_dir']}/conservative_results.json"
```

Which resolves to: `/data/user_data/anshulk/dsgen/results/conservative_results.json`

The save uses the atomic write function from `ds_sgen/utils.py` (tempfile + os.replace),
which is safe against SLURM preemption. However, since Method 2 runs in minutes (not
hours), preemption is unlikely to be an issue.

---

## 43. Running Method 2

### Prerequisites

1. All six cache files must exist (from a complete run of `run_baseline.py`)
2. Python environment with numpy, scipy, pyyaml
3. No GPU required

### Command

```bash
python run_conservative.py
```

Or with explicit config:

```bash
python run_conservative.py --config configs/default.yaml
```

### SLURM submission

The SLURM script `scripts/run_conservative.sh` is provided:

```bash
sbatch scripts/run_conservative.sh
```

This script requests GPU resources because some SLURM configurations require it for the
Python environment to load correctly. The actual computation is CPU-only.

### Expected output

The script will print:
1. Configuration summary
2. Cache loading confirmation with record counts
3. Progress for each option (Option A, B, C) with per-gamma/k/frac results
4. Final summary table (see Section 41)
5. Total execution time

---

## 44. Runtime Estimates

### Per-split computation

Each split involves:
1. Random permutation: O(n_nq) = O(3610) — negligible
2. Conformal threshold: O(n_ze × log(n_ze)) = O(632 × 10) — negligible
3. Pseudo-labeling: O(n_zu) = O(1895) — negligible
4. Grid construction: O(n_zu × log(n_zu)) per score — negligible
5. Grid search: O(H × n_zu) = O(2500 × 1895) ≈ 4.7M operations — this is the bottleneck
6. Evaluation: O(n_test + n_tqa) = O(1083 + 3610) — negligible

Step 5 dominates. Each iteration of the inner loop computes a boolean mask, counts
failures, and evaluates one Clopper-Pearson bound (which calls scipy.stats.beta.ppf —
a single transcendental function evaluation).

### Estimated wall time per split

The grid search is a double loop in Python (not vectorized). The outer loop over tau1
values is ~50 iterations. The inner loop over tau2 values is ~50 iterations. Each
iteration does numpy operations on arrays of ~1895 elements.

Rough estimate: ~0.1-0.5 seconds per split, depending on CPU speed and scipy
performance. This is a guess — actual timing will depend on the hardware.

### Total estimated wall time

```
1,300 splits × 0.1-0.5 sec/split = 130-650 seconds ≈ 2-11 minutes
```

Plus overhead (data loading, merging, logging): probably 1-2 minutes.

**Total estimate: approximately 5-15 minutes.** This is orders of magnitude faster than
Method 1 (which takes hours because of LLM generation and NLI scoring).

### Why Method 2 is so fast

Method 2 does zero neural network inference. It operates entirely on cached floating-point
numbers (fM1, fM2, entail_score, entail_label) using numpy and scipy. The most expensive
operation is scipy.stats.beta.ppf, which is a scalar function called once per grid point
per split.

---

## 45. Current Status

### As of April 6, 2026

| Component | Status |
|-----------|--------|
| `ds_sgen/conservative.py` | Complete, code-reviewed |
| `run_conservative.py` | Complete |
| `configs/default.yaml` | Complete, 500 splits configured |
| NQ data cache | **Complete** (3,610 records, 811 KB) |
| TQA data cache | **Complete** (3,610 records, ~2.1 MB) |
| NQ generation cache (GPT-4o-mini) | **Complete** (3,610 records, 18.9 MB) |
| TQA generation cache (GPT-4o-mini) | **Complete** (3,610 records, ~15 MB) |
| NQ entailment cache (DeBERTa-xxl) | **Complete** (3,610 records, 2.1 MB — 43.1% correct) |
| TQA entailment cache (DeBERTa-xxl) | **Complete** (3,610 records, ~2.1 MB — 71.6% correct) |
| Method 1 baseline results | **Complete** |
| **Method 2 results** | **Complete** |

### Execution order

```
Method 1 baseline (500 splits, CPU-only)
  → Method 2 conservative (500 splits, CPU-only)
    → Method 3 importance reweighting (500 splits, needs GPU for embeddings)
      → Epsilon sweep (all methods × 4 epsilon values)
```

All data caches are ready. Method 1 baseline and Method 2 conservative variants
have both completed successfully.

---

## 46. What Method 2 Already Tells Us (in Theory)

Even without results, Method 2's design reveals several things:

### 1. Conservative approaches have a structural limitation

The validity-efficiency tradeoff is inherent. Without domain information, the only way
to improve validity is to answer fewer questions. This is a mathematical consequence of
the selection rule structure, not an implementation detail.

### 2. The three injection points are not equivalent

Options A, B, and C operate at different levels of the algorithm and have different
effectiveness. This tells us that the "where" of the conservatism matters, not just
the "how much."

### 3. The delta budget is not the bottleneck

Our theoretical analysis (Section 21) predicts that Option C has the weakest effect,
because the Bonferroni correction already divides delta into very small per-hypothesis
budgets. Further reducing delta has diminishing returns. This means the domain shift
problem is not about confidence levels — it's about the mismatch between calibration
and test distributions.

### 4. The grid search structure is preserved

All three options maintain the same grid search procedure. They modify what's accepted,
not how candidates are generated. This means the computational cost is essentially
the same as Method 1's grid search, just with different acceptance criteria.

### 5. Self-consistency (fM2) threshold inflation is bounded

The `min(tau2 * gamma, 1.0)` cap in Option A means that for gamma ≥ 2.0 and typical
tau2 values (0.4-0.8), the fM2 threshold is capped at 1.0, requiring perfect
self-consistency. This is an implicit "ceiling effect" that limits Option A's range
on the fM2 dimension.

---

## 47. Connection to Method 3: What Method 2 Motivates

### The bridge between Methods 2 and 3

Method 2 establishes that:
1. Domain shift is a real problem (from Method 1's failure on TQA)
2. Naive conservatism can partially fix it (from Method 2's improved validity)
3. The fix is too expensive (from Method 2's reduced efficiency)
4. We need something smarter (motivates Method 3)

### What Method 3 (DS-SGen) does differently

Method 3 replaces blind conservatism with informed adaptation:

| Aspect | Method 2 | Method 3 |
|--------|---------|---------|
| Domain information | None (uses only calibration data) | Unlabeled test-domain samples |
| Conservatism | Uniform across all questions | Adaptive per question |
| Mechanism | Threshold inflation / epsilon reduction / delta allocation | Importance reweighting |
| Formal guarantee | None (empirical improvement only) | PAC bound with shift-dependent error term |
| Expected result | Higher validity, much lower efficiency | Higher validity, maintained efficiency |

### The specific innovations of Method 3

1. **Embed questions** using a sentence transformer (e.g., all-MiniLM-L6-v2)
2. **Train a domain classifier** (e.g., XGBoost) to distinguish NQ from TQA
3. **Convert classifier probabilities to importance weights**: w(x) = P_TQA(x) / P_NQ(x)
4. **Use weighted conformal prediction** for pseudo-labeling (instead of standard CP)
5. **Use weighted Clopper-Pearson bounds** in the grid search (instead of standard CP)
6. **Add a domain similarity score** as a third selection signal alongside fM1 and fM2

### Why Method 2's results matter for Method 3

Method 2's results provide the **baseline** against which Method 3 is evaluated.
Specifically:

- If Method 2 at gamma = 2.0 achieves 95% TQA validity with 5% efficiency,
  and Method 3 achieves 98% TQA validity with 30% efficiency,
  then Method 3 has improved efficiency by 25 percentage points while matching or
  exceeding validity. This is the headline result.

- If Method 2's best configuration already achieves 98% TQA validity with reasonable
  efficiency (say 20%), then Method 3 needs to do significantly better to justify its
  complexity. The improvement from 20% to 30% efficiency is less dramatic than from
  5% to 30%.

The severity of Method 2's efficiency penalty directly determines how impressive Method 3
needs to be.

---

## Appendix A: Code Listing — conservative.py Key Functions

### _run_single_split signature and return type

```
Input:
  nq_merged: list[dict]     — merged NQ records (3,610 entries)
  tqa_merged: list[dict]    — merged TQA records (3,610 entries)
  split_seed: int           — seed for this split's random permutation
  sgen_cfg: dict            — SGen config section from YAML
  epsilon_effective: float  — Option B override (default: epsilon from config)
  delta_shift: float        — Option C override (default: 0.0)
  tau_safety_factor: float  — Option A override (default: 1.0)

Output: dict with keys:
  split_seed: int
  cal_size: int             — size of calibration set (2527)
  zu_size: int              — size of Z_U (1895)
  ze_size: int              — size of Z_E (632)
  tau_cp: float             — conformal pseudo-labeling threshold
  tau1: float | None        — selected fM1 threshold (after inflation if Option A)
  tau2: float | None        — selected fM2 threshold (after inflation if Option A)
  grid_size_H: int          — |tau1_grid| × |tau2_grid|
  epsilon_effective: float  — the epsilon used in grid search
  delta_shift: float        — the delta reserved for shift
  tau_safety_factor: float  — the safety factor applied
  nq_test: dict             — {fdr_e, efficiency, valid, n_selected, n_total}
  tqa: dict                 — {fdr_e, efficiency, valid, n_selected, n_total}
```

### run_conservative_experiment signature

```
Input:
  cfg: dict                 — full config dict
  nq_records: list[dict]    — NQ data records
  nq_gen: list[dict]        — NQ generation records
  nq_ent: list[dict]        — NQ entailment records
  tqa_records: list[dict]   — TQA data records
  tqa_gen: list[dict]       — TQA generation records
  tqa_ent: list[dict]       — TQA entailment records

Output: dict with keys:
  option_a: dict            — keyed by str(gamma), each value is a sweep result
  option_b: dict            — keyed by str(k), each value is a sweep result
  option_c: dict            — keyed by str(frac), each value is a sweep result
```

Each sweep result has keys: nq (dict with 5 stats), tqa (dict with 5 stats), per_split (list).

---

## Appendix B: Mathematical Symbols Reference

| Symbol | Meaning | Value in our config |
|--------|---------|-------------------|
| ε (epsilon) | Target FDR-E level | 0.25 |
| δ (delta) | PAC failure probability | 0.02 |
| δ_p (delta_p) | Pseudo-labeling failure budget | 1e-5 |
| δ_s (delta_shift) | Budget reserved for domain shift (Option C) | 0 to 0.015 |
| δ_cp | Budget for Clopper-Pearson correction | δ - δ_p - δ_s |
| δ_adj | Per-hypothesis confidence level | δ_cp / H |
| ε_e (epsilon_e) | Conformal pseudo-labeling error rate | 0.10 |
| ε_eff (epsilon_effective) | Reduced epsilon for grid search (Option B) | ε / k |
| τ₁ (tau1) | fM1 selection threshold | Learned from data |
| τ₂ (tau2) | fM2 selection threshold | Learned from data |
| τ_CP (tau_cp) | Conformal pseudo-labeling threshold | Learned from Z_E |
| γ (gamma) | Safety factor (Option A) | 1.0 to 2.0 |
| k | Epsilon divisor (Option B) | 1.0 to 4.0 |
| frac | Delta shift fraction (Option C) | 0.0 to 0.75 |
| H | Number of candidate threshold pairs | ≤ n_grid² = 2500 |
| n_grid | Grid points per score dimension | 50 |
| K | Number of sampled answers for self-consistency | 5 |
| fM1 | Mean log-probability of greedy answer | Continuous, typically negative |
| fM2 | Self-consistency score | [0, 1] |
| CP_upper(f, m, α) | Clopper-Pearson upper bound | beta.ppf(1-α, f+1, m-f) |
| FDR-E | False Discovery Rate with Entailment | (wrong selected) / (total selected) |

---

## Appendix C: Tracing One Complete Computation

This appendix traces a single split of Option A with gamma = 1.5, using hypothetical
but realistic values, to show every computation step from input to output.

### Input data

Assume:
- 3,610 NQ merged records, each with fM1, fM2, entail_score, entail_label
- 3,610 TQA merged records, same fields
- split_seed = 42 (first split)
- gamma = 1.5

### Step 1: NQ split

```python
rng = np.random.RandomState(42)
indices = rng.permutation(3610)
# indices[0:2527] → calibration
# indices[2527:3610] → NQ test (1083 questions)
```

### Step 2: Calibration split

```python
zu_size = floor(2527 * 0.75) = 1895
# cal_data[0:1895] → Z_U (unlabeled, will be pseudo-labeled)
# cal_data[1895:2527] → Z_E (labeled, used for conformal threshold)
# Z_E has 632 records
```

### Step 3: Conformal threshold

```python
ze_scores = [0.12, 0.45, 0.89, 0.03, ...]  # 632 entailment scores
sorted_scores = np.sort(ze_scores)
k = ceil((632 + 1) * (1 - 0.10)) = ceil(633 * 0.90) = ceil(569.7) = 570
tau_cp = sorted_scores[569]
# Hypothetical: tau_cp ≈ 0.72
```

### Step 4: Pseudo-labeling Z_U

```python
for r in z_u:
    r["pseudo_label"] = 1 if r["entail_score"] >= 0.72 else 0
# Hypothetical: ~60% labeled as correct (1), ~40% as wrong (0)
```

### Step 5: Grid construction

```python
tau1_grid = _build_percentile_grid(zu_fM1, 50)
# Hypothetical: 47 unique values from -8.2 to -0.3
tau2_grid = _build_percentile_grid(zu_fM2, 50)
# Hypothetical: 42 unique values from 0.0 to 1.0
H = 47 * 42 = 1974
```

### Step 6: Delta computation (no Option C modification)

```python
delta_cp = 0.02 - 1e-5 - 0.0 = 0.01999
delta_adj = 0.01999 / 1974 = 1.013e-5
```

### Step 7: Grid search (no Option B modification)

For each of 1,974 threshold pairs (t1, t2):
```python
sel = (zu_fM1 >= t1) & (zu_fM2 >= t2)
m = sel.sum()
failures = (sel & (zu_pseudo == 0)).sum()
cp_upper = beta.ppf(1 - 1.013e-5, failures + 1, m - failures)
if cp_upper <= 0.25:  # epsilon (not epsilon_effective, since Option B is not active)
    efficiency = m / 1895
    # track best efficiency and corresponding thresholds
```

Hypothetical best result:
- best_tau1 = -2.5
- best_tau2 = 0.60
- best_efficiency = 42.1% (798 out of 1895 Z_U records selected)

### Step 8: Option A threshold inflation

```python
tau_safety_factor = 1.5
best_tau1 = -2.5 + np.log(1.5) = -2.5 + 0.4055 = -2.0945
best_tau2 = min(0.60 * 1.5, 1.0) = min(0.90, 1.0) = 0.90
```

### Step 9: NQ-test evaluation

```python
fM1 = np.array([r["fM1"] for r in nq_test])  # 1083 values
fM2 = np.array([r["fM2"] for r in nq_test])  # 1083 values
labels = np.array([r["entail_label"] for r in nq_test])  # 1083 values

selected = (fM1 >= -2.0945) & (fM2 >= 0.90)
n_selected = selected.sum()
# Hypothetical: 183 selected

n_wrong = (selected & (labels == 0)).sum()
# Hypothetical: 28 wrong

fdr_e = 28 / 183 = 0.153
efficiency = 183 / 1083 = 0.169
valid = 0.153 <= 0.25  # True
```

### Step 10: TQA evaluation

```python
fM1 = np.array([r["fM1"] for r in tqa_merged])  # 3610 values
fM2 = np.array([r["fM2"] for r in tqa_merged])  # 3610 values
labels = np.array([r["entail_label"] for r in tqa_merged])

selected = (fM1 >= -2.0945) & (fM2 >= 0.90)
n_selected = selected.sum()
# Hypothetical: 421 selected

n_wrong = (selected & (labels == 0)).sum()
# Hypothetical: 89 wrong

fdr_e = 89 / 421 = 0.211
efficiency = 421 / 3610 = 0.117
valid = 0.211 <= 0.25  # True
```

### Step 11: Return value

```python
return {
    "split_seed": 42,
    "cal_size": 2527,
    "zu_size": 1895,
    "ze_size": 632,
    "tau_cp": 0.72,
    "tau1": -2.0945,
    "tau2": 0.90,
    "grid_size_H": 1974,
    "epsilon_effective": 0.25,
    "delta_shift": 0.0,
    "tau_safety_factor": 1.5,
    "nq_test": {"fdr_e": 0.153, "efficiency": 0.169, "valid": True,
                "n_selected": 183, "n_total": 1083},
    "tqa": {"fdr_e": 0.211, "efficiency": 0.117, "valid": True,
            "n_selected": 421, "n_total": 3610},
}
```

This is ONE split. The sweep repeats this 100 times with seeds 42-141, then aggregates.

---

## Appendix D: Differences Between conservative.py and sgen_semi.py

### Structural differences

| Aspect | sgen_semi.py | conservative.py |
|--------|-------------|----------------|
| Lines | 263 | 479 |
| Logging | print() | logging.getLogger() |
| Entry point | run_experiment() | run_conservative_experiment() |
| Parameters | Fixed from config | Three optional overrides |
| Sweep | None (single config) | Three option sweeps |
| Output | Single result dict | Three nested result dicts |
| Per-split progress | Print every 10 splits | No per-split printing |
| Results saved | With per_split data | Per_split data stripped |
| Summary | None (caller handles) | print_conservative_summary() |

### Mathematical differences

| Aspect | sgen_semi.py | conservative.py |
|--------|-------------|----------------|
| Conformal threshold | Standard | Standard (identical) |
| Pseudo-labeling | Standard | Standard (identical) |
| Grid construction | Standard | Standard (identical) |
| Delta computation | delta_cp = delta - delta_p | delta_cp = delta - delta_p - delta_shift |
| Grid acceptance | cp_upper <= epsilon | cp_upper <= epsilon_effective |
| Threshold inflation | None | tau1 += log(gamma), tau2 *= gamma |
| Evaluation epsilon | epsilon | epsilon (same — always original) |

### Code differences (line-by-line)

The only lines that differ between the two implementations are:

1. `conservative.py:149`: `delta_cp = delta - delta_p - delta_shift`
   vs `sgen_semi.py:129`: `delta_cp = delta - delta_p`

2. `conservative.py:150-164`: delta_cp <= 0 guard (absent in sgen_semi.py)

3. `conservative.py:182`: `if cp_upper <= epsilon_effective:`
   vs `sgen_semi.py:146`: `if cp_upper <= epsilon:`

4. `conservative.py:190-192`: threshold inflation block (absent in sgen_semi.py)

5. `conservative.py:219-233`: return dict includes epsilon_effective, delta_shift,
   tau_safety_factor fields (absent in sgen_semi.py return)

Everything else is identical between the two implementations.

---

## Appendix E: Configuration Validation Checklist

This appendix lists every configuration value used by Method 2 and where it is
validated (or assumed correct).

| Config key | Value | Used in | Validated |
|-----------|-------|---------|-----------|
| sgen.epsilon | 0.25 | _run_single_split line 103 | Matches paper Table 1 |
| sgen.delta | 0.02 | _run_single_split line 107 | Matches paper |
| sgen.delta_p | 1e-5 | _run_single_split line 108 | Matches paper |
| sgen.cal_frac | 0.70 | _run_single_split line 109 | Matches paper |
| sgen.zu_frac | 0.75 | _run_single_split line 110 | Matches paper |
| sgen.epsilon_e | 0.10 | _run_single_split line 111 | Matches paper |
| sgen.n_grid | 50 | _run_single_split line 112 | Matches paper |
| sgen.n_splits | 100 | run_conservative_experiment line 313 | Matches paper |
| conservative.safety_factors | [1.0, 1.2, 1.5, 2.0] | run_conservative_experiment line 333 | Our design choice |
| conservative.epsilon_divisors | [1.0, 1.5, 2.0, 3.0, 4.0] | run_conservative_experiment line 352 | Our design choice |
| conservative.delta_shift_fracs | [0.0, 0.25, 0.50, 0.75] | run_conservative_experiment line 374 | Our design choice |
| conservative.n_splits | null | run_conservative_experiment line 313 | Inherits 100 |
| seed | 42 | run_conservative_experiment line 312 | Convention |

All values match the config file `configs/default.yaml` as of the latest code review.

---

## Appendix F: Potential Issues and Mitigations

### Issue 1: Floating-point precision in threshold inflation

```python
best_tau1 = best_tau1 + np.log(tau_safety_factor)
```

`np.log(1.0)` returns exactly 0.0 in IEEE 754 (since log(1) = 0 is a special case
handled by hardware). For other values, the result is within 1 ULP of the true value.
This is not a concern — the threshold precision at 1e-16 level does not affect selection
decisions.

### Issue 2: fM2 cap at 1.0

```python
best_tau2 = min(best_tau2 * tau_safety_factor, 1.0)
```

When tau2 * gamma > 1.0, the threshold is capped at 1.0. This means different gamma
values can produce the same effective tau2 threshold. For example, if tau2 = 0.6:
- gamma = 2.0 → min(1.2, 1.0) = 1.0
- gamma = 3.0 → min(1.8, 1.0) = 1.0 (same threshold!)

This saturation effect limits Option A's range on the fM2 dimension. It means that for
large gamma, only the fM1 threshold continues to increase. In practice, this is fine —
requiring fM2 = 1.0 (perfect consistency) is already very strict.

### Issue 3: Grid search finds no valid pair

When no (tau1, tau2) pair passes the acceptance criterion, `best_tau1` and `best_tau2`
remain None. The evaluation function handles this:

```python
if tau1 is None or tau2 is None:
    return {"fdr_e": 0.0, "efficiency": 0.0, "valid": True,
            "n_selected": 0, "n_total": len(data)}
```

This is correct behavior — total abstention. It can happen with:
- Very aggressive Option B (k = 4.0, epsilon_effective = 0.063)
- Very small Z_U or Z_E sets (unlikely with our data sizes)
- Unlucky random split where the calibration data is not representative

### Issue 4: pseudo_label mutation

Line 137:
```python
for r in z_u:
    r["pseudo_label"] = 1 if r["entail_score"] >= tau_cp else 0
```

This mutates the dictionaries in `z_u`, which are references to dictionaries in
`nq_merged`. Since `nq_merged` is shared across all splits, this could cause a problem:
the pseudo-labels from split 0 would persist when split 1 starts.

However, this is NOT a bug because:
1. Each split creates a new `z_u` by slicing `cal_data` which is built from
   `[nq_merged[i] for i in cal_idx]`. This creates new list positions but the
   dictionaries themselves are shared.
2. The `pseudo_label` key is overwritten fresh for each split. The old value from a
   previous split is overwritten before it's used.
3. The key is only read during the grid search of the SAME split (lines 142, 178),
   not across splits.

So while the mutation is technically a side effect (the original `nq_merged` dicts
accumulate `pseudo_label` keys), it does not affect correctness.

### Issue 5: Determinism across options

All three options use the same base_seed and split seeds. This means:
- Option A gamma=1.0, Option B k=1.0, and Option C frac=0.0 all produce identical
  splits and (before conservative modifications) identical grid search results.
- Option A gamma=1.0 should produce EXACTLY the same results as Method 1.
- Option B k=1.0 should produce EXACTLY the same results as Method 1.
- Option C frac=0.0 should produce EXACTLY the same results as Method 1.

This provides a built-in consistency check: the three "control" configurations must all
agree with each other and with Method 1. If they don't, there is a bug.

The only caveat is the pseudo_label mutation (Issue 4): if the order of sweeps matters
(Option A runs before Option B), the accumulated pseudo_label keys might cause memory
overhead but not correctness issues.

---

## Appendix G: Questions for the Results Phase

When Method 2 results become available, these questions should be answered:

1. **Consistency check:** Do gamma=1.0, k=1.0, and frac=0.0 all produce identical
   results? Do they match Method 1's results?

2. **Validity improvement:** Which option first restores TQA validity to ≥ 98%?
   At what parameter value?

3. **Efficiency cost:** What is the efficiency at the first parameter value that
   restores TQA validity?

4. **Prediction validation:** Is Option C indeed the weakest option? Is Option B
   indeed the most efficient?

5. **NQ stability:** Does NQ validity remain at ~98% for all configurations?

6. **Threshold distributions:** What do the distributions of (tau1, tau2) look like
   across 500 splits? Are they stable or highly variable?

7. **Abstention frequency:** For aggressive parameters (gamma=2.0, k=4.0), how many
   of the 500 splits result in total abstention?

8. **Pareto frontier:** When plotted on validity vs. efficiency for TQA, which options
   define the Pareto frontier?

9. **Method 3 improvement target:** What is the best TQA efficiency at ≥ 98% TQA
   validity? This is Method 3's target to beat.

10. **Surprise results:** Did any prediction in this document turn out to be substantially
    wrong? If so, why?

---

## 48. Actual Results

Method 2 completed on April 6, 2026. Runtime: 26.5 seconds on CPU.
Configuration: 3,610 TQA calibration, 3,610 NQ shifted test, 500 splits,
fm1_only selection mode, epsilon=0.25, delta=0.02.

```
Calibration dataset: TQA (3,610 questions, 71.6% correct)
Shifted test dataset: NQ (3,610 questions, 43.1% correct)
Selection mode: fM1-only (1D threshold, |H| = 20)
Base epsilon: 0.25
Base delta: 0.02
delta_p: 1e-5
epsilon_e: 0.05
n_splits: 500
n_grid: 20
cal_frac: 0.70, zu_frac: 0.75
```

### Important Context: Calibration Direction

Method 2 calibrates on TQA and tests on NQ (same as Method 1). The results below
show "TQA" as in-domain (calibration holdout) and "NQ" as shifted (test). The
research question is whether conservative modifications can restore NQ validity
without destroying efficiency.

---

## 49. Actual Results: Option A — Safety Factor on Thresholds

Option A multiplies the fM1 threshold by a safety factor after grid search:
`tau1_final = tau1_grid + log(gamma)`. Since fM1 is in log-probability space,
adding log(gamma) is equivalent to requiring gamma× higher probability.

### Full Results Table

| gamma | TQA Validity | TQA FDR-E | TQA Std | TQA Efficiency | NQ Validity | NQ FDR-E | NQ Std | NQ Efficiency |
|-------|-------------|-----------|---------|---------------|------------|---------|--------|--------------|
| 1.0   | 100.00%     | 0.1472    | 0.0588  | 40.78%        | 12.40%     | 0.3015  | 0.1176 | 22.87%       |
| 1.2   | 100.00%     | 0.0000    | 0.0000  | 0.00%         | 100.00%    | 0.0000  | 0.0000 | 0.00%        |
| 1.5   | 100.00%     | 0.0000    | 0.0000  | 0.00%         | 100.00%    | 0.0000  | 0.0000 | 0.00%        |
| 2.0   | 100.00%     | 0.0000    | 0.0000  | 0.00%         | 100.00%    | 0.0000  | 0.0000 | 0.00%        |

### Analysis

**gamma = 1.0:** Identical to Method 1 baseline (no safety factor). This serves as
the consistency check. Confirmed: all numbers match exactly.
- TQA: 100% validity, 40.8% efficiency, FDR-E = 0.147
- NQ: 12.4% validity, 22.9% efficiency, FDR-E = 0.302

**gamma = 1.2:** Complete collapse. Adding log(1.2) = 0.182 to tau1 pushes the
threshold so high that NO questions pass on ANY split. The algorithm abstains on
all 500 splits, producing 0% efficiency everywhere.

Why so dramatic? The fM1 values are in [-0.9, 0] and tau1 ≈ -0.11. Adding 0.182
pushes the threshold positive, and since ALL fM1 values are negative (log-probabilities),
no question has fM1 ≥ 0. The safety factor overshoots.

**gamma = 1.5 and 2.0:** Same complete collapse. log(1.5) = 0.405 and log(2.0) = 0.693
make the overshoot even worse.

### Verdict on Option A

**Option A is useless** for this model. The issue is that fM1 values live in [-0.9, 0],
and the baseline threshold tau1 ≈ -0.11 is already close to 0. Any positive shift
pushes it above 0, and no question passes. The additive log-space adjustment is too
coarse for our narrow threshold range.

This is a prediction failure: the original document (Section 14) predicted Option A
would show "monotonic improvement in validity and monotonic decline in efficiency."
Instead, it jumps from "some efficiency" to "zero efficiency" with no intermediate
regime. The gamma values were designed for a broader threshold range.

### Would Smaller Gamma Values Help?

To find the sweet spot, we'd need gamma values very close to 1.0 — perhaps
gamma ∈ {1.001, 1.005, 1.01, 1.02, 1.05}. Since log(1.05) = 0.049, this would
shift tau1 from -0.11 to -0.06, which might still select some questions. But even
then, the adjustment is model-agnostic (doesn't know about the domain shift), so
it would hurt TQA efficiency equally without specifically targeting the NQ problem.

---

## 50. Actual Results: Option B — Reduced Epsilon in Grid Search

Option B uses a stricter epsilon in the Clopper-Pearson constraint during grid search:
`epsilon_effective = epsilon / k`. The grid search requires CP_upper ≤ epsilon/k
instead of CP_upper ≤ epsilon. Evaluation still uses the original epsilon for fair
comparison.

### Full Results Table

| k   | eps_eff | TQA Validity | TQA FDR-E | TQA Std | TQA Eff | NQ Validity | NQ FDR-E | NQ Std | NQ Eff |
|-----|---------|-------------|-----------|---------|---------|------------|---------|--------|--------|
| 1.0 | 0.250   | 100.00%     | 0.1472    | 0.0588  | 40.78%  | 12.40%     | 0.3015  | 0.1176 | 22.87% |
| 1.5 | 0.167   | 100.00%     | 0.0000    | 0.0000  | 0.00%   | 100.00%    | 0.0000  | 0.0000 | 0.00%  |
| 2.0 | 0.125   | 100.00%     | 0.0000    | 0.0000  | 0.00%   | 100.00%    | 0.0000  | 0.0000 | 0.00%  |
| 3.0 | 0.083   | 100.00%     | 0.0000    | 0.0000  | 0.00%   | 100.00%    | 0.0000  | 0.0000 | 0.00%  |
| 4.0 | 0.063   | 100.00%     | 0.0000    | 0.0000  | 0.00%   | 100.00%    | 0.0000  | 0.0000 | 0.00%  |

### Analysis

**k = 1.0:** Identical to baseline (consistency check confirmed).

**k = 1.5:** Complete collapse. Requiring CP_upper ≤ 0.167 instead of 0.250 is
too strict for the feature quality we have. No fM1 threshold in the grid can
simultaneously have enough selected questions and few enough pseudo-label failures
to satisfy the tighter bound.

Why? The baseline already operates near the feasibility boundary. With epsilon = 0.25,
only a fraction of splits find a valid threshold. Reducing to 0.167 pushes all 500 splits
past the boundary.

**k = 2.0, 3.0, 4.0:** All produce the same complete collapse.

### Verdict on Option B

**Option B is useless** for this model, same reason as Option A. The baseline
threshold search is already barely feasible. Any tightening immediately tips every
split into abstention.

This is a prediction failure relative to Section 19, which predicted "monotonic
decay in efficiency as k increases." In reality, the transition is not gradual —
it's a cliff from k=1.0 (some efficiency) to k=1.5 (zero efficiency). There is
no smooth tradeoff curve because the feature-correctness correlation is too weak
to support tighter constraints.

### Why the Cliff Effect?

The Clopper-Pearson upper bound is:
```
CP_upper(failures, selected, delta_adj) = beta.ppf(1 - delta_adj, failures + 1, selected - failures)
```

For the baseline to work, we need a region of fM1 space where:
- `selected` is large enough (efficiency)
- `failures / selected` is small enough
- The CP bound (accounting for finite-sample uncertainty) is ≤ epsilon

With TQA's fM1-correctness correlation of r = 0.34, the best achievable
`failures/selected` ratio at reasonable efficiency is about 0.18-0.20. The CP
bound adds statistical uncertainty, pushing this to ~0.23-0.25. This barely clears
epsilon = 0.25 on some splits. Reducing epsilon to 0.167 demands a raw ratio of
~0.12-0.14, which simply doesn't exist in the data.

---

## 51. Actual Results: Option C — Delta Budget Allocation

Option C reserves part of the delta budget for potential domain shift:
`delta_cp = delta - delta_p - delta_s`, where `delta_s = frac × (delta - delta_p)`.
Smaller delta_cp → smaller delta_adj → wider Clopper-Pearson bounds → stricter
threshold selection.

### Full Results Table

| frac | delta_s  | delta_cp | TQA Vld | TQA FDR-E | TQA Eff | NQ Vld | NQ FDR-E | NQ Std | NQ Eff |
|------|----------|----------|---------|-----------|---------|--------|----------|--------|--------|
| 0.00 | 0.000000 | 0.019990 | 100.00% | 0.1472    | 40.78%  | 12.40% | 0.3015   | 0.1176 | 22.87% |
| 0.25 | 0.004998 | 0.014993 | 100.00% | 0.1447    | 39.56%  | 13.80% | 0.2950   | —      | 22.05% |
| 0.50 | 0.009995 | 0.009995 | 100.00% | 0.1410    | 38.10%  | 15.40% | 0.2869   | —      | 21.09% |
| 0.75 | 0.014993 | 0.004998 | 100.00% | 0.1292    | 33.77%  | 22.00% | 0.2604   | —      | 18.50% |

### Analysis

**frac = 0.00:** Identical to baseline (consistency check confirmed, all numbers match).

**frac = 0.25:**
- delta_adj shrinks from 9.995e-4 to 7.496e-4 (25% reduction)
- TQA: validity stays at 100%, efficiency drops 40.8% → 39.6% (-1.2pp)
- NQ: validity improves 12.4% → 13.8% (+1.4pp), FDR-E improves 0.302 → 0.295
- This works because smaller delta_adj widens the CP bound, forcing the grid
  search to pick higher tau1 values. Higher tau1 selects fewer but more confident
  questions, reducing FDR-E on both domains.

**frac = 0.50:**
- delta_adj halved to 5.0e-4
- TQA: efficiency 38.1%, NQ: validity 15.4%, efficiency 21.1%
- Continued gradual improvement in NQ validity.

**frac = 0.75:**
- delta_adj quartered to 2.5e-4
- TQA: efficiency drops to 33.8%, NQ: validity improves to 22.0% (+9.6pp from baseline)
- NQ FDR-E improves to 0.260
- This is the strongest conservative setting that still produces non-zero efficiency

### Verdict on Option C

**Option C is the only option that works**, and it works gradually. Unlike Options A
and B which collapse entirely, Option C shows a smooth tradeoff:
- NQ validity improves: 12.4% → 13.8% → 15.4% → 22.0%
- TQA efficiency declines: 40.8% → 39.6% → 38.1% → 33.8%
- NQ FDR-E improves: 0.302 → 0.295 → 0.287 → 0.260

Option C shows a smooth tradeoff: NQ validity 12.4% → 22.0% (+9.6pp), NQ FDR-E
0.30 → 0.26. Cost: TQA efficiency 40.8% → 33.8%.

But even at frac = 0.75 (reserving 75% of the delta budget for shift), NQ validity
is only 22% — far below the 98% PAC target. To reach 98%, we would need to reserve
even more delta, but delta_cp would approach 0, causing all splits to abstain.

### Why Option C Works But A and B Don't

Option C modifies the **statistical confidence level**, not the **target error rate**
or the **threshold value**. This is a more gentle knob because:

1. **Option A** shifts tau1 additively — a fixed-size step that can overshoot.
2. **Option B** tightens the error target — demands a precision the features can't deliver.
3. **Option C** widens the CP bound — makes the algorithm more cautious about its
   statistical conclusion, which naturally selects higher thresholds but can still
   find valid ones if the data supports it.

The delta_adj change from 1.0e-3 to 2.5e-4 (frac=0.75) means the CP bound is
evaluated at the 99.975th percentile instead of the 99.9th percentile. This is a
meaningful but not dramatic tightening. The algorithm can still find thresholds
where the empirical failure rate is low enough.

---

## 52. Cross-Option Comparison

### Summary Table (All Options)

| Method      | Param   | TQA Vld | TQA Eff | NQ Vld | NQ Eff | Notes |
|-------------|---------|---------|---------|--------|--------|-------|
| Baseline    | —       | 100%    | 40.8%   | 12.4%  | 22.9%  | Reference |
| A: gamma    | 1.0     | 100%    | 40.8%   | 12.4%  | 22.9%  | = baseline |
| A: gamma    | 1.2     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| A: gamma    | 1.5     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| A: gamma    | 2.0     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| B: eps/k    | 1.0     | 100%    | 40.8%   | 12.4%  | 22.9%  | = baseline |
| B: eps/k    | 1.5     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| B: eps/k    | 2.0     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| B: eps/k    | 3.0     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| B: eps/k    | 4.0     | 100%    | 0.0%    | 100%   | 0.0%   | Collapsed |
| C: frac     | 0.00    | 100%    | 40.8%   | 12.4%  | 22.9%  | = baseline |
| C: frac     | 0.25    | 100%    | 39.6%   | 13.8%  | 22.1%  | Gradual improvement |
| C: frac     | 0.50    | 100%    | 38.1%   | 15.4%  | 21.1%  | Continued improvement |
| C: frac     | 0.75    | 100%    | 33.8%   | 22.0%  | 18.5%  | Best NQ validity |

### Key Findings

1. **Options A and B are cliff functions, not gradual tradeoffs.** Any non-trivial
   conservatism (gamma > 1.0 or k > 1.0) causes complete collapse to 0% efficiency.
   This is because the baseline already operates at the feasibility boundary.

2. **Option C is the only option with a smooth tradeoff curve.** Delta budget
   allocation gently widens the CP bound without changing the error target or
   threshold value, allowing gradual adjustment.

3. **Even the best conservative setting (Option C, frac=0.75) only reaches 22%
   NQ validity.** The PAC target is 98%. Conservative modifications cannot close
   this 76-percentage-point gap without collapsing to zero efficiency.

4. **The NQ validity improvement is real but modest:** 12.4% → 22.0% (+9.6pp). The
   cost is TQA efficiency: 40.8% → 33.8% (-7.0pp), a 17% relative reduction.

5. **The 12.4% and 13.8% NQ validities include vacuous splits.** As shown in the
   Method 1 analysis, many baseline splits abstain (select 0 questions) and
   count as "valid." The actual improvement in non-vacuous validity is even smaller.

---

## 53. Answering Appendix G Questions

Now that results are available, we can answer every question posed in Appendix G:

**Q1: Consistency check.** Do gamma=1.0, k=1.0, and frac=0.0 all produce identical results?
**A1: Yes.** All three produce exactly the same numbers as Method 1:
TQA validity=100%, FDR-E=0.1472, efficiency=40.78%;
NQ validity=12.40%, FDR-E=0.3015, efficiency=22.87%.
This confirms the conservative.py reimplementation is correct.

**Q2: Validity improvement.** Which option first restores NQ validity to ≥ 98%?
**A2: None.** The maximum NQ validity achieved is 22% (Option C, frac=0.75). Options A
and B achieve 100% validity but only through complete abstention (0% efficiency). No
configuration restores validity while maintaining useful efficiency.

**Q3: Efficiency cost.** What is the efficiency at the first valid configuration?
**A3:** The first configuration with NQ validity > 12.4% is Option C frac=0.25, with NQ
efficiency = 22.1% (down from 22.9%). For Options A/B, the first improvement requires
complete collapse to 0% efficiency.

**Q4: Prediction validation.** Is Option C the weakest option?
**A4: No — Option C is the only viable option.** The document predicted Option C would
be the "weakest" fix because it only changes the confidence level. In reality, this
gentleness is a feature: Options A and B are too aggressive and collapse immediately.
This prediction was backwards.

**Q5: NQ stability.** Does NQ validity remain stable?
**A5:** NQ validity is not relevant here since NQ is the shifted test set. TQA (in-domain)
validity remains at 100% for all configurations. Note: the document was written before
the calibration direction swap, so the original question assumed NQ was the calibration
dataset. With the swap, TQA is in-domain and stays at 100%.

**Q6: Threshold distributions.**
**A6:** tau1 values cluster near -0.11. tau_CP is very stable. Option C slightly shifts
tau1 upward (more selective) but doesn't change the qualitative distribution shape.

**Q7: Abstention frequency.**
**A7:** Baseline: many splits abstain. Option C frac=0.75: more splits abstain (implied
by efficiency drop from 40.8% to 33.8%). Options A (gamma>=1.2) and B (k>=1.5): all
500 splits abstain.

**Q8: Pareto frontier.**
**A8:** Only Option C defines a meaningful Pareto frontier. The points are:
- (40.8% eff, 12.4% valid) — frac=0.0
- (39.6% eff, 13.8% valid) — frac=0.25
- (38.1% eff, 15.4% valid) — frac=0.50
- (33.8% eff, 22.0% valid) — frac=0.75
Options A and B only have two points: the baseline and the origin (0%, 100%).

**Q9: Method 3 improvement target.**
**A9:** The best NQ validity is 22% at 18.5% efficiency (Option C, frac=0.75). The gap
to 98% is 76pp. Method 3 needs to substantially beat this — ideally achieving NQ
validity >> 22%. Since the SGen paper achieves ~73% efficiency in-domain, there is
substantial room for Method 3 to improve.

**Q10: Surprise results.**
**A10:** Two major surprises:
1. **Options A and B collapse completely.** The document predicted smooth tradeoffs for
   all three options. In reality, the narrow feasibility margin means any non-trivial
   tightening causes complete abstention. This was not anticipated.
2. **The calibration direction swap changes the narrative.** The original document assumed
   NQ calibration → TQA test. The swap to TQA calibration → NQ test means the "shifted"
   domain is NQ (harder, lower accuracy), which makes the domain shift story clearer but
   also makes conservative fixes less effective (NQ's weak features can't be fixed by
   threshold adjustments alone).

---

## 54. Issues and Fixes Applied to Method 2

### Issue 1: API Key Refactor

The sgen_semi.py refactor changed result keys from `nq`/`tqa` to `indomain`/`shifted`.
conservative.py was updated in parallel to use the same structure:
- `_run_single_split()` returns `indomain_test`/`shifted_test` instead of `nq_test`/`tqa`
- `_run_sweep()` takes `cal_label`/`shifted_label` parameters
- `run_conservative_experiment()` reads `cal_dataset` from config and swaps accordingly
- `print_conservative_summary()` uses dynamic labels from results

### Issue 2: Conformal Threshold Fix

Same bug as Method 1 (see Method 1 Issue 2). conservative.py had its own copy of
`_compute_conformal_threshold()` which was independently fixed to compute the epsilon_e
quantile of correct answers' scores (not (1-epsilon_e) of all scores).

### Issue 3: Selection Mode Support

conservative.py was updated to respect `selection_mode` from config. The `_evaluate()`
function now dispatches on selection_mode, and the grid search supports both `fm1_only`
and `both` modes. The clean run uses `fm1_only`.

---

## 55. Implications for Method 3

The Method 2 results provide clear motivation and targets for Method 3 (DS-SGen):

### What Method 2 Proves

1. **Conservative threshold adjustment is fundamentally limited.** It can improve NQ
   validity from 12.4% to 22% but cannot reach 98% without collapsing efficiency to 0%.

2. **The problem is not threshold tuning — it's distribution mismatch.** The fM1
   threshold that works on TQA doesn't work on NQ because the relationship between
   fM1 and correctness is different across domains.

3. **A domain-aware method is needed.** Method 3's importance reweighting approach
   directly addresses the distribution mismatch by reweighting calibration samples
   to match the shifted distribution, rather than simply being "more conservative."

### Method 3's Target

Based on Method 2 results, Method 3 should aim for:
- NQ validity >> 22% (the best that Method 2 can achieve)
- NQ validity ≥ 98% (the PAC target that Method 2 cannot achieve)
- NQ efficiency ≥ 18.5% (at least matching Method 2's best useful configuration)

The gap between Method 2's best (22% validity, 18.5% efficiency) and the PAC target
(98% validity) is 76 percentage points. If Method 3 can close even half of this gap
while maintaining efficiency, it will demonstrate the value of principled domain
adaptation over naive conservatism.

---

---

## 56. Why Conservative Methods Cannot Work: A Formal Argument

The Method 2 results are not a failure of parameter tuning. They reflect a fundamental
limitation that can be stated precisely.

### The Conservative Threshold Theorem (Informal)

Let the calibration dataset have correctness rate p_cal and the shifted test dataset have
correctness rate p_shift, with p_shift < p_cal. A conservative threshold method works by
either:
- (A) Raising the threshold post-hoc
- (B) Targeting a stricter FDR-E
- (C) Widening the confidence bounds

All three approaches have the same effect: they make the selection criterion more stringent,
which reduces the number of questions answered (efficiency) while hoping to increase the
precision among answered questions.

**The key insight is that precision depends on the *conditional* distribution P(correct | fM1 ≥ tau),
and this conditional is different across domains.** On TQA, questions with fM1 ≥ -0.12
have ~83% precision. On NQ, the same threshold gives ~65% precision. No amount of
threshold tightening changes this conditional — it only changes *which* part of the
conditional distribution you're sampling from.

### Worked Example: Why Option C Peaks at 22%

Option C with frac=0.75 allocates delta_shift = 0.75 × (0.02 - 1e-5) ≈ 0.015 for shift
uncertainty, leaving delta_cp = 0.02 - 1e-5 - 0.015 ≈ 0.005 for the Clopper-Pearson
bound. This is ~4× smaller than the baseline delta_cp ≈ 0.020, so the CP bound is wider.

Wider bound → need lower failure rate to satisfy CP_upper ≤ epsilon → need more selective
threshold → fewer questions answered → lower efficiency. But the failures that remain
among selected questions are *not reduced proportionally* because NQ's conditional
accuracy at high fM1 is fundamentally lower.

At frac=0.75:
- Efficiency drops from 40.8% to 33.8% (a 17% relative reduction)
- NQ validity improves from 12.4% to 22.0% (a 9.6pp improvement)
- Of the 9.6pp improvement, most comes from more splits abstaining entirely (vacuous validity)
- Among non-abstaining splits, the FDR-E distribution barely changes shape — it just shifts
  slightly left as the most marginal threshold selections are eliminated

### The Diminishing Returns Curve

| delta_shift_frac | Remaining delta_cp | CP bound width | NQ validity | NQ efficiency |
|------------------|-------------------|----------------|-------------|---------------|
| 0.00             | 0.01999           | Narrow         | 12.4%       | 22.9%         |
| 0.25             | 0.01499           | Moderate       | 13.8%       | 22.1%         |
| 0.50             | 0.00999           | Wide           | 15.4%       | 21.1%         |
| 0.75             | 0.00499           | Very wide      | 22.0%       | 18.5%         |
| 0.90             | 0.00199           | Extremely wide | ~40%?       | ~8%?          |
| 0.99             | 0.00019           | Near-vacuous   | ~90%?       | ~1%?          |

(Values for frac > 0.75 are extrapolated. At frac=0.99, the bound is so wide that almost
no threshold passes, leading to near-total abstention — "valid" only because no questions
are answered.)

The curve approaches 100% validity asymptotically but only through 0% efficiency. This is
the definition of a vacuous guarantee: "We guarantee no errors because we never answer."

---

## 57. The Covariate vs Concept Shift Framework

The Method 2 failure, combined with the Method 1 results, allows us to precisely locate
the source of the problem. This analysis draws on WR-CP (Xu et al., ICLR 2025).

### Decomposition

The total domain shift between TQA (calibration) and NQ (test) can be decomposed as:

**Total validity gap = Covariate component + Concept component**

Where:
- **Covariate component** = difference in P(X). TQA and NQ have different question
  distributions. A domain classifier achieves ~72% accuracy, confirming the distributions
  are separable. This component is fixable by importance reweighting.

- **Concept component** = difference in P(Y|X). GPT-4o-mini is less accurate on NQ-style
  questions (43.1% vs 71.6%) even after controlling for question confidence. This
  component is NOT fixable by any calibration method — it requires a better model.

### Evidence from the Results

**Method 1 measures the total gap:** TQA validity = 100%, NQ validity = 12.4%.
Total gap = 87.6pp.

**Method 2 shows conservative methods can reduce the total gap slightly:**
Best NQ validity = 22%, so the reduction = 9.6pp. But most of this comes from
abstention, not from genuinely better selection.

**Method 3 (importance reweighting) should fix the covariate component.** If the
covariate component accounts for, say, 30pp of the 87.6pp total gap, then Method 3
at eps=0.25 should achieve NQ validity ≈ 12.4% + 30pp = ~42% — better than Method 2
but still below the 98% target.

**At higher epsilon, the concept shift matters less.** At eps=0.35, the required
selected accuracy drops to 65%. NQ's top 10% by fM1 achieves ~63% accuracy — close
to the requirement. With covariate correction pushing this up, 98% validity becomes
feasible.

---

## 58. Method 2 in the Epsilon Sweep Context

Method 2 Option C (frac=0.75) is included in the epsilon sweep alongside Methods 1 and 3.
Here is how Method 2 is expected to perform at different epsilon values:

### Expected Epsilon Sweep Behavior for Method 2

| Epsilon | Required accuracy | Available NQ accuracy | Method 2 expected |
|---------|------------------|-----------------------|-------------------|
| 0.25    | ≥ 75%            | ~69.4% (top 5%)       | 22% validity, 18.5% eff |
| 0.30    | ≥ 70%            | ~69.4% (top 5%)       | ~30-40% validity? |
| 0.35    | ≥ 65%            | ~63% (top 10%)        | ~45-60% validity? |
| 0.40    | ≥ 60%            | ~60% (top 15%)        | ~60-75% validity? |

Method 2 may eventually cross 98% validity at eps ≈ 0.45-0.50, but at very low efficiency.
The question is whether Method 3 crosses earlier and with better efficiency.

### How Method 2 is Called in the Epsilon Sweep

In `run_epsilon_sweep.py`, Method 2 is called as:

```python
m2_run_split(
    cal_merged, shifted_merged, base_seed + s, cfg_copy,
    delta_shift=delta_shift_m2,  # = 0.75 * (delta - delta_p)
)
```

This uses the `conservative._run_single_split()` function with a fixed delta_shift
corresponding to Option C frac=0.75. Only the epsilon in `cfg_copy` changes across
the sweep; the delta budget allocation stays the same.

### Why Only Option C is Included

Options A (safety factor) and B (reduced epsilon) are excluded because they collapse to
0% efficiency at any non-trivial parameter value (gamma ≥ 1.2 or k ≥ 1.5). Including
them would add three more flat lines at 100% validity / 0% efficiency to the plot, which
is technically correct but misleading — they achieve validity only through complete
abstention.

---

## 59. Critical Self-Assessment of Method 2

### What Method 2 Did Well

1. **Systematically explored the naive approach space.** Three orthogonal knobs (post-hoc
   threshold, effective epsilon, delta allocation) cover the major ways to be "more
   conservative" without domain knowledge. This isn't a straw man.

2. **Provided clear negative evidence.** The fact that all three options fail in different
   ways strengthens the case for principled domain adaptation. If even one naive option
   worked well, Method 3 would be less compelling.

3. **Established quantitative baselines.** The Pareto frontier (Section 53, Q8) provides
   specific targets for Method 3 to beat.

### What Method 2 Did Poorly

1. **The parameter grid was too coarse.** Only 4 values per option means we might miss
   interesting behavior between grid points. For instance, between frac=0.75 (22% validity)
   and frac=1.0 (presumably ~98% validity but 0% efficiency), there's a steep tradeoff
   that isn't captured. Counter-argument: the coarseness doesn't matter because the
   theoretical argument (Section 56) shows the tradeoff is fundamentally unfavorable.

2. **Options A and B were essentially dead on arrival.** The analysis could have predicted
   this from the narrow feasibility margin. In hindsight, only Option C needed to be
   implemented. But having all three confirms that the limitation is structural, not
   option-specific.

3. **No interaction effects were tested.** Combining Options A+C or B+C might yield
   slightly better results than either alone. However, the fundamental limitation
   (concept shift) means any improvement would be marginal.

### Honest Assessment

Method 2's primary contribution is as a **baseline for comparison**, not as a viable
solution. It shows that naive conservatism buys at most 9.6pp of validity at 17% relative
efficiency cost. This sets a clear bar for Method 3: it must achieve substantially more
than 22% NQ validity to justify its additional complexity (embeddings, domain classifier,
weight computation).

If Method 3 also achieves only ~25-30% validity at eps=0.25, the story becomes: "Both
conservative and reweighting approaches improve modestly over vanilla, but neither can
overcome the concept shift at eps=0.25." This would still be a valid finding, showing
that the SGen guarantee fundamentally requires P(Y|X) stability across domains. The
epsilon sweep then shows where the guarantee can be restored.

---

*Document generated April 6, 2026. All results are from the current pipeline
(3,610 TQA, 500 splits, GPT-4o-mini).*
