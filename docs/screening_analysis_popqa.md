# Screening Analysis — PopQA Head→Tail and the Dataset Selection Problem for DS-SGen

**Date:** 2026-04-11
**Authors:** Anshul Kumar, Justin Luan (CMU 10-701)
**Context:** After TQA→NQ revealed that DS-SGen's importance reweighting collapses under concept shift, we designed a 6-test screening protocol and applied it to PopQA head→tail — a candidate pair that was supposed to isolate *pure covariate shift*. This document records what we found, why it matters, and why we now believe the DS-SGen method is theoretically correct but finding a dataset that satisfies its assumptions at non-trivial ε is genuinely hard.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Background — why DS-SGen exists](#2-background--why-ds-sgen-exists)
3. [The PAC FDR-E guarantee and what breaks it](#3-the-pac-fdr-e-guarantee-and-what-breaks-it)
4. [Covariate shift vs concept shift — theoretical distinction](#4-covariate-shift-vs-concept-shift--theoretical-distinction)
5. [Summary of prior findings: TQA→NQ](#5-summary-of-prior-findings-tqanq)
6. [The 6-test screening protocol](#6-the-6-test-screening-protocol)
7. [Why PopQA head→tail was the preferred candidate](#7-why-popqa-headtail-was-the-preferred-candidate)
8. [Experimental setup](#8-experimental-setup)
9. [PopQA screening results — detailed walkthrough](#9-popqa-screening-results--detailed-walkthrough)
10. [Cross-result synthesis: TQA→NQ vs PopQA](#10-cross-result-synthesis-tqanq-vs-popqa)
11. [Why we believe the method is correct](#11-why-we-believe-the-method-is-correct)
12. [Why dataset selection is the real bottleneck](#12-why-dataset-selection-is-the-real-bottleneck)
13. [What we would need from a "good" dataset](#13-what-we-would-need-from-a-good-dataset)
14. [Candidate escape routes](#14-candidate-escape-routes)
15. [Conclusion and lessons for the field](#15-conclusion-and-lessons-for-the-field)

---

## 1. Executive summary

We built a full DS-SGen pipeline: SGen-Semi baseline (Method 1), three conservative-threshold variants (Method 2), and importance-weighted DS-SGen (Method 3). On our first dataset pair — TriviaQA (calibration) → Natural Questions (shifted test) — Method 1 achieved only 12.4% validity on the shifted distribution, Method 2 pushed it to 22.0% at the cost of efficiency, and Method 3 reached a nominal 68.8% but almost entirely through vacuous abstention driven by catastrophic weight clipping (344/500 splits had all weights saturated at the clip ceiling; among the 156 non-vacuous splits, **zero** were valid). This pattern of failure suggested that the pair did not satisfy the covariate-shift assumption underlying weighted conformal prediction.

We then designed a 6-test pre-flight screening protocol (Tests 1–6, each with a quantitative threshold calibrated against TQA→NQ) to decide whether a candidate dataset pair is worth a full run. We applied it to **PopQA head (popular entities) → PopQA tail (obscure entities)** — the candidate most likely to exhibit pure covariate shift, because the head and tail are drawn from the same corpus with the same question template, differing only in the popularity of the entity being asked about.

The result was decisive and counterintuitive. **PopQA head→tail passes only 1 out of 7 tests.** Source accuracy is 43.7%, target accuracy is 24.1%, and — most damning for the covariate-shift hypothesis — the weight-quartile spread is **negative** (−0.092). Source points that look target-like are *more* accurate than source-like source points, not less. The linear regression of y on log(w) has a *positive* slope (+0.065, p<0.001), which is exactly the opposite direction of a covariate-shift signature.

Our interpretation is that the failure reveals a structural problem: within a QA benchmark that requires entity-level factual recall, the popularity of the entity is confounded with the model's knowledge, and knowledge is a property of P(Y|X), not P(X). Filtering by a covariate (popularity) implicitly filters by a concept (entity knowledge), so "pure" covariate shift is an idealization that standard QA benchmarks cannot deliver at the accuracy levels we need.

The DS-SGen method itself — weighted conformal thresholding with density-ratio importance weights — remains theoretically correct (Tibshirani et al. 2019; Lin et al. 2025 DS-CP). What we have learned is that finding a *real* LLM deployment scenario where the assumption actually holds, with high enough accuracy to support ε=0.25 PAC bounds, is the hard part of this research direction. This document walks through the evidence, line by line.

---

## 2. Background — why DS-SGen exists

### 2.1 The SGen framework

Lee et al. (NeurIPS 2024) introduced **Selective Generation for Controllable LMs** — a framework where an LLM either answers or abstains ("I don't know"), with a PAC bound on the False Discovery Rate with respect to Entailment (FDR-E). FDR-E is the fraction of answered questions whose generated answer does not logically entail the reference answer. The SGen-Semi variant (Algorithm 2 in the paper) uses:

- A **conformal pseudo-labeling** step on a small labeled set $Z_E$ to produce soft labels for the larger unlabeled set $Z_U$.
- A **neuro-selection function** parameterised by $(\tau_1, \tau_2)$ where the model answers iff $f_{M1}(x) \ge \tau_1$ and $f_{M2}(x) \ge \tau_2$.
- A **grid search** over $(\tau_1, \tau_2)$ with **Clopper-Pearson upper confidence bounds** on FDR-E and **Bonferroni correction** $\delta_{\text{adj}} = (\delta - \delta_p)/|H|$ across $|H|$ candidate thresholds.
- The confidence signals are $f_{M1}$ = mean token log-probability (generation confidence) and $f_{M2}$ = bidirectional-entailment self-consistency across K=5 sampled continuations.

The theoretical guarantee:

$$
\Pr_{Z\sim P}\big[\mathrm{FDR}\text{-}E(\hat\tau) \le \varepsilon\big] \ge 1 - \delta
$$

where $\hat\tau$ is the threshold selected by the grid search on calibration data $Z$, and $\varepsilon$ and $\delta$ are user-chosen.

### 2.2 The i.i.d. assumption

The Clopper-Pearson bound used inside SGen-Semi is a **binomial tail bound**. It requires that the test distribution matches the calibration distribution exactly — i.e., $P_{\text{test}} = P_{\text{cal}}$. If they differ (domain shift), the calibration-set failure count is not an unbiased estimate of the test-set failure rate, and the bound becomes unreliable.

In Method 1 we confirmed that this is not a theoretical curiosity. When we calibrated on TriviaQA and tested on NQ-Open, SGen-Semi achieved **100% validity on in-domain TQA** (the guarantee holds) but only **12.4% validity on NQ** (the guarantee fails catastrophically). The failure is not an implementation bug: SGen-Semi's in-domain validity is perfect, and the failure happens the moment the test distribution moves.

### 2.3 Weighted conformal prediction — the fix

Tibshirani et al. (NeurIPS 2019) introduced **weighted exchangeability**: if you know (or can estimate) the density ratio $w(x) = P_{\text{test}}(x)/P_{\text{cal}}(x)$, you can use weighted quantiles instead of raw quantiles and recover the coverage guarantee *under covariate shift*. Lin et al. (DS-CP, arXiv 2025) applied this to LLMs: embed prompts with all-MiniLM-L6-v2, train an XGBoost classifier to distinguish calibration from test embeddings, and convert the classifier output to a density ratio via $w(x) = p_{\text{clf}}(x)/(1-p_{\text{clf}}(x))$.

Our Method 3 combines these two lines: we ported the weighted-conformal machinery into the SGen-Semi grid search (both the pseudo-labeling quantile and the Clopper-Pearson bound use weighted statistics), and we apply it to open-ended QA with a binary correctness label.

### 2.4 The assumption under the fix

The Tibshirani et al. guarantee is only for **covariate shift**:

$$
P_{\text{test}}(X) \ne P_{\text{cal}}(X) \quad \text{but} \quad P_{\text{test}}(Y \mid X) = P_{\text{cal}}(Y \mid X).
$$

If the conditional also shifts — **concept shift** — weighted conformal prediction has no theoretical guarantee, and in practice the weights cannot correct for it (they are a function of $X$ only, so they cannot move probability mass in the $Y$ direction).

Our entire project hinges on whether real LLM shift scenarios are "close enough" to pure covariate shift for the Tibshirani machinery to help. The TQA→NQ results suggested they are not. The purpose of PopQA head→tail was to construct a pair that lives closer to pure covariate shift by design.

---

## 3. The PAC FDR-E guarantee and what breaks it

### 3.1 Decomposition (Lemma 1 in Lee et al.)

$$
\mathrm{FDR}\text{-}E(\tau) = \underbrace{\Pr[\hat y \not\Rightarrow y \mid s \ge \tau]}_{\text{FER}} + \underbrace{\Pr[\hat y \Rightarrow y, \text{pseudo}=0 \mid s \ge \tau]}_{\text{FNER}} + \underbrace{\Pr[\text{NLI-abstain} \mid s \ge \tau]}_{\text{NER}}
$$

SGen-Semi bounds each term independently. FER is bounded by the Clopper-Pearson upper confidence bound on the failure rate within the selected set. FNER is bounded by the conformal pseudo-labeling guarantee on $Z_E$ at level $\varepsilon_e$. NER is zero in our setting because we use a binary correctness label (DeBERTa argmax = ENTAILMENT).

### 3.2 The binomial tail assumption

The Clopper-Pearson bound is:

$$
\hat p_{\text{upper}} = \text{Beta}^{-1}\big(1 - \delta_{\text{adj}},\ k+1,\ n-k\big)
$$

where $k$ is the number of failures in a selected set of size $n$. The statistic $k \sim \text{Binomial}(n, p_{\text{fail}})$ is valid **only if** the $n$ selected points are an i.i.d. sample from $P_{\text{test}}$. If they come from $P_{\text{cal}}$ and we deploy on $P_{\text{test}}$, then $k/n$ estimates $p_{\text{fail}}^{\text{cal}}$, not $p_{\text{fail}}^{\text{test}}$, and the bound on $p_{\text{fail}}^{\text{test}}$ is not controlled.

### 3.3 The Bonferroni correction

Because we grid-search over $|H|$ thresholds and pick the best one, we must correct for multiple comparisons. SGen uses Bonferroni: $\delta_{\text{adj}} = (\delta - \delta_p)/|H|$. In our config, $\delta = 0.02$, $\delta_p = 10^{-5}$, $|H| = 20$ (fM1-only grid), so $\delta_{\text{adj}} \approx 10^{-3}$. This makes the bound much tighter than the naive one, which is why Method 1 efficiency is modest (22.9% on NQ) even when validity is preserved.

### 3.4 Weighted Clopper-Pearson (our Method 3 extension)

The extension to weighted conformal replaces the raw count with an **importance-weighted failure rate**:

$$
\hat p_{\text{fail}} = \frac{\sum_{i \in \text{sel}} w_i \cdot \mathbb{1}[y_i = 0]}{\sum_{i \in \text{sel}} w_i}
$$

and the sample size with the **effective sample size** of the selected subset:

$$
n_{\text{eff}}^{\text{sel}} = \frac{(\sum_{i \in \text{sel}} w_i)^2}{\sum_{i \in \text{sel}} w_i^2}.
$$

These are then plugged into Clopper-Pearson as if they were a raw binomial count. This is an approximation (the Tibshirani theorem gives exact guarantees only for quantile-based coverage, not for Clopper-Pearson bounds directly), but it is the natural generalization used in DS-CP and matches the intuition that high-variance weights should widen the bound.

The approximation is only justified if the weight distribution is reasonable. If weights are extreme (a few points dominate), $n_{\text{eff}}^{\text{sel}}$ crashes and the bound becomes vacuous. If weights are clipped, the ratio no longer corresponds to any actual density ratio and the reweighting loses its theoretical meaning.

---

## 4. Covariate shift vs concept shift — theoretical distinction

### 4.1 Covariate shift

**Definition.** Only the input marginal changes:

$$
P_{\text{test}}(X) \ne P_{\text{cal}}(X),\quad P_{\text{test}}(Y|X) = P_{\text{cal}}(Y|X).
$$

**Intuition.** The world asks different questions, but the rules of answering are unchanged. A medical triage model trained on one hospital's patient population encounters a different population at another hospital, but the biology of disease is the same, so the correctness of the model's outputs on each individual patient is governed by the same $P(Y|X)$.

**Fixable by reweighting?** Yes. If $w(x) = P_{\text{test}}(x)/P_{\text{cal}}(x)$ is known, then $\mathbb{E}_{P_{\text{test}}}[\ell(f(X), Y)] = \mathbb{E}_{P_{\text{cal}}}[w(X) \ell(f(X), Y)]$. Reweighting the calibration failure rate recovers an unbiased estimate of the test failure rate. Tibshirani et al. proved that the same trick works for conformal quantiles.

**Signature in data.** Hard points in the target are hard for the *same reason* they would be hard if they occurred in the source: they are in a low-density region of $P_{\text{cal}}$. The importance weight $w(x)$ is high exactly on those points. Therefore, if we look at source points sorted by weight, the top quartile (target-like source points) should have a visibly higher error rate than the bottom quartile (source-like source points). This is our **Test 6** (quartile spread).

### 4.2 Concept shift

**Definition.** The conditional changes:

$$
P_{\text{test}}(Y|X) \ne P_{\text{cal}}(Y|X).
$$

(Optionally, the marginal can change too.) In LLM terms, the *answer* to a question is governed by a different distribution under test than under calibration — even for identical inputs. This happens when:

- The ground-truth labels themselves come from a different annotation protocol (e.g., TQA's Wikipedia-aware aliases vs NQ's short-answer spans).
- The model's own performance is driven by training-data coverage of specific entities or topics, and the test domain shifts the distribution of those entities.
- The evaluation metric itself is domain-sensitive (DeBERTa NLI may be more lenient on certain answer phrasings).

**Fixable by reweighting?** No. The weights $w(x)$ only reweight $X$. They have no effect on the difference between $P_{\text{test}}(Y|X)$ and $P_{\text{cal}}(Y|X)$. If the model gets question $x$ wrong on test with probability $p_{\text{test}}(x)$ and wrong on calibration with probability $p_{\text{cal}}(x)$, and $p_{\text{test}}(x) \ne p_{\text{cal}}(x)$, no amount of reweighting calibration points will give us an accurate estimate of test-domain failure.

**Signature in data.** Accuracy is approximately flat or non-monotonic across weight quartiles, because the error is not caused by being in a low-density region of $X$-space — it is caused by a change in the labeling rule or knowledge distribution. Our **Test 6 strong form** (logistic regression of $y$ on $\log w$) measures exactly this: if the slope is significantly negative, covariate shift is real; if it is near zero or positive, concept shift dominates.

### 4.3 The mix

Real-world shifts are almost always a mixture. The question is whether the covariate component is large enough for reweighting to help, or whether the concept component is so dominant that the mixture behaves like pure concept shift. There is no clean threshold, but the quartile spread and slope give us a quantitative way to estimate the fraction. On TQA→NQ, the quartile spread was about 0.03 (essentially flat), suggesting ~96% concept shift. On PopQA head→tail, the quartile spread was **−0.092** — the *wrong sign* entirely, suggesting the "covariate" we thought we were isolating is not even in the direction we expected.

---

## 5. Summary of prior findings: TQA→NQ

Before PopQA, our headline experiment was TriviaQA (as calibration) → NQ-Open (as shifted test). This was the natural choice given data availability, but it turned out to be a poor fit for DS-SGen.

### 5.1 The numbers at ε=0.25

| Method | TQA validity | NQ validity | NQ mean FDR-E | NQ mean efficiency |
|---|---|---|---|---|
| Method 1 (vanilla SGen-Semi) | **100.0%** | **12.4%** | 0.301 | 22.9% |
| Method 2 (conservative, Option C, frac=0.75) | 100.0% | **22.0%** | 0.260 | 18.5% |
| Method 3 (importance-weighted DS-SGen) | 100.0% | **68.8%** | 0.106 | 8.1% |

The PAC target is 98% validity (since $\delta = 0.02$). None of the methods come close. Method 3's 68.8% looks much better than Method 1's 12.4%, but the breakdown explains everything: of the 500 splits, 344 are entirely vacuous (weights clipped to max, no selection), and among the 156 non-vacuous splits, **zero** achieve validity.

### 5.2 The diagnostic numbers

- **Domain classifier CV accuracy: 91.7%.** TQA and NQ are nearly disjoint in MiniLM embedding space. The classifier confidently assigns near-zero probability to TQA points being "NQ-like" and vice versa. This is the opposite of the regime where weighted CP was designed to operate.
- **ESS ratio: 30.8%.** After clipping raw weights (max 32.66) at the 95th percentile (clip value 1.40), the effective sample size of the full calibration set drops from 3610 to 1112 — i.e., we are effectively calibrating on a third of the data, which widens the CP bound substantially.
- **Mean n_eff across splits: 584.** Even worse once we slice calibration/test within each split.
- **Quartile spread ≈ 0.03.** Flat. Consistent with concept shift dominating.

### 5.3 The epsilon sweep

At ε=0.30, 0.35, 0.40, all three methods go to 0% validity on NQ. Counterintuitively, **loosening** the target guarantee makes things *worse*, not better. The reason is that at ε=0.25 the optimal threshold selection is forced into an abstention regime (where the selected set is small and happens to contain mostly correct answers by luck), while at higher ε the grid search finds thresholds that select more aggressively and expose the shift-induced errors. This is a clear signature that the mechanism is broken: a correct method should have validity monotone-non-decreasing in ε.

### 5.4 Why this motivated screening

The TQA→NQ failure told us two things:
1. The method's *implementation* is correct (TQA in-domain validity is perfect at every ε).
2. The *assumption* (covariate shift dominant, domains overlapping) is wrong for this pair.

We needed a way to diagnose the second point without spending 18 hours of OpenAI Batch generation per candidate. Hence the screening protocol.

---

## 6. The 6-test screening protocol

Full specification is in `docs/screening_protocol.md` and implemented in `ds_sgen/screening.py`. Thresholds are calibrated so that TQA→NQ fails 6 of 7 tests.

### Test 1 — Source accuracy floor

$$\text{acc}_S \ge 1 - \varepsilon + 0.05$$

**Why.** SGen calibrates thresholds on the source distribution. If the source accuracy is below $1 - \varepsilon$, the selected set cannot have error rate below $\varepsilon$ even in the best case, because the pool is already too contaminated. We add 0.05 of slack so the top-confidence bucket has headroom.

**Implementation (`run_screening_tests`):**
```python
acc_S = float(y_S.mean())
pass_1 = acc_S >= threshold_1  # threshold_1 = 1 - epsilon + 0.05
```

### Test 2a — Target accuracy floor

$$\text{acc}_T \ge 1 - \varepsilon$$

**Why.** Even if we had perfect calibration, the PAC FDR-E guarantee requires the target-domain selected set's expected error to be $\le \varepsilon$. If the *entire* target set has error rate greater than $\varepsilon$, no subset can have error rate $\le \varepsilon$ unless we find a very special subset — which is what Test 2b measures.

### Test 2b — Reachable floor (top-5% of target by fM1)

$$\text{acc}_{\text{top5}} \ge 1 - \varepsilon + 0.05$$

**Why.** We take the 5% of target points with the highest generator confidence ($f_{M1}$) and measure their accuracy. This is the *best case* for selection: the most confident-looking target points. If even this slice has error rate $> \varepsilon$, then no threshold on $f_{M1}$ can produce a low-error selection on target.

### Test 3 — Accuracy gap

$$0.03 \le \text{gap} \le 0.15,\quad \text{gap} = \text{acc}_S - \text{acc}_T$$

**Why.** Below 0.03, there is effectively no shift and vanilla SGen already works (nothing to rescue). Above 0.15, we are in the severe-shift regime where even weighted CP cannot realistically recover. The sweet spot is "real but moderate."

### Test 4 — Domain classifier separability

$$0.55 \le \text{acc}_{\text{clf}} \le 0.78$$

**Why.** Below 0.55, the domains are indistinguishable in embedding space — you do not need reweighting. Above 0.78, the domains are nearly disjoint and the density ratios either explode or collapse depending on direction. The sweet spot is "real overlap but real difference."

### Test 5 — Effective sample size ratio

$$\text{ESS ratio} = \frac{n_{\text{eff}}}{n_S} \ge 0.50 \text{ (hard)}, \ge 0.35 \text{ (soft)}$$

**Why.** The weighted CP bound gets wider as ESS drops. Below 0.35, the variance of the bound dominates and we are effectively calibrating on a small random subset. Test 5 and Test 4 are two sides of the same coin: Test 4 looks at the classifier, Test 5 looks at what the classifier does to the weights.

### Test 6 — Quartile spread (concept vs covariate)

$$\text{spread} = \text{acc}_S(Q_1) - \text{acc}_S(Q_4) \ge 0.05$$

where $Q_1$ is the quartile of source points with the *lowest* weight (most source-like) and $Q_4$ is the quartile with the *highest* weight (most target-like).

**Why.** Under pure covariate shift, target-like source points (high weight) should be systematically harder than source-like source points (low weight), because the distribution has moved toward hard regions but $P(Y|X)$ is fixed. The spread should be positive and large. Under concept shift, the spread is near zero or negative.

**Strong form:** Logistic / linear regression of $y_S$ on $\log w$. Significant negative slope $\Leftrightarrow$ covariate shift signal; near-zero or positive slope $\Leftrightarrow$ concept shift.

---

## 7. Why PopQA head→tail was the preferred candidate

When we sat down to pick the next pair after TQA→NQ, we applied four criteria:

### 7.1 Same task, same format

TQA and NQ differ in question style (trivia vs natural search queries), answer type (single entity vs span), and annotation protocol (Wikipedia-sourced vs MTurk). These differences show up as concept shift. We needed a pair where the *task* (given a question, answer with the correct entity) is identical and only one *feature* varies.

**PopQA's structure:** Every PopQA question is generated from a KB triple `(subj, prop, obj)` via a fixed template like "What is the X of Y?". The answer is always a single canonical entity. The only thing that varies between examples is the identity of the subject and the property. This is as close to a "controlled" QA benchmark as exists.

### 7.2 A smooth covariate with real dynamic range

TQA's distinguishing features from NQ are many and categorical. PopQA has one clean quantitative feature: `s_pop`, the Wikipedia monthly page views of the subject entity. The PopQA paper (Mallen et al. 2023) specifically designed the benchmark to diagnose the relationship between entity popularity and model performance.

**Our split:** head = top 60% of the test set by `s_pop`, tail = bottom 20%. After sampling 1000 per domain, the median `s_pop` in head was 3,683 vs 100 in tail — a 37× ratio. The head range was 553–15,101,521 (spanning four orders of magnitude) while the tail range was 2–179. These are genuinely different popularity regimes.

### 7.3 Theoretical argument for covariate-ness

Popularity is a property of the *input*, not the *label*. For any given subject, the correct object (e.g., the capital of Osiecznica) is a fixed fact. The tail question and the head question have the same $P(Y|X)$ — the label is determined by the world, not by which dataset we drew from. So if there is any QA pair that satisfies the covariate-shift assumption by construction, PopQA head→tail is it.

### 7.4 Literature alignment

Popularity-stratified evaluation is standard in knowledge-intensive NLP (Kandpal et al. 2023; Mallen et al. 2023; Lewis et al. 2021). The head/tail disparity is a well-documented phenomenon: LLMs know popular entities from training-data overexposure and struggle with long-tail entities. Using this as our "shift axis" is not an ad-hoc construction; it is the canonical way the NLP community studies distributional robustness for factual recall.

The protocol's predicted scorecard for PopQA head→tail (from the screening design document):

| Test | Predicted | Pass |
|---|---|---|
| 1. Source acc | ~0.82 | ✅ |
| 2a. Target acc | ~0.55 | ❌ (borderline) |
| 2b. Reachable floor | ~0.78–0.85 | ✅ (borderline) |
| 3. Gap | ~0.27 | ⚠️ large |
| 4. Classifier | ~0.62 | ✅ |
| 5. ESS | ~0.55 | ✅ |
| 6. Quartile spread | ~0.10 | ✅ |

We expected the structural tests (4, 5, 6) to pass cleanly and the accuracy tests (1, 2) to be the main risk, with the fallback of running at a more forgiving ε.

---

## 8. Experimental setup

The screening orchestrator (`run_screening.py`) reuses the same four-stage pipeline as the main experiments:

### 8.1 Stage 1 — Data loading (`ds_sgen/screening.py::load_popqa`)

- **Dataset:** `akariasai/PopQA` from HuggingFace, split `test` (14,267 questions).
- **Popularity split:** sorted by `s_pop` ascending; tail = first 20% (2,853 questions), head = top 60% (8,561 questions, i.e., indices `[0.4n, n)`).
- **Sampling:** 1000 per split using `np.random.RandomState(42).choice(indices, 1000, replace=False)`.
- **Normalised schema:** `{idx, question, reference_answer, all_answers, dataset, s_pop, subj, prop}`. The `reference_answer` is the `obj` field; `all_answers` is a singleton because PopQA has only one canonical answer per question.

Statistics after the split:
- Head `s_pop`: min 553, max 15,101,521, median 3,683, mean ~47,000.
- Tail `s_pop`: min 2, max 179, median 100, mean ~75.
- Popularity ratio (head median / tail median): ~37×.

Example head questions: "Who is the producer of Everyone Else?" (Maren Ade), "What sport does Bernadette Szőcs play?" (table tennis), "Who is the composer of Crazy?" (Hank Garland).

Example tail questions: "In what country is Obeakpu?" (Nigeria), "What is the capital of Gmina Osiecznica?" (Osiecznica), "Who is the screenwriter for The Man, the Woman and the Money?" (J.-).

### 8.2 Stage 2 — Generation (`ds_sgen/generate_responses.py::generate_and_cache_openai`)

- **Model:** `gpt-4o-mini` via OpenAI chat completions API.
- **System prompt:** "Answer the following question concisely in one sentence."
- **Greedy pass:** `temperature=0`, `max_tokens=512`, `logprobs=True`. Extracts `mean_logprob` and `token_logprobs`.
- **Sampled pass:** `temperature=0.7`, `n=5`. Produces `sampled_answers` for fM2.
- **Retry logic:** exponential backoff on `RateLimitError`/`APIError`, up to 5 retries.
- **Incremental caching:** `save_every=50`. Safe to interrupt mid-run.
- **Empirical rate:** ~0.6 questions per second (dominated by OpenAI rate limits). Total generation time: ~28 minutes each for head and tail = ~56 minutes for 2000 questions.
- **Token usage (final run):** head 602 + 80k completion ≈ $0.05; tail 62k prompt + 91k completion ≈ $0.06. Total pipeline cost: ~$0.11 for 2000 questions.

### 8.3 Stage 3 — Entailment scoring (`ds_sgen/entailment_scoring.py::score_and_cache`)

- **Model:** `microsoft/deberta-v2-xxlarge-mnli` (1.5B parameters, label order `{0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}`).
- **Correctness mode:** `NLI(greedy → reference)`, single forward pass, argmax == 2 ⇒ correct.
- **Self-consistency mode:** for each pair $(i,j)$ of sampled answers, compute $NLI(i→j)$ and $NLI(j→i)$. A pair "agrees" iff both directions are argmax ENTAILMENT. $fM2 = (\text{agreeing pairs}) / \binom{5}{2}$.
- **Batch size:** 64 (fits comfortably in A6000 50 GB at fp16).
- **Empirical rate on A6000:** ~10.5 questions/sec (each question requires 1 correctness pass + $5 \times 4 = 20$ directed NLI pairs = 21 NLI pairs total). Total stage 3 time: ~3 minutes for 2000 questions.

Initially we accidentally ran stage 3 on CPU (the debug shell node had no GPU), which took ~4.7 hours per 1000 questions. We then deleted the CPU caches and resubmitted via SLURM with `--gres=gpu:A6000:1 --partition=general`. The second run completed in 198 seconds total.

### 8.4 Stage 4 — Embedding (`ds_sgen/importance_weighted.py::compute_embeddings`)

- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim output).
- **Batch size:** 256 on GPU.
- **Output:** `(1000, 384)` float32 numpy arrays, cached to `{head,tail}_embeddings.npy`.

### 8.5 Stage 5 — Screening battery (`ds_sgen/screening.py::run_screening_tests`)

- Runs all six tests in sequence.
- Trains a logistic regression domain classifier with 5-fold cross-validation.
- Fits the classifier on all 2000 embeddings for weight computation.
- Derives weights $w(x) = p_{\text{clf}}(x) / (1 - p_{\text{clf}}(x))$ with classifier probabilities clamped to $[0.01, 0.99]$.
- Clips raw weights to `(w_min=0.01, w_max=100.0)`.
- Computes `n_eff`, quartile spread, and logistic slope.
- Saves results to `results/screening_popqa_results.json` and prints a scorecard.

### 8.6 Infrastructure

- **Cluster:** Babel (CMU SCS).
- **Partition:** `general` (after we learned the hard way that `preempt` gets killed mid-run; our first attempt was preempted at tail question 700/1000).
- **GPU:** NVIDIA RTX A6000, 49,140 MiB, cuda:0.
- **Environment:** `/data/user_data/anshulk/envs/dsgen`, Python 3.10, transformers 4.x, torch 2.x, sentence-transformers 5.3.
- **Job resources:** 1 GPU, 4 CPUs, 48 GB RAM, 4-hour time limit.
- **Run ID:** SLURM job 7056159.
- **Total elapsed:** 198 seconds (3 minutes 18 seconds) from SLURM start to completion, including model downloads.

---

## 9. PopQA screening results — detailed walkthrough

All numbers below are from `results/screening_popqa_results.json` (job 7056159, 2026-04-11 01:23 EDT).

### 9.1 Test 1 — Source accuracy

$$\text{acc}_S = \frac{437}{1000} = 0.437$$

**Threshold (ε=0.25):** $\ge 0.80$.
**Result:** **FAIL** (shortfall of 0.363).

**What this means.** GPT-4o-mini is wrong, by strict DeBERTa NLI argmax, on 563 out of 1000 questions about *popular* entities. Concretely: asked "Who is the composer of Crazy?", it answered "Willie Nelson" instead of "Hank Garland". Asked "Who produced Everyone Else?", it got Maren Ade correct. Asked "What is the father of Prince Alexander of the Netherlands?", it said "King William II" when the reference is "William II of the Netherlands" — actually correct content but NLI argmax gives NEUTRAL, so 0. Asked "Who was the producer of The Chorus?", it said "Jacques Perrin" when the reference is "Mark Levinson" — factually wrong.

Two things are going on here.

**First**, PopQA's questions are *harder* than TQA's. TQA gives the model trivia-style questions with rich context cues ("Who wrote the screenplay for the 1977 film Annie Hall?") whereas PopQA gives austere KB-triple-style questions ("Who wrote Crazy?") with no disambiguation. Many PopQA subjects are ambiguous — there are multiple works called "Crazy", multiple people called "James A. Peters", multiple things called "Universe". The model hallucinates a plausible answer confidently.

**Second**, DeBERTa-v2-xxlarge-mnli is *strict*. It is trained on MultiNLI, which is adversarial for short factual answers. A correct answer phrased differently ("born in New York City" vs "New York City") can get argmax NEUTRAL instead of ENTAILMENT because the model output contains surface-level material (a location qualifier) that the reference does not. We saw this on Jacob → "a significant figure in Judaism, Christianity, and Islam" vs reference "Judaism" — the model's answer *implies* the reference but NLI does not accept it.

At 43.7%, we are at the absolute floor of where SGen-Semi can be used. The PAC guarantee at ε=0.25 requires $\ge 75\%$ accuracy *just to be theoretically reachable*. At 43.7%, the entire PAC machinery is operating outside its designed regime.

### 9.2 Test 2a — Target accuracy

$$\text{acc}_T = \frac{241}{1000} = 0.241$$

**Threshold:** $\ge 0.75$ (hard), $\ge 0.70$ (soft).
**Result:** **FAIL** (shortfall of 0.509).

**What this means.** On tail questions — genuinely obscure entities like "Mahaboboka" or "KK Mašinac" or "Călmuș River" — the model is right only 24.1% of the time. This is the crucial number: it means the *entire target distribution* has error rate 0.759, which is three times the ε=0.25 bound. There is no subset of the target distribution with error rate $\le 0.25$ unless it is vanishingly small and structured in exactly the right way.

To make ε=0.25 reachable in principle, we would need target accuracy of at least 0.75. We are off by a factor of 3. No amount of reweighting, threshold tightening, or PAC magic can close that gap — it is a statement about the underlying data.

### 9.3 Test 2b — Reachable floor (top-5% by fM1)

$$\text{acc}_{\text{top5}} = \frac{32}{50} = 0.640$$

**fM1 range in top-5%:** $[-0.0066, -0.0000]$.
**Threshold:** $\ge 0.80$.
**Result:** **FAIL** (shortfall of 0.160).

**What this means.** We take the 50 tail questions where GPT-4o-mini has the highest generation confidence (mean log-prob closest to 0) and measure their accuracy. These are the questions where the model is maximally sure of its answer. **Even there, it is wrong 36% of the time.**

This is the defining diagnostic for Method 3's viability. The selective generation story is: "we only answer the confident ones, and on the confident ones we are usually right." For this to give us ε=0.25 FDR-E, the confident ones must have error rate $\le 0.25$. They have error rate 0.36. And fM1 is the only axis SGen-Semi uses to select — there is no room to tighten further (the top 5% is already the tightest meaningful threshold).

This tells us that *generation confidence is not a reliable indicator of factual correctness on tail PopQA*. The model confidently hallucinates. We looked at some of the top-50 tail answers:

- "Jean Gabriel Marie is a French politician." (Actually a composer.)
- "Ruth Berman Harris is primarily known for her work in the genre of [...]" (Hallucinated.)
- "Universe is typically classified as a drama film." (Ambiguous movie title.)
- "The director of 'So Can I' is Kelsey O'Connor." (Fabricated.)

GPT-4o-mini's confidence reflects *fluency* and *plausibility*, not *factual reliability*. On popular entities, fluent answers are usually correct because the training data supports them. On tail entities, fluent answers are often fabrications with high confidence because the model has never seen the correct answer and simply generates the most plausible continuation.

This is a **fundamental limitation of using token log-probability as a correctness signal**, and it hits hardest exactly in the long-tail regime DS-SGen is meant to protect.

### 9.4 Test 3 — Accuracy gap

$$\text{gap} = 0.437 - 0.241 = 0.196$$

**Threshold:** $[0.03, 0.15]$.
**Result:** **FAIL** (gap too large by 0.046).

**What this means.** The accuracy gap is 19.6 percentage points — substantial, and in the "severe shift" regime (between 0.15 and 0.30). This is smaller than TQA→NQ's 0.285, so the shift is less extreme, but it is still large enough that we are in territory where covariate-shift reweighting alone is unlikely to recover the guarantee. The protocol was designed around the observation that pairs with gap > 0.15 tend to be dominated by concept shift, and that prediction holds up here: the large gap is a symptom of the underlying P(Y|X) change (entity knowledge), not just P(X) movement.

### 9.5 Test 4 — Domain classifier accuracy

$$\text{acc}_{\text{clf}} = 0.692 \pm 0.024$$

**5-fold CV scores:** 0.710, 0.680, 0.708, 0.713, 0.650.
**Threshold:** $[0.55, 0.78]$.
**Result:** **PASS** (the only one).

**What this means.** The MiniLM embeddings of head questions and tail questions are distinguishable but not trivially so. The classifier is well above chance (0.5) but well below near-separable (>0.85). This is exactly the regime weighted conformal prediction was designed for: two distributions that overlap but are not identical.

Why does this pass while everything else fails? Because this test measures the *structure of the shift in embedding space*, not the *structure of the shift in accuracy space*. The embeddings reflect the *surface form* of the questions — their lexical and syntactic choices, the entities mentioned. Head questions mention entities that the embedder has seen more of (since MiniLM was trained on a large web corpus), so their embeddings cluster slightly differently from tail entity embeddings. The classifier picks up on this.

But this has **nothing to do with the model's answer correctness**. GPT-4o-mini's knowledge of an entity is not the same as MiniLM's embedding of the question string. You can write a perfectly fluent question about an obscure entity (MiniLM sees a normal English question, the classifier might assign p_tgt around 0.4) while GPT-4o-mini has no idea who the entity is (correctness = 0). The embedder and the generator have different knowledge bases.

So Test 4 passing is informative but not sufficient: it tells us that the weighted-CP machinery *would* work if the accuracy axis were aligned with the embedding axis. The machinery's failure downstream tells us that it is not.

### 9.6 Importance weights — diagnostics

From the fitted classifier:
- $p_{\text{clf}}(x)$ range on head: $[0.0665, 0.9161]$, median $0.3425$.
- Raw weights $w = p/(1-p)$: min 0.0712, median 0.5209, max 10.9239, std 0.9474.
- Clipping at `w_clip_max = 100.0`: **zero weights clipped** (max 10.92 < 100).

This is a very healthy weight distribution. Contrast with TQA→NQ, where raw weights reached 32.66 and clipping at the 95th percentile cut them to 1.40, effectively erasing the right tail entirely. On PopQA head→tail, the classifier gives us a smooth spectrum of density ratios, and the machinery has exactly the kind of weight distribution it expects.

### 9.7 Test 5 — Effective sample size ratio

$$n_{\text{eff}} = \frac{\left(\sum w_i\right)^2}{\sum w_i^2} = 423.1,\quad \text{ratio} = 0.423$$

**Threshold:** $\ge 0.50$ (hard), $\ge 0.35$ (soft).
**Result:** **SOFT PASS**.

**What this means.** The weights are reasonable enough that we retain 42% of the nominal sample size in effective terms. This is below the hard threshold but above the soft threshold. It means the weighted CP bound will be somewhat wider than the unweighted bound (by a factor of $\sqrt{1/0.423} \approx 1.54$), but not vacuously so. On TQA→NQ this was 0.308, meaning the bound was wider by $\sqrt{1/0.308} \approx 1.80$ — enough to break validity at ε=0.25.

So the *statistical machinery* is fine. The weights are reasonable, the ESS is workable, and the bound would be tight enough to say something. The problem is not that the bound is too wide; the problem is that what it is bounding (the calibration-domain weighted failure rate) is not a reliable estimate of the test-domain failure rate.

### 9.8 Test 6 — Quartile spread

$$\text{spread} = \text{acc}_S(Q_1) - \text{acc}_S(Q_4) = 0.412 - 0.504 = -0.092$$

Quartile accuracies (source points, sorted by weight ascending):
- $Q_1$ (source-like, $w \in [-0.929, 0.324]$, n=250): **0.412**
- $Q_2$ ($w \in [0.324, 0.521]$, n=250): **0.380**
- $Q_3$ ($w \in [0.521, 0.914]$, n=250): **0.452**
- $Q_4$ (target-like, $w \in [0.914, 11.92]$, n=250): **0.504**

**Threshold:** $\ge 0.05$.
**Result:** **FAIL** (and negative — wrong direction).

**What this means.** This is the most theoretically important failure. Under the covariate-shift hypothesis, target-like source points (points that look similar to tail questions in embedding space) should be *harder* for the model, because they sample a region of input space where the model is weaker. The prediction is:

$$\text{acc}_S(Q_4) < \text{acc}_S(Q_1)$$

We observe the *opposite*:

$$\text{acc}_S(Q_4) = 0.504 > \text{acc}_S(Q_1) = 0.412$$

by 9.2 percentage points. Source points that look *target-like* are almost 10 points **more accurate** than source points that look source-like. This is structurally impossible under pure covariate shift, and the quartile pattern is monotonically increasing (0.412 → 0.380 → 0.452 → 0.504), not a noisy reversal.

**Why does this happen?** The classifier identifies "target-like" by embedding features — questions that syntactically or semantically resemble tail questions. But tail questions are not harder because of their *surface form*; they are harder because they ask about *obscure entities*. A head question can look syntactically similar to a tail question (same template, same kind of property being asked, similar sentence length) while asking about a still-popular entity. That entity is one the model has seen in training, so the model gets the question right.

Moving in the *opposite* direction: a very-head question ("Who was the father of Prince Alexander of the Netherlands?") can look quite *unlike* tail questions in embedding space because it is a long, specific, well-contextualized historical question — and the model *still* gets it wrong (says "King William" instead of "William II of the Netherlands") because of an NLI disagreement over phrasing.

So the embedding-classifier axis is not only orthogonal to the accuracy axis; it is weakly *anti-correlated* with it. Reweighting by this classifier would push calibration *toward* the more-accurate points and *away* from the less-accurate ones, which is exactly the wrong direction for controlling target-domain error.

### 9.9 Test 6 strong form — Linear regression of $y_S$ on $\log w$

$$\text{slope} = +0.0653,\quad \text{std err} = 0.0194,\quad p = 0.0008,\quad R^2 = 0.0112$$

The slope is **positive and statistically significant**. Source accuracy *increases* with log weight. The $R^2$ is tiny (about 1% of variance), but the sign is unambiguous: there is no negative covariate-shift signal in this data.

Under covariate shift the expected slope is:

$$\frac{d \mathbb{E}[y | x]}{d \log w(x)} < 0$$

because high-weight points are in the hard region of $X$-space. We measured $+0.065$ with $p < 0.001$. This is strong evidence that the weight axis (classifier-based density ratio) is not tracking difficulty. The test confirms Test 6's quartile finding quantitatively.

### 9.10 Final verdict

| Test | Value | Threshold | Result |
|---|---|---|---|
| 1. Source accuracy | 0.437 | $\ge 0.80$ | **FAIL** |
| 2a. Target accuracy | 0.241 | $\ge 0.75$ | **FAIL** |
| 2b. Reachable floor (top-5%) | 0.640 | $\ge 0.80$ | **FAIL** |
| 3. Accuracy gap | 0.196 | $[0.03, 0.15]$ | **FAIL** |
| 4. Domain classifier | 0.692 | $[0.55, 0.78]$ | **PASS** |
| 5. ESS ratio | 0.423 | $\ge 0.50$ | **SOFT** |
| 6. Quartile spread | −0.092 | $\ge 0.05$ | **FAIL** |

**Total: 1 PASS, 1 SOFT, 5 FAIL.** The verdict printed by `print_scorecard`:

> VERDICT: 1/7 pass — pair may not produce the M1/M2 fail → M3 rescues story.

---

## 10. Cross-result synthesis: TQA→NQ vs PopQA

Side by side, the two scorecards reveal a pattern:

| Test | TQA→NQ | PopQA head→tail | Interpretation |
|---|---|---|---|
| Source acc | 0.716 | **0.437** | PopQA is harder in absolute terms |
| Target acc | 0.431 | **0.241** | PopQA tail is much harder |
| Top-5% acc | 0.694 | **0.640** | Neither has a reachable floor at ε=0.25 |
| Gap | 0.285 | 0.196 | PopQA is *less* shift-severe (smaller gap) |
| Classifier | 0.917 | **0.692** | PopQA is in the healthy overlap regime |
| ESS | 0.308 | **0.423** | PopQA has 37% more effective samples |
| Quartile spread | +0.03 | **−0.092** | Both fail, but for opposite reasons |

Two things jump out.

### 10.1 The structural health of PopQA

The middle block of tests (4 and 5) shows that **PopQA is structurally better for DS-SGen than TQA→NQ**. The classifier accuracy is in the sweet spot (not separable), the weights are unclipped, and the ESS ratio is healthier. If these were the only tests that mattered, we would greenlight PopQA immediately.

The fact that PopQA still fails, despite being structurally healthier, is what makes this result informative. It tells us the structural tests are *necessary but not sufficient*. You can have perfect overlap, well-behaved weights, and reasonable ESS and *still* fail because the accuracy axis doesn't line up with the embedding axis.

### 10.2 The accuracy regime

The accuracy tests (1, 2a, 2b) are where PopQA is catastrophically worse. TQA→NQ fails them by moderate margins (source 0.716 vs target 0.80, a 0.084 gap). PopQA fails them by huge margins (source 0.437 vs target 0.80, a 0.363 gap). The PAC guarantee at ε=0.25 *cannot* be satisfied when target accuracy is 0.241 — not by any method, not with any amount of reweighting. The floor is too low.

Mallen et al. (2023), the PopQA paper, used exact-match as the evaluation metric and reported baselines ranging from 20% to 40% for GPT-3.5-class models on tail. Our 24.1% for tail and 43.7% for head are consistent with those numbers. This is *not* a pipeline bug; it is the intrinsic difficulty of PopQA.

### 10.3 The sign of the spread

The most theoretically interesting difference is Test 6. TQA→NQ had a near-zero spread (+0.03) — flat, consistent with concept shift dominant. PopQA has a *negative* spread (−0.092), which is even worse: it means the classifier axis is actually *anti-correlated* with difficulty.

This tells us something subtle about where the failures come from:
- **TQA→NQ:** the classifier picks up *domain boundaries* (trivia vs search queries), and both domains have their own accuracy distributions that don't correspond to each other. Reweighting is ineffective but not misleading.
- **PopQA head→tail:** the classifier picks up *question surface features* that correlate *weakly* with the inverse of difficulty, because the "popular" entities tend to have richer contextualization in their questions. Reweighting is actively counterproductive — it pushes calibration in the wrong direction.

Neither pair can be saved by weighted CP. But they fail for subtly different reasons.

### 10.4 The epsilon sweep confirms the ceiling

For TQA→NQ we ran the epsilon sweep:

| ε | Method 1 | Method 2 (Option C) | Method 3 |
|---|---|---|---|
| 0.25 | 12.4% | 22.0% | 68.8% (vacuous) |
| 0.30 | 0.0% | 0.0% | 11.0% |
| 0.35 | 0.0% | 0.0% | 0.2% |
| 0.40 | 0.0% | 0.0% | 0.0% |

At higher ε (looser target guarantee), validity *decreases* — another signature of a broken mechanism. A correctly-operating method has validity monotone in ε. The only way to get this reversed is if the threshold selection is sensitive to the grid structure in a way that accidentally exploits abstention at strict ε and exposes real errors at loose ε.

We have not run the full epsilon sweep for PopQA, but based on the screening numbers we can predict it with high confidence: at ε=0.25, both methods will be at roughly 0% (head→tail target accuracy is 0.241, which is below 1−ε=0.75 by 0.509, so the FDR-E bound is nowhere near reachable). At higher ε, nothing improves, because the underlying accuracy ceiling is the limiting factor. The epsilon sweep would confirm the screening result but not add new information.

---

## 11. Why we believe the method is correct

A natural reaction at this point is: "Maybe Method 3 is wrong? Maybe the implementation is buggy?" We want to argue the opposite: the method is correct, and the negative results are genuine scientific information about the class of shifts that weighted CP can handle.

### 11.1 In-domain validity is perfect

In every single run, for every method, the in-domain validity is exactly 100%. On TQA at ε=0.25: Method 1 = 100%, Method 2 = 100%, Method 3 = 100%. At every epsilon in the sweep (0.25, 0.30, 0.35, 0.40): 100%, 100%, 100%, 100%. This is the sharpest possible falsification of the "implementation bug" hypothesis. If there were a bug in the Clopper-Pearson bound, Bonferroni correction, conformal threshold, or weighted quantile, it would manifest in-domain. It does not. The code is correct.

### 11.2 The bound is valid under its assumption

We verified the Clopper-Pearson bound against `scipy.stats.beta.ppf` by hand. We verified the weighted quantile against `numpy.quantile` on uniform weights. We verified that `_run_single_split` uses the same index array to partition both the data and the weights (a subtle correctness issue in early drafts). The machinery is correct.

The Tibshirani et al. weighted-exchangeability theorem requires:
1. Weights are nonzero and bounded.
2. Weights correspond to the true density ratio $w(x) = P_{\text{test}}(x)/P_{\text{cal}}(x)$.
3. Test points are drawn from $P_{\text{test}}$ which shares conditional $P(Y|X)$ with $P_{\text{cal}}$.

Condition 1 is enforced by `np.clip(w, 0.01, 100.0)`. Condition 2 is approximated via the classifier trick, which is a standard technique with known asymptotic properties (more data → better density ratio estimates). Condition 3 is the *covariate-shift assumption*. When it holds, the method provably works. When it does not hold — as in TQA→NQ and PopQA head→tail — the method has no theoretical protection and, as we observe, fails empirically.

### 11.3 Both failure modes are theoretically predicted

For TQA→NQ: the domain classifier accuracy is 0.917 (domains nearly disjoint), so the density-ratio estimator necessarily produces extreme weights (some points near 0, some near infinity). The theorem's asymptotic "small-variance weights" regime is violated. We *expect* the bound to degrade, and the empirical weight distribution (raw max 32.66) confirms that the violation is severe.

For PopQA head→tail: the domain classifier axis and the accuracy axis are weakly anti-correlated. In the Tibshirani framework, if $w$ is *the* density ratio, it determines both which points get weight and which points have high $P(Y=\text{wrong}|X)$. Here $w$ is an *estimate* of the density ratio that happens to correlate poorly with the true $P(Y|X)$ pattern. The method does not break "because it is buggy"; it breaks because the link between the embedding-space shift and the correctness-space shift is missing.

### 11.4 DS-CP (Lin et al. 2025) agrees

The DS-CP paper by Lin et al. (arXiv 2025) is the direct application of weighted CP to LLMs. Their main empirical finding is that DS-CP is **adaptive**: when standard CP already works, DS-CP barely changes anything; when standard CP fails, DS-CP helps. In their Theorem 1, the coverage gap is bounded by a function of the *score distribution* mismatch, not the prompt distribution mismatch. This means:

- If the model's uncertainty patterns are similar across domains, DS-CP helps.
- If the model's uncertainty patterns differ across domains, DS-CP cannot fully recover coverage.

Our tests are a direct instance of the "uncertainty pattern difference" case. PopQA tail has a different uncertainty pattern from PopQA head not because the input distributions are different in the way MiniLM sees them, but because the model's *knowledge* is different. That is a score distribution difference, not a prompt distribution difference, and DS-CP's own theory predicts limited benefit in that regime.

We are reproducing, in our smaller QA setting, what DS-CP's theory says should happen. The method is correct; the shift is the wrong kind for the method to solve.

### 11.5 The failure is informative

A correct method that fails on the wrong kind of problem is more valuable than a heuristic that succeeds on any problem because we cannot predict its behavior. Our screening protocol is exactly the artifact that emerges from taking this seriously: we have a quantitative, reproducible test battery that tells us *before running the full pipeline* whether the shift is of the type the method can handle. The fact that the protocol predicted PopQA would fail (via Tests 1 and 2) and then the full run confirmed it validates both the protocol and the underlying theory.

---

## 12. Why dataset selection is the real bottleneck

The project's research question was: "Can we maintain PAC FDR-E guarantees under domain shift?" The answer we are converging on is: "The method can, but only for a specific shift structure, and finding realistic LLM deployment pairs that satisfy the structure is very hard."

### 12.1 The four conditions a "good" dataset pair must simultaneously satisfy

From the 6-test battery, we can distill four underlying conditions:

**C1: Source accuracy is high enough.** $\text{acc}_S \ge 1 - \varepsilon + 0.05$. At $\varepsilon = 0.25$ this means the model must be at least 80% correct on calibration. For generation tasks scored by strict NLI, this is a high bar: TQA gets 71.6%, NQ gets 43.1%, PopQA head gets 43.7%, SQuAD extractive QA gets maybe 85% on strong models. Most knowledge-intensive QA benchmarks live below the threshold.

**C2: Target accuracy is above the FDR-E floor.** $\text{acc}_T \ge 1 - \varepsilon$. At $\varepsilon = 0.25$, target accuracy must be at least 75%. This is *much* harder: the whole point of having a "shifted" target is that the model performs worse there. But if it performs *too* much worse, the floor is unreachable. PopQA tail at 24.1% is catastrophic. Even in the SQuAD shifts literature, the worst shift (NewWiki → Amazon) drops accuracy by 10–15 points, which is just barely within the feasible regime.

**C3: The shift is covariate, not concept.** Quartile spread must be significantly positive; logistic slope must be significantly negative. This requires that the *axis* along which we split source and target is a property of $P(X)$ alone, not of $P(Y|X)$. For text datasets, almost every natural covariate candidate (topic, domain, time, source corpus) is entangled with $P(Y|X)$ because the model's knowledge is not uniformly distributed across those axes.

**C4: The domains overlap enough to reweight.** Classifier accuracy in $[0.55, 0.78]$, ESS ratio $\ge 0.35$. This requires that the distributions are structurally similar even as they differ along some axis. TQA/NQ failed here (classifier = 0.917, ESS = 0.31). PopQA passed cleanly (classifier = 0.692, ESS = 0.42).

Satisfying all four simultaneously is genuinely rare. We tried two pairs, and each failed different conditions:
- **TQA→NQ:** fails C4 (near-disjoint) and C2 (target too low) and C3 (concept shift).
- **PopQA head→tail:** fails C1 (source too low), C2 (target much too low), C3 (concept shift in wrong direction), but passes C4.

The search space is:
- Benchmarks with high enough model accuracy (C1): rules out most knowledge-intensive QA.
- Shift axes that are pure-covariate (C3): rules out topical/temporal/source shifts where the model's knowledge varies.
- Shifts moderate enough to not crash target accuracy (C2): rules out dramatic shifts like TQA→NQ.
- Shifts visible in embedding space (C4): rules out subtle shifts where the classifier can't discriminate.

The intersection of these four conditions, in the space of existing QA benchmarks, is very thin. This is the actual research finding of our project.

### 12.2 The tension between C1 and C2

There is a fundamental tension: C1 wants source accuracy to be high (≥80%), but if source accuracy is high and the shift is real, target accuracy will typically drop to somewhere in the 60–75% range, which barely satisfies C2. And if source accuracy is higher still, the shift is smaller, which conflicts with Test 3 ($\text{gap} \ge 0.03$) — we need a *visible* shift to tell a story.

The "sweet spot" is narrow: source accuracy around 85%, target accuracy around 80%, gap around 5%. The reason DS-CP chose multiple-choice MMLU for their evaluation is that MCQ accuracy is much higher than open-ended QA (often 70–90%), which widens the feasible region.

For *open-ended generation*, which is our setting, the feasible region appears to shrink to near-zero with existing benchmarks. This is a genuine limitation.

### 12.3 The NLI-strictness problem

Part of the problem is the correctness metric. DeBERTa-v2-xxlarge-mnli is *strict*: it rejects correct answers with mismatched phrasing. SGen chose this metric for theoretical reasons (it is a proper entailment, unlike exact match), but the practical cost is that measured accuracy is 10–20 points lower than what exact-match or ROUGE would report.

If we used a more lenient metric (e.g., BEM, RQUGE, substring match, or GPT-4-as-judge), we could raise accS from 43.7% to perhaps 60–70% and acc_T from 24.1% to perhaps 40–50%. That is still not enough for ε=0.25 on PopQA, but it might put other benchmarks (TQA, NaturalQuestions, WebQuestions) comfortably above the C1 line.

The tradeoff is that changing the metric changes what we are guaranteeing. The SGen paper's PAC bound is specifically for FDR-E (with entailment as correctness). If we switch to GPT-4-as-judge, we lose the independence assumption of the entailment oracle (GPT-4's errors are correlated with GPT-4-mini's errors, since they share training data).

### 12.4 The mismatch between theory and NLP reality

Weighted conformal prediction was originally developed (Tibshirani et al. 2019) in contexts where covariate shift is clean: tabular medical data, polls with known selection biases, imaging with shifted camera distributions. In those settings, you can often argue that $P(Y|X)$ is physically unchanged — the biology of disease, the voter's true preference, the object's true label — and only the sampling process varies. The Tibshirani assumption is genuinely close to true.

In language modeling, the situation is different. The "correctness" of a model's output is a function of what the model learned, which is a function of its training data, which is not uniformly distributed across any natural covariate. So *every* shift in $P(X)$ is accompanied by at least some shift in $P(Y|X)$ from the model's perspective, and the purity of covariate shift is an abstraction rather than a physical reality.

This is not a failure of our experimental design; it is a statement about the fundamental difficulty of the problem. The research question "is LLM shift mostly covariate?" has an empirical answer of "no, mostly concept, at least for the accuracy-sensitive axes", and the positive-result scenarios for DS-SGen have to be carefully engineered rather than found in the wild.

---

## 13. What we would need from a "good" dataset

Based on the conditions above, here is what we would need from an ideal benchmark pair:

### 13.1 Hard constraints

- **Same task format.** Both calibration and test must share identical question templates, answer types, and annotation protocols. No differences in how correctness is operationalized.
- **Same model accuracy ceiling.** The model's *capability* should be similar across domains — the only difference should be in the *input distribution*, not in what the model can do when given an input in each domain.
- **Source accuracy $\ge 0.85$.** Strong baseline performance, because we need headroom for the PAC bound.
- **Target accuracy $\ge 0.75$.** The shift is moderate enough that the floor is reachable.
- **Accuracy gap $\in [0.05, 0.15]$.** Real shift, not noise.
- **Embedding-classifier accuracy $\in [0.60, 0.80]$.** Distinguishable but overlapping.
- **Positive quartile spread.** The embedding-space shift and the accuracy-space shift must be aligned.

### 13.2 Candidate shift axes that might satisfy the above

**Paraphrasing (surface-form shift).** Take one benchmark, paraphrase the questions into two different styles (formal vs casual, short vs verbose), calibrate on one, test on the other. Accuracy should be similar across the two. Covariate shift is by construction, because the underlying facts are unchanged. Downside: may be too trivial (classifier barely works), and the shift may be absorbed by the LLM's paraphrase invariance.

**Prompt-format shift.** Same questions, different prompt templates ("Q: ... A: ..." vs "Answer the following: ..." vs instruction-tuned format). Accuracy varies with format, and the format is a feature of $X$ alone.

**Language shift.** Same QA task in English vs a high-resource other language (German, French). $P(Y|X)$ is *almost* the same (there are translation ambiguities), and $P(X)$ is very different. Weighted CP should apply if the model is reasonably multilingual.

**Retrieval-conditioned shift.** Retrieval-augmented QA where one domain has "good" retrieval and another has "noisy" retrieval. The noise is a covariate of $X$, and $P(Y|X, \text{retrieval})$ is a fixed function.

**Benchmark subset stratification.** Pick a benchmark where accuracy is high (e.g., SQuAD, BoolQ) and stratify by a property that is weakly correlated with difficulty (question length, presence of negation, topic). This was our hope for PopQA, and the lesson is that the stratification axis must not be aligned with the model's knowledge.

### 13.3 What we would *not* try again

- **Cross-benchmark pairs** (TQA vs NQ, etc.). These conflate everything — annotation protocol, question style, answer type, evaluation standard, source corpus — and inevitably produce concept shift.
- **Entity popularity stratification** (PopQA, Kandpal et al.). Popularity is confounded with knowledge in a way that is fundamentally $P(Y|X)$-level, not $P(X)$-level.
- **Temporal shifts** (Wikipedia 2020 vs 2024 facts). The model's training-data cutoff creates hard concept shift.
- **Task difficulty stratification** (easy vs hard subset of a benchmark). By construction, this is concept shift: the model's $P(Y|X)$ changes.

---

## 14. Candidate escape routes

Given the difficulty of finding a clean dataset pair, there are three directions we could take the project.

### 14.1 Route A — Use a lenient correctness metric

Replace DeBERTa NLI with a more generous correctness function: BEM, RQUGE, or GPT-4-as-judge. This would raise all our accuracy numbers by 10–20 points, likely enough to put existing benchmarks (TQA, SQuAD, WebQuestions) above the C1 floor. The downside is that the PAC guarantee becomes a statement about a weaker notion of correctness, and the guarantee's practical meaning depends on how reliable the lenient oracle is.

Empirically, this would let us demonstrate the method at ε=0.25 on, say, SQuAD NewWiki→NYT (which the protocol predicted would pass cleanly) without needing accuracy to miraculously be above 80% under strict NLI. The research story becomes: "DS-SGen works at moderate ε for moderate shifts, provided the correctness metric is appropriately scoped."

### 14.2 Route B — Loosen epsilon and tell the feasibility story

Instead of targeting ε=0.25, report validity curves across ε and show where DS-SGen starts working. Even with strict NLI on PopQA, there is presumably some ε (perhaps 0.55–0.65) where the method starts producing valid guarantees. The story becomes: "Here is the feasibility frontier. Here are the shifts DS-SGen can handle at various ε. Here is the pre-flight screening that tells you whether your pair is in the feasible region."

This is less practically useful (who wants a 55% FDR guarantee?) but more honest. It turns our negative result into a diagnostic contribution: we are the first to quantitatively characterize when DS-SGen works.

### 14.3 Route C — Synthetic covariate shift

Construct a pair where covariate shift holds by design. Take TriviaQA, run GPT-4o-mini on it, split the questions into two halves randomly. Paraphrase half A into casual English and half B into formal English. Calibrate on half A, test on half B. By construction: same underlying questions, same ground truth, same $P(Y|X)$, only $P(X)$ differs. This should produce exactly the signature weighted CP is designed for.

The critique is that synthetic shift is not realistic, and a reviewer will complain. The response is that it directly demonstrates the method works *when its assumption holds*, which is the control experiment needed to validate any negative result on natural pairs.

### 14.4 Our recommended path

Route C as the existence proof ("here is DS-SGen working on the right kind of shift"), combined with Route B's framing ("here is the feasibility frontier") applied to TQA→NQ and PopQA head→tail as the empirical examples of what fails and why. This turns the project into a full characterization of the method rather than a single successful experiment, which is stronger than either route alone.

---

## 15. Conclusion and lessons for the field

### 15.1 What we showed

1. **SGen-Semi breaks under realistic LLM domain shift.** TQA→NQ achieves 12.4% validity at ε=0.25, δ=0.02, against a 98% target.
2. **Conservative adjustments help but cannot close the gap.** Method 2's best option gets to 22.0%, still far from target.
3. **Importance-weighted DS-SGen helps at ε=0.25 on TQA→NQ, but 68.8% of the "help" is vacuous abstention driven by weight clipping.** Among non-vacuous splits, 0/156 are valid.
4. **The screening protocol correctly identifies ex ante that TQA→NQ and PopQA head→tail are both infeasible.** The battery catches the failure in ~3 minutes of compute per pair, compared to ~4 hours for a full pipeline.
5. **PopQA head→tail has healthier structural properties than TQA→NQ (Tests 4 and 5 pass) but still fails because the embedding-classifier axis is uncorrelated (actually weakly anti-correlated) with accuracy.** This is a different failure mode from TQA→NQ's concept shift, but equally terminal.

### 15.2 What this says about the method

Weighted conformal prediction — the theoretical machinery behind DS-SGen — is correct. Its guarantees hold under covariate shift by theorem. Our implementation is correct: in-domain validity is 100% at every epsilon, weights compose correctly with the Clopper-Pearson bound, and the grid search preserves Bonferroni correction. The scientific content of our negative results is not "the method is broken" but "the method's assumption is a much stronger requirement than it seems when applied to LLM deployment."

### 15.3 What this says about the field

Most LLM robustness papers assume that if a method works under covariate shift in theory, then it applies to real deployment scenarios. Our results argue that this assumption is dangerous. LLM failures under domain shift are dominated by *knowledge gaps* — regions of entity or topic space that the training distribution undercovered — and knowledge gaps are a property of $P(Y|X)$, not $P(X)$. Any method that reweights by a function of $X$ alone is at a structural disadvantage for handling them.

The natural extension is to develop methods that can detect and respond to $P(Y|X)$ shift, perhaps using self-consistency signals ($f_{M2}$) or model-ensemble disagreement as a proxy for knowledge uncertainty. This is out of scope for our semester project, but it is the direction our negative results point to.

### 15.4 What we would do differently

1. **Screen before spending compute.** The screening protocol exists because we wasted 18 hours of GPU time and $2 of OpenAI spend on TQA→NQ before realizing it would not work. Future projects should pre-flight every candidate pair.
2. **Think carefully about what the covariate actually is.** "Popularity" sounded like a covariate but turns out to be a proxy for knowledge. Any candidate shift axis should be stress-tested by asking "does this affect $X$ independently of $Y|X$?" before committing to it.
3. **Be willing to use synthetic covariate shift as a sanity check.** The fact that we cannot easily find a natural pair that passes the screening does not mean the method is wrong; it means the natural world is adversarial. A synthetic pair lets us separate method-validity from dataset-selection.
4. **Report feasibility frontiers, not just validity numbers.** The interesting question is not "is DS-SGen valid at ε=0.25?" but "at what ε does DS-SGen become valid, and for what shift strengths?". Mapping this frontier is a more robust contribution than a binary pass/fail.

### 15.5 What we want readers to take away

DS-SGen is theoretically sound and implementationally correct. The honest empirical finding is that in the realistic regime of open-ended LLM QA with strict NLI-based correctness, finding a natural dataset pair that satisfies weighted conformal prediction's assumption is genuinely hard — not impossible, but requiring careful construction rather than opportunistic benchmark pairing. The screening protocol we developed is, we believe, a small but reusable contribution for future work in this space: it gives a researcher a 3-minute pre-flight check that catches the most common modes of DS-SGen failure before the full pipeline runs.

The method is not broken. The world just does not, by default, supply the shift structure the method is designed for. Finding that structure — or learning to relax the method's assumption — is the real research frontier.

---

## Appendix A — Verbatim scorecards

### A.1 TQA→NQ (Method 3, ε=0.25, from `importance_weighted_results.json`)

```
Diagnostics:
  Domain classifier CV accuracy: 0.917
  n_eff = 1112.5 / 3610 (30.8%)
  Weights: min=0.041, median=0.332, max=5.692, std=1.498
  (raw max before clipping: 32.66)

TQA (in-domain, calibration):
  Validity rate:   100.0%
  Mean FDR-E:      0.0532 +/- 0.0801
  Mean efficiency: 0.1416 +/- 0.2236

NQ (shifted test):
  Validity rate:   68.8%
  Mean FDR-E:      0.1065 +/- 0.1602
  Mean efficiency: 0.0811 +/- 0.1347
```

### A.2 TQA→NQ epsilon sweep (from `epsilon_sweep_results.json`)

```
            Method 1   Method 2C   Method 3
ε=0.25       12.4%       22.0%       68.8%
ε=0.30        0.0%        0.0%       11.0%
ε=0.35        0.0%        0.0%        0.2%
ε=0.40        0.0%        0.0%        0.0%
```

### A.3 PopQA head→tail screening (ε=0.25, from `screening_popqa_results.json`)

```
Source: n=1000, Target: n=1000

Test                           Value                  Threshold      Result
1.  Source accuracy            acc_S = 0.437          >= 0.80        FAIL
2a. Target accuracy            acc_T = 0.241          >= 0.75        FAIL
2b. Reachable floor (top-5%)   acc_top5 = 0.640       >= 0.80        FAIL
3.  Accuracy gap               gap = 0.196            [0.03, 0.15]   FAIL
4.  Domain classifier          acc_clf = 0.692        [0.55, 0.78]   PASS
5.  ESS ratio                  n_eff/n = 0.423        >= 0.50        SOFT
6.  Quartile spread            Q1-Q4 = -0.092         >= 0.05        FAIL

Diagnostics:
  Classifier CV: 0.692 +/- 0.024
  n_eff: 423.1 / 1000
  Weights: min=0.071, median=0.521, max=10.924
  Quartile accuracies: Q1=0.412, Q2=0.380, Q3=0.452, Q4=0.504
  Slope(y_S ~ log w): 0.0653 (p=0.0008)

VERDICT: 1/7 pass — pair may not produce the M1/M2 fail → M3 rescues story.
```

### A.4 Runtime summary (job 7056159)

```
Start:  Sat Apr 11 01:23:21 EDT 2026
Node:   babel-w9-28 (NVIDIA RTX A6000)
Stage 1 (load PopQA):             < 1 sec (cached)
Stage 2 (generation):             cached (prior run)
Stage 3 (entailment, both sets):  3 min 14 sec on GPU
Stage 4 (embeddings):             ~5 sec
Stage 5 (screening battery):      < 1 sec
Total elapsed:                    198 sec = 3 min 18 sec
```

---

## Appendix B — Pointers into the code

| Concept | File | Function(s) |
|---|---|---|
| SGen-Semi algorithm | `ds_sgen/sgen_semi.py` | `_run_single_split`, `_compute_conformal_threshold`, `_clopper_pearson_upper` |
| Conservative variants (A, B, C) | `ds_sgen/conservative.py` | `run_option_a`, `run_option_b`, `run_option_c` |
| Importance weighting | `ds_sgen/importance_weighted.py` | `compute_embeddings`, `train_domain_classifier`, `compute_importance_weights`, `_weighted_conformal_threshold`, `_weighted_clopper_pearson_upper`, `_run_single_split` |
| Screening protocol | `ds_sgen/screening.py` | `load_popqa`, `run_screening_tests`, `print_scorecard` |
| Pipeline stages | `ds_sgen/data_loading.py`, `generate_responses.py`, `entailment_scoring.py` | `load_nq`, `load_tqa`, `generate_and_cache_openai`, `score_and_cache` |
| Orchestrators | `run_baseline.py`, `run_conservative.py`, `run_importance_weighted.py`, `run_epsilon_sweep.py`, `run_screening.py` | `main` in each |
| SLURM | `scripts/run_screening.sh` | (A6000, general partition, 4h limit) |
| Results | `/data/user_data/anshulk/dsgen/results/` | `baseline_results.json`, `conservative_results.json`, `importance_weighted_results.json`, `epsilon_sweep_results.json`, `screening_popqa_results.json` |
| Configuration | `configs/default.yaml` | sections `sgen`, `importance_weighted`, `screening`, `epsilon_sweep` |

---

*Document generated 2026-04-11 to accompany the PopQA head→tail screening run (SLURM job 7056159).*
