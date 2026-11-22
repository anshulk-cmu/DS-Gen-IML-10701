# DS-SGen: Domain-Shift-Aware Selective Generation with PAC Guarantees for Reliable LLMs

## A Full Research Paper Proposal — Targeting NeurIPS Submission

---

## 1. Problem Statement and Motivation

Large language models hallucinate — they produce confident, fluent text that is factually wrong. In high-stakes settings like medicine, law, and finance, a wrong answer can be catastrophic. The natural solution is **selective prediction**: let the model say "I don't know" when it is uncertain, and guarantee that when it *does* answer, the answer is usually correct. The SGen framework (Lee et al., NeurIPS 2024) achieved exactly this under one critical assumption — that calibration data and test data come from the same distribution (the i.i.d. assumption).

But in practice, this assumption almost never holds. A medical QA system calibrated on general health questions will encounter specialized oncology queries. A legal assistant trained on US case law will face questions about EU regulations. This mismatch — called **domain shift** — silently breaks every statistical guarantee that SGen provides. The PAC bound that promised "at most 25% of answered questions are wrong" might actually allow 60% errors under a new domain. The user has no warning that the guarantee has failed.

**Our core research question:** Can we build a selective generation system for LLMs that maintains provable PAC guarantees on its false discovery rate even when test queries come from a different domain than the calibration data?

---

## 2. Gap Analysis: What Exists and What Is Missing

We conducted a systematic review of five foundational papers and identified a precise, unfilled gap at their intersection.

**Paper 1 — SGen (Lee et al., NeurIPS 2024):** Introduces selective generation with entailment-based correctness. The model either answers or says "I don't know," and controls the False Discovery Rate with entailment (FDR-E) via PAC bounds: P{FDR-E ≤ ε} ≥ 1−δ. The semi-supervised variant (SGen-Semi) uses conformal prediction to pseudo-label unlabeled data, reducing annotation costs. **Limitation:** All guarantees require i.i.d. data — they break completely under domain shift.

**Paper 2 — Weighted Conformal Prediction (Tibshirani et al., NeurIPS 2019):** The foundational theory for adapting conformal prediction to covariate shift. By reweighting calibration scores using the likelihood ratio w(x) = P_test(x)/P_train(x), coverage guarantees are restored. Introduces "weighted exchangeability" as a theoretical tool. **Limitation:** Requires knowing or estimating the density ratio, which is intractable in high-dimensional text space. No connection to selective prediction or LLMs.

**Paper 3 — DS-CP (Lin et al., arXiv 2025):** Applies weighted conformal prediction to LLMs by embedding prompts into a low-dimensional space (using sentence transformers), training a domain classifier (XGBoost), and converting classifier probabilities into density ratio weights. Tested on 16 models across 272 MMLU domain pairs. **Limitation:** Only works for multiple-choice QA (finite output space). Cannot abstain. Uses approximate coverage guarantees with unknowable error terms. Not peer-reviewed.

**Paper 4 — Conformal Factuality (Mohri & Hashimoto, Stanford, 2024):** Shows that every LLM output implicitly defines an uncertainty set through textual entailment — making conformal prediction tractable for open-ended generation for the first time. Removes uncertain sub-claims to guarantee output correctness. **Limitation:** Marginal-only guarantee (fails for hard subgroups). Removes too many claims (~76% filtered). Assumes i.i.d. data.

**Paper 5 — Enhanced CP (Cherian, Gibbs, Candès, NeurIPS 2024):** Fixes Conformal Factuality's problems with conditional boosting (learning optimal scoring functions by differentiating through conformal calibration) and level-adaptive CP (per-prompt guarantee levels that maintain ~70% claim retention). Provides Corollary A.1 showing the method handles certain covariate shifts within the function class F. **Limitation:** The covariate shift handling requires F to be chosen a priori to capture the shift — not adaptive to unknown shifts.

**Paper 6 — Subpopulation CP (Wang et al., arXiv 2025):** Addresses subpopulation shift (test distribution is a different mixture of known subpopulations). Proposes three algorithms with progressively weaker assumptions on domain classifier quality. **Limitation:** Requires known subpopulation structure. Not tested on open-ended generation.

### The Precise Gap

No existing method provides all four of the following simultaneously: (1) PAC-style finite-sample guarantees (not just marginal coverage), (2) open-ended text generation with entailment-based correctness (not just multiple-choice), (3) selective prediction with abstention capability ("I don't know"), and (4) robustness to domain shift between calibration and test data. DS-SGen fills exactly this gap.

---

## 3. Proposed Method: DS-SGen

### 3.1 Framework Overview

DS-SGen wraps around any black-box LLM and operates in three stages:

**Stage 1 — Domain-Aware Importance Weight Estimation.** We embed all calibration prompts and the test prompt using a sentence transformer (e.g., all-MiniLM-L6-v2 or BGE-large). We train a domain classifier (XGBoost or logistic regression) to distinguish calibration embeddings from test-time embeddings (collected from a small unlabeled pool of test-domain queries). The classifier's output probabilities are converted to density ratio weights: ŵ(x) = p̂(x)/(1−p̂(x)), where p̂(x) is the predicted probability that x comes from the test domain. We apply weight clipping (cap at the 95th percentile) and normalization to prevent degeneracy from extreme weights.

**Stage 2 — Weighted Entailment-Based Calibration.** We inherit SGen's entailment-based correctness metric: for a question q with reference answer a* and model response â, correctness is determined by whether â textually entails a* (using a fine-tuned DeBERTa-large NLI model). The semi-supervised pseudo-labeling from SGen-Semi is modified: instead of using standard conformal prediction for pseudo-labels, we use *weighted* conformal prediction with our importance weights. This ensures pseudo-labels are calibrated to the test domain. The binomial tail bounds used in SGen's PAC guarantee are replaced with **weighted binomial bounds** that account for the non-uniform contribution of each calibration sample.

**Stage 3 — Domain-Aware Selective Generation.** The neuro-selection function from SGen (which combines multiple confidence scores with learnable thresholds) is augmented with a third signal: **domain similarity score** — the cosine similarity between the test prompt embedding and the calibration distribution centroid. This signal helps the system abstain more aggressively on queries that are far from any calibration data, where guarantees are weakest. The final selection threshold is optimized to maximize answer efficiency (fraction of questions answered) subject to the weighted PAC constraint.

### 3.2 Theoretical Contribution

**Theorem (Informal):** Under the covariate shift assumption (P_test(Y|X) = P_cal(Y|X) but P_test(X) ≠ P_cal(X)), and given importance weight estimates ŵ with bounded estimation error, DS-SGen achieves: P{FDR-E(ĝ) ≤ ε} ≥ 1 − δ − O(Δ_w), where Δ_w is the total variation distance between the true and estimated weight distributions, and ĝ is the learned selection function.

The key technical insight is that SGen's FDR-E decomposition (Lemma 1 of the original paper) is purely algebraic and distribution-agnostic — it holds regardless of domain shift. What changes is how we bound each component term (FER, FNER, NER). By replacing uniform binomial bounds with importance-weighted Hoeffding-type bounds, each component remains controllable under shift. The union bound across components then yields the overall PAC guarantee with an additive penalty proportional to the weight estimation error.

### 3.3 Two Alternative Domain-Shift Strategies

We propose to investigate two complementary approaches and compare them empirically:

**Approach A — Importance Reweighting (from DS-CP):** Estimate density ratios via embedding + classifier, reweight all calibration data, use weighted PAC bounds. This is more flexible (handles arbitrary covariate shift) but depends on density ratio estimation quality.

**Approach B — Conditional Conformal (from Enhanced CP):** Include domain-related features (embedding distance to calibration centroid, predicted domain membership probability) in the function class F of conditional conformal prediction. The resulting guarantee holds conditionally on these features, providing domain-shift robustness "for free" without explicit density estimation. This is cleaner but requires F to be designed a priori.

---

## 4. Proposed Datasets

We design experiments to test across progressively harder domain shifts:

**Dataset 1 — Natural Questions → TriviaQA → SciQ (Open-ended QA cross-domain).** Calibrate on NQ (web search queries), test on TriviaQA (trivia) and SciQ (science). These represent mild-to-moderate shift. Open-ended generation setting with entailment evaluation. This is the primary benchmark.

**Dataset 2 — MMLU cross-domain pairs (Multiple-choice, for comparison with DS-CP).** Use the same 17-subject MMLU setup as DS-CP (272 ordered domain pairs). This allows direct comparison with DS-CP baselines and validates our importance weight estimation pipeline.

**Dataset 3 — MedLFQA medical QA (High-stakes domain shift).** Calibrate on HealthSearchQA (general health), test on MedicationQA (pharmacology) and LiveQA (consumer medical questions). This tests the framework in a safety-critical setting where domain shift is most dangerous. Following Cherian et al.'s experimental setup.

**Dataset 4 — TruthfulQA (Adversarial/out-of-distribution).** Test on questions specifically designed to trigger hallucinations. This represents an extreme shift scenario where the model's typical confidence patterns are unreliable.

**Dataset 5 — CoQA (Conversational QA).** Multi-turn conversational questions that shift naturally across topics within a conversation. Tests robustness to within-session domain drift.

---

## 5. Proposed Models

We test across a range of LLM architectures to ensure generality:

**Primary model (PoC):** LLaMA-3.1-8B-Instruct (open-source, white-box access for confidence scores and token probabilities). This is the workhorse for the 10-701 PoC and initial ablation studies.

**Scale test:** LLaMA-3.1-70B-Instruct (to verify that results hold at larger scale).

**Black-box test:** GPT-4o-mini via API (to test the framework without logit access, using sampling-based confidence like self-consistency).

**Comparison baselines:** Qwen2.5-7B-Instruct and Mistral-7B-v0.3 (to verify model-agnosticity across families).

**Entailment model:** DeBERTa-v2-xxlarge-mnli (1.5B params, label order: {0:CONTRADICTION, 1:NEUTRAL, 2:ENTAILMENT}) for entailment-based correctness scoring, following SGen.

**Embedding model:** all-MiniLM-L6-v2 or BGE-large-en-v1.5 (for prompt embeddings in the density ratio estimation pipeline).

---

## 6. Experimental Design

**Experiment 1 — Coverage Validity under Domain Shift.** For each dataset and model, compare: (a) vanilla SGen (ignoring shift), (b) DS-CP (reweighting but no abstention), (c) DS-SGen-A (importance reweighting), (d) DS-SGen-B (conditional conformal). Measure: actual FDR-E vs. target ε across 100+ random calibration/test splits. The key metric is whether the PAC guarantee holds empirically.

**Experiment 2 — Selection Efficiency.** Measure the fraction of questions answered (not abstained) at the target FDR-E level. DS-SGen should achieve higher efficiency than naive approaches (which must be overly conservative to account for shift) while maintaining validity.

**Experiment 3 — Ablation Studies.** (a) Effect of weight clipping threshold on guarantee validity vs. efficiency tradeoff. (b) Embedding model choice (MiniLM vs. BGE vs. OpenAI embeddings). (c) Domain classifier choice (XGBoost vs. logistic regression vs. neural network). (d) Calibration set size sensitivity. (e) Severity of domain shift (near vs. far domains).

**Experiment 4 — Comparison of Approaches A vs. B.** When does importance reweighting outperform conditional conformal, and vice versa? We hypothesize that Approach A handles arbitrary shift better while Approach B is tighter when the shift aligns with the chosen features.

---

## 7. Expected Contributions and Novelty

This work makes four contributions that are each individually meaningful and collectively represent a significant advance:

**Contribution 1 (Theoretical):** The first PAC guarantee for selective generation under covariate shift. We extend SGen's binomial-based PAC bounds to the importance-weighted setting with a precise characterization of the additional error term due to weight estimation.

**Contribution 2 (Methodological):** A complete, practical framework (DS-SGen) that combines entailment-based correctness, selective prediction, semi-supervised calibration, and domain-shift adaptation into a unified pipeline. No prior work connects all four components.

**Contribution 3 (Empirical):** The first comprehensive evaluation of selective generation under domain shift, spanning 5 datasets, 5+ models, and hundreds of domain-shift pairs. This establishes benchmarks for future work.

**Contribution 4 (Practical):** A deployable system for reliable LLM abstention under distribution shift — directly useful for any production LLM system that encounters queries outside its calibration domain (which is essentially all of them).

---

## 8. Risk Assessment and Mitigation

**Risk 1:** Density ratio estimation fails in high dimensions. **Mitigation:** We operate in embedding space (384-768 dimensions), not raw text space, and apply weight clipping. DS-CP showed this works across 272 domain pairs.

**Risk 2:** Weighted PAC bounds are too loose (vacuous). **Mitigation:** We follow the effective sample size analysis from Tibshirani et al. If effective sample size is too small, the system abstains on everything — a safe failure mode.

**Risk 3:** Entailment model errors compound with domain shift. **Mitigation:** We evaluate NLI model calibration on the target domain and include it as a controlled variable.

---

## 9. Timeline

Weeks 1-3: Implement SGen-Semi baseline and DS-CP baseline. Weeks 4-6: Implement DS-SGen (both approaches), prove theoretical bounds. Weeks 7-9: Run all experiments across datasets and models. Weeks 10-12: Ablation studies, analysis, paper writing.

---

*This proposal targets NeurIPS 2026 as the primary venue, with ICML 2026 as backup. The work sits squarely in the "Trustworthy ML" and "Uncertainty Quantification" tracks.*