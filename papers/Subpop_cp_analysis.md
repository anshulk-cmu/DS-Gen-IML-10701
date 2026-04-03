# Conformal Prediction Adaptive to Unknown Subpopulation Shifts — Complete Paper Analysis

**Paper:** Conformal Prediction Adaptive to Unknown Subpopulation Shifts
**Authors:** Nien-Shao Wang, Duygu Nur Yaldiz, Yavuz Faruk Bakman & Sai Praneeth Karimireddy (University of Southern California)
**Venue:** arXiv preprint (arXiv:2506.05583v2), November 2025
**Type:** Empirical / Methods paper with theoretical analysis
**Peer-Reviewed:** No (under review — stated on every page as "Preprint. Under review")
**Reading Purpose:** Deep understanding — this paper proposes a different approach to domain-shift-aware conformal prediction for LLMs, directly relevant to your DS-SGen research project

---

# PHASE -1: PAPER CLASSIFICATION

This paper is an **Empirical / Methods** paper with meaningful theoretical contributions. It proposes three new algorithms (Algorithms 1, 2, and 3) for adapting conformal prediction to subpopulation shifts — a specific type of domain shift where the test environment is a *different mixture* of the same underlying domains as the training data. The paper provides formal coverage guarantees under assumptions about the quality of a domain classifier (Bayes-optimal, multicalibrated, or multiaccurate), and validates everything with experiments on vision (ImageNet + Vision Transformers) and language (LLaMA-3-8B for hallucination detection) tasks.

Because this paper has both theory and experiments, and directly connects to your DS-SGen research on domain shift in conformal prediction for LLMs, we apply the full Pass 1-2-3 framework.

---

# PHASE 0: PRE-READING CONTEXT

**Authors:**

- **Sai Praneeth Karimireddy** (senior author): Assistant Professor at USC. Well-known for work in distributed optimization, federated learning, and fairness in ML. Has publications at top venues (ICML, NeurIPS, ICLR). Strong credibility signal. He's the intellectual leader of this work.
- **Nien-Shao Wang, Duygu Nur Yaldiz, Yavuz Faruk Bakman**: PhD students at USC. Bakman has prior work on MARS (Meaning-Aware Response Scoring), which is cited in this paper and used as one of the score functions for the LLM experiments.

**Venue quality:** This is an arXiv preprint explicitly "under review" (probably submitted to ICML 2026 or a similar top venue). It has NOT been through peer review yet, so we should apply the Credibility Assessment checklist. However, the senior author's track record is strong, which provides some confidence.

**Credibility Assessment (for non-peer-reviewed work):**

- **Author track record:** Strong. Karimireddy has published at top ML venues. Bakman has relevant prior work (MARS, ACL 2024).
- **Institutional affiliation:** USC — a well-known research university.
- **Reproducibility signals:** Code is not explicitly released for the algorithms themselves (mild concern), but the experimental setup is described in detail and uses publicly available datasets (ImageNet, TriviaQA, GSM8K) and models (LLaMA-3-8B).
- **What's NOT disclosed?** The paper is quite transparent about limitations (Section 7). The hyperparameters for Algorithm 3 (σ, β) require tuning, and this tuning process is somewhat heuristic.
- **Independent verification:** Not yet (too recent).

**Why this matters for you:** This paper tackles the same core problem as the DS-CP paper (Lin et al., 2025) — adapting conformal prediction to domain shift — but takes a fundamentally different approach. Instead of estimating density ratios between a single source and target domain, this paper models domain shift as a *subpopulation shift* (the test environment is a different *mixture* of known subpopulations). This gives you a third conceptual tool for your DS-SGen project, alongside the SGen framework and the DS-CP density ratio approach.

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## The Real-World Problem (No Math Yet)

Imagine you're building a medical AI system. During development, you train and calibrate it on data from a hospital that treats a balanced mix of patients: some with heart problems, some with lung problems, some with bone problems. Your conformal prediction system (which gives "prediction sets" — a list of possible diagnoses with a guarantee that the true diagnosis is in the list 90% of the time) works great in this balanced hospital.

Now you deploy the same system at a *different* hospital that happens to specialize in heart conditions. Most of the patients there have heart problems. Heart problems are hard to diagnose, so the AI's uncertainty scores tend to be higher on heart patients. But your conformal prediction threshold was calibrated for the balanced hospital. Since the new hospital has way more hard-to-diagnose heart patients, the threshold is too low — the prediction sets are too small, and the true diagnosis is missing from the set more than 10% of the time. Your "90% guarantee" is broken.

This is **subpopulation shift**: both hospitals draw patients from the same underlying types (heart, lung, bone), but the *proportions* (the mix) are different. The first hospital has equal proportions. The second hospital has mostly heart patients.

The paper asks: **How do we fix conformal prediction so the 90% guarantee holds no matter what mix of subpopulations the test environment has?**

## Q1: What Is the Problem Being Solved?

**In simple words:** When an AI model is calibrated on data from one mix of subpopulations (like 1/3 heart, 1/3 lung, 1/3 bone patients) but tested on a different mix (like 80% heart, 10% lung, 10% bone), the standard conformal prediction guarantee breaks. This paper develops algorithms that adapt the conformal prediction threshold at test time to restore the coverage guarantee, even when the test mix is unknown.

**More precisely:** Given K domain-specific distributions P₁, P₂, ..., P_K, the test data comes from:

P_test = λ₁P₁ + λ₂P₂ + ... + λ_K P_K

where the mixing weights λ₁, ..., λ_K are **unknown** and potentially very different from the calibration data's mixing weights. The paper proposes methods to construct prediction sets that achieve coverage ≥ 1 − α for any possible mixing weights λ.

**In one sentence:** "This paper studies how to adapt conformal prediction to unknown subpopulation shifts by using either a learned domain classifier (Algorithms 1 & 2) or embedding-based similarity reweighting (Algorithm 3) to adjust the calibration threshold, proving coverage guarantees under multicalibration/multiaccuracy assumptions."

## Q2: Why Is This Problem Hard and Interesting?

There are three key difficulties:

### Difficulty 1: The Mixing Weights Are Unknown

If you knew the test environment's mixing weights λ (like "80% heart, 10% lung, 10% bone"), you could simply reweight your calibration data accordingly and standard conformal prediction would work. But in practice, you don't know what mix of patients will show up at a new hospital. The test mix is completely arbitrary and unknown.

Think of it like this: you made a study guide for a test by evenly covering all chapters. But the actual test might focus 80% on Chapter 3. Your preparation was evenly spread, but the test isn't — and you don't know the test's focus in advance.

### Difficulty 2: Group-Conditional CP Seems Like the Answer But Isn't

An obvious fix is "group-conditional conformal prediction" — which means: instead of having one threshold for everyone, have a separate threshold for each subpopulation (heart patients get one threshold, lung patients get another, etc.). If each group's threshold is correct, then any mix of groups will also be correct.

BUT this approach requires knowing which group each test patient belongs to. In practice, you often don't know this — is this patient primarily a "heart" case or a "lung" case? You need a classifier to guess. And the paper proves (Theorem 2.1) that if this group classifier makes mistakes, group-conditional CP can fail *catastrophically*. If the classifier has accuracy γ, the coverage can drop to as low as γ − α (which could be ZERO for a moderate classifier).

This is a crucial insight: **even a good-but-imperfect domain classifier can completely break group-conditional CP.**

### Difficulty 3: The Coverage-Efficiency Tradeoff

You could use a "worst-case" approach — calibrate for the hardest possible subpopulation mix. This guarantees coverage, but it's way too conservative. The prediction sets become huge (containing almost all possible answers), which makes them useless. You want to be *adaptive* — tight prediction sets when the test environment is easy, and larger sets only when necessary.

**In one sentence:** "This is nontrivial because the test mixing weights are unknown, standard group-conditional CP degrades with imperfect group information (shown formally in Theorem 2.1), and worst-case methods are too conservative — you need to adapt to the actual difficulty of the test environment."

## Q3: What Is the Main Claim?

The paper proposes three algorithms:

1. **Algorithm 1 (with domain classifier, per-test-point):** Uses a domain classifier c(X_test) to predict the mixing weights for each individual test point, then reweights domain-specific calibration thresholds accordingly. Requires the classifier to be **multicalibrated** for the guarantee to hold.

2. **Algorithm 2 (with domain classifier, averaged):** Same idea but averages the classifier's predictions over the entire test set instead of using per-point predictions. Requires only the weaker **multiaccuracy** assumption.

3. **Algorithm 3 (no domain classifier needed):** When you don't know the domains at all, filters calibration data by similarity to the test point (using embeddings) and reweights by softmax-normalized similarity scores. No formal guarantee, but works well empirically.

The main theoretical guarantee (Theorem 3.1) says:

**If the domain classifier c is Bayes-optimal (perfect), then Algorithm 1's prediction set satisfies:**

P(Y_test ∈ C_α(X_test)) ≥ 1 − α

And this relaxes to multicalibrated classifiers (Theorem 3.3) and multiaccurate classifiers (Theorem 3.5), both still giving the same ≥ 1 − α guarantee.

**In one sentence:** "They show that Algorithms 1 and 2 provably maintain valid coverage under arbitrary subpopulation shifts when equipped with a multicalibrated or multiaccurate domain classifier, and Algorithm 3 achieves near-oracle coverage empirically even without any domain knowledge, using only embedding-based similarity."

---

# PASS 2: THE SCUBA DIVE — How Does It Work?

## Q1: What Was the Main Technical Hurdle Before This Paper?

### The State of the Art Before This Paper

Before this paper, there were several approaches to handling distribution shift in conformal prediction:

**Approach A: Weighted CP with known density ratios (Tibshirani et al., 2020).** If you know the ratio P_test(x) / P_train(x) for every point, you can reweight calibration samples and recover valid coverage. The problem? This ratio requires knowing the test distribution, which is exactly what you don't have.

**Approach B: Robust/Max CP (Cauchois et al., 2024).** Use the threshold from the "hardest" domain — this guarantees coverage for any test environment. The problem? It's extremely conservative. If the test environment isn't actually the worst case (which it almost never is), your prediction sets are needlessly huge. Imagine always bringing a winter coat everywhere because it might be cold somewhere — even to a beach vacation.

**Approach C: Group-conditional CP (Gibbs et al., 2024; Jung et al., 2022).** Maintain separate thresholds for each subpopulation. The problem? You need to know which group each test point belongs to. This paper's Theorem 2.1 shows this approach breaks badly when group membership is uncertain.

**Approach D: DS-CP (Lin et al., 2025 — the paper you already studied).** Estimates density ratios in embedding space and uses non-exchangeable CP with data-dependent weights. The problem? It handles one-to-one domain shift (old domain → new domain) but doesn't model subpopulation structure. Also, its theoretical guarantee has unknowable error terms.

### The Barrier

The fundamental barrier was: **there was no method that could (a) adapt to unknown subpopulation shifts, (b) work with imperfect group information, and (c) provide formal coverage guarantees simultaneously.** Existing methods either required perfect knowledge (density ratios or exact group labels) or gave up on formal guarantees.

### How This Paper Overcomes It

The key insight is a **two-stage framework**: first learn a domain classifier (which group does this input belong to?), then use its predictions to reweight the calibration threshold. The theoretical innovation is showing that the classifier doesn't need to be perfect — it just needs to satisfy a statistical calibration property called **multicalibration** (or the weaker **multiaccuracy**). These are much easier to achieve than perfect accuracy.

For the case where you don't even know the domains exist, Algorithm 3 bypasses domain classification entirely by using embedding-based similarity — the idea that semantically similar inputs should have similar score distributions.

## Q2: The Core Technical Machinery — Explained Step by Step

### Building Block 1: What Is a "Subpopulation Shift" Exactly?

Let's make this very concrete. Imagine you have 3 types of questions:

- **Domain 1 (Easy):** Basic arithmetic ("What is 5 + 3?")
- **Domain 2 (Medium):** Geography ("What is the capital of France?")
- **Domain 3 (Hard):** Advanced reasoning ("Explain the butterfly effect in chaos theory.")

Your **training/calibration environment** has an equal mix: 1/3 easy, 1/3 medium, 1/3 hard.

Your **test environment** might be anything:
- Test Environment A: 90% hard, 5% medium, 5% easy (tough test!)
- Test Environment B: 80% easy, 10% medium, 10% hard (easy test)
- Test Environment C: Equal mix (same as training — no shift)

Mathematically, the test distribution is:

P_test = λ₁ · P₁ + λ₂ · P₂ + λ₃ · P₃

where λ₁ + λ₂ + λ₃ = 1, and the λ's are **unknown** and could be anything.

The KEY constraint that makes this different from general distribution shift: **the same underlying domains exist in both training and test data — only the proportions change.** This is a more structured (and arguably more realistic) model of shift than arbitrary covariate shift.

### Building Block 2: Why Does Standard CP Fail Under Subpopulation Shift?

Standard conformal prediction computes a **single threshold** q̂_α from all calibration data mixed together. This threshold is the (1 − α)-quantile of all the scores.

Now, each domain has its own score distribution:
- Easy questions: scores are mostly low (the model is confident and correct)
- Hard questions: scores are mostly high (the model is uncertain or wrong)

When you mix these together in the calibration data (1/3 each), you get a blended score distribution, and the threshold sits somewhere in the middle.

If the test environment has 90% hard questions, the actual score distribution at test time is very different from calibration. The threshold is too low — it was calibrated for a balanced mix but the test is mostly hard stuff. Result: the prediction sets are too small and miss the correct answer too often. Coverage drops below 1 − α.

Conversely, if the test environment has 90% easy questions, the threshold is too high — the prediction sets are unnecessarily large. Coverage exceeds 1 − α by a lot, wasting precision.

**The left panel of Figure 1 in the paper** illustrates this beautifully: the gray calibration score distribution and the blue test score distributions are different for each test environment, and the calibration threshold q̂_0.9 lands in the wrong place for environments 2 and 3.

### Building Block 3: What Is a Domain Classifier?

A domain classifier c is a model that takes an input X and predicts a probability distribution over the K domains:

c(X) = (c₁(X), c₂(X), ..., c_K(X))

where c_k(X) ≈ P(X came from domain k).

Think of it like this: given a question, the classifier guesses "this looks like an easy arithmetic question with 70% probability, a medium geography question with 20% probability, and a hard reasoning question with 10% probability."

If the classifier is **perfect** (Bayes-optimal), then for any test input X_test, c(X_test) gives us the exact probability distribution over which domain this test input came from.

### Building Block 4: Algorithm 1 — Weighted CP with Domain Classifier

Here's the algorithm in simple steps:

**Input:** A pretrained model, a domain classifier c, calibration data from each domain, a test point X_test, and the desired error rate α.

**Step 1:** Compute nonconformity scores for all calibration data in each domain.

A nonconformity score is just a number measuring "how surprised is the model by the correct answer?" For example, using the LAC score:

S(X, Y) = 1 − f(X)_Y

where f(X)_Y is the probability the model assigns to the correct answer Y. If the model gives 90% probability to the correct answer, the score is 0.10 (not very surprising). If the model gives only 20%, the score is 0.80 (very surprising).

**Step 2:** Predict domain probabilities for the test point: λ̂ = c(X_test).

This tells us: "for this particular test input, the classifier thinks it's 70% likely from domain 1, 20% from domain 2, and 10% from domain 3."

**Step 3:** Find the threshold q̂_α that satisfies:

Σ_{k=1}^{K} λ̂_k · m_k(q̂_α) / (n_k + 1) ≥ (1 − α)

Let me unpack this formula:
- m_k(q̂_α) = the number of calibration points in domain k whose score is ≤ q̂_α
- n_k = the total number of calibration points in domain k
- m_k(q̂_α) / (n_k + 1) ≈ the fraction of domain-k data covered by this threshold (this is the within-domain coverage)
- λ̂_k = the weight we give to domain k (from the domain classifier)
- The whole sum = the expected coverage across domains, weighted by how likely the test point is from each domain

So we find the **smallest threshold** such that the weighted average coverage across domains is at least 1 − α.

**Step 4:** Build the prediction set: C_α = {answers y : S(X_test, y) ≤ q̂_α}.

**In plain English:** "Weight the domains by how relevant they are to this test point, find a threshold that achieves the desired coverage under this weighting, and include all answers below the threshold."

**The intuition is beautiful:** If the test point looks like a hard question (λ̂₃ is high), the algorithm automatically increases the threshold (making the prediction set bigger) because it needs to cover a harder domain. If it looks easy (λ̂₁ is high), the threshold stays low (tight prediction set).

### Building Block 5: Why Does Algorithm 1 Give Valid Coverage?

The proof (Theorem 3.1) works like this. It builds on a concept called **partial exchangeability** from federated conformal prediction (Lu et al., 2023).

Here's the key idea at a high level:

**Within each domain k**, the calibration data and the test data are exchangeable (they come from the same distribution P_k). This means that within domain k, conformal prediction works perfectly — the calibration scores from domain k and the test score (if the test point is from domain k) are interchangeable.

If the domain classifier is perfect (c = c*), then:

P(S_test ≤ q̂_α) = Σ_{k=1}^{K} λ_k · P(S_test ≤ q̂_α | test point is from domain k)

For each domain k, because of within-domain exchangeability:

P(S_test ≤ q̂_α | from domain k) ≥ m_k(q̂_α) / (n_k + 1)

(This is exactly the standard CP guarantee applied within domain k.)

Therefore:

P(S_test ≤ q̂_α) ≥ Σ_{k=1}^{K} λ_k · m_k(q̂_α) / (n_k + 1) ≥ 1 − α

The last inequality holds by the way we chose q̂_α in Step 3.

**In plain English:** "Since data within each domain is exchangeable, and the classifier correctly tells us the domain probabilities, combining the within-domain guarantees with the correct weights gives us the overall guarantee."

### Building Block 6: What Are Multicalibration and Multiaccuracy?

A perfect domain classifier is unrealistic. The paper relaxes the assumption in two stages:

**Multicalibration** (Definition 3.2): A domain classifier c is multicalibrated if, for any predicted probability vector v and any test distribution D:

E[c*(X) | c(X) = v, X ~ D] = v

**In simple words:** "Whenever the classifier says 'I think this input is 70% domain 1, 20% domain 2, 10% domain 3,' it should be RIGHT ON AVERAGE. That is, among all inputs where the classifier predicted this exact probability vector, the actual domain proportions should indeed be 70/20/10."

This is like a weather forecaster saying "70% chance of rain." Multicalibration means: on all days where the forecaster said "70% chance of rain," it actually rained about 70% of the time.

**Multiaccuracy** (Definition 3.4): A domain classifier c is multiaccurate if, for any test distribution D:

E[c*(X) | X ~ D] = E[c(X) | X ~ D]

**In simple words:** "On average across a test environment, the classifier's predicted domain probabilities match the true domain probabilities." This is weaker — it doesn't need to be correct for each predicted value v, just correct on average.

Continuing the weather analogy: multiaccuracy means the forecaster's predictions are correct on average over the whole week, even if they're off on individual days.

**Why these matter:** Theorem 3.3 says Algorithm 1 works with a multicalibrated classifier. Theorem 3.5 says Algorithm 2 works with just a multiaccurate classifier. Since multiaccuracy is easier to achieve in practice (Hansen et al., 2024 show that well-trained models tend to be relatively multicalibrated), this makes the guarantees more practical.

### Building Block 7: Algorithm 2 — Averaging Over the Test Set

Algorithm 2 is a small but important modification: instead of predicting λ̂ for each individual test point (as Algorithm 1 does), Algorithm 2 **averages the classifier's predictions across the entire test set**:

λ̂ = (1/n_test) Σ_{i=1}^{n_test} c(X_i^test)

This is like saying: "Instead of asking 'what type is THIS specific patient?', let's ask 'what's the overall patient mix in this hospital?'"

**Why this helps:** Averaging smooths out individual prediction errors. Even if the classifier is wrong about specific inputs, the average tends to be correct (by multiaccuracy). This lets us use the weaker assumption.

**The tradeoff:** Algorithm 2 uses the same threshold for ALL test points (it's not adaptive per-point like Algorithm 1). This is fine when all test inputs come from the same environment, but less ideal if different test inputs face different levels of shift.

### Building Block 8: Algorithm 3 — No Domain Knowledge at All

This is the most practically useful algorithm. It doesn't need:
- Knowledge of what the domains are
- A domain classifier
- Domain labels on calibration data

Instead, it uses **embedding-based similarity** to reweight calibration data.

**Step 1:** Embed all calibration data and the test point using an embedding function z (they use all-mpnet-base-v2, a sentence transformer for the language tasks).

**Step 2:** Keep only the top β fraction of calibration data that is most similar to the test point. (They use β = 0.10, meaning keep the top 10%.) This is a filtering step that throws away calibration data that's too different from the test point.

**Step 3:** Compute similarity weights:
- γ_i = h(z(X_test), z(X_i)) for each remaining calibration point
- m = Softmax({γ_i / σ}) — this normalizes the similarities into weights

The temperature parameter σ controls how "peaked" the weights are:
- Small σ → weights are very concentrated on the most similar points
- Large σ → weights are nearly uniform (reduces to standard CP as σ → ∞)

**Step 4:** Set the test point's score to ∞ (s_{n+1} = ∞) — this is the "safety valve" from weighted CP, ensuring the method is slightly conservative.

**Step 5:** Find the (1 − α)-quantile of the weighted score distribution and build the prediction set.

**In plain English:** "Find calibration points that look like the test point (using embeddings), give more weight to the most similar ones, and use their scores to determine the threshold."

**Connection to DS-CP (Lin et al., 2025):** Algorithm 3 is very similar to the DS-CP approach! Both use embeddings + similarity-based reweighting. The differences are:
1. DS-CP uses density ratio estimation (XGBoost classifier), while Algorithm 3 uses direct similarity with softmax normalization
2. DS-CP doesn't filter (keeps all calibration data), while Algorithm 3 filters to top β fraction
3. DS-CP has formal theoretical guarantees (approximate coverage bounds), while Algorithm 3's guarantees are only empirical for this paper

### Building Block 9: Extension to LLM Hallucination Detection

Section 4.2 extends Algorithm 3 to a particularly interesting application: detecting when an LLM is hallucinating (generating incorrect or fabricated answers).

The setup is:
- An LLM generates answers to questions using greedy decoding
- A correctness evaluator (GPT-4o) determines whether each generated answer is a hallucination
- The calibration set consists of ONLY hallucinated examples (where the LLM was wrong)
- The score function measures the model's uncertainty on each query (using LNS, MARS, or Degree Matrix Uncertainty)

The goal is to achieve a target **recall** for hallucination detection: among all actual hallucinations, at least r_test fraction should be flagged.

This is framed as a **conformal risk control** problem: find a threshold such that scores above it are labeled "hallucination," and the recall is at least the target.

Algorithm 3's similarity-based reweighting is applied to adjust this threshold for subpopulation shifts — the idea being that if the test environment has a different mix of question types (more math questions vs. trivia, for example), the hallucination detection threshold should adapt accordingly.

## Q2: What Is the Simplest Baseline and How Much Better Is This?

### Baselines

1. **Unweighted (Standard CP):** Uses a single threshold from all calibration data, ignoring any domain structure. This is the default.

2. **Max (Robust CP):** Uses the threshold from the hardest domain. Guarantees coverage for any test environment but is very conservative.

3. **Conditional Calibration (CC):** The two-stage group-conditional approach from Gibbs et al. (2024), which also uses a domain classifier but assumes it provides perfect group membership.

4. **Oracle:** Uses the TRUE mixing weights λ (which are unknown in practice). This is the best possible — an upper bound on what any method could achieve.

### Results Summary

**Vision experiments (ImageNet with Vision Transformers, Figure 2):**

For 100 different test environments with α = 0.05 (targeting 95% coverage):

| Method | Mean Coverage | Standard Deviation | Behavior |
|--------|-------------|-------------------|----------|
| Unweighted | 0.955 | ±0.014 | Often under-covers; high variance |
| CC (Conditional Calibration) | 0.953 | ±0.010 | Worse than unweighted at low 1−α! |
| Max (robust) | 0.997 | ±0.002 | Severe over-coverage (wastes precision) |
| Algorithm 1 | 0.963 | ±0.006 | Tight coverage, low variance |
| Algorithm 2 | 0.963 | ±0.004 | Tight coverage, lowest variance |
| Oracle | 0.963 | ±0.004 | Practically matched by A1 and A2! |

**Key observations:**

1. **Algorithms 1 and 2 essentially match the oracle.** This means the domain classifier is good enough that having the true weights barely helps more.

2. **Standard CP has high variance** — it achieves the right coverage on average but some test environments get badly under-covered (as low as 0.92 when the target is 0.95). Algorithms 1 and 2 dramatically reduce this variance.

3. **Conditional Calibration (CC) is worse than unweighted CP** in some settings! This validates the paper's Theorem 2.1 — imperfect group information degrades group-conditional CP.

4. **Max CP is way too conservative** — 99.7% coverage when you only need 95%. The prediction sets are huge and uninformative.

**Vision experiments without domain knowledge (Algorithm 3, Figure 3):**

| Method | Mean Coverage | Standard Deviation |
|--------|-------------|-------------------|
| Unweighted | 0.955 | ±0.014 |
| Algorithm 3 | 0.965 | ±0.009 |
| Oracle | 0.963 | ±0.004 |

Algorithm 3 is remarkably close to the oracle, even though it has NO knowledge of domains at all! The standard deviation is much lower than unweighted CP, though not quite as low as Algorithms 1 and 2 (which do use domain knowledge).

**LLM hallucination detection (Figure 4):**

Using LLaMA-3-8B with three different score functions (LNS, MARS, Degree Matrix Uncertainty), across 100 test environments mixing TriviaQA and GSM8K:

The standard (unweighted) method's test recall has **high variance** across test environments — sometimes far above the target, sometimes far below. Algorithm 3 produces recall that **tightly tracks the target** with much lower variance across environments.

This is the most practically relevant result: in a real deployment where the mix of question types is unknown and variable, Algorithm 3 provides much more reliable hallucination detection.

### Comprehensive Results Across Settings (Tables 1–4 in Appendix)

The paper tests across:
- 3 model architectures (ViT, ResNet50, CLIP)
- 3 score functions (LAC, APS, RAPS)
- Different numbers of domains (26 domains with 3 classes each, 15 domains with 17 classes each)
- Different degrees of shift (Dirichlet parameter α' = 0.1 for strong shift, α' = 1 for mild shift)

**Consistent finding across ALL settings:** Algorithms 1, 2, and 3 achieve similar mean coverage to unweighted CP but with dramatically lower standard deviation across test environments. The improvement is largest when the subpopulation shift is strong (α' = 0.1) and smallest when the shift is mild (α' = 1).

## Q3: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: The Subpopulation Shift Model Is Restrictive

The paper assumes that the test distribution is a *mixture of the same K domains* that exist in the calibration data. This means:
- No completely new domains at test time
- The K domains are the same; only their proportions change

If a genuinely new type of question appears at test time (one that doesn't belong to any of the K calibration domains), this framework doesn't handle it. This is a significant restriction compared to the more general covariate shift model in DS-CP.

### Limitation 2: Algorithm 3's Hyperparameters Are Heuristic

Algorithm 3 has two hyperparameters:
- β (what fraction of calibration data to keep): set to 0.10 in most experiments
- σ (temperature for softmax weighting): tuned per α level (see Table 5 in appendix)

The paper acknowledges that choosing σ is a "trade-off between mean and standard deviation across test environments." There's no principled way to set these — you'd need validation data from the test distribution, which defeats the purpose.

### Limitation 3: Over-Coverage Due to Independence Not Being Exploited

The paper's Theorem 3.1 doesn't exploit the independence between samples from different domains. Because it treats the test point's domain membership as uncertain, the bound is slightly conservative. This means the algorithms tend to slightly over-cover (e.g., achieving 96.3% when targeting 95%), especially when the shift is mild. The authors acknowledge this in Section 7.

### Limitation 4: No Guidance on Score Function Selection

The paper tests three score functions (LAC, APS, RAPS for vision; LNS, MARS, DegreeMatrix for language) but doesn't provide guidance on which to choose. Different score functions lead to different behavior, and the "best" choice depends on the specific problem.

### Limitation 5: Scalability of the Domain Classifier Approach

Algorithms 1 and 2 require:
- Knowing the domains at calibration time (having labeled calibration data per domain)
- Training a domain classifier
- The classifier being multicalibrated or multiaccurate

In practice, identifying the "right" set of domains is itself a challenge. If you choose too few domains, you miss important structure. If you choose too many, the domain classifier becomes unreliable and calibration data per domain becomes sparse.

## Q4: Does This Insight Apply to Other Problems?

### Connection 1: Your DS-SGen Research Project

This paper provides a **third conceptual framework** for handling domain shift, alongside the SGen i.i.d. framework and the DS-CP density ratio approach. Here's how it relates:

**What you can borrow:**

1. **The subpopulation shift framing** is potentially a better model for many real-world LLM deployment scenarios. Instead of thinking "the distribution shifts from domain A to domain B," you can think "the distribution shifts from one mix of topics to another mix." This is arguably more realistic — a chatbot deployed in a new setting sees different proportions of the same types of questions, not entirely new question types.

2. **Algorithm 3's embedding + similarity + filtering approach** is a practical tool you could directly use in your DS-SGen framework. Instead of (or in addition to) density ratio estimation, you could filter calibration data to the most similar points and reweight by similarity.

3. **The multicalibration/multiaccuracy framework** for analyzing imperfect domain classifiers could inform your theoretical analysis. When you use an NLI model (like DeBERTa) as part of your entailment-based correctness metric, the NLI model is itself imperfect. The idea that "calibration" of auxiliary models can be formally characterized and its effect on guarantees quantified is valuable.

4. **The hallucination detection application** directly overlaps with your DS-SGen use case. Their use of conformal risk control for hallucination recall is related to your FDR-E control for selective generation. Both aim to make LLM outputs more reliable under distribution shift.

**Where your work differs and adds value:**

1. **Open-ended generation vs. multiple choice:** This paper (like DS-CP) works with finite output spaces. Your DS-SGen targets open-ended text generation — a fundamentally harder problem.

2. **Selective prediction vs. prediction sets:** This paper constructs prediction sets (sets of possible answers). Your DS-SGen adds the ability to abstain ("I don't know"), which is more useful in practice than a set of possible answers.

3. **Entailment-based correctness:** This paper uses exact match for correctness. Your DS-SGen uses textual entailment, which is semantically richer.

4. **PAC guarantees:** This paper's guarantees are asymptotic/expected-value. Your DS-SGen aims for PAC guarantees (finite-sample, high-probability bounds).

### Connection 2: Federated Learning and Multi-Source Data

The subpopulation shift model is closely related to federated learning, where data comes from multiple clients (each a "domain") with different distributions. The partial exchangeability proof technique (from Lu et al., 2023) that underlies this paper came from federated conformal prediction.

### Connection 3: Fairness and Group-Wise Guarantees

The multicalibration framework connects to the fairness literature. Ensuring that a predictor is multicalibrated is a key goal in algorithmic fairness (Hébert-Johnson et al., 2017). This paper shows that fairness-style calibration properties are useful beyond fairness — they enable robust conformal prediction.

## Q5: Caveats and Takeaways

### Strengths

1. **Clear problem formulation.** The subpopulation shift model is well-defined, practically motivated, and sits between "no shift" (too easy) and "arbitrary shift" (too hard/conservative).

2. **Strong theoretical results.** Theorems 2.1, 3.1, 3.3, and 3.5 form a coherent hierarchy of increasingly practical assumptions, each with a formal guarantee.

3. **The negative result (Theorem 2.1) is extremely valuable.** Showing that group-conditional CP fails with imperfect group info motivates the entire paper and is independently useful.

4. **Comprehensive experiments.** 3 model architectures × 3 score functions × multiple domain structures × multiple shift levels × 100 test environments × 15 random splits is very thorough.

5. **Algorithm 3 is a pleasant surprise.** The fact that a simple similarity-based reweighting (with no domain knowledge at all) nearly matches the oracle is remarkable and very practically useful.

6. **LLM application is timely and relevant.** Applying this to hallucination detection under distribution shift directly addresses a real problem in LLM deployment.

### Weaknesses

1. **Not yet peer-reviewed.** The theoretical results should be verified by independent reviewers.

2. **Algorithm 3 lacks theoretical guarantees.** The paper's formal guarantees are for Algorithms 1 and 2 (which need domain knowledge). Algorithm 3 (the most practical one) only has empirical evidence.

3. **Limited LLM experiments.** Only one model (LLaMA-3-8B), two datasets (TriviaQA, GSM8K), and a binary domain structure. Real LLM deployment involves many more domains and much more complex shifts.

4. **Hyperparameter sensitivity for Algorithm 3.** The σ and β parameters need tuning per setting (Table 5 shows σ varying from 0.20 to 2.05 depending on α and the model). No principled selection method is provided.

5. **The subpopulation shift assumption may not hold.** Real-world distribution shifts can involve genuinely new domains (not just different mixtures of existing ones). The paper doesn't test this scenario.

### Devil's Advocate — Three Weakest Points

1. **The LLM experiments are too simple.** Two domains (TriviaQA vs. GSM8K) is a very coarse domain structure. Real-world LLM deployment might involve dozens of topic domains with subtle differences. The binary setting may not stress-test Algorithm 3 adequately. Would it still work with 20 domains where the differences are subtle?

2. **No comparison with DS-CP.** The paper doesn't compare against the density-ratio-based approach from Lin et al. (2025), which addresses a very similar problem. Since DS-CP is the closest competitor, this omission makes it hard to assess the relative merits of the subpopulation shift framing vs. the general covariate shift framing.

3. **The theory-practice gap.** The formal guarantees require multicalibration or multiaccuracy — properties that are hard to verify in practice. The paper cites Hansen et al. (2024) saying "well-trained models tend to be relatively multicalibrated," but doesn't actually test whether their domain classifiers satisfy these properties. Meanwhile, Algorithm 3 (which is the most practically useful) has no formal guarantee at all.

---

# PASS 3: THE SWAMP — Deep Dive into the Mathematical Machinery

## Proof Architecture Overview

The paper has four main theoretical results:

1. **Theorem 2.1:** Negative result showing group-conditional CP fails with imperfect classifiers
2. **Theorem 3.1:** Coverage guarantee with Bayes-optimal domain classifier
3. **Theorem 3.3:** Coverage guarantee with multicalibrated domain classifier
4. **Theorem 3.5:** Coverage guarantee with multiaccurate domain classifier

The proofs build on each other and all leverage the key idea of **partial exchangeability** from Lu et al. (2023). Let me walk through each in detail.

## Proof 1: Theorem 2.1 — Why Group-Conditional CP Fails

**What the theorem says:** If you have a group-conditional CP system that works perfectly with exact group labels, and you replace the exact labels with predictions from a classifier with accuracy γ, the coverage can drop to as low as max(0, γ − α).

**The construction (Appendix A.1):**

The proof constructs a worst-case example with K = 2 domains. The key trick is to make the score distributions completely non-overlapping:

- If X comes from domain 1: S(X) ∈ [0, 1)
- If X comes from domain 2: S(X) ∈ [1, 2]

This means domain 1 is "easy" (low scores) and domain 2 is "hard" (high scores), and you can tell which domain a point is from just by looking at its score.

Now, the group-conditional CP system has:
- Threshold q_{α,1} for domain 1: the (1−α)-quantile of [0, 1) scores
- Threshold q_{α,2} for domain 2: the (1−α)-quantile of [1, 2] scores

Since q_{α,2} > q_{α,1} (domain 2 is harder, needs a higher threshold), the system works perfectly with correct labels.

**Now introduce classifier errors:** Suppose the classifier has accuracy γ, meaning it correctly identifies the domain γ fraction of the time and makes mistakes (1 − γ) fraction of the time.

Consider the **worst case for the classifier's mistakes:** All (1 − γ) fraction of mistakes are on domain-2 inputs (hard questions) that the classifier wrongly labels as domain 1 (easy). These hard inputs get the domain-1 threshold q_{α,1}, which is too low. Their scores are in [1, 2] but the threshold is q_{α,1} < 1, so S(X) > q_{α,1} for all these points. They are ALL miscovered.

**Counting coverage:** 
- γ fraction of domain-2 inputs are correctly identified → covered with probability 1 − α
- (1 − γ) fraction are misidentified → covered with probability 0 (since all their scores exceed q_{α,1})

Overall coverage for domain 2: γ · (1 − α) + (1 − γ) · 0 = γ(1 − α) = γ − γα ≤ γ − α.

**Why this matters:** Even a classifier with 90% accuracy (γ = 0.9), at α = 0.1, could give coverage as low as 0.9 − 0.1 = 0.8 — far below the target 0.9! And for higher α values, coverage can drop to zero.

**The lesson:** Group-conditional CP is extremely brittle to classification errors. This motivates the entire rest of the paper.

## Proof 2: Theorem 3.1 — Coverage with a Perfect Classifier

**What the theorem says:** If c is a Bayes-optimal domain classifier, Algorithm 1 satisfies P(Y_test ∈ C_α(X_test)) ≥ 1 − α.

**The proof (Appendix A.2):**

**Step 1: Condition on the score values.**

Define event E = {all scores within each domain have a specific sorted order}. This is a technical trick that lets us work with specific score values.

Formally:

E = {∀k ∈ [K], ∃π_k, (S^k_{π_k(1)}, ..., S^k_{π_k(n_k)}, S^k_{π_k(n_k+1)}) = (s^k_1, ..., s^k_{n_k}, s^k_{n_k+1})}

where {s^k_i} are the sorted score values. Think of E as "fixing which scores we observe, just shuffling which data point gets which score."

**Step 2: Use the perfect classifier to decompose by domain.**

Since c is perfect (c = c*), we have λ̂ = λ (the true mixing weights). Therefore:

P(S(X_test, Y_test) ≤ q̂_α | E) = Σ_{k=1}^K λ_k · P(S_test ≤ q̂_α | test point is from domain k, E)

**Step 3: Apply within-domain exchangeability.**

Within domain k, the calibration data (X^k_1, Y^k_1), ..., (X^k_{n_k}, Y^k_{n_k}) and the test point (if it's from domain k) are all drawn i.i.d. from P_k. Therefore, their scores are exchangeable.

By exchangeability, conditioned on E, the test score is equally likely to be any of the n_k + 1 sorted score values. The probability that the test score falls at or below q̂_α is:

P(S_test ≤ q̂_α | from domain k, E) ≥ m_k(q̂_α) / (n_k + 1)

where m_k(q̂_α) is the number of domain-k calibration scores at or below q̂_α.

**Step 4: Combine using the algorithm's threshold selection.**

From the algorithm's definition of q̂_α:

Σ_{k=1}^K λ_k · m_k(q̂_α) / (n_k + 1) ≥ 1 − α

Therefore:

P(S_test ≤ q̂_α | E) ≥ Σ_{k=1}^K λ_k · m_k(q̂_α) / (n_k + 1) ≥ 1 − α

**Step 5: Remove the conditioning.**

Since this holds for every possible realization of E (every possible set of sorted score values), taking the expectation (averaging over all possible score realizations) gives:

P(S_test ≤ q̂_α) ≥ 1 − α

This means P(Y_test ∈ C_α(X_test)) ≥ 1 − α. ∎

**The key insight in this proof:** The magic is that within-domain exchangeability is preserved even under subpopulation shift. It's only the *between-domain* proportions that change. By decomposing the coverage into per-domain contributions and weighting correctly, you get the overall guarantee.

**Think of it like a classroom analogy:** If you know that easy questions are fair (the test's easy questions are drawn from the same pool as the practice easy questions), and hard questions are fair (same pool), then ANY mix of easy and hard questions is fair — you just need to know the mix to compute the overall score.

## Proof 3: Theorem 3.3 — Coverage with a Multicalibrated Classifier

**What the theorem says:** Same guarantee as Theorem 3.1, but only requires the classifier to be multicalibrated (not Bayes-optimal).

**The proof (Appendix A.3):**

This proof is nearly identical to Theorem 3.1's, with one key modification at Step 2.

In Theorem 3.1, we used c = c* (perfect classifier), so λ̂ = λ exactly. 

In Theorem 3.3, c is not perfect, but it IS multicalibrated. This means:

E[c*(X_test) | c(X_test) = λ̂, X_test ~ P_test] = λ̂

In other words: conditioned on the classifier predicting λ̂, the TRUE domain probabilities are λ̂ on average.

**The crucial step:** We write:

P(S_test ≤ q̂_α | X_test ~ P_test, c(X_test) = λ̂, E)
= Σ_{k=1}^K λ̂_k · P(S_test ≤ q̂_α | from domain k, E)

This step uses the multicalibration property: conditioned on c(X_test) = λ̂, the effective mixing weights ARE λ̂ (because multicalibration ensures that c's predictions are correct on average whenever it makes a specific prediction).

From here, the proof proceeds exactly as in Theorem 3.1, using within-domain exchangeability to bound each term.

**The final step uses the law of total probability:**

P(S_test ≤ q̂_α | X_test ~ P_test) = Σ_{λ̂} P(S_test ≤ q̂_α | P_test, c(X_test) = λ̂) · P(c(X_test) = λ̂ | P_test) ≥ (1 − α) · Σ_{λ̂} P(c(X_test) = λ̂ | P_test) = 1 − α

**In plain English:** "Even though the classifier makes mistakes on individual inputs, its predictions are calibrated in the sense that when it says 'this looks like domain 1 with probability 0.7,' the actual probability is indeed 0.7 on average. This is enough to make the weighted threshold correct."

## Proof 4: Theorem 3.5 — Coverage with a Multiaccurate Classifier

**What the theorem says:** Same guarantee, but with the even weaker multiaccuracy assumption, using Algorithm 2 (which averages classifier predictions over the test set).

**The proof (Appendix A.4):**

The structure is similar, but instead of conditioning on c(X_test) = λ̂ for each test point, we use the **averaged** λ̂ = E[c(X_test) | X_test ~ P_test].

Multiaccuracy gives us:

E[c*(X_test) | X_test ~ P_test] = E[c(X_test) | X_test ~ P_test] = λ̂

This means the averaged predicted weights match the true weights in expectation.

The rest follows the same pattern: decompose by domain, apply within-domain exchangeability, combine with the threshold selection rule.

**The key difference from Theorem 3.3:** Multiaccuracy only requires the predictions to be correct on average over the test distribution (not for each specific predicted value). This is why Algorithm 2 averages over the test set — it needs the law of large numbers to make the averaged predictions accurate.

## Understanding the Relationship Between the Three Guarantee Levels

The three theorems form a hierarchy from strongest to weakest assumptions:

1. **Bayes-optimal** (Theorem 3.1): c(X) = c*(X) for all X. Strongest assumption, strongest result (works per-point).

2. **Multicalibrated** (Theorem 3.3): c(X) is correct on average for each specific predicted value. Middle assumption, works per-point (Algorithm 1).

3. **Multiaccurate** (Theorem 3.5): c(X) is correct on average over the test distribution. Weakest assumption, works on average (Algorithm 2, needs to average predictions).

**The practical implication:** In practice, well-trained neural network classifiers tend to be approximately multicalibrated (especially after temperature scaling calibration). So Algorithm 1 should work well, and Algorithm 2 provides a backup for cases where multicalibration is questionable.

## Techniques You Can Borrow for Your DS-SGen Research

### Technique 1: Partial Exchangeability Decomposition

The core proof technique — decomposing the overall coverage into per-domain exchangeable components — is directly applicable whenever you have calibration data from multiple sources/domains. In your DS-SGen framework, if you know the calibration data comes from K domains, you could use the same decomposition to get domain-weighted PAC bounds.

### Technique 2: The Multicalibration → Coverage Pipeline

The insight that "classifier calibration → conformal prediction coverage" is a general principle. In your DS-SGen, when you use an NLI model for entailment scoring, the quality of coverage will depend on the NLI model's calibration. You could characterize this dependency using a similar multicalibration framework.

### Technique 3: Algorithm 3's Filtering + Similarity Reweighting

The practical recipe of "filter to top-β similar points, then reweight by softmax-normalized similarity" is a simple, effective alternative to full density ratio estimation. You could use this in your DS-SGen as:
1. Embed all calibration QA pairs using a sentence transformer
2. For each test question, filter to the most similar calibration data
3. Reweight the binomial bounds using these similarity weights
4. Apply the weighted bounds in the SGen-Semi framework

### Technique 4: Dirichlet-Based Evaluation Framework

The experimental design — sampling 100 test environments from a Dirichlet distribution with varying concentration parameter — is an excellent evaluation methodology. You could use the same approach to systematically evaluate DS-SGen across a wide range of subpopulation shift scenarios.

---

# KEY CONCEPTS GLOSSARY (for Grade 12 Level)

| Concept | Simple Explanation |
|---------|-------------------|
| **Subpopulation shift** | When the test data has a different mix of "types" than the training data — like a hospital that sees more heart patients than what the training data reflected |
| **Conformal prediction** | A method that gives a SET of possible answers with a guarantee that the true answer is in the set at least (1−α)% of the time |
| **Coverage** | The probability that the prediction set contains the correct answer — we want this to be at least 1 − α |
| **Domain** | A specific subtype of data (like "easy math" vs "hard reasoning" questions) |
| **Mixing weights (λ)** | The proportions of each domain in the data. Like "40% easy, 30% medium, 30% hard" |
| **Domain classifier** | A model that, given an input, predicts which domain(s) it likely belongs to |
| **Bayes-optimal classifier** | A perfect classifier that knows the true probability of each domain for every input — an ideal that can't be achieved in practice |
| **Multicalibration** | The classifier's predictions are correct on average for every specific prediction value it outputs. Like a weather forecaster being right about "70% rain" predictions 70% of the time |
| **Multiaccuracy** | The classifier's predictions are correct on average over any test distribution. Weaker than multicalibration — like a forecaster being right on average over the week, even if individual days are off |
| **Exchangeability** | A statistical property meaning the data's order doesn't matter — like shuffling a deck of cards. Required for standard conformal prediction to work |
| **Partial exchangeability** | Data is exchangeable WITHIN each domain, but not across domains. The key assumption that enables the paper's approach |
| **Nonconformity score** | A number measuring how "surprised" the model is by the correct answer. High = model didn't expect this answer |
| **LAC (Least Ambiguous Classifier)** | A simple nonconformity score: S(X,Y) = 1 − P(Y\|X). If the model assigns 90% probability to the right answer, the score is 0.10 |
| **APS (Adaptive Prediction Sets)** | A nonconformity score that considers how many answers the model has to list before it reaches the correct one |
| **Softmax** | A function that converts a list of numbers into a probability distribution (all positive, sum to 1). Used to normalize similarity weights |
| **Temperature (σ)** | A parameter controlling how "peaked" the softmax distribution is. Low σ = extreme (one weight dominates). High σ = uniform (all weights similar) |
| **Dirichlet distribution** | A probability distribution over probability distributions. Used to generate random mixing weights λ. The parameter α' controls how extreme the mixing is |
| **Conformal risk control** | An extension of conformal prediction that controls not just coverage but any loss function (like recall for hallucination detection) |
| **Hallucination** | When an LLM generates text that is factually incorrect or fabricated |
| **Recall** | Among all actual positive cases (e.g., actual hallucinations), what fraction did the system correctly identify? |
| **Sentence transformer** | A neural network that converts text into a fixed-length vector capturing the text's meaning. Used to compute similarity between texts |
| **Embedding** | The vector representation of text produced by a sentence transformer. Similar texts have similar embeddings |

---

# CONNECTION TO YOUR DS-SGen RESEARCH PROJECT

## How This Paper Relates to the Other Two Papers You've Studied

| Aspect | SGen (Lee et al., 2024) | DS-CP (Lin et al., 2025) | This Paper (Wang et al., 2025) |
|--------|------------------------|------------------------|-------------------------------|
| **Problem** | Selective generation under i.i.d. | CP under domain shift | CP under subpopulation shift |
| **Shift model** | None (i.i.d.) | General covariate shift | Mixture of known subpopulations |
| **Output space** | Open-ended text | Multiple choice (finite) | Classification + open-ended (binary) |
| **Key technique** | Entailment + binomial bounds | Density ratio estimation | Domain classifier + partial exchangeability |
| **Guarantee type** | PAC (finite sample) | Approximate (TV distance error) | Exact (under calibration assumptions) |
| **Handles abstention?** | Yes (IDK) | No | No (sets only, no abstention) |
| **Needs domain knowledge?** | No | No | A1/A2: Yes; A3: No |
| **Tested on LLMs?** | Yes (GPT-3.5, Alpaca) | Yes (16 models on MMLU) | Yes (LLaMA-3-8B on TriviaQA/GSM8K) |

## What This Paper Adds to Your DS-SGen Toolkit

### New Conceptual Tool: Subpopulation Shift as a Middle Ground

DS-CP assumes general covariate shift (any change in P(X) is possible). This paper assumes subpopulation shift (P_test is a mixture of known components with unknown weights). Your DS-SGen could benefit from considering BOTH frameworks:

- **When you know the calibration data has natural subpopulations** (e.g., different QA datasets, different topic categories), use the subpopulation shift framework for tighter guarantees.
- **When the shift is truly unknown**, use the DS-CP density ratio approach.
- **Always** use the SGen entailment-based correctness and selective prediction machinery.

### New Practical Tool: Algorithm 3 for DS-SGen

Algorithm 3's filter-and-reweight approach could be directly integrated into your DS-SGen framework:

1. **In the SGen-Semi pseudo-labeling step:** Instead of treating all calibration data equally, filter to the most similar data and reweight. This would make pseudo-labels more domain-appropriate.

2. **In the binomial bound computation:** Use similarity-weighted samples instead of uniform samples. This is analogous to importance weighting but with a simpler, more stable estimation method.

3. **In the neuro-selection function:** Add embedding similarity as a third scoring function alongside LLM confidence and self-consistency. Test points that are more similar to the calibration distribution might be more reliable.

### New Theoretical Tool: Partial Exchangeability Proofs

The partial exchangeability proof technique is cleaner and more elegant than the union-bound approach in SGen for handling multi-domain data. If your DS-SGen has access to calibration data from identifiable subpopulations, you could use this proof technique to get tighter bounds than generic importance weighting.

### New Evaluation Tool: Dirichlet-Based Test Environments

The evaluation methodology (sampling 100 test environments from Dirichlet distributions) is excellent. You should use this for evaluating DS-SGen:

1. Choose a multi-domain QA benchmark (like using TriviaQA + SciQ + GSM8K + MedQA as domains)
2. Sample mixing weights from Dirichlet distributions with varying concentration
3. Evaluate DS-SGen's FDR-E and efficiency across all test environments
4. Compare against SGen (no shift handling), DS-CP (density ratios), and this paper's Algorithm 3

---

# SUMMARY: ONE-PAGE CHEAT SHEET

**Problem:** Standard conformal prediction breaks when the test environment has a different mix of subpopulations than the calibration data.

**Key Innovation:** Use a domain classifier to predict the test environment's domain composition, then reweight domain-specific calibration thresholds accordingly. Three algorithms for three levels of domain knowledge.

**Three Algorithms:**
- **A1 (per-point, needs domain classifier):** Predicts domain weights for each test point. Guarantee: coverage ≥ 1−α if classifier is multicalibrated.
- **A2 (averaged, needs domain classifier):** Averages domain predictions across test set. Guarantee: coverage ≥ 1−α if classifier is multiaccurate.
- **A3 (no domain knowledge):** Filters calibration data by embedding similarity, reweights with softmax-normalized similarity scores. No formal guarantee, but empirically matches the oracle.

**Theoretical Contribution:** Hierarchy of increasingly relaxed classifier assumptions (Bayes-optimal → multicalibrated → multiaccurate), each preserving coverage guarantee. Plus a negative result showing group-conditional CP fails with imperfect group information.

**Key Results:**
- All three algorithms maintain coverage within ±1% of target across 100 test environments, while standard CP has ±2.6% variance
- A1 and A2 essentially match the oracle (which knows the true mixing weights)
- A3 nearly matches the oracle with NO domain knowledge at all
- For LLM hallucination detection, A3 dramatically reduces variance in recall across test environments

**Main Limitations:**
- Assumes subpopulation structure (no genuinely new domains at test time)
- A3 has no formal guarantee
- Hyperparameters (σ, β) require tuning
- Not peer-reviewed yet
- LLM experiments limited to binary domain structure

**For Your DS-SGen Project:** Use Algorithm 3's filter-and-reweight approach as a practical tool, the Dirichlet evaluation methodology for systematic testing, and the partial exchangeability proof technique as a theoretical tool for multi-domain scenarios. The subpopulation shift framing offers a middle ground between "no shift" (SGen) and "arbitrary shift" (DS-CP) that may better capture real LLM deployment scenarios.