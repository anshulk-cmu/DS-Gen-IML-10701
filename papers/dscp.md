# Domain-Shift-Aware Conformal Prediction for Large Language Models — Complete Paper Analysis

**Paper:** Domain-Shift-Aware Conformal Prediction for Large Language Models
**Authors:** Zhexiao Lin (UC Berkeley), Yuanyuan Li, Neeraj Sarna, Michael von Gablenz (Munich RE), Yuanyuan Gao (UC Berkeley)
**Venue:** arXiv preprint (arXiv:2510.05566v1), October 2025
**Type:** Empirical / Methods paper with theoretical analysis
**Peer-Reviewed:** No (arXiv preprint, not yet published at a conference)
**Reading Purpose:** Deep understanding — this paper directly addresses the domain shift problem in conformal prediction for LLMs, which is central to your DS-SGen research project

---

# PHASE -1: PAPER CLASSIFICATION

This paper is an **Empirical / Methods** paper with supporting theoretical results. It proposes a new method (DS-CP) for adapting conformal prediction to work under domain shift for LLMs, provides coverage guarantee theorems, and demonstrates effectiveness through experiments on the MMLU benchmark. The theoretical results (Theorems 1 and 2) are generalizations of existing work (Barber et al., 2023), while the main novelty lies in the **practical framework** for choosing weights systematically using embeddings and density ratio estimation.

Because this paper sits at the intersection of conformal prediction, domain shift, and LLMs — all three pillars of your DS-SGen research — we should apply the full Pass 1-2-3 framework with extra attention to how the techniques can be borrowed.

---

# PHASE 0: PRE-READING CONTEXT

**Authors:**
- Zhexiao Lin: PhD student at UC Berkeley (Statistics). Has published in Econometrica and Biometrika — top-tier statistics journals. This is a strong credibility signal. The paper was done while Lin was an intern at Munich RE.
- Yuanyuan Li, Neeraj Sarna, Michael von Gablenz: Munich RE (one of the world's largest reinsurance companies). Industry motivation for reliable AI — they care about this because wrong AI predictions in insurance/finance cost real money.
- Yuanyuan Gao: UC Berkeley Statistics. Similar academic pedigree.

**Venue quality:** This is an arXiv preprint, meaning it has NOT been through peer review at a conference or journal. This means we should apply extra scrutiny — the results haven't been validated by independent reviewers. However, the first author's track record (Econometrica, Biometrika) provides some credibility.

**Code released:** Not explicitly mentioned. The experiments use data from an existing GitHub repository (LLM-Uncertainty-Bench). This is a mild concern for reproducibility of the DS-CP method itself.

**Why this matters for you:** This paper attempts to solve *exactly* the problem your DS-SGen project targets — extending conformal prediction to handle domain shift for LLMs. It's a different approach than what you've proposed (they work with prediction sets for multiple-choice QA, you're working with selective generation for open-ended QA), but the core technical machinery (embedding-based density ratio estimation, reweighting calibration data) is directly relevant.

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## The Real-World Problem (No Math Yet)

Imagine you're a doctor using an AI system that was trained and calibrated (fine-tuned for reliability) on general medical textbooks. Now a patient comes in with a rare tropical disease that wasn't well covered in those textbooks. The AI gives you a set of possible diagnoses with a "confidence guarantee" — it says "I'm 90% sure the real diagnosis is somewhere in this set."

But here's the problem: the AI's guarantee was calibrated on general medicine questions. Tropical diseases are a different "domain" — the questions look different, the difficulty is different, the patterns are different. When you shift to this new domain, the AI's 90% guarantee might actually only be correct 60% of the time. That's **domain shift breaking conformal prediction**.

This paper builds a system that fixes this problem. It says: "Before I give you my confidence set for this tropical disease question, let me look at how similar this question is to the ones I was calibrated on, and adjust my confidence accordingly." If the question is very different from what it's seen before, it will make the set bigger (more cautious). If it's similar, it keeps the set tight (more useful).

## Q1: What Is the Problem Being Solved?

**In simple words:** Standard conformal prediction (a method for giving AI models reliable uncertainty estimates) breaks down when the test data comes from a different domain than the calibration data. This paper proposes a way to fix this for LLMs by reweighting calibration data based on how similar they are to the test input.

**More precisely:** Given a pre-trained LLM and calibration data from an "old domain," the paper wants to construct prediction sets for inputs from a "new domain" that still achieve the target coverage probability (e.g., 90%), even though the calibration and test data come from different distributions.

**In one sentence:** "This paper studies how to maintain valid conformal prediction coverage guarantees for LLMs when the test-time distribution of prompts shifts away from the calibration distribution, by leveraging sentence embeddings to estimate and correct for the distributional difference."

## Q2: Why Is This Problem Hard and Interesting?

There are three key difficulties:

### Difficulty 1: The Exchangeability Assumption Breaks

Standard conformal prediction requires that calibration data and test data are "exchangeable" — roughly, they come from the same distribution and any ordering is equally likely. When you calibrate on humanities questions but test on physics questions, this assumption is violated. The prediction sets can become unreliable — they might contain the right answer only 60% of the time when you wanted 90%.

Think of it this way: if you practiced free throws in your driveway and made 90% of them, that doesn't guarantee you'll make 90% in a packed arena with crowd noise. The "distribution" has shifted.

### Difficulty 2: LLM Prompts Live in a Huge, Messy Space

The standard fix for domain shift in conformal prediction (from Tibshirani et al., 2019) involves estimating how the data distribution changes — specifically computing a "density ratio" between the old and new domains. But LLM prompts are text — long sequences of tokens with complex structure. Estimating density ratios directly on text is effectively impossible because the space is so high-dimensional and unstructured.

Imagine trying to measure "how different is this physics question from this humanities question" by looking at the raw characters. You'd need to understand the *meaning*, not just the characters.

### Difficulty 3: Extreme Weights Can Make the Method Useless

Even if you can estimate the density ratio, when the old and new domains are very different, the weights become extreme — a few calibration points get huge weights and the rest get nearly zero. This makes the prediction set balloon to include ALL possible answers, which is technically "valid" (it contains the right answer!) but completely useless (it doesn't narrow anything down).

It's like a weather forecaster saying "the temperature tomorrow will be between -50°C and 60°C" — technically correct, but not helpful.

**In one sentence:** "This is nontrivial because the exchangeability assumption of standard CP breaks under domain shift, density ratio estimation in the raw prompt space is infeasible due to high dimensionality, and naive reweighting produces degenerate prediction sets."

## Q3: What Is the Main Claim?

The paper proposes **DS-CP** (Domain-Shift-Aware Conformal Prediction), which:

1. **Embeds** prompts into a lower-dimensional semantic space using a sentence transformer
2. **Estimates density ratios** in this embedding space using a classifier (XGBoost)
3. **Regularizes** the weights to prevent degenerate prediction sets
4. **Constructs prediction sets** using the reweighted calibration scores

The theoretical guarantee (Theorem 1) says:

$$P(Y_{n+1} \in \hat{C}_n(X_{n+1})) \geq 1 - \alpha - \text{(error term depending on domain shift)}$$

The error term depends on the total variation distance between score distributions across domains — if the domains produce similar nonconformity scores, the coverage is close to 1 − α even under large prompt-level shifts.

**In one sentence:** "They show that DS-CP, by embedding prompts and reweighting calibration scores with estimated density ratios plus regularization, achieves more reliable coverage than standard CP across 16 LLMs and 272 domain-shift pairs on MMLU, while keeping prediction sets only modestly larger."

---

# PASS 2: THE SCUBA DIVE — How Does It Work?

## Q1: What Was the Main Technical Hurdle Before This Paper?

### The State of the Art Before DS-CP

There were three existing approaches to conformal prediction under domain shift:

**Approach A: Weighted Conformal Prediction (Tibshirani et al., 2019).** This is the "classical" fix. Under the assumption of **covariate shift** (the distribution of inputs X changes, but the relationship between X and Y stays the same), you reweight calibration samples by the density ratio r(x) = P_new(x) / P_old(x). This works beautifully when X is low-dimensional (like a handful of numeric features). But for LLM prompts (thousands of tokens), estimating this density ratio is a statistical nightmare — the "curse of dimensionality" makes it unreliable.

**Approach B: Non-exchangeable Conformal Prediction (Barber et al., 2023).** This is a very general framework that allows arbitrary fixed weights on calibration points. The coverage guarantee becomes approximate: you get 1 − α *minus* an error term that depends on the total variation distance between score distributions. The problem? The weights are chosen *heuristically* — there's no principled, data-driven way to pick them. The framework tells you "if you pick good weights, you get good coverage" but doesn't tell you HOW to pick good weights.

**Approach C: Ad-hoc nearest-neighbor methods (Ulmer et al., 2024).** Some work applied non-exchangeable CP to LLMs using heuristic similarity-based weights, but without a systematic framework or density ratio estimation.

### The Barrier

The fundamental barrier was: **there was no systematic, data-driven way to choose weights for non-exchangeable CP applied to LLMs**. You either needed to estimate density ratios in an impossibly high-dimensional space, or you picked weights by gut feeling with no guarantees about their quality.

### How This Paper Overcomes It

The key insight is a **two-step dimension reduction and regularization strategy**:

1. **Project prompts into embedding space**: Instead of working with raw text, map every prompt through a sentence transformer (a neural network trained to capture meaning) into a 384-dimensional vector. This makes density ratio estimation feasible.

2. **Estimate density ratios in embedding space**: Train a simple classifier (XGBoost) to distinguish "old domain" embeddings from "new domain" embeddings. The classifier's predicted probabilities directly give you density ratio estimates.

3. **Regularize the test point weight**: Instead of using the test point's actual density ratio (which can be extreme), replace it with a fixed regularization parameter λ (defaulting to 1). This prevents the prediction set from degenerating.

This is clever because it bridges the gap between the theoretical framework (non-exchangeable CP) and practical application (LLMs) by providing a principled, data-driven weighting scheme.

## Q2: The Core Technical Machinery — Explained Step by Step

### Building Block 1: What Is Conformal Prediction?

Think of conformal prediction as a way to build a "safety net" around an AI model's predictions. Instead of the model saying "the answer is B," it says "I'm confident the answer is one of {A, B, C}."

The mathematical guarantee is: with probability at least 1 − α, the true answer is in the prediction set. If you set α = 0.10, you get a 90% guarantee.

Here's how standard conformal prediction works, step by step:

**Step 1: Define a "nonconformity score."** This is a number that measures how "surprised" the model is by the true answer. For multiple-choice questions:

$$S(X, Y) = 1 - f(X)_Y$$

where f(X)_Y is the probability the model assigns to the correct answer Y. If the model is very confident in the right answer (say, 95% probability), the score is small (0.05). If the model is unsure (say, 20% probability), the score is large (0.80).

Think of it as a "surprise meter." Low score = not surprising (model predicted well). High score = very surprising (model got it wrong or was uncertain).

**Step 2: Compute scores on calibration data.** You have n calibration examples with known answers. You compute S₁, S₂, ..., Sₙ for each one.

**Step 3: Find the threshold.** Sort all the scores and find the (1 − α)-quantile — the value below which (1 − α) fraction of the scores fall. Let's call this q̂.

**Step 4: Build the prediction set.** For a new test question x, include answer y in the prediction set if and only if S(x, y) ≤ q̂. In other words, include all answers that the model isn't "too surprised" by.

**Why this works:** If the calibration and test data come from the same distribution, then the test score is "exchangeable" with the calibration scores — it's like drawing one more sample from the same pile. The quantile ensures that the test score falls below it about (1 − α) of the time, which means the true answer is in the prediction set about (1 − α) of the time.

**Why it FAILS under domain shift:** If the test data comes from a harder domain, the test scores might be systematically higher (the model is more often wrong). The threshold q̂ was calibrated for the old domain, so it's too low for the new domain — the prediction sets are too small and miss the correct answer too often.

### Building Block 2: Weighted Conformal Prediction

The fix for covariate shift is to reweight the calibration scores. The idea: some calibration points are "more representative" of the new domain than others. Give those points more weight.

The weight for calibration point i is:

$$w_i = r(X_i) = \frac{P_{\text{new}}(X_i)}{P_{\text{old}}(X_i)}$$

This is the **density ratio** — how much more likely is this input under the new distribution compared to the old one?

Think of it like a political poll. If you surveyed a college campus (old domain) but want to know what the whole country (new domain) thinks, you'd give more weight to survey respondents who are demographically representative of the broader population. The density ratio tells you how much to upweight or downweight each respondent.

The weighted empirical distribution becomes:

$$\sum_{i=1}^{n} \frac{w_i}{\sum_{j=1}^{n} w_j + r(x_{n+1})} \delta_{S_i} + \frac{r(x_{n+1})}{\sum_{j=1}^{n} w_j + r(x_{n+1})} \delta_\infty$$

The δ_∞ term is a "safety valve" — it puts some probability mass on infinity, which inflates the quantile slightly upward, making the prediction set a bit larger. This is what provides the coverage guarantee.

**The problem for LLMs:** The density ratio r(x) lives in the space of text prompts, which is enormous. Estimating it reliably is essentially impossible.

### Building Block 3: The Embedding Trick

Here's the paper's first key move. Instead of estimating density ratios in the raw text space, they:

1. Take every prompt X and run it through a **sentence transformer** h: X → Z (specifically, all-MiniLM-L6-v2), producing a 384-dimensional vector Z = h(X).

2. Estimate density ratios in the **embedding space** Z instead of the text space X.

**Why this works:** Sentence transformers are trained so that texts with similar meanings get mapped to nearby vectors. A physics question and a similar physics question will have nearby embeddings, even if they use different words. This captures the "semantic" distance between domains in a way that raw text can't.

**In everyday terms:** Imagine trying to compare two books by looking at every individual letter on every page versus summarizing each book in a paragraph. The paragraph comparison is much easier and still captures the important differences. The embedding is like the "paragraph summary" of a prompt.

The density ratio in embedding space is:

$$r_e(z) = \frac{P'_Z(z)}{P_Z(z)}$$

where P_Z is the distribution of calibration embeddings and P'_Z is the distribution of test embeddings.

### Building Block 4: Estimating the Density Ratio with a Classifier

How do you actually compute r_e(z)? The paper uses a clever trick based on **Bayes' theorem**.

Label all calibration embeddings as W = 0 ("old domain") and all test embeddings as W = 1 ("new domain"). Train a classifier to predict P(W = 1 | Z = z) — the probability that an embedding z came from the new domain.

Then the density ratio is:

$$r_e(z) = \frac{P(W = 1 | Z = z)}{P(W = 0 | Z = z)} \cdot \frac{P(W = 0)}{P(W = 1)}$$

This is just a mathematical identity — you can verify it by plugging in Bayes' rule. The beauty is that estimating a classifier (which domain does this embedding belong to?) is much easier and more standard than directly estimating a density ratio.

**In everyday terms:** Instead of asking "how much more common are physics-like questions in the new domain?", you ask "given a question's embedding, can I tell whether it came from the old domain or the new domain?" If the classifier says "this calibration point looks a lot like a new-domain point" (high P(W=1)), it gets a high weight.

The paper uses **XGBoost** (a popular, robust machine learning algorithm) as the classifier.

### Building Block 5: The Regularization Step — Preventing Degeneracy

Here's the second key move. In standard weighted CP, the test point also gets a weight — its density ratio r_e(h(X_{n+1})). When the new domain is very different from the old domain, this weight can be enormous, and the resulting empirical distribution puts almost all mass on δ_∞. The quantile becomes infinity, and the prediction set becomes everything (all possible answers). Valid, but useless.

The fix: **replace the test-point weight with a fixed constant λ** (default λ = 1).

The empirical distribution of scores becomes:

$$\mu_n = \sum_{i=1}^{n} \frac{\hat{w}_i}{\sum_{j=1}^{n} \hat{w}_j + \lambda} \delta_{S_i} + \frac{\lambda}{\sum_{j=1}^{n} \hat{w}_j + \lambda} \delta_\infty$$

**Why λ = 1 is a good default:** When there's no domain shift, all weights ŵ_i ≈ 1, and λ = 1 gives you exactly standard CP. When there IS domain shift, the calibration points with high density ratios (those that "look like" the new domain) get upweighted, while the regularization keeps the prediction sets from exploding.

**In everyday terms:** It's like a temperature control. Without regularization, the system can "panic" when it sees very different test data and become uselessly cautious. The regularization keeps it calm — still adaptive, but not overreacting.

### The Complete DS-CP Algorithm

Putting it all together:

1. **Embed** all calibration prompts and the test prompt using the sentence transformer
2. **Train a domain classifier** (XGBoost) on the embeddings to estimate density ratios
3. **Compute weights** ŵ_i = r̂(h(X_i)) for each calibration point
4. **Build the weighted empirical distribution** of calibration scores using these weights and λ = 1
5. **Find the (1 − α)-quantile** of this weighted distribution
6. **Include** answer y in the prediction set if S(x, y) ≤ this quantile

### The Theory: Coverage Guarantees (Theorems 1 and 2)

**Theorem 1 (Lower Bound on Coverage):** Under the condition λ ≥ max_i ŵ_i,

$$P(Y_{n+1} \in \hat{C}_n(X_{n+1})) \geq 1 - \alpha - \sum_{i=1}^{n} \left\| \frac{\hat{w}_i}{\sum_{j=1}^{n} \hat{w}_j + \lambda} \right\|_\infty \text{TV}(S^i, S)$$

Let's break this down into plain English:

- **The left side** is the probability that the prediction set contains the correct answer — the thing we want to be at least 1 − α.

- **1 − α** is the target coverage level (e.g., 0.90 for 90%).

- **The error term** (the big sum) measures how much coverage we might lose due to domain shift. It's a sum over all calibration points, where each term is:
  - The **normalized weight** of calibration point i (how influential it is)
  - Times the **total variation distance** between the original joint score vector S and the "swapped" version S^i (where scores i and n+1 are exchanged)

**What does total variation distance mean here?** TV(S^i, S) measures how different the distribution of scores is when you swap a calibration score with the test score. If the domains produce similar nonconformity scores, this distance is small — even if the prompts themselves look very different!

**Key insight from the theory:** The coverage gap depends on the **score distributions**, not the **prompt distributions**. If the model produces similar uncertainty patterns across domains (even if the questions look different), DS-CP still works well. This is a stronger statement than you might expect.

**In everyday terms:** Even if a physics question looks nothing like a history question (very different prompts), if the model is equally confident/uncertain on both types, the conformal prediction still works.

**Theorem 2 (Upper Bound on Coverage):** Under the same conditions plus distinct scores:

$$P(Y_{n+1} \in \hat{C}_n(X_{n+1})) < 1 - \alpha + \left\| \frac{\lambda}{\sum_{j=1}^{n} \hat{w}_j + \lambda} \right\|_\infty + \text{(same error term)}$$

The upper bound shows the method can be **conservative** — the prediction sets might provide more coverage than needed. The extra term λ/(Σŵ_j + λ) comes from the regularization and represents the "price" of the safety valve δ_∞.

**Special case (no domain shift):** When there's no shift, all weights equal 1, TV distances are 0, and the bounds become:

$$1 - \alpha \leq P(\text{coverage}) < 1 - \alpha + \frac{1}{n+1}$$

This is exactly the standard CP guarantee — the method reduces gracefully to the baseline when there's no shift.

### The Proof Architecture (Theorem 1)

The proof generalizes Theorem 2 in Barber et al. (2023) to data-dependent weights. Here's the conceptual structure:

**Step 1: Introduce a random index K.** Define K as a random variable drawn from the weighted distribution over {1, ..., n+1}, where calibration point i gets weight ŵ_i and the test point gets weight λ. Think of K as "randomly picking a data point according to how important it is."

**Step 2: Connect miscoverage to K.** Show that if Y_{n+1} is NOT in the prediction set, then K must land in a "bad" set F(S^K) — a set of indices whose scores exceed the quantile threshold.

**Step 3: Bound the probability of K landing in F.** By construction, the weighted probability of the bad set is at most α. The complication is that F depends on the score vector, and exchanging scores (going from S to S^i) can change F.

**Step 4: Handle the non-exchangeability.** The key step: bound the difference P(i ∈ F(S^i)) − P(i ∈ F(S)) using total variation distance. Since S^i is S with scores i and n+1 swapped, the TV distance between S^i and S captures how different the calibration and test score distributions are.

**Step 5: Assemble via union.** Sum up all the differences across calibration points, weighted by their normalized weights, to get the total error term.

**The independent scores case:** When scores are independent (a reasonable approximation in practice), the proof simplifies because TV(S^i, S) ≤ 2·TV(S_i, S_{n+1}) — the joint TV distance reduces to just comparing the marginal distributions of the i-th calibration score and the test score. This is much easier to interpret.

## Q2: What Is the Simplest Baseline and How Much Better Is DS-CP?

### Baselines

1. **Standard CP:** Uses uniform weights (ignores domain shift entirely). This is the main comparison.

2. **Weighted CP (Tibshirani et al., 2019):** Uses density ratios but in the raw space — not tested here because it's infeasible for LLM prompts.

3. **Non-exchangeable CP with heuristic weights (Ulmer et al., 2024):** Uses similarity-based weights without the systematic density ratio estimation.

### Results Summary

The experiments use the MMLU benchmark with 17 subjects grouped into 4 categories, creating 272 ordered subject pairs (calibrate on one subject, test on another).

**Key finding 1 — Coverage improvement:** For standard CP at α = 0.10 (targeting 90% coverage):
- Standard CP frequently under-covers — for every model tested, there are subject pairs where coverage drops well below 90%. For several models, even the MEDIAN coverage is below 90%.
- DS-CP consistently lifts coverage: median coverage is higher across all 16 models, and the lower tail (severe under-coverage cases) is substantially reduced.

**Key finding 2 — Set size remains reasonable:** DS-CP produces modestly larger prediction sets than standard CP. This is expected — to fix under-coverage, you need to include more answers. But the increase is moderate, not catastrophic.

**Key finding 3 — DS-CP is adaptive, not just inflating:** The most compelling result is Figure 3 — the scatter plot comparing coverage of DS-CP vs standard CP for each subject pair:
- When standard CP already achieves 90% coverage (blue points), DS-CP makes minimal changes — it doesn't unnecessarily inflate the sets.
- When standard CP under-covers (orange points), DS-CP provides the largest improvements — the worse the under-coverage, the more DS-CP helps.

This shows DS-CP is genuinely detecting and correcting for domain shift, not just blindly making everything bigger.

**Across models:** The results hold across 16 diverse LLMs from 9 model families (Llama-2, Mistral, Falcon, MPT, Gemma, Qwen, Yi, DeepSeek, InternLM), spanning 1.8B to 72B parameters. This breadth is impressive and suggests the method is model-agnostic.

### Quantifying the Improvement

While the paper doesn't give a single summary number, the violin plots (Figure 1) show that:
- Standard CP's coverage distribution extends far below 0.90, with long downward tails reaching 0.50 or lower for some models
- DS-CP's coverage distribution is much more concentrated around 0.90, with the lower tail pulled up significantly

The APS score variant (Appendix) shows similar patterns, confirming robustness to the choice of nonconformity score.

## Q3: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: Dependence on Embedding Quality

DS-CP's effectiveness hinges on the sentence transformer capturing meaningful semantic similarity between domains. If the embedding model doesn't capture the relevant aspects of domain shift (e.g., if two domains differ in ways the embeddings don't reflect), the density ratio estimates will be wrong and the reweighting won't help.

The paper uses a general-purpose sentence transformer (all-MiniLM-L6-v2), which works for the MMLU benchmark. But for specialized domains (medical, legal), a domain-specific embedding model might be needed.

### Limitation 2: Only Multiple-Choice QA

The experiments are exclusively on multiple-choice question answering (6 answer options). This is the simplest setting for conformal prediction because the output space is small and finite. Extending to **open-ended text generation** — where there are infinitely many possible outputs — is a fundamentally harder problem.

The paper acknowledges this in its Discussion section and frames it as future work. This is a significant limitation for your DS-SGen project, which targets open-ended generation.

### Limitation 3: The Regularization Parameter λ

The paper defaults to λ = 1 and acknowledges that more principled tuning strategies remain open. The choice of λ trades off between:
- **Large λ:** More conservative (bigger sets), more robust to severe shift
- **Small λ:** More adaptive (tighter sets), but might under-cover if shift is large

There's no automatic way to pick the "right" λ based on the data.

### Limitation 4: Not a Hard Guarantee

Unlike the SGen paper's PAC guarantee (which gives a probability bound that holds with at most δ failure probability), DS-CP's coverage guarantee is approximate — it's 1 − α *minus* an error term that depends on total variation distances. These TV distances are not directly computable from data (you'd need to know the true score distributions), so you can't verify whether the guarantee is actually met.

### Limitation 5: No Comparison with Other Domain Adaptation Methods

The paper only compares DS-CP against standard CP. It doesn't compare against:
- Other non-exchangeable CP methods with different weight choices
- Methods that use model fine-tuning or adaptation
- Ensemble methods or other UQ approaches under domain shift

This makes it hard to assess whether the improvement comes specifically from the density-ratio-based reweighting or whether simpler approaches would also work.

## Q4: Does This Insight Apply to Other Problems?

### Connection 1: Your DS-SGen Research Project

This is the most important connection. The DS-CP paper provides:

**What you can borrow:**
- The **embedding + density ratio estimation** pipeline is directly applicable. You could use the same approach (sentence transformer → XGBoost classifier → density ratio) to estimate importance weights for your DS-SGen framework.
- The **regularization trick** (replacing the test-point weight with λ) solves the degeneracy problem you would also face.
- The **theoretical framework** (generalizing Barber et al. to data-dependent weights) provides a template for your own theoretical analysis.

**Where your work differs and adds value:**
- DS-CP works with **prediction sets** for multiple-choice QA. Your DS-SGen works with **selective generation** for open-ended QA — a fundamentally different and harder setting.
- DS-CP uses the entailment-free LAC/APS nonconformity scores. Your DS-SGen would use the **entailment-based** correctness metric from the SGen paper — a more semantically meaningful notion of "correct."
- DS-CP's guarantee is approximate (1 − α minus error). Your DS-SGen aims for a **PAC guarantee** (P{FDR-E ≤ ε} ≥ 1 − δ) — a formally stronger type of guarantee.
- DS-CP doesn't do selective prediction (the model always gives a prediction set). Your DS-SGen adds the ability to **abstain** ("I don't know"), which is more useful in practice.

### Connection 2: Domain-Adaptive Conformal Prediction More Broadly

The embedding + density ratio + regularization recipe could apply to any conformal prediction problem where inputs are high-dimensional and unstructured: image classification under distribution shift, speech recognition across accents, etc.

### Connection 3: Active Learning and Data Collection

The density ratio estimates tell you which calibration points are most "relevant" to the new domain. This could guide active learning: collect new calibration labels for test-like inputs to improve coverage most efficiently.

## Q5: Caveats and Takeaways

### Strengths
- Clean, practical framework that actually works across 16 models
- The adaptive behavior (Figure 3) is convincing — DS-CP helps where help is needed
- The theoretical framework properly generalizes existing work
- Comprehensive experiments (16 models, 272 domain pairs, two score functions)
- Honest about limitations

### Weaknesses
- Not peer-reviewed (arXiv only)
- Only tested on multiple-choice QA (not open-ended generation)
- No code release for the DS-CP method itself
- The theory doesn't provide computable bounds (TV distances are unknowable)
- No comparison with other domain adaptation approaches
- Density ratio estimation quality isn't evaluated independently
- The XGBoost hyperparameters are set without tuning — no sensitivity analysis

### Devil's Advocate — Three Weakest Points

1. **The MMLU benchmark may understate the problem.** MMLU subjects share a common format (multiple-choice with 6 options) and were designed for the same purpose. Real-world domain shift (e.g., general QA → medical QA) might involve much more dramatic shifts where the density ratio estimates break down entirely.

2. **No ablation on the embedding model.** What if you used a different (better or worse) embedding model? The paper uses one fixed model but doesn't study sensitivity. If DS-CP's performance depends critically on embedding quality, this is a vulnerability.

3. **The coverage improvement might come partly from set size inflation.** While Figure 3 shows adaptivity, the paper doesn't disentangle "genuinely better coverage through smarter weighting" from "slightly bigger sets that happen to catch the right answer." A more rigorous comparison would fix set size and compare coverage, or fix coverage and compare set size.

---

# PASS 3: THE SWAMP — Deep Dive into the Proofs and Machinery

## Proof Architecture Overview

The paper has a relatively lean theoretical contribution — the main results (Theorems 1 and 2) are generalizations of Theorem 2 in Barber et al. (2023) to handle data-dependent weights. The novelty is more in the *method* (how to choose weights) than in the *proof technique* (which follows the existing blueprint closely).

### Detailed Walkthrough of Theorem 1's Proof

Let me walk through every step in the proof (Appendix A.1), explaining the logic like a story.

**The setup:** We have n calibration pairs (X₁, Y₁), ..., (Xₙ, Yₙ) and a test pair (X_{n+1}, Y_{n+1}). We've computed nonconformity scores S₁, ..., Sₙ, S_{n+1} and weights ŵ₁, ..., ŵₙ. We have a regularization parameter λ.

**Step 1: Introduce a random index K.**

Define a random variable K ∈ {1, ..., n+1} whose distribution, *conditioned on all the data*, is:

- P(K = i | all data) = ŵᵢ / (Σⱼŵⱼ + λ) for i = 1, ..., n
- P(K = n+1 | all data) = λ / (Σⱼŵⱼ + λ)

Think of K as "randomly picking a data point, where points with higher weights are more likely to be chosen."

**Why this random index?** This is the classic proof trick from exchangeable conformal prediction, adapted to the weighted setting. The idea is to show that the test point Y_{n+1} is NOT in the prediction set only if a certain "bad event" happens for K, and then bound the probability of that bad event.

**Step 2: Connect non-coverage to K.**

By definition, Y_{n+1} ∉ Ĉₙ(X_{n+1}) means:

$$S_{n+1} > \text{Quantile}\left(1 - \alpha; \sum_{i=1}^{n} \frac{\hat{w}_i}{\sum_j \hat{w}_j + \lambda} \delta_{S_i} + \frac{\lambda}{\sum_j \hat{w}_j + \lambda} \delta_\infty \right)$$

Because λ ≥ max_i ŵᵢ (the assumption in the theorem), the mass on δ_∞ is at least as large as any individual weight. This means the quantile computed with δ_∞ is at least as large as the quantile computed with δ_{S_{n+1}} (since ∞ ≥ S_{n+1}). Therefore:

If Y_{n+1} ∉ Ĉₙ(X_{n+1}), then S_{n+1} is also above the quantile computed from the empirical distribution that uses S^K (the scores with K-th and (n+1)-th entries swapped) instead of the δ_∞ term.

**In simpler terms:** This step uses the assumption λ ≥ max ŵᵢ to replace the "infinity mass" with the actual test score. This is a conservative replacement (making the quantile smaller or equal), so if the test score exceeds the bigger quantile (with δ_∞), it certainly exceeds the smaller one (with S_{n+1}).

**Step 3: Define the "bad set" function F.**

Define F(s) for any score vector s = (s₁, ..., s_{n+1}) as:

$$F(s) = \{i \in \{1, ..., n+1\} : s_i > \text{Quantile}(\text{weighted distribution of other scores})\}$$

This is the set of indices whose scores exceed the quantile threshold. By the definition of a quantile, the total weighted probability of indices in F(s) is at most α.

In math: Σᵢ∈F(s) (weight of i) ≤ α.

**Step 4: Show non-coverage implies K ∈ F(S^K).**

From Step 2, if Y_{n+1} is not covered, then the K-th entry of the swapped score vector S^K exceeds the quantile, which means K is in the bad set F(S^K).

**Step 5: Compute P(K ∈ F(S^K)).**

This is the core computation. By the law of total expectation:

$$P(K \in F(S^K)) = \sum_{i=1}^{n+1} P(K = i \text{ and } i \in F(S^i))$$

Expand each term using the conditional distribution of K:

$$= \sum_{i=1}^{n+1} E\left[\text{weight}_i \cdot \mathbf{1}(i \in F(S^i))\right]$$

Now here's the key trick. We want to relate this to F(S) (the bad set for the *original* scores, not the swapped ones). Split each term:

$$= \sum_{i=1}^{n+1} E\left[\text{weight}_i \cdot \mathbf{1}(i \in F(S))\right] + \sum_{i=1}^{n+1} E\left[\text{weight}_i \cdot (\mathbf{1}(i \in F(S^i)) - \mathbf{1}(i \in F(S)))\right]$$

The first sum equals Σᵢ∈F(S) weight_i ≤ α (by the quantile property from Step 3, since S^{n+1} = S).

The second sum captures the "error" from non-exchangeability — the fact that swapping scores i and n+1 can change whether i is in the bad set. Each term is bounded by:

$$\text{weight}_i \cdot |P(i \in F(S^i)) - P(i \in F(S))| \leq \text{weight}_i \cdot \text{TV}(S^i, S)$$

This last step uses the definition of total variation distance: the maximum difference in probability of any event between two distributions is their TV distance.

**Step 6: Assemble the bound.**

$$P(Y_{n+1} \notin \hat{C}_n(X_{n+1})) \leq P(K \in F(S^K)) \leq \alpha + \sum_{i=1}^{n} \left\|\frac{\hat{w}_i}{\sum_j \hat{w}_j + \lambda}\right\|_\infty \text{TV}(S^i, S)$$

Taking the complement:

$$P(Y_{n+1} \in \hat{C}_n(X_{n+1})) \geq 1 - \alpha - \sum_{i=1}^{n} \left\|\frac{\hat{w}_i}{\sum_j \hat{w}_j + \lambda}\right\|_\infty \text{TV}(S^i, S) \quad \square$$

**The independent scores simplification:** When scores are independent, swapping S_i and S_{n+1} only changes two coordinates of the joint vector, so:

$$\text{TV}(S^i, S) \leq 2 \cdot \text{TV}(S_i, S_{n+1})$$

This follows because the TV distance of the joint is bounded by the TV distance of the changed marginals (a standard result in probability theory). This gives the cleaner second statement in Theorem 1.

### Understanding the Error Term — When Is It Small?

The error term Σᵢ (weight_i · TV(S^i, S)) is small when:

1. **The weights are spread out evenly** — no single calibration point dominates. If one weight is huge, it contributes a large term to the sum.

2. **The score distributions are similar across domains** — TV(Sᵢ, S_{n+1}) is small. This happens when the model's uncertainty is well-calibrated across domains, even if the inputs look different.

This gives a nice practical message: **DS-CP works best when the model transfers well across domains** (produces similar uncertainty patterns), which is exactly the setting where you'd want conformal prediction to work.

### Proof of Theorem 2 (Upper Bound)

The upper bound proof follows a similar structure but is slightly trickier because it needs to account for the mass on δ_∞. The extra term λ/(Σŵⱼ + λ) comes from the fact that the test point always has weight λ in the empirical distribution, and this λ can inflate the quantile beyond what's needed for exact coverage. The paper states it follows the same pattern as Barber et al. (2023, Theorem 3).

### The Assumption λ ≥ max_i ŵ_i

This assumption is needed for Step 2 of the proof (the conservative replacement of δ_∞). In practice, this may not hold — some calibration points might have density ratios larger than λ = 1 (meaning they look more like the new domain than average). The paper doesn't discuss what happens when this assumption is violated.

For the default λ = 1, this assumption holds when all density ratios are at most 1, which means the new domain has *lower* density everywhere compared to the old domain — a restrictive condition. In practice, some calibration points will likely have ŵ_i > 1.

This is a gap between the theory and practice that future work could address, perhaps by setting λ = max_i ŵ_i adaptively.

## Techniques You Can Borrow for Your DS-SGen Research

### Technique 1: The Embedding + Classifier → Density Ratio Pipeline

**What it is:** Map prompts to embeddings, train a domain classifier, convert classifier probabilities to density ratios.

**How to use it in DS-SGen:** The SGen framework needs i.i.d. samples for its binomial tail bounds. Under domain shift, you need importance weights. You can use exactly this pipeline to estimate importance weights w(x) for each calibration sample, then use weighted binomial bounds (or weighted conformal prediction) in the SGen-Semi framework.

**Specific implementation:**
1. Embed all calibration QA pairs and test QA pairs using a sentence transformer
2. Train XGBoost (or logistic regression) to classify old vs. new domain
3. Compute weights ŵᵢ = r̂(h(Xᵢ))
4. Replace the uniform binomial tail bounds in SGen with weighted versions

### Technique 2: The Regularization of Test-Point Weights

**What it is:** Replace the test-point's density ratio with a fixed constant to prevent degenerate prediction sets.

**How to use it in DS-SGen:** In your reweighted binomial bounds, extreme weights can make the effective sample size very small, leading to vacuous bounds. You could apply a similar regularization — cap the weights or normalize them — to maintain informative bounds.

### Technique 3: The "Score Distribution Similarity" Insight

**What it is:** The key theoretical insight that coverage depends on score distribution similarity, not prompt distribution similarity.

**How to use it in DS-SGen:** This suggests a diagnostic: before deploying DS-SGen, check whether the *confidence score distributions* (not the prompts themselves) are similar across domains. If they are, your reweighted approach will work well. If they're very different, you might need domain-specific calibration data.

### Technique 4: The Experimental Design

**What it is:** Using all n×(n-1) ordered domain pairs for systematic evaluation.

**How to use it in DS-SGen:** When evaluating your DS-SGen method, you could use the same MMLU setup (or a similar multi-domain dataset) to create hundreds of domain shift pairs, giving you statistically powerful evidence of improvement.

---

# KEY CONCEPTS GLOSSARY (Grade 12 Level)

| Concept | Simple Explanation |
|---------|-------------------|
| **Conformal prediction** | A method that builds "prediction sets" (a set of possible answers) with a mathematical guarantee that the true answer is included with high probability |
| **Domain shift** | When the test data comes from a different "world" than the training/calibration data — like studying for a math test but taking a history exam |
| **Exchangeability** | A statistical assumption that the order of data doesn't matter — like drawing balls from a well-mixed bag. Needed for standard conformal prediction. |
| **Nonconformity score** | A number measuring how "surprised" the model is by the true answer — high score means the model didn't expect this answer |
| **Density ratio** | A number measuring how much more (or less) likely a data point is under the new distribution compared to the old one |
| **Sentence transformer / embedding** | A neural network that converts text into a fixed-length vector of numbers that captures the text's meaning |
| **XGBoost** | A popular machine learning algorithm used here to train a classifier that distinguishes old-domain from new-domain embeddings |
| **Regularization** | A technique to prevent a model from being too extreme — here, it prevents the prediction sets from becoming uselessly large |
| **Total variation distance** | A measure of how different two probability distributions are — ranges from 0 (identical) to 1 (completely different) |
| **Coverage** | The probability that the prediction set contains the correct answer — we want this to be at least 1 − α |
| **Set size** | How many answers are in the prediction set — smaller is better (more informative), but must maintain coverage |
| **MMLU** | A benchmark with thousands of multiple-choice questions across 17 subjects, used to test LLM knowledge |
| **Covariate shift** | A type of domain shift where the inputs change but the relationship between inputs and outputs stays the same |
| **LAC score** | "Least Ambiguous Classifier" — a simple nonconformity score: S(X,Y) = 1 − P(Y|X) |
| **APS score** | "Adaptive Prediction Sets" — a nonconformity score that considers the ranking of the correct answer among alternatives |

---

# CONNECTION TO YOUR DS-SGen RESEARCH PROJECT

## What DS-CP Does
- Adapts conformal prediction to handle domain shift for LLMs
- Uses embedding-based density ratio estimation
- Provides approximate coverage guarantees (1 − α minus error)
- Tested on multiple-choice QA (MMLU)

## What DS-CP DOESN'T Handle (Your Research Gap)
- **Open-ended generation:** DS-CP only works with finite output spaces (multiple choice). Your DS-SGen targets open-ended text generation with infinite output spaces.
- **Selective prediction / abstention:** DS-CP always gives a prediction set. It can't say "I don't know." Your DS-SGen adds the ability to abstain.
- **Entailment-based correctness:** DS-CP uses standard exact-match correctness (is the answer A, B, C, D, E, or F?). Your DS-SGen uses textual entailment, which is semantically richer.
- **PAC guarantees:** DS-CP's coverage guarantee is approximate with unknowable error terms. Your DS-SGen aims for proper PAC guarantees (P{FDR-E ≤ ε} ≥ 1 − δ) with computable bounds.
- **Semi-supervised learning:** DS-CP doesn't use unlabeled data. Your DS-SGen inherits the semi-supervised pseudo-labeling from SGen-Semi.

## What Your DS-SGen Uniquely Proposes
Your DS-SGen sits at the intersection of SGen and DS-CP:
- **From SGen:** The selective generation framework, entailment-based correctness, semi-supervised pseudo-labeling, PAC guarantees, neuro-selection functions
- **From DS-CP:** The embedding-based density ratio estimation, the reweighting strategy, the regularization to prevent degeneracy
- **Your novel contribution:** Combining these into a unified framework that provides PAC-guaranteed selective generation under domain shift, using weighted binomial bounds

## Specific Technical Bridges

1. **DS-CP's density ratio estimation → Your importance weights:** Use the exact same embedding + XGBoost pipeline to estimate importance weights for your calibration data.

2. **DS-CP's regularization → Your weight clipping:** Apply similar regularization to prevent extreme weights from making your PAC bounds vacuous.

3. **DS-CP's Theorem 1 → Your coverage analysis:** The TV-distance-based error term gives you a template for analyzing how domain shift affects your weighted binomial bounds.

4. **DS-CP's MMLU evaluation → Your experimental baseline:** Include DS-CP (or its variants) as a baseline in your experiments to show that DS-SGen provides strictly stronger guarantees.

5. **DS-CP's limitation (only multiple-choice) → Your advantage:** Highlight that DS-SGen extends domain-shift-aware CP to the much harder open-ended generation setting with entailment-based evaluation.

---

# SUMMARY: ONE-PAGE CHEAT SHEET

**Problem:** Standard conformal prediction fails under domain shift — prediction sets are unreliable when test data differs from calibration data.

**Key Innovation:** Embed prompts → estimate density ratios in embedding space → regularize to prevent degeneracy → reweight calibration scores for domain-adaptive prediction sets.

**Algorithm (DS-CP):**
1. Embed prompts using sentence transformer (all-MiniLM-L6-v2)
2. Train domain classifier (XGBoost) on embeddings
3. Convert classifier probabilities to density ratio weights
4. Set regularized test-point weight λ = 1
5. Construct weighted empirical distribution of calibration scores
6. Build prediction set using weighted quantile

**Theoretical Guarantee:** Coverage ≥ 1 − α − (error depending on score distribution similarity across domains)

**Key Result:** DS-CP consistently improves coverage over standard CP across 16 LLMs and 272 domain pairs on MMLU, with only modest increases in prediction set size. Improvements are largest where standard CP fails most.

**Main Limitations:**
- Only tested on multiple-choice QA (not open-ended generation)
- Not peer-reviewed
- Theory gives approximate bounds with unknowable error terms
- No comparison with other domain adaptation methods
- Regularization parameter λ chosen heuristically

**For Your DS-SGen Project:** Borrow the density ratio estimation pipeline, but combine it with SGen's stronger guarantees (PAC bounds), entailment-based correctness, and selective prediction capability to handle the much harder open-ended generation setting under domain shift.