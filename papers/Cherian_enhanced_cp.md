# Large Language Model Validity via Enhanced Conformal Prediction Methods — Complete Paper Analysis

**Paper:** Large Language Model Validity via Enhanced Conformal Prediction Methods
**Authors:** John J. Cherian, Isaac Gibbs, Emmanuel J. Candès (Stanford University, Department of Statistics)
**Venue:** NeurIPS 2024 (arXiv:2406.09714v2)
**Type:** Breakthrough / Theory paper with empirical validation
**Peer-Reviewed:** Yes (NeurIPS 2024 — one of the top 3 ML conferences)
**Reading Purpose:** Deep understanding — all three passes. This paper directly extends the Conformal Factuality paper (Mohri & Hashimoto, 2024) that is foundational to your DS-SGen research project.

---

# PHASE -1: PAPER CLASSIFICATION

This paper is a **Breakthrough / Theory** paper. It introduces two genuinely new methods — **conditional boosting** and **level-adaptive conformal prediction** — backed by formal mathematical guarantees (Theorems 3.1, 3.2, and Proposition 3.1) and demonstrates they work in practice on real LLM outputs (medical QA and biography generation).

What makes this a "breakthrough" rather than just "incremental" is that it solves two fundamental problems that made the prior work (Mohri & Hashimoto's Conformal Factuality) impractical:

1. The prior method's guarantee only held "on average" and could be terrible for specific types of questions
2. The prior method removed too many correct claims, making the filtered output useless

This paper fixes BOTH problems simultaneously, which was previously believed to be impossible (they seem like opposing goals — stronger guarantees typically mean removing MORE claims, not fewer).

The paper sits in the same research landscape as your other three papers (SGen, DS-CP, Conformal Factuality), providing the most advanced version of the "conformal factuality" approach.

---

# PHASE 0: PRE-READING CONTEXT

## Authors — Who Wrote This?

- **Emmanuel J. Candès:** This is the BIG name. Candès is one of the most influential statisticians alive today. He's a professor at Stanford in both Statistics and Mathematics, a MacArthur Fellow ("genius grant"), a member of the National Academy of Sciences, and he co-invented compressed sensing (a breakthrough in signal processing). In recent years, he has been a driving force behind conformal prediction research. Having Candès as senior author is the strongest possible credibility signal.

- **Isaac Gibbs:** PhD student at Stanford Statistics, working with Candès. Already has highly cited work on conditional conformal prediction (the Gibbs et al. 2023 paper that THIS paper builds on). This is the person who developed the core conditional conformal framework being extended here.

- **John J. Cherian:** PhD student at Stanford Statistics, supported by the Hertz Foundation (a highly prestigious fellowship). First author on this paper.

**Bottom line:** This team is arguably the world's best for conformal prediction theory. The methods they develop here carry enormous credibility.

## Venue — Where Was This Published?

NeurIPS 2024 — one of the top 3 machine learning conferences alongside ICML and ICLR. This means the paper passed rigorous peer review. The fact that Stanford's statistics powerhouse published this at NeurIPS (rather than a statistics journal) signals they're targeting the ML/NLP community.

## Code Released?

Yes — at two GitHub repositories:
- `github.com/jjcherian/conformal-safety` (experiments and data)
- `github.com/jjcherian/conditional-conformal` (Python package, installable from PyPI)

This is an excellent reproducibility signal.

## Why This Matters for Your Research

This paper is the "next generation" of the Conformal Factuality paper (Mohri & Hashimoto) that you've already analyzed. While Mohri & Hashimoto introduced the idea of filtering LLM claims using conformal prediction, THIS paper makes it:
1. **Conditionally valid** — the guarantee holds for specific GROUPS of prompts, not just on average
2. **More practical** — it keeps more of the original output while still being safe
3. **Aware of difficulty** — it can issue weaker guarantees for harder prompts to avoid destroying useful content

For your DS-SGen project, this paper provides:
- A more sophisticated baseline to compare against
- Technical tools (conditional conformal prediction, score boosting) you could adapt
- A covariate shift interpretation (Appendix A.2) that directly connects to your domain-shift work

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## The Real-World Problem (No Math Yet)

Let's start with a story. Imagine you're a doctor using an AI assistant for medical questions. A patient asks: "How often do I need a shingles vaccine?"

The AI (GPT-3.5-Turbo) responds with several claims:

1. "The shingles vaccine is recommended for adults aged 50 and older" ✓ TRUE
2. "The vaccine is given in two doses" ✓ TRUE
3. "The second dose is administered 2 to 6 months after the first dose" ✓ TRUE
4. "Individuals should receive the shingles vaccine once in their lifetime" ✗ FALSE (you need it periodically)
5. "Consult with a healthcare provider for personalized recommendations" ✓ TRUE

The previous method (Conformal Factuality by Mohri & Hashimoto) tries to filter out the wrong claims. It gives you a guarantee: "After filtering, the remaining claims are all correct with at least 90% probability."

But here's the problem: **to achieve that 90% guarantee, the method has to remove almost ALL the claims** — not just claim 4 (the wrong one), but also claims 2, 3, and 5 (which are correct!). Why? Because the scoring function that measures "how confident is the model?" is imperfect. It can't perfectly distinguish true from false claims. To be safe enough to guarantee 90% correctness, it has to be extremely aggressive with its filtering.

After filtering, you're left with: "The shingles vaccine is recommended for adults aged 50 and older." That's it. One claim out of five. The doctor has learned almost nothing useful.

**This paper's solution:** Instead of always demanding a 90% guarantee, let the system say: "For THIS particular question, I'm 63% confident the remaining claims are all correct." At 63% confidence, the system can keep 4 out of 5 claims — everything except the false one. The doctor gets a useful answer, AND an honest probability of correctness.

The key insight: **a 63% guarantee with a useful answer is more valuable than a 90% guarantee with a useless answer.** But the 63% must be HONEST — it must actually mean 63%. If the system says "63% confident" but is really only 40% correct, that's dangerous. This paper proves that the issued probabilities are genuinely calibrated.

And on top of that, the paper also finds a way to improve the scoring function itself (the thing that decides which claims to remove), so that even at a fixed guarantee level, fewer correct claims get removed.

## Q1: What Is the Problem Being Solved?

**In the simplest words:** The previous method for filtering false claims from AI outputs (Conformal Factuality) has two problems: (1) its guarantee only works "on average" — for some topics it might fail badly, and (2) it removes too many correct claims, making the filtered output useless. This paper fixes both problems.

**More precisely:** The paper addresses two specific limitations of marginal conformal factuality for LLMs:

### Problem 1: The guarantee is only "marginal" (it works on average, but not for every topic)

Think about this analogy. Imagine a weather forecaster who is 90% accurate overall. But they're 99% accurate for sunny predictions and only 50% accurate for storm predictions. Their OVERALL accuracy is 90%, but when you really need them (during potential storms), they're unreliable.

The same thing happens with conformal factuality. The method calibrates its threshold using a mix of different types of questions. For common, well-known topics (like "Who was George Washington?"), the model rarely halluccinates, so the filtering works great. For rare, obscure topics (like "Who was Zamfir Ralli-Arbore?"), the model hallucinates heavily, and the filtering fails to catch everything.

The average performance is fine, but the worst-case performance (which tends to happen on exactly the topics where safety matters most) can be terrible.

### Problem 2: The filtering removes too many correct claims

Because the scoring functions (which predict whether a claim is true or false) are imperfect — they're only "weakly correlated" with the actual truth — the filtering threshold has to be set very conservatively. This means many correct claims get removed along with the false ones. The paper reports that the previous method often retains only 24% of claims on medical QA. That's removing 76% of the content — most of which was correct!

**In one sentence:** "This paper studies how to improve the conditional validity and practical utility of conformal prediction methods for filtering false claims from LLM outputs, by developing conditional boosting (to learn better scoring functions) and level-adaptive conformal prediction (to adapt the guarantee strength to each prompt)."

## Q2: Why Is This Problem Hard and Interesting?

There are three key difficulties:

### Difficulty 1: Conditional guarantees and utility seem fundamentally at odds

In the conformal prediction literature, there's a well-known impossibility result (by Barber, Candès, Ramdas, and Tibshirani — note that Candès, a co-author of THIS paper, is also behind the impossibility result!). It says: **you cannot achieve exact conditional coverage without making your prediction sets uselessly large.**

Think of it this way: if you want to guarantee that your forecast is correct for EVERY possible type of question (not just on average), you'd need to be so cautious that you'd basically have to say "I don't know" every time.

So the authors face a fundamental theoretical barrier: making the guarantee hold conditionally (for specific groups of prompts) seems to require removing even MORE claims, not fewer. How do you improve conditional validity AND keep more claims at the same time?

### Difficulty 2: Optimizing the scoring function through conformal prediction is non-trivial

The scoring function decides which claims to keep and which to remove. A better scoring function → fewer correct claims removed → more useful output. So you want to optimize the scoring function.

But the conformal prediction algorithm sits between the scoring function and the output. You compute scores → run conformal prediction to find a threshold → filter claims. If you want to optimize the scores, you need to "backpropagate" through the conformal prediction algorithm (like in neural network training). The problem is that conformal prediction involves computing quantiles (the value below which a certain fraction of data falls), and quantiles are NOT smooth functions — they can jump discontinuously. This makes standard calculus-based optimization (gradient descent) impossible without careful mathematical work.

### Difficulty 3: Adapting the guarantee level requires a new calibration theory

Standard conformal prediction issues the SAME guarantee (like 90% coverage) for every test point. Adapting the level per-prompt (like saying "90% for this easy question, 63% for that hard question") requires a whole new theoretical framework. The issued levels must be provably honest — the 63% must really mean 63%. Developing the theory for this is nontrivial.

**In one sentence:** "This is nontrivial because exact conditional coverage is known to be impossible without vacuous sets, optimizing scoring functions requires differentiating through a non-smooth conformal quantile, and adapting the guarantee level per-prompt requires a new calibration framework."

## Q3: What Is the Main Claim?

The paper makes three main claims:

### Claim 1: Conditional Boosting improves claim retention by 62%

By learning an optimal linear combination of four candidate scoring functions through differentiation of the conditional conformal algorithm, the method increases the fraction of retained claims from 24% to 39% (on medical QA), while maintaining the same validity guarantee.

### Claim 2: Level-Adaptive Conformal Prediction produces calibrated, prompt-specific guarantees

Instead of issuing a fixed 90% guarantee for all prompts (which forces massive claim removal), the method adapts the guarantee per-prompt to ensure at least 70% of claims are retained. The guarantee levels range from 50% to 85%, and they are provably well-calibrated:

$$P(\text{output is correct} \mid \text{issued probability} \in [0.7, 0.8], \text{prompt} \in G) = E[\text{issued probability} \mid \text{issued probability} \in [0.7, 0.8], \text{prompt} \in G]$$

**In plain English:** Among similar prompts where the system claims a 70-80% probability of correctness, the output actually IS correct 70-80% of the time.

### Claim 3: Combining both methods retains most claims while issuing non-trivial guarantees

When you combine conditional boosting with level-adaptive CP, you retain ~70% of claims while still issuing factuality guarantees that range from 50% to 85%.

### The Formal Guarantee (Theorem 3.2)

For any finite-dimensional linear function class F and any level function α(·):

$$E[f(X_{n+1}) \cdot (\mathbf{1}\{L(\hat{F}(C_{n+1}; \hat{\tau}(X_{n+1})), W_{n+1}) \leq \lambda\} - (1 - \alpha(X_{n+1})))] = 0 \quad \forall f \in F$$

Don't worry if this looks scary — I'll explain every piece of this in Pass 2. The key point is that this equation guarantees the issued probabilities are well-calibrated.

**In one sentence:** "They show that conditional boosting + level-adaptive CP achieves well-calibrated, prompt-specific factuality guarantees for LLM outputs while retaining ~70% of the original claims, dramatically improving over the fixed-level method's ~24% retention on medical QA."

---

# PASS 2: THE SCUBA DIVE — How Does It Work?

## Q1: What Was the Main Technical Hurdle Before This Paper?

### The State of the Art Before This Paper

The direct predecessor is Mohri & Hashimoto's "Language Models with Conformal Factuality Guarantees" (2024), which you've already analyzed. Let me recap what they did and where they fell short.

**What Mohri & Hashimoto Did:**

They proposed a clever idea: break an LLM's output into sub-claims, score each sub-claim by confidence, and use conformal prediction to find a threshold. Remove all sub-claims with confidence below the threshold. The guarantee: with probability at least 1 − α, all remaining claims are correct.

The conformal prediction machinery is simple (standard split conformal) and the guarantee is clean:

$$P(\text{all retained claims are correct}) \geq 1 - \alpha$$

**Where They Fell Short — Problem 1: Marginal-only guarantee**

The guarantee holds "on average" over a random test prompt. But the model hallucinates much more on rare topics (obscure people, specialized medical questions) than on common topics. The paper demonstrates this concretely: for Wikipedia biographies, the probability of correctness after filtering varies dramatically based on how famous the person is. For very famous people (millions of Wikipedia views), the filtering works great. For obscure people (few hundred views), the filtering often fails.

This is a serious safety concern: the guarantee is weakest precisely where it matters most.

**Where They Fell Short — Problem 2: Too much claim removal**

The scoring functions (frequency scoring, self-evaluation, etc.) are only weakly correlated with actual correctness. This means the conformal threshold must be set high to ensure safety, which removes many correct claims along with incorrect ones.

Think of it like airport security. If your metal detector beeps on 30% of innocent passengers (false alarms) but also catches 90% of actual threats, then to catch nearly all threats, you need to search EVERYONE who beeps — even though most of them are innocent. You end up "removing" (searching/delaying) a lot of innocent people to catch the few bad ones.

### The Barrier

The fundamental barrier was the trade-off between two competing goals:

1. **Conditional validity** (the guarantee should hold for specific groups of prompts, not just on average) — this generally INCREASES the amount of filtering needed
2. **Utility** (the filtered output should retain enough content to be useful) — this requires DECREASING the amount of filtering

Prior theoretical results (Barber et al., 2020; Vovk, 2012) proved that exact conditional coverage requires vacuous (useless) prediction sets. So it seemed like you had to choose: either a strong guarantee that's useless in practice, or a useful output with a weak guarantee.

### How This Paper Overcomes the Barrier — TWO KEY INSIGHTS

**Insight 1: You don't need EXACT conditional coverage — approximate conditional coverage suffices.**

Instead of demanding exact conditional coverage (which is impossible), the paper targets a relaxed version: coverage that holds for any function in a linear class F. By choosing F to include group indicators (e.g., "medical topic = cardiology" or "Wikipedia page popularity > 100,000 views"), you get coverage that holds for each group. This is called **group-conditional coverage**, and it's achievable!

**Insight 2: Adapting the guarantee level per prompt can INCREASE utility.**

Instead of demanding 90% correctness for every prompt, allow the system to issue weaker guarantees (like 63%) for hard prompts. The weaker guarantee means fewer claims need to be removed, so the output is more useful. As long as the issued guarantee is honest (63% really means 63%), this is a better deal for the user.

These two insights are brought to life through two concrete methods: conditional boosting and level-adaptive CP.

## Q2: The Core Technical Machinery — Explained Step by Step

### Background: What You Need to Know First

Before diving into this paper's methods, let me make sure you understand the building blocks.

#### Building Block 1: What Is a Sub-Claim?

When an LLM writes a paragraph, that paragraph contains multiple individual facts. For example:

> "Abraham Lincoln was born in Kentucky in 1809. He served as the 16th President of the United States."

This contains three sub-claims:
1. "Abraham Lincoln was born in Kentucky" (TRUE)
2. "Abraham Lincoln was born in 1809" (TRUE)
3. "Abraham Lincoln served as the 16th President of the United States" (TRUE)

The paper uses GPT-4o to automatically extract these sub-claims from the model's output.

#### Building Block 2: The Scoring Function p(P, C)

For each sub-claim C generated in response to prompt P, we need a number that measures "how confident is the model that this claim is correct?" This is the scoring function p(P, C).

Higher score → more likely to be correct. The paper considers four scoring functions:

1. **Frequency score:** Generate 5 alternative answers to the same question. For each sub-claim, check how many of the 5 alternatives support it. A claim that appears in 4 out of 5 alternatives gets a score of 0.8. This is the best single scoring function.

2. **Self-evaluation score:** Directly ask the LLM "what's the probability this claim is correct?" and use the number it returns.

3. **Ordinal score:** Earlier claims in the output get higher scores (the model tends to state more confident facts first).

4. **Log-probability score:** Ask the LLM to classify the claim as True (T) or False (F), and use the log-probability of the T token as the score.

#### Building Block 3: The Filtering Function

Given a scoring function p and a threshold τ, the filtered set of claims is:

$$\hat{F}(C; \tau) = \{C_j : p(P, C_j) \geq \tau\}$$

**In plain English:** Keep all claims with confidence score at least τ, remove the rest.

Higher threshold τ → fewer claims kept → more likely that all remaining claims are correct.

#### Building Block 4: The Conformity Score (What Mohri & Hashimoto Defined)

For a prompt-response pair with claims C and ground-truth labels W (where W_j = 1 if claim j is true, W_j = 0 if false), the conformity score is:

$$S(C, W) = \inf\{\tau \mid \hat{F}(C; \tau) \text{ contains no false claims}\}$$

**In plain English:** S is the MINIMUM threshold needed to filter out ALL false claims for this particular example.

Think of it like a difficulty score:
- If the model made no mistakes → S is very low (you don't need to filter much)
- If the model made mistakes but the scoring function ranks them low → S is moderate
- If the model made mistakes AND the scoring function wrongly ranked them high → S is very high (you need an extremely high threshold to catch them)

#### Building Block 5: Split Conformal Prediction (The Previous Method)

Mohri & Hashimoto used standard split conformal prediction:

1. Compute conformity scores S₁, S₂, ..., Sₙ for n calibration examples
2. Find the quantile: q̂_α = the ⌈(1-α)(n+1)⌉/(n+1)-th quantile of these scores
3. For a new prompt, filter claims at threshold q̂_α

The guarantee: P(all retained claims are correct) ≥ 1 − α

**The problem:** This guarantee is MARGINAL — it holds on average over a random test prompt. For specific groups of prompts (rare topics, hard questions), the actual correctness probability can be much lower than 1 − α.

### Method 1: Conditional Conformal Prediction — The Foundation

Before we get to this paper's new methods, let me explain the **conditional conformal framework** from Gibbs et al. (2023), which this paper builds on. This framework is the mathematical foundation for everything that follows.

#### The Core Idea

Instead of using a single global threshold q̂_α for all prompts, use a **prompt-dependent threshold** g(X), where X are features of the prompt/response (like prompt length, topic, response complexity, etc.).

Think of it like this: instead of one speed limit for all roads, different roads get different speed limits based on their characteristics (highway vs. residential, weather conditions, etc.).

#### How It Works — The Pinball Loss

The key mathematical tool is the **pinball loss** (also called the quantile regression loss):

$$\ell_\alpha(r) = (1-\alpha) \cdot [r]_+ + \alpha \cdot [r]_-$$

where [r]₊ = max(r, 0) and [r]₋ = max(-r, 0).

**What does this mean in simple terms?** The pinball loss is a way to find the (1-α)-quantile of data using optimization. If you minimize the average pinball loss, the solution gives you the quantile.

Think of it like a seesaw that's deliberately unbalanced. If you want the 90th percentile (α = 0.1), you make underestimation 9× more expensive than overestimation. This "tips" the optimal solution toward the 90th percentile.

The conditional conformal method finds a function g from a class F (like linear functions of prompt features) that minimizes:

$$g_S = \arg\min_{g \in F} \frac{1}{n+1} \sum_{i=1}^{n} \ell_\alpha(S_i - g(X_i)) + \frac{1}{n+1} \ell_\alpha(S - g(X_{n+1}))$$

**In plain English:** Find the function g(X) that best predicts the (1-α)-quantile of the conformity score S, given the features X of the prompt/response.

Then filter at threshold τ̂(X_{n+1}) = g(X_{n+1}).

#### The Guarantee (Theorem 2.1 from Gibbs et al.)

For any function f in the class F:

$$E[f(X_{n+1}) \cdot (\mathbf{1}\{S_{n+1} \leq g(X_{n+1})\} - (1-\alpha))] = 0$$

**What does this mean in English?** Let me break this down piece by piece:

- **1{S_{n+1} ≤ g(X_{n+1})}** is 1 if the filtering works (all retained claims are correct) and 0 if it fails
- **(1-α)** is the target probability of success
- **f(X_{n+1})** is any function from the class F — this is what makes it CONDITIONAL

If we choose f(X) = 1{X ∈ G} (an indicator for group G), the equation becomes:

$$E[\mathbf{1}\{X_{n+1} \in G\} \cdot (\mathbf{1}\{\text{correct}\} - (1-\alpha))] = 0$$

Dividing by P(X_{n+1} ∈ G), we get:

$$P(\text{correct} \mid X_{n+1} \in G) = 1 - \alpha$$

This says: **the probability of correctness is exactly 1 − α for every group G.** That's the group-conditional guarantee!

For example, if your groups are "common topics" and "rare topics", the method guarantees 1 − α coverage for EACH group separately.

### Method 2: Generalization to Monotone Losses (Theorem 3.1 — Section 3.1)

This is the paper's first theoretical contribution. Gibbs et al. (2023) proved their conditional conformal result for prediction sets (the standard conformal prediction setting). This paper extends it to handle **any monotone loss function**.

#### What Is a Monotone Loss?

A loss function L(F̂(C), W) measures how bad the filtered output is. "Monotone" means: if you include MORE claims, the loss can only go up or stay the same — it can never go down.

Examples of monotone losses:
- **Zero-error loss:** L = 1 if any retained claim is false, L = 0 otherwise. More claims → more chances for error → loss can only increase.
- **Count-error loss:** L = number of false claims in the retained set. More claims → more potential errors → loss can only increase.
- **Fraction-error loss:** L = fraction of retained claims that are false. (This one is NOT monotone — removing a true claim could increase the fraction of false ones.)

Also required: **L(∅, ·) = 0** — if you output nothing (empty set), the loss is 0. This makes sense: an empty output contains no errors.

#### Why This Generalization Matters

The original Conformal Factuality paper only handled the zero-error loss (either all retained claims are true, or at least one is false). But the biography experiments show that requiring ALL claims to be correct can be too strict — the system can only issue very weak guarantees (often below 50%).

With the generalized framework, you can say "at most 3 false claims" instead of "zero false claims." This is more practical for tasks where a small number of errors is tolerable (like biography generation where the user might cross-check key facts).

#### The Guarantee (Theorem 3.1)

Under the same conditions as before, the method satisfies:

$$E[f(X_{n+1}) \cdot (\mathbf{1}\{L(\hat{F}(C_{n+1}; \hat{\tau}(X_{n+1})), W_{n+1}) \leq \lambda\} - (1-\alpha))] = 0 \quad \forall f \in F$$

This says: the probability that the loss is at most λ (the user-specified tolerance) equals 1 − α, not just marginally but also for every group defined by F.

The proof (in Appendix A) is elegant: it reduces to the original Gibbs et al. result by defining the conformity score as:

$$S(C, W) = \inf\{\tau \mid L(F(C; \tau), W) \leq \lambda\}$$

Because L is monotone, S is well-defined (higher threshold → fewer claims → lower loss → eventually below λ). Because L(∅, ·) = 0, there always exists a safe threshold (in the worst case, remove everything).

### Method 3: Level-Adaptive Conformal Prediction (Section 3.2) — THE FIRST BIG CONTRIBUTION

This is where things get really interesting.

#### The Idea

Instead of using a fixed α for all prompts, allow α to depend on the prompt:

- For an easy medical question where the model is rarely wrong → use α = 0.1 (90% guarantee, filter little)
- For a hard medical question where the model often hallucinates → use α = 0.4 (60% guarantee, but keep most claims)

The user doesn't pick α — the method picks it automatically, with the goal of retaining at least 70% of the original claims.

#### How It Works

**Step 1: Learn α(·)** — Split the data. On the first half, run the conditional conformal method at many different fixed levels α ∈ {0.01, 0.02, ..., 0.99}. For each prompt, find the maximum α at which at least 70% of claims are retained:

$$\alpha^*_i = \inf\{\alpha : \forall \beta \geq \alpha, \text{ at least 70% of claims are retained at level } \beta\}$$

Then fit a quantile regression to predict α* from prompt features. This gives you the function α(·).

**Step 2: Use α(·) in calibration** — Replace the fixed pinball loss ℓ_α with a prompt-dependent one ℓ_{α(X_i)}:

$$g_S = \arg\min_{g \in F} \frac{1}{n+1} \sum_{i=1}^{n} \ell_{\alpha(X_i)}(S_i - g(X_i)) + \frac{1}{n+1} \ell_{\alpha(X_{n+1})}(S - g(X_{n+1}))$$

**In plain English:** Instead of fitting a single quantile for all prompts, fit a different quantile for each prompt based on how hard it is.

**Step 3: Report α(X_{n+1}) to the user** — Along with the filtered output, tell the user "this output is correct with probability 1 − α(X_{n+1})."

#### The Guarantee (Theorem 3.2)

$$E[f(X_{n+1}) \cdot (\mathbf{1}\{L(\hat{F}(C_{n+1}; \hat{\tau}(X_{n+1})), W_{n+1}) \leq \lambda\} - (1 - \alpha(X_{n+1})))] = 0 \quad \forall f \in F$$

The key difference from Theorem 3.1: **(1-α)** is replaced by **(1 − α(X_{n+1}))** — a prompt-specific guarantee level.

**Why this is powerful:** Choose F to include functions that depend on α(X_{n+1}). For example, let F include indicator functions like 1{α(X_{n+1}) ∈ [0.2, 0.3]}. Then the theorem says:

$$P(\text{correct} \mid \alpha(X_{n+1}) \in [0.2, 0.3]) = E[1 - \alpha(X_{n+1}) \mid \alpha(X_{n+1}) \in [0.2, 0.3]]$$

**In everyday English:** Among all the outputs where the system claims a probability of correctness between 70% and 80%, the actual fraction of correct outputs is indeed between 70% and 80%. The claimed probabilities are HONEST.

This is exactly what we mean by "calibration" — the system's stated confidence matches reality.

#### Why the Function Class F Matters

The strength of the guarantee depends on the choice of F. The paper demonstrates this in Figure 8:

- **F = constant functions** (equivalent to split conformal): The nominal levels are completely uncorrelated with true levels. The system might say "70% confident" but actually be 30% correct, or vice versa. This is USELESS.

- **F = functions of α(X) and X**: The nominal levels closely track the true levels. The system's stated confidence is honest. This is what you want.

The trade-off: a larger F gives a stronger guarantee but requires more calibration data and can slow computation. The paper uses a moderate F that includes prompt metadata (prompt length, response length, mean/standard deviation of claim scores) and group indicators for the data source.

### Method 4: Conditional Boosting (Section 3.3) — THE SECOND BIG CONTRIBUTION

#### The Idea

The scoring function p(P, C) determines which claims get filtered. A better scoring function → fewer correct claims removed → higher utility. Can we LEARN a better scoring function?

Yes! The paper proposes to learn an optimal linear combination of the four existing scoring functions by optimizing through the conformal prediction algorithm.

#### What Exactly Is Being Optimized?

Let p_θ(P, C) = θ₁·(frequency score) + θ₂·(self-evaluation score) + θ₃·(ordinal score) + θ₄·(log-probability score).

The goal: find θ that maximizes the number of retained claims on a hold-out set, AFTER running the conditional conformal algorithm:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{m} \sum_{j=1}^{k_{n+i}} \mathbf{1}\{p_\theta(P_{n+i}, C_{(n+i)j}) \geq \hat{\tau}_i(\theta)\}$$

**In plain English:** "Find the scoring weights that, after conformal calibration, let us keep the most claims."

#### The Technical Challenge: Differentiating Through the Conformal Algorithm

The threshold τ̂_i(θ) depends on θ through the conformity scores. To optimize θ using gradient descent, we need the derivative ∂τ̂_i/∂θ.

But τ̂_i is the output of a quantile regression — and quantiles are not generally differentiable! (Imagine a step function: it's flat everywhere except at the step, where the derivative doesn't exist.)

**How the paper solves this — Proposition 3.1:**

The key observation is that for a linear function class F, the quantile regression is a **linear program** (LP). An LP's solution is determined by its "optimal basis" — a specific subset of the data points at which the regression line passes through the scores.

When the optimal basis is unique and non-degenerate (which almost always holds), small changes in θ don't change the basis. This means τ̂_i(θ) is LOCALLY LINEAR in θ, and the derivative exists:

$$\frac{\partial \hat{\tau}_i}{\partial \theta} = \Phi(X_{n+i})^\top \left(\Phi(X)_B^{-1} \cdot \frac{\partial S_B}{\partial \theta}\right)$$

where:
- Φ(X_{n+i}) is the feature vector of the test prompt
- Φ(X)_B is the feature matrix for the "basis" points (the data points where the quantile regression interpolates)
- S_B(θ) are the conformity scores of the basis points
- ∂S_B/∂θ is how the basis scores change with θ (this is easily computable)

**In plain English:** The formula says: "the threshold's sensitivity to the scoring weights equals the test point's features, projected through the inverse of the basis features, times how the basis scores change with the weights."

#### The Complete Boosting Procedure (Algorithm 1)

1. Split data into boosting set and final calibration set
2. For T iterations:
   a. Randomly split the boosting set into a "calibration" and "test" half
   b. Run conditional conformal on the calibration half
   c. On the test half, compute how many claims are retained
   d. Compute the gradient using Proposition 3.1
   e. Update θ using Adam optimizer (a standard gradient descent variant)
3. Use the final θ to define the scoring function for deployment

**Two tricks to handle non-differentiability:**
- The indicator function 1{score ≥ threshold} is not differentiable → approximate with a sigmoid function (a smooth S-curve)
- The basis may change for large perturbations of θ → but for small perturbations (typical in gradient descent), it stays the same

#### Why Conditional Boosting Is Better Than Marginal Boosting

The paper compares their conditional boosting against ConfTr (Stutz et al., 2021), which optimizes through the MARGINAL split conformal algorithm.

Figure 3 shows the difference dramatically on synthetic data:
- **ConfTr (marginal):** Achieves 90% average coverage, but coverage for high-variance data points drops to near 0%. The method sacrifices safety on hard examples to achieve efficiency on easy ones.
- **Conditional Boosting:** Achieves 90% coverage UNIFORMLY across all data points. The prediction sets are slightly larger on average, but there are no catastrophic failures.

**The lesson:** Optimizing for marginal coverage can actually make things WORSE for the hardest cases. Conditional boosting avoids this pitfall.

## Q2: What Is the Simplest Baseline and How Much Better Is This?

### Baselines

1. **Conformal Factuality (Mohri & Hashimoto):** Fixed level α = 0.1, uses frequency scoring, marginal guarantee only. This is the main comparison.

2. **Conditional Conformal without boosting:** Uses the conditional conformal framework but with the original frequency score. This isolates the effect of conditional boosting.

3. **Conditional Boosting without level-adaptation:** Uses boosted scores but at a fixed level α = 0.1. This isolates the effect of boosting.

### Results on Medical QA (MedLFQA — 5 datasets combined)

The experiment uses GPT-3.5-Turbo on a combined medical question-answering benchmark with 4,854 questions from 5 medical QA datasets: HealthSearchQA (n=3,047), K-QA (n=1,077), LiveQA (n=100), and MedicationQA (n=627).

**Claim retention at fixed level α = 0.1 (targeting 90% correctness):**

| Method | Mean % Claims Retained |
|--------|----------------------|
| Conformal Factuality (baseline) | ~24% |
| Conditional Boosting (no adaptation) | ~39% |
| Level-Adaptive + Boosting (combined) | ~70% (target met) |

**Key observations:**

1. **Boosting alone improves retention by 62%** (from 24% to 39%). This is from finding a better linear combination of the four scoring functions.

2. **Level-adaptive further improves to 70%** by allowing weaker guarantees (50-85% instead of fixed 90%) where needed.

3. **The level-adaptive guarantees are well-calibrated** — Figure 2 (left panel) shows that the issued probabilities closely match the realized probabilities across all bins.

### Results on Wikipedia Biographies (FActScore)

For biographies, the scoring functions are less well-correlated with factuality than for medical QA. As a result, the zero-error guarantee (all retained claims must be correct) would force guarantees below 50%. The paper instead uses a "≤ 3 errors" tolerance.

**The most striking result (Figure 6):**

Split conformal (Mohri & Hashimoto) gives dramatically different coverage for popular vs. obscure topics:
- **Very Frequent** (>1M views): ~93% correctness (great!)
- **Very Rare** (<100 views): ~73% correctness (well below the 90% target!)

Conditional conformal maintains ~90% correctness for ALL groups, by automatically adjusting claim retention:
- **Very Frequent:** Retains ~85% of claims (many claims are correct, so keep more)
- **Very Rare:** Retains ~55% of claims (more hallucinations, so filter more)

This is the GROUP-CONDITIONAL guarantee in action — the method trades off claim retention between easy and hard groups to maintain uniform coverage.

### The Level-Adaptive Result on Biographies (Figure 5)

The center panel of Figure 5 is particularly revealing. It plots claim retention vs. Wikipedia page views:
- **Fixed level method (blue):** Claim retention drops precipitously for obscure people (~60%)
- **Adaptive level method (orange):** Maintains ~80% retention across all popularity levels

The trick: for obscure people where the model hallucinates more, the system issues a weaker guarantee (lower 1 − α) but keeps more claims. For famous people, it issues a stronger guarantee while still retaining most claims.

## Q3: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: The i.i.d. Assumption

The theoretical guarantees assume that calibration and test data are i.i.d. (identically and independently distributed). In the real world, user prompts change over time and across contexts.

The paper acknowledges this and provides Corollary A.1 in the appendix: the method can handle certain covariate shifts (where the distribution of prompts changes but the relationship between prompts and correctness stays the same). Specifically, if f ∈ F is non-negative, the guarantee holds under a covariate shift where the prompt distribution is reweighted by f.

**Connection to your DS-SGen project:** This covariate shift result is directly relevant! The function class F essentially defines which distribution shifts the method can handle. Your DS-SGen project could explore choosing F specifically to handle the domain shifts you care about.

### Limitation 2: Quality of the Scoring Function

The method is a "wrapper" around existing scoring functions. If the scoring functions are weakly correlated with actual correctness, even boosting can only help so much. On the biography dataset, the boosted scores improve retention but the guarantees are still weaker than on medical QA.

The paper is honest about this: "the utility of our method depends on the quality of the underlying scoring algorithm."

### Limitation 3: The Function Class F Is Hard to Choose

The function class F controls the strength of the conditional guarantee. Too small → the guarantee is essentially marginal (useless conditionally). Too large → computation slows down and claim retention drops.

The paper uses hand-crafted features (prompt length, response length, data source indicators, claim score statistics). Finding the RIGHT features for F is an art, not a science.

### Limitation 4: Computational Cost

The boosting procedure requires running the conformal calibration algorithm at each gradient step, which means solving a linear program per iteration. The paper uses a simplified version (computing the basis from calibration data alone, ignoring the test-point augmentation) to reduce computational cost, but this is an approximation.

### Limitation 5: GPT-3.5-Turbo Only

All experiments use GPT-3.5-Turbo as the base model, with GPT-4o for claim parsing. It's unclear how well the methods transfer to other models with different hallucination patterns.

## Q4: Does This Insight Apply to Other Problems?

### Connection 1: Your DS-SGen Research Project

This paper is highly relevant to your DS-SGen project in several ways:

**The covariate shift interpretation (Appendix A.2):** The conditional guarantee automatically handles covariate shifts that lie in the function class F. If you design F to capture the domain shift between your source and target domains, the conditional conformal guarantee would hold across both domains.

Concretely: if your domain shift can be described by a density ratio function that lies in the span of F, then the conditional conformal method provides valid guarantees under domain shift "for free" — no importance reweighting needed!

**This could be a cleaner alternative to the DS-CP approach** for handling domain shift. Instead of estimating density ratios and reweighting, you could include domain-related features (e.g., similarity to source domain) in the function class F.

**The boosting technique:** You could adapt conditional boosting to learn better selection functions for your SGen framework. Instead of using fixed neuro-selection functions, learn the optimal combination of confidence scores through differentiation of the conformal calibration.

### Connection 2: General Uncertainty Quantification

The level-adaptive framework is general — it applies to any setting where conformal prediction is used and the practitioner wants output quality to be maintained even at the cost of weaker guarantees. Applications include medical diagnosis (weaker guarantee but complete report vs. strong guarantee but sparse report), autonomous driving (weaker safety guarantee but functional behavior vs. strong guarantee but overly conservative driving).

### Connection 3: Score Optimization Beyond LLMs

The conditional boosting technique for differentiating through conformal prediction is general and could be applied to any conformal prediction problem where you want to optimize the scoring function: image classification, regression, anomaly detection, etc.

## Q5: Caveats and Takeaways

### Strengths

1. **Two genuinely useful innovations** that address real practical problems (conditional validity + claim retention)
2. **Clean theoretical framework** with formal guarantees (Theorems 3.1, 3.2, Proposition 3.1)
3. **Large-scale experiments** (4,854 medical QA examples, 8,516 biographies) — much larger than the 50 examples in Mohri & Hashimoto
4. **Authors are THE top team** in conformal prediction — methods are built on a decade of foundational work
5. **Code released** with a pip-installable Python package
6. **Honest about limitations** — the paper acknowledges when their methods help less (e.g., biography dataset)
7. **The covariate shift extension** (Appendix A.2) adds theoretical value

### Weaknesses / Devil's Advocate

**Weakness 1: The ground-truth labels are LLM-generated, not human-verified.**

For medical QA, ground-truth responses come from the MedLFQA benchmark — a mix of human-written and LLM-generated reference answers. The factuality labels (W_j) are obtained by asking GPT-3.5-Turbo to evaluate whether claims are supported by these references. This creates a pipeline where GPT-3.5 judges its own outputs against references that may themselves be imperfect. The paper acknowledges this but argues it's the best available approach at scale.

**Weakness 2: The "70% claim retention" target is arbitrary.**

The paper sets the level-adaptive method to target 70% claim retention. But there's no principled reason why 70% is the right target. For some applications (high-stakes medical decisions), you might want higher retention even at the cost of weaker guarantees. For others (casual information lookup), lower retention with stronger guarantees might be better.

**Weakness 3: The conditional guarantee is only as good as the features in F.**

If the features in F don't capture the relevant variation in difficulty (e.g., if claim correctness depends on the medical specialty but F doesn't include specialty indicators), the conditional guarantee is no better than the marginal one. The paper's choice of F is reasonable but not systematically optimized.

---

# PASS 3: THE SWAMP — Deep Dive into the Proofs

## Proof Architecture Overview

The paper's theoretical contribution has three layers:

1. **Layer 1: Theorem 3.1** — Extending conditional conformal to monotone losses (relatively straightforward generalization)
2. **Layer 2: Theorem 3.2** — Level-adaptive conformal prediction (the crown jewel of the theory)
3. **Layer 3: Proposition 3.1** — Differentiability of the conformal threshold (enables boosting)

All three are proved through a master theorem (Theorem A.1 in the appendix), which handles the most general case. Theorems 3.1 and 3.2 are then just special cases.

## Layer 1: The Master Theorem (Theorem A.1) — Full Proof Walkthrough

### Setup

We have n+1 exchangeable data points: D₁, D₂, ..., D_{n+1}, where each D_i = (P_i, R_i, X_i, C_i, W_i) contains the prompt, response, features, claims, and ground-truth labels.

We have a level function α(·), a monotone loss L(·,·), and a function class F (a vector space with a convex penalty P).

The conformity score is S(C_i, W_i) = inf{τ : L(F(C_i; τ), W_i) ≤ λ}.

The conditional conformal regression solves:

$$g_S = \arg\min_{g \in F} \frac{1}{n+1} \sum_{i=1}^{n} \ell_{\alpha(X_i)}(S_i - g(X_i)) + \frac{1}{n+1} \ell_{\alpha(X_{n+1})}(S - g(X_{n+1})) + P(g)$$

The randomized threshold uses the dual variable η^S_{n+1} of this optimization:

$$\hat{\tau}_{\text{l.a. rand.}}(X_{n+1}) = \max\{S : \eta^S_{n+1} \leq U\}$$

where U ~ Uniform([-α(X_{n+1}), 1-α(X_{n+1})]).

### The Proof — Step by Step

**Step 1: Connect the loss event to the dual variable.**

The key insight from Gibbs et al. (2023): the dual variable η^S_{n+1} is non-decreasing in S. By the monotonicity of L and the definition of the conformity score:

$$\mathbf{1}\{L(\hat{F}(C_{n+1}; \hat{\tau}), W_{n+1}) \leq \lambda\} = \mathbf{1}\{\eta^{S(C_{n+1}, W_{n+1})}_{n+1} \leq U\}$$

**In plain English:** The event "the filtered output is good" is equivalent to "the dual variable at the true conformity score is at most the random uniform U." This converts a filtering problem into a comparison between two numbers.

**Step 2: Compute the conditional expectation over U.**

Since U is uniform on [-α(X_{n+1}), 1-α(X_{n+1})]:

$$E_U[\mathbf{1}\{\eta^{S_{n+1}}_{n+1} \leq U\} - (1-\alpha(X_{n+1})) \mid \text{data}, X_{n+1}] = -\eta^{S_{n+1}}_{n+1}$$

**Why?** The probability that U ≥ η is:
P(U ≥ η) = (1-α - η) / (1-α+α) = (1-α-η) when η ∈ [-α, 1-α].

So E[1{η ≤ U}] = (1-α(X_{n+1}) - η) / 1 = 1-α(X_{n+1}) - η.

Then E[1{η ≤ U} - (1-α)] = -η.

**Step 3: Use KKT conditions and exchangeability.**

From the KKT (Karush-Kuhn-Tucker) stationarity condition of the quantile regression:

$$0 = \frac{d}{d\epsilon}(n+1)P(g_{S_{n+1}} + \epsilon f)\Big|_{\epsilon=0} - \sum_{i=1}^{n+1} \eta^{S_{n+1}}_i f(X_i)$$

**In plain English:** At the optimal solution, the derivative of the regularized objective equals zero. This relates the sum of all dual variables weighted by f to the penalty term.

**Step 4: Apply exchangeability to symmetrize.**

Since the data points are exchangeable, each one is equally likely to be in any position. So:

$$E[f(X_{n+1}) \cdot \eta^{S_{n+1}}_{n+1}] = \frac{1}{n+1} E\left[\sum_{i=1}^{n+1} f(X_i) \eta^{S_{n+1}}_i\right]$$

By the KKT condition, this equals:

$$E\left[\frac{d}{d\epsilon}(n+1)P(g_{S_{n+1}} + \epsilon f)\Big|_{\epsilon=0}\right]$$

**Step 5: Assemble the final result.**

Combining Steps 2, 3, and 4:

$$E[f(X_{n+1})(\mathbf{1}\{\text{correct}\} - (1-\alpha(X_{n+1})))] = -E[f(X_{n+1}) \eta^{S_{n+1}}_{n+1}] = -E\left[\frac{d}{d\epsilon}P(g + \epsilon f)\Big|_{\epsilon=0}\right]$$

### Special Cases: Theorems 3.1 and 3.2

**For P(·) = 0 (no regularization) and finite-dimensional F:**

The right-hand side becomes zero! The dual program is a linear program, and by Slater's condition (the interior point η = 0 is feasible), strong duality holds.

So:

$$E[f(X_{n+1})(\mathbf{1}\{\text{correct}\} - (1-\alpha(X_{n+1})))] = 0$$

This proves BOTH Theorem 3.1 (when α is constant) and Theorem 3.2 (when α depends on X).

**Why the proof is elegant:** The entire argument reduces to three ingredients: (1) the randomization trick converts the loss event to a comparison with a uniform random variable, (2) the KKT conditions relate dual variables to penalty derivatives, and (3) exchangeability symmetrizes over data points.

## Layer 2: Proposition 3.1 — Differentiability of the Threshold

### The Problem

We need ∂τ̂_i/∂θ. The threshold τ̂_i(θ) is defined as the sup of S values for which S ≤ g_S(X_{n+i}), where g_S is the solution to a quantile regression that depends on θ through the conformity scores.

### The Key Observation

For a linear F with no penalty, the quantile regression is a linear program. Linear programs have solutions determined by a "basis" — a subset B of d data points (where d is the dimension of F) at which the regression interpolates the scores.

When the basis is unique and non-degenerate, the solution is:

$$\beta^S = \Phi(X)_B^{-1} S_B(θ)$$

and the threshold is:

$$\hat{\tau}_i(\theta) = \Phi(X_{n+i})^\top \beta^S = \Phi(X_{n+i})^\top (\Phi(X)_B^{-1} S_B(\theta))$$

This is a LINEAR function of S_B(θ), and S_B(θ) is differentiable in θ (it's the conformity scores, which depend on the scoring function p_θ). So:

$$\frac{\partial \hat{\tau}_i}{\partial \theta} = \Phi(X_{n+i})^\top (\Phi(X)_B^{-1} \frac{\partial S_B}{\partial \theta})$$

### Why the Basis Doesn't Change

The proof establishes that for S > τ̂_i(θ), the basis is CONSTANT (doesn't depend on S). This is because the augmented quantile regression problem with the (n+1)-st score imputed at S only changes the constraint for the (n+1)-st data point. When S is above the threshold, the (n+1)-st point is not in the basis, so the basis is unchanged.

For small perturbations of θ, the basis is also unchanged (by LP sensitivity analysis — the reduced costs of non-basic variables are bounded away from zero under non-degeneracy).

Combining these two facts: the basis is locally constant in θ, so τ̂_i(θ) is locally a linear function of θ, and therefore differentiable.

## Techniques You Can Borrow for Your DS-SGen Research

### Technique 1: The Conditional Conformal Framework for Domain Shift

The covariate shift corollary (A.1 and A.2) tells you that if you include domain-relevant features in F, the conditional guarantee automatically holds under distribution shifts described by functions in F. This is a MUCH simpler approach to domain shift than density ratio estimation!

**Concrete idea for DS-SGen:** Include features like "embedding distance to the nearest source-domain example" or "predicted domain membership probability" in your feature vector X. Then run conditional conformal prediction. The resulting guarantee would hold conditionally on these features, effectively providing domain-shift robustness.

### Technique 2: Level-Adaptive Selection for Open-Ended QA

Instead of the binary "answer/abstain" decision in SGen, you could use level-adaptive conformal prediction to provide a spectrum of responses:
- High confidence → full answer with strong guarantee
- Medium confidence → partial answer (some claims removed) with moderate guarantee
- Low confidence → "I don't know" with no claims

This gives users more information than the binary SGen approach.

### Technique 3: Boosting the Selection Function

The conditional boosting technique for learning optimal scoring function combinations could be directly applied to your neuro-selection function in DS-SGen. Instead of trying a few fixed configurations, learn the optimal combination of confidence scores by differentiating through the PAC bound computation.

### Technique 4: The Monotone Loss Generalization

Instead of using FDR-E (a single threshold for error tolerance), you could use the monotone loss framework to control different types of errors: "at most k false answers" or "at most fraction a of shown answers are wrong." This gives more flexible control than the binary "FDR-E ≤ ε" guarantee.

---

# KEY CONCEPTS GLOSSARY (for Grade 12 Level)

| Concept | Simple Explanation |
|---------|-------------------|
| **Marginal guarantee** | A guarantee that holds "on average" — like saying a restaurant is good on average, even though some dishes are terrible |
| **Conditional guarantee** | A guarantee that holds for each GROUP — like saying every dish category (appetizers, mains, desserts) is good, not just the average |
| **Group-conditional coverage** | The probability of correctness is the same (e.g., 90%) for every pre-specified group of inputs |
| **Calibration** | When stated probabilities match reality — if a forecaster says "70% chance of rain" and it rains 70% of those times, they're calibrated |
| **Pinball loss** | An asymmetric loss function used to find quantiles through optimization — it penalizes under-prediction more heavily than over-prediction |
| **Quantile regression** | A type of regression that predicts a specific percentile (like the 90th percentile) instead of the average |
| **Level-adaptive** | The strength of the guarantee (the probability of correctness) changes based on how hard the question is |
| **Conditional boosting** | Learning better scoring functions by optimizing through the conformal prediction algorithm with conditional guarantees |
| **Optimal basis** | In a linear program, the specific subset of data points that determine the solution — like the "pivot points" of the solution |
| **Linear program (LP)** | An optimization problem where both the objective and constraints are linear functions — solvable efficiently |
| **KKT conditions** | Mathematical conditions that must be satisfied at the optimal solution of a constrained optimization problem — like the "first derivative = 0" condition but for constrained problems |
| **Dual variable** | A number associated with each constraint in an optimization problem that measures "how much would the optimal value change if we relaxed this constraint slightly?" |
| **Exchangeability** | Data points can be reordered without changing their joint distribution — like cards shuffled in a deck |
| **Monotone loss** | A loss function that can only increase (or stay the same) when you include more claims in the output |
| **Covariate shift** | A type of distribution change where the input distribution changes but the relationship between inputs and outputs stays the same |
| **Strong duality** | When the optimal value of an optimization problem equals the optimal value of its "dual" (mirror image) problem — this holds for linear programs |
| **Slater's condition** | A sufficient condition for strong duality: there exists a strictly feasible point for the dual problem |
| **FActScore** | A benchmark for evaluating the factual accuracy of LLM-generated biographies using Wikipedia as ground truth |
| **MedLFQA** | A benchmark combining several medical question-answering datasets for evaluating LLM factuality in healthcare |

---

# HOW THIS PAPER CONNECTS TO YOUR OTHER THREE PAPERS

## The Four-Paper Landscape

Your DS-SGen research project now sits at the intersection of FOUR papers:

| Paper | What It Does | Key Innovation | Limitation |
|-------|-------------|---------------|------------|
| **SGen** (Lee et al., NeurIPS 2024) | Binary: answer or "IDK" | Entailment-based correctness + PAC guarantee | i.i.d. only, marginal guarantee |
| **Conformal Factuality** (Mohri & Hashimoto, 2024) | Removes uncertain claims | Entailment sets make CP tractable for text | i.i.d. only, marginal guarantee, removes too much |
| **THIS paper** (Cherian et al., NeurIPS 2024) | Removes claims with conditional + adaptive guarantees | Conditional boosting + level-adaptive CP | i.i.d. only (but covariate shift extension exists) |
| **DS-CP** (Lin et al., arXiv 2025) | Handles domain shift for CP | Embedding-based density ratio estimation | Multiple-choice only, approximate guarantee |

## Key Relationships

### This Paper extends Conformal Factuality
- Same basic setup (score claims, threshold, filter)
- Adds conditional validity (group-specific guarantees)
- Adds level adaptation (prompt-specific guarantee levels)
- Adds score boosting (learning better scoring functions)
- Uses larger calibration sets (thousands vs. 50)
- Result: dramatically more practical (70% retention vs. ~30%)

### This Paper partially addresses the DS-CP problem
- Corollary A.1 shows the conditional conformal method handles covariate shifts where the density ratio function lies in F
- This means: if you choose F to include features that capture domain shift, you get domain-shift robustness
- This is a cleaner approach than DS-CP's density ratio estimation, but less flexible (F must be chosen in advance)

### What Your DS-SGen Project Adds
Your project combines ALL of these:
- **From SGen:** The selective generation framework (binary answer/abstain) with PAC guarantees
- **From Conformal Factuality:** Entailment-based correctness for open-ended generation
- **From THIS paper:** Conditional guarantees, adaptive levels, better scoring functions
- **From DS-CP:** Handling arbitrary domain shift through importance reweighting
- **Your novel combination:** Domain-shift-aware selective generation with conditional PAC guarantees

## Specific Technical Bridges to Your DS-SGen

1. **This paper's Corollary A.1 → Your domain shift strategy:** Consider using conditional conformal prediction with domain-related features in F as an ALTERNATIVE to importance reweighting. This avoids density ratio estimation entirely.

2. **This paper's boosting → Your neuro-selection function:** Apply conditional boosting to learn optimal combinations of selection scores under domain shift constraints.

3. **This paper's level-adaptive CP → Your abstention mechanism:** Instead of binary "answer/IDK," provide a spectrum: full answer (high guarantee), partial answer (moderate guarantee), IDK (no answer). This is more informative than SGen's binary decision.

4. **This paper's monotone loss framework → Your FDR-E control:** The monotone loss generalization lets you control more nuanced error metrics than binary "FDR-E ≤ ε."

5. **This paper's MedLFQA evaluation → Your experimental setup:** The medical QA datasets and evaluation methodology provide a ready-made experimental testbed for evaluating DS-SGen on domain-shift scenarios (e.g., calibrate on HealthSearchQA, test on MedicationQA).

---

# SUMMARY: ONE-PAGE CHEAT SHEET

**Problem:** Conformal factuality (Mohri & Hashimoto) has marginal-only guarantees that fail on hard topics, and removes too many correct claims to be useful.

**Two Key Innovations:**

1. **Conditional Boosting:** Learn an optimal linear combination of scoring functions by differentiating through the conditional conformal algorithm. Result: 62% more claims retained (24% → 39%) at the same guarantee level.

2. **Level-Adaptive CP:** Adapt the guarantee probability per-prompt to maintain claim retention above 70%. Harder prompts get weaker but honest guarantees; easier prompts get stronger guarantees. Result: 70% claim retention with calibrated probabilities between 50-85%.

**Key Theoretical Results:**
- Theorem 3.1: Conditional conformal prediction controls any monotone loss
- Theorem 3.2: Level-adaptive CP provides calibrated, prompt-specific guarantees
- Proposition 3.1: The conformal threshold is differentiable (enables boosting)
- Corollary A.1: The method handles covariate shifts within the function class F

**Key Practical Results (Medical QA, GPT-3.5-Turbo):**
- Fixed conformal factuality: 24% claims retained, 90% fixed guarantee
- Conditional boosting alone: 39% claims retained, 90% fixed guarantee  
- Boosting + level-adaptive: 70% claims retained, 50-85% adaptive guarantees
- All issued probabilities are well-calibrated (realized matches nominal)

**Main Limitations:**
- i.i.d. assumption (partially addressed by covariate shift extension)
- Depends on quality of underlying scoring functions
- Function class F requires careful design
- Only tested on GPT-3.5-Turbo
- Ground-truth labels are LLM-generated, not human-verified

**For Your DS-SGen Project:** Consider using conditional conformal prediction with domain-related features as an alternative to density ratio estimation for domain shift. Borrow the boosting technique for learning optimal selection functions. Explore the level-adaptive framework for providing graded responses (full answer → partial answer → IDK) under domain shift.