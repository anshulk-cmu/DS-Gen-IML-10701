# Conformal Prediction Under Covariate Shift — Complete Paper Analysis

**Paper:** Conformal Prediction Under Covariate Shift  
**Authors:** Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas  
**Venue:** NeurIPS 2019 (one of the top machine learning conferences)  
**Type:** Breakthrough / Theory paper  
**Peer-Reviewed:** Yes — published at NeurIPS, a highly competitive peer-reviewed venue  
**Purpose:** Deep understanding — this is a foundational paper for your DS-SGen research project

---

# PHASE -1: PAPER CLASSIFICATION

This is a **Breakthrough / Theory** paper. Here's why:

- It introduces a genuinely new idea: extending conformal prediction to work when training and test data come from different distributions
- It has formal mathematical theorems with proofs
- It introduces a new concept ("weighted exchangeability") that generalizes a classical notion
- It comes from four top statisticians at CMU, Chicago, and Stanford
- It's been cited hundreds of times and spawned an entire research direction

**Reading strategy:** Apply the full Pass 1-2-3 framework. This is the default mode for breakthrough theory papers.

---

# PHASE 0: PRE-READING — Context and Authors

Before diving in, let's understand who wrote this and why it matters.

**Authors:**
- **Ryan Tibshirani** (CMU) — Son of Robert Tibshirani (inventor of the Lasso), a leading statistician himself. Known for work in nonparametric statistics, optimization, and conformal prediction.
- **Rina Foygel Barber** (University of Chicago) — Top statistician working on high-dimensional inference, selective inference, and conformal prediction. Won the COPSS Presidents' Award (the "Nobel Prize of statistics" for young researchers).
- **Emmanuel Candès** (Stanford) — One of the most influential statisticians alive. Co-inventor of compressed sensing. MacArthur "Genius" Fellow.
- **Aaditya Ramdas** (CMU) — Expert in sequential testing, conformal prediction, multiple testing. The person whose reading framework we're using!

**Publication:** NeurIPS 2019 — one of the top 3 machine learning conferences globally. Getting a paper here is highly competitive.

**Why this paper matters for your project:** Your DS-SGen project extends SGen (selective generation for LLMs) to handle domain shift. This paper provides THE foundational technique — weighted conformal prediction — that makes conformal prediction work when the training and test data come from different distributions. The DS-CP paper you've already studied directly builds on this paper.

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## Q1: What Is the Problem Being Solved?

### The Setup — What Are We Trying to Do?

Imagine you're a weather forecaster. You have historical data from the past year, and you want to predict tomorrow's temperature. But you don't just want to say "it'll be 25°C" — you want to give a **range** like "it'll be between 22°C and 28°C" that you're 90% confident about.

In machine learning, this is called a **prediction interval**. You have:
- **Training data**: past examples where you know both the input (X) and the output (Y)
- **A new test input** (X_{n+1}): for which you want to predict the output (Y_{n+1})

Your goal: build a prediction interval C(X_{n+1}) such that:

$$P\big(Y_{n+1} \in C(X_{n+1})\big) \geq 1 - \alpha$$

where α is your error tolerance (e.g., α = 0.10 means you want 90% coverage).

**The magical thing about conformal prediction** is that it gives you this guarantee **without any assumptions** about the underlying data distribution. You don't need to assume the data is normally distributed, or linear, or anything. The ONLY thing you need is that the training and test data are **exchangeable** (more on this below).

### The Problem: What If Training and Test Data Are Different?

Here's where the trouble starts. Standard conformal prediction assumes your training data and test data come from the **same** distribution. But in many real-world situations, they don't!

**Everyday example:** Imagine you train a medical diagnosis model on data from Hospital A (mostly young patients in a city). Now Hospital B (mostly elderly patients in a rural area) wants to use your model. The patients look different (different ages, different backgrounds — this is the "covariate" X changing), but the underlying biology (how symptoms relate to diseases — this is Y|X) is the same.

This mismatch between training and test distributions is called **covariate shift**. Under covariate shift:
- The distribution of inputs X changes between training and test
- But the relationship between X and Y stays the same: P(Y|X) is unchanged
- Standard conformal prediction **breaks** — the 90% guarantee might only give you 80% coverage

**In one sentence:** "This paper studies how to build prediction intervals with guaranteed coverage when the test data comes from a different distribution than the training data, under the covariate shift assumption."

## Q2: Why Is This Problem Interesting and Nontrivial?

### Why Can't We Just Use Regular Conformal Prediction?

Conformal prediction relies on a property called **exchangeability**. Let me explain this with a simple analogy.

**Exchangeability — The Card Analogy:**

Imagine you have a deck of cards. You deal 10 cards face-up (training data) and 1 card face-down (test point). Exchangeability means: it doesn't matter WHICH card ended up face-down. If you shuffled all 11 cards randomly and then dealt them, any card was equally likely to be the face-down one.

Mathematically, random variables Z₁, Z₂, ..., Z_{n+1} are exchangeable if their joint distribution doesn't change when you rearrange their order. If you drew all of them from the same distribution (i.i.d.), they're automatically exchangeable.

**Why exchangeability matters for conformal prediction:**

The core logic is: if all n+1 data points are "equivalent" (exchangeable), then the test point's nonconformity score (how "weird" it is) has no special reason to be bigger or smaller than any training point's score. So the test score should rank uniformly among all n+1 scores. This means: the probability that the test score exceeds the (1-α)-quantile of the training scores is at most α. That's the coverage guarantee!

**What breaks under covariate shift:**

When test data comes from a different distribution, the test point IS special — it was drawn from a different place! The exchangeability breaks. The test point's score might be systematically higher or lower than training scores, and the coverage guarantee fails.

**Why fixing this is hard:**

You need to somehow "correct" for the distribution mismatch without knowing what the test distribution looks like in advance. The correction involves the **likelihood ratio** — how much more (or less) likely each training point would be under the test distribution. But:
1. You need to know (or estimate) this ratio
2. Even if you know it, you need a principled way to incorporate it into conformal prediction
3. The "effective sample size" shrinks when you reweight, making intervals wider

**In one sentence:** "This is nontrivial because the exchangeability assumption that makes conformal prediction work breaks under covariate shift, and correcting for it requires knowing the likelihood ratio between test and training distributions."

## Q3: What Is the Main Claim?

The paper's main claim has two parts:

### Part 1: Weighted Conformal Prediction (Corollary 1)

If training data comes from distribution P and test data comes from distribution P̃, and you know the likelihood ratio w(x) = dP̃_X(x) / dP_X(x), then you can build a **weighted** conformal prediction interval that still guarantees:

$$P\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \geq 1 - \alpha$$

The trick: instead of treating all training nonconformity scores equally (uniform weights 1/(n+1) each), you weight them by how "relevant" they are to the test distribution. A training point whose X looks more like a typical test X gets more weight.

### Part 2: Weighted Exchangeability (Theorem 2)

The paper introduces a new concept called **weighted exchangeability** and shows that conformal prediction can be extended to any data satisfying this property. The covariate shift result (Part 1) is just one special case of this more general theory.

**In one sentence:** "They show that by weighting each training point's nonconformity score by the likelihood ratio of test-to-training covariate distributions, conformal prediction maintains its coverage guarantee under covariate shift, and this extends to a general notion of weighted exchangeability."

---

# PASS 2: THE SCUBA DIVE — How Does This Paper Actually Work?

Now let's go deeper into the mechanics. I'll explain every concept from scratch.

## Q1: What Was the Main Technical Barrier Before This Paper? How Does It Overcome It?

### The Barrier: Exchangeability Was Considered Essential

Before this paper, the entire theory of conformal prediction was built on exchangeability. Every theorem, every guarantee, every result assumed the data was exchangeable. Nobody had figured out how to give provable guarantees when this assumption was violated.

People knew covariate shift was a problem in practice. There was a whole literature on correcting classifiers and estimators for covariate shift (using importance weighting). But nobody had connected these ideas to conformal prediction to give distribution-free prediction intervals under covariate shift.

### The Key Insight: Weighted Exchangeability

The paper's conceptual breakthrough is recognizing that **you don't need full exchangeability** — you just need a weaker property that still captures enough symmetry for the quantile argument to work.

Let me build this up step by step.

**Step 1: Recall how the quantile argument works for exchangeable data**

If V₁, ..., V_{n+1} are exchangeable (think: drawn from the same bag), then for any permutation σ, the joint distribution of (V_{σ(1)}, ..., V_{σ(n+1)}) is the same as (V₁, ..., V_{n+1}). This means V_{n+1} has "no special position" — it's equally likely to be the smallest, second smallest, ..., or largest among all n+1 values.

So the probability that V_{n+1} is above the (1-α)-quantile of V₁, ..., V_n is at most α. That's the basic coverage guarantee.

**Step 2: What happens without exchangeability?**

Under covariate shift, V_{n+1} IS different from V₁, ..., V_n. Some training points might produce scores that are more "relevant" to what we'd expect at the test point, and others might be less relevant.

**Step 3: The fix — introduce weights**

The key move: instead of saying "V_{n+1} is equally likely to be in any rank position," say "V_{n+1} is in rank position i with probability proportional to a weight w_i."

Think of it like a weighted lottery. In an unweighted lottery (exchangeability), each ticket has equal chance. In a weighted lottery (weighted exchangeability), some tickets are "heavier" — they have higher probability of being drawn. The paper shows that if you use the right weights (the likelihood ratios), then comparing V_{n+1} to a **weighted** quantile of V₁, ..., V_n still gives the coverage guarantee.

### Formal Definition of Weighted Exchangeability

Random variables V₁, ..., V_n are **weighted exchangeable** with weight functions w₁, ..., w_n if their joint density can be written as:

$$f(v_1, ..., v_n) = \prod_{i=1}^n w_i(v_i) \cdot g(v_1, ..., v_n)$$

where g is a **symmetric** function (doesn't depend on the order of its inputs).

**What does this mean in simple terms?**

The joint distribution has two parts:
1. **Individual weights** w_i(v_i): each variable has its own "personality" — it might tend to be larger or smaller
2. **Symmetric core** g: the underlying structure is symmetric (doesn't care about ordering)

Regular exchangeability is the special case where all weights are 1 (all variables have the same "personality").

**Why covariate shift gives weighted exchangeability:**

Under covariate shift:
- Training points (X_i, Y_i) come from P = P_X × P_{Y|X}
- Test point (X_{n+1}, Y_{n+1}) comes from P̃ = P̃_X × P_{Y|X}

The joint density of all n+1 points can be written as:

$$\prod_{i=1}^n p_X(x_i) \cdot \tilde{p}_X(x_{n+1}) \cdot \prod_{i=1}^{n+1} p_{Y|X}(y_i|x_i)$$

We can rewrite this as:

$$\frac{\tilde{p}_X(x_{n+1})}{p_X(x_{n+1})} \cdot \prod_{i=1}^{n+1} p_X(x_i) \cdot p_{Y|X}(y_i|x_i)$$

The first factor is w(x_{n+1}) = dP̃_X/dP_X evaluated at x_{n+1} — the likelihood ratio. The second factor is the joint density of n+1 i.i.d. draws from P, which IS symmetric. So we have weighted exchangeability with w_i ≡ 1 for training points and w_{n+1}(x,y) = w(x) for the test point.

## The Weighted Conformal Prediction Algorithm — Step by Step

Now let's see exactly how the algorithm works. I'll use a concrete example throughout.

### Setup

**Your data:**
- Training data: (X₁, Y₁), ..., (X_n, Y_n) — e.g., n = 100 patients from Hospital A
- Test input: X_{n+1} — a new patient from Hospital B
- Likelihood ratio: w(x) = dP̃_X(x)/dP_X(x) — how much more likely is input x under Hospital B's distribution vs. Hospital A's

**Your score function:**
You pick any score function S that measures how "weird" a point is. A common choice:

$$S\big((x, y), Z\big) = |y - \hat{\mu}(x)|$$

This is just: "how far is the actual Y from my model's prediction?" If the model predicts well, the score is low. If the model is wrong, the score is high.

### Step 1: Compute weights for each training point

For each training point i = 1, ..., n, compute:

$$p_i^w(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}$$

And for the test point:

$$p_{n+1}^w(x) = \frac{w(x)}{\sum_{j=1}^n w(X_j) + w(x)}$$

**What this means in plain English:**

Each training point gets a weight proportional to its likelihood ratio w(X_i). Training points that "look like" test data (high likelihood ratio) get MORE weight. Training points that are very different from test data (low likelihood ratio) get LESS weight. The test point also gets weight proportional to its own likelihood ratio.

All weights sum to 1 (they form a probability distribution).

**Concrete example:**

Suppose Hospital A has mostly young patients (ages 20-40) and Hospital B has mostly elderly patients (ages 60-80). Then:
- A training patient who is 70 years old gets a HIGH weight (they're representative of Hospital B's patients)
- A training patient who is 25 years old gets a LOW weight (they're not representative)

### Step 2: Compute nonconformity scores

For each possible test value y, compute the scores:

$$V_i^{(x,y)} = S\big(Z_i, Z_{1:n} \cup \{(x,y)\}\big), \quad i = 1, ..., n$$
$$V_{n+1}^{(x,y)} = S\big((x,y), Z_{1:n} \cup \{(x,y)\}\big)$$

These scores measure how well each point "conforms" to the entire dataset (including the hypothetical test point (x,y)).

### Step 3: Compute the weighted quantile

Form a weighted discrete distribution:

$$\sum_{i=1}^n p_i^w(x) \cdot \delta_{V_i^{(x,y)}} + p_{n+1}^w(x) \cdot \delta_\infty$$

This is a mixture distribution. Think of it as putting "blobs" of probability at each training score value (with height equal to the weight) and one blob at infinity (with height equal to the test weight).

Find the (1-α)-quantile of this weighted distribution.

**The infinity term is crucial.** It's a "safety valve" that prevents the prediction set from being too small. Even if all training scores are small, this infinity blob pushes the quantile up, ensuring the prediction set is large enough for coverage.

### Step 4: Include y in the prediction interval if the test score is below the quantile

$$\hat{C}_n(x) = \Big\{y \in \mathbb{R} : V_{n+1}^{(x,y)} \leq \text{Quantile}\Big(1-\alpha;\; \sum_{i=1}^n p_i^w(x)\delta_{V_i^{(x,y)}} + p_{n+1}^w(x)\delta_\infty\Big)\Big\}$$

**In plain English:** Try each possible y value. If the test score at y is "not too extreme" compared to the weighted training scores, include y in the prediction interval.

### Why This Gives Valid Coverage

The proof relies on the weighted exchangeability we established earlier. Here's the intuition:

1. Under weighted exchangeability, the test score V_{n+1} is NOT equally likely to be in any rank — but it IS in rank position i with probability p_i^w
2. The weighted quantile is set so that the total probability mass below it is exactly (1-α)
3. Therefore, the probability that V_{n+1} falls below the weighted quantile is at least (1-α)
4. This means Y_{n+1} is in the prediction set with probability at least (1-α)

## Q2: What Is the Simplest Baseline? How Much Better Is This?

### The Baselines

**Baseline 1: Standard (unweighted) conformal prediction.** Just ignore the covariate shift and use the ordinary conformal method. This is the simplest approach.

**The experiment:** The paper uses the UCI Airfoil dataset (1503 observations, 5-dimensional features related to airfoil noise). They create covariate shift by exponential tilting:

$$w(x) = \exp(x^T \beta), \quad \text{where } \beta = (-1, 0, 0, 0, 1)$$

This makes certain types of airfoils more likely in the test set than in the training set.

### Results (Figure 1):

**Without covariate shift (red histogram):** Standard conformal prediction achieves 90.2% average coverage. Works perfectly — close to the target 90%.

**With covariate shift, standard conformal (blue histogram):** Coverage drops to 82.2%. That's a HUGE gap — you're losing 8 percentage points of coverage. Your "90% guarantee" is actually only giving you 82%.

**With covariate shift, weighted conformal with oracle weights (orange):** Coverage is 90.8%. The guarantee is restored!

**With covariate shift, weighted conformal with estimated weights:**
- Using logistic regression to estimate weights: 91.0% coverage
- Using random forests to estimate weights: 91.0% coverage

Both estimated methods match the oracle nearly perfectly.

### The Price: Wider Intervals

Weighted conformal prediction restores coverage, but the prediction intervals are wider (more spread out across trials). This makes sense — by reweighting, you're effectively using fewer data points (reduced "effective sample size"). The paper shows this with a clever comparison: they run unweighted conformal on a smaller dataset (matching the effective sample size), and the results are very similar to weighted conformal. So the extra dispersion is fully explained by the reduced effective sample size, not by any inefficiency of the method.

### Effective Sample Size

The paper uses this formula for effective sample size:

$$\hat{n} = \frac{\big(\sum_{i=1}^n |w(X_i)|\big)^2}{\sum_{i=1}^n |w(X_i)|^2} = \frac{\|w(X_{1:n})\|_1^2}{\|w(X_{1:n})\|_2^2}$$

**Intuition:** If all weights are equal (no covariate shift), then n̂ = n (full sample). If one weight dominates (extreme shift), then n̂ ≈ 1 (you effectively have one data point). The more "spread out" the weights are, the higher the effective sample size.

## Q3: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: You Need to Know (or Estimate) the Likelihood Ratio

The whole method depends on knowing w(x) = dP̃_X(x)/dP_X(x). In the paper's experiment, they estimate it by training a classifier to distinguish training from test points and using the odds ratio:

$$\hat{w}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)}$$

where p̂(x) is the estimated probability that x comes from the test distribution. This works well when X is low-dimensional (5 dimensions in the airfoil example). But in high dimensions (like text data for LLMs), this estimation becomes very hard.

### Limitation 2: The Coverage Guarantee Assumes Perfect Knowledge of w

If you use estimated weights ŵ instead of true weights w, the formal guarantee from Corollary 1 doesn't strictly hold. The paper shows empirically that estimated weights work well, but there's no theorem quantifying how estimation error in ŵ affects coverage.

### Limitation 3: Extreme Weights Cause Problems

If the training and test distributions are very different, some weights become huge while most are near zero. This makes the effective sample size tiny, and the prediction intervals become enormous (technically valid, but useless).

### Limitation 4: No Conditional Coverage

The paper provides **marginal** coverage: averaged over all possible test points, the coverage is at least 1-α. But for any SPECIFIC test point x₀, the coverage might be much higher or lower than 1-α.

The paper discusses using weighted conformal for "approximate conditional coverage" (Section 4), using kernel weights K((x - x₀)/h) to get local coverage around x₀. But this is limited — the band must be recomputed for each new center point x₀.

### Limitation 5: Computational Cost of Full Conformal

The full conformal method requires recomputing the scores for EVERY possible test value y, which is computationally infeasible for most real problems. The paper uses split conformal (which avoids this) in experiments, but the theory is presented for full conformal.

## Q4: Does the Insight Apply to Other Problems?

The paper itself discusses several:

### Application 1: Graphical Models with Covariate Shift

If you have a causal chain Z → X → Y where Z is low-dimensional and X is high-dimensional, and only Z's distribution shifts, you can estimate the likelihood ratio in the Z space (easy) instead of the X space (hard). This is because:

$$\frac{\tilde{P}_{Z,X}(z,x)}{P_{Z,X}(z,x)} = \frac{\tilde{P}_Z(z)}{P_Z(z)}$$

The X parts cancel because P(X|Z) doesn't change.

### Application 2: Missing Covariates / Privacy Settings

Hospital B wants predictions from Hospital A's model, but can't share full patient data for privacy reasons. If Hospital B only shares summary statistics about a sensitive variable Z (like the fraction of male vs. female patients), you can compute the likelihood ratio for Z and use weighted conformal prediction.

### Application 3: Approximate Conditional Coverage

Using kernel weights around a point x₀, you can get a "locally-weighted" coverage guarantee. The bandwidth h controls the trade-off: smaller h gives more local coverage but with less effective data.

### Connection to Your DS-SGen Project

This paper is the theoretical foundation for what DS-CP (Lin et al., 2025) does for LLMs. The chain is:
1. **This paper (2019):** Establishes that weighted conformal prediction works under covariate shift
2. **DS-CP (2025):** Applies this to LLMs by estimating density ratios in embedding space
3. **Your DS-SGen project:** Extends SGen's FDR control to handle domain shift, using these same reweighting ideas

## Q5: Caveats and Takeaways

**Strengths:**
- Beautiful, clean theory with a simple but powerful idea
- The weighted exchangeability concept is genuinely new and useful
- Empirical results strongly support the theory
- Written by four of the most respected statisticians in the field
- Published at a top venue

**Caveats:**
- The empirical evaluation is on a single, low-dimensional dataset
- No theoretical guarantees for the case of estimated weights
- The effective sample size reduction can be severe in practice
- The connection to split conformal (what people actually use) is deferred to the supplement

**Key takeaway:** Weighted conformal prediction is the RIGHT way to handle covariate shift in conformal prediction. The theory is clean and the method is simple to implement.

---

# PASS 3: THE SWAMP — Deep Dive Into the Mathematical Details

Now let's go through the proofs and mathematical machinery in detail. I'll explain every step as if you're seeing this math for the first time.

## Prerequisite Concepts You Need

### Concept 1: What Is a Quantile?

You probably know the **median** — the value where 50% of the data is below and 50% is above. A quantile generalizes this.

The β-quantile of a distribution F is the smallest value z such that at least a fraction β of the probability mass is at or below z:

$$\text{Quantile}(\beta; F) = \inf\{z : P(Z \leq z) \geq \beta\}$$

**Example:** If your data is {3, 7, 12, 15, 20}, the 0.6-quantile is 12, because 3/5 = 60% of the data is ≤ 12.

For a **weighted** distribution, each data point carries a different amount of probability mass. If the weights are (0.1, 0.1, 0.3, 0.3, 0.2) on the values {3, 7, 12, 15, 20}, the 0.6-quantile is 15, because the cumulative weight at 12 is only 0.1 + 0.1 + 0.3 = 0.5 (not enough), but at 15 it's 0.8 (enough).

### Concept 2: What Is a Point Mass (δ)?

δ_a is the simplest possible probability distribution — it puts ALL its probability at the single value a. Think of it as "I am 100% certain the value is a."

When we write:

$$\frac{1}{3}\delta_5 + \frac{1}{3}\delta_7 + \frac{1}{3}\delta_\infty$$

this means: "with probability 1/3 the value is 5, with probability 1/3 it's 7, and with probability 1/3 it's infinity."

### Concept 3: What Is a Likelihood Ratio?

If distribution P has density p(x) and distribution P̃ has density p̃(x), the likelihood ratio is:

$$w(x) = \frac{\tilde{p}(x)}{p(x)}$$

**Intuition:** w(x) tells you how much more likely point x is under P̃ compared to P.
- w(x) = 2: x is twice as likely under P̃
- w(x) = 0.5: x is half as likely under P̃
- w(x) = 1: x is equally likely under both

**Why it's useful:** If you have samples from P and want to "pretend" they came from P̃, multiply each sample's importance by w(x). This is the core of importance sampling.

### Concept 4: What Is Absolute Continuity?

We say P̃_X is "absolutely continuous with respect to" P_X if: whenever P_X says an event has probability zero, P̃_X also says it has probability zero.

**In plain terms:** P̃_X doesn't put probability where P_X doesn't. If a region of space is impossible under the training distribution, it's also impossible under the test distribution.

**Why this matters:** This ensures the likelihood ratio w(x) = dP̃_X/dP_X exists and is well-defined. If P̃_X put probability somewhere P_X didn't, we'd be dividing by zero.

### Concept 5: What Is a Multiset?

A multiset is like a set, but it allows repeated elements. The multiset {3, 5, 5, 7} has four elements, with 5 appearing twice. Unlike a regular set (where {3, 5, 7} and {5, 3, 7} are the same), a multiset is also unordered but keeps track of how many times each element appears.

## The Proofs — Step by Step

### Lemma 1: The Quantile Lemma (Foundation of Everything)

**Statement:** If V₁, ..., V_{n+1} are exchangeable, then for any β ∈ (0,1):

$$P\big(V_{n+1} \leq \text{Quantile}(\beta;\; V_{1:n} \cup \{\infty\})\big) \geq \beta$$

**What this says in plain English:** If you have n+1 exchangeable random variables, and you compute the β-quantile of the first n values plus infinity, then the (n+1)-th value falls below this quantile with probability at least β.

**Proof intuition (building it from scratch):**

*Step 1: Understand the ranking argument.*

Since V₁, ..., V_{n+1} are exchangeable, they are "statistically identical" — no variable is special. So V_{n+1}'s rank among all n+1 variables is uniformly distributed. That is:

$$P(\text{V_{n+1} has rank } k) = \frac{1}{n+1} \quad \text{for each } k = 1, 2, ..., n+1$$

(assuming no ties for simplicity)

*Step 2: Count how many ranks lead to the event.*

The event "V_{n+1} ≤ Quantile(β; V_{1:n} ∪ {∞})" happens when V_{n+1} is not too large relative to the other values. How many of the n+1 possible rank positions for V_{n+1} would make this event true?

The quantile is computed over n values plus infinity, so it's the β-quantile of n+1 values (V₁, ..., V_n, and ∞). The number of values at or below this quantile is at least ⌈β(n+1)⌉.

Since V_{n+1} has equal probability of being in any rank, and at least ⌈β(n+1)⌉ out of n+1 ranks satisfy the event:

$$P\big(V_{n+1} \leq \text{Quantile}(\beta; V_{1:n} \cup \{\infty\})\big) \geq \frac{\lceil\beta(n+1)\rceil}{n+1} \geq \beta$$

*Step 3: The upper bound.*

If there are no ties (all values are distinct), at most ⌈β(n+1)⌉ ranks satisfy the event, giving:

$$P(\text{event}) \leq \beta + \frac{1}{n+1}$$

The 1/(n+1) term is the "discretization error" from working with finitely many values.

**Why add infinity?** Without the infinity term, we'd be looking at the quantile of just V₁, ..., V_n. Adding infinity ensures the quantile is large enough — it's a conservative adjustment that guarantees coverage. Think of it as padding: you're making the prediction interval slightly larger than necessary to be safe.

### Theorem 1: Standard Conformal Prediction Coverage

**Statement:** Under exchangeability, the conformal band:

$$\hat{C}_n(x) = \{y \in \mathbb{R} : V_{n+1}^{(x,y)} \leq \text{Quantile}(1-\alpha; V_{1:n}^{(x,y)} \cup \{\infty\})\}$$

satisfies P(Y_{n+1} ∈ Ĉ_n(X_{n+1})) ≥ 1 - α.

**Proof (conceptual reconstruction):**

1. Fix the test point to be (X_{n+1}, Y_{n+1})
2. The scores V₁^{(X_{n+1}, Y_{n+1})}, ..., V_{n+1}^{(X_{n+1}, Y_{n+1})} are computed symmetrically — each score S(Z_i, Z_{1:n+1}) treats all data points the same way
3. Since Z₁, ..., Z_{n+1} are exchangeable, and the scores are computed symmetrically, the scores are also exchangeable
4. Apply Lemma 1 with β = 1-α: V_{n+1} is below the (1-α)-quantile with probability ≥ 1-α
5. By definition of Ĉ_n, this means Y_{n+1} is in the prediction set with probability ≥ 1-α

**The beauty:** The guarantee holds for ANY score function S and ANY distribution P. The only requirement is exchangeability.

### Lemma 3: The Weighted Quantile Lemma (The New Result)

**Statement:** If Z₁, ..., Z_{n+1} are weighted exchangeable with weight functions w₁, ..., w_{n+1}, and V_i = S(Z_i, Z_{1:(n+1)}), then define weights:

$$p_i^w(z_1, ..., z_{n+1}) = \frac{\sum_{\sigma: \sigma(n+1) = i} \prod_{j=1}^{n+1} w_j(z_{\sigma(j)})}{\sum_\sigma \prod_{j=1}^{n+1} w_j(z_{\sigma(j)})}$$

Then for any β ∈ (0,1):

$$P\Big(V_{n+1} \leq \text{Quantile}\big(\beta;\; \sum_{i=1}^n p_i^w \delta_{V_i} + p_{n+1}^w \delta_\infty\big)\Big) \geq \beta$$

**Okay, this looks scary. Let me break it down completely.**

**What are these weights p_i^w?**

The formula involves summing over **permutations**. A permutation σ of {1, ..., n+1} is a rearrangement. For n+1 = 3, the permutations of {1,2,3} are: (1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1).

The weight p_i^w is the "weighted probability that position n+1 gets the value z_i." Here's the intuition:

Imagine you're randomly assigning the values z₁, ..., z_{n+1} to positions 1, ..., n+1, but NOT uniformly — with probability proportional to ∏ᵢ wᵢ(z_{σ(i)}). The weight p_i^w is the probability that value z_i ends up in position n+1 under this weighted random assignment.

**Simplified form for the covariate shift case:**

When w₁ = ... = w_n ≡ 1 (training weights are all 1) and w_{n+1}(z) = w(x) (test weight depends only on x), the complicated permutation formula simplifies dramatically to:

$$p_i^w(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}, \quad i = 1, ..., n$$

$$p_{n+1}^w(x) = \frac{w(x)}{\sum_{j=1}^n w(X_j) + w(x)}$$

**This is just a normalized version of the likelihood ratios!** Each training point gets weight proportional to how likely its X value is under the test distribution.

**Why does this simplification happen?**

Because most permutations are "equivalent" when all training weights are 1. The only thing that matters is which value z_i gets assigned to the test position (position n+1). The w_{n+1} weight "picks" which value should be the test point, and it prefers values whose x-component has high likelihood ratio w(x).

**Proof intuition for Lemma 3:**

1. By weighted exchangeability, the joint density is f(v₁, ..., v_{n+1}) = ∏ wᵢ(vᵢ) · g(v₁, ..., v_{n+1})
2. Consider randomly permuting the indices. Under the weighted model, permutation σ has probability proportional to ∏ wⱼ(z_{σ(j)})
3. The probability that V_{n+1} (the value at the last position) equals zᵢ is exactly p_i^w
4. Now, p_i^w is the probability mass assigned to score V_i in the weighted distribution
5. The quantile of this weighted distribution is set so that the total mass below it is β
6. Since V_{n+1} = V_i with probability p_i^w, and the mass below the quantile is at least β, we get:

$$P(V_{n+1} \leq \text{Quantile}) = \sum_{i: V_i \leq \text{Quantile}} p_i^w + p_{n+1}^w \cdot \mathbb{1}[\infty \leq \text{Quantile}] \geq \beta$$

The infinity term means the quantile has to be above all the V_i values to "use up" p_{n+1}^w's mass, which only helps the inequality.

### Theorem 2: Weighted Conformal Prediction (The Main Result)

**Statement:** Under weighted exchangeability, the weighted conformal band:

$$\hat{C}_n(x) = \Big\{y : V_{n+1}^{(x,y)} \leq \text{Quantile}\big(1-\alpha;\; \sum_{i=1}^n p_i^w(Z_1, ..., Z_n, (x,y))\delta_{V_i^{(x,y)}} + p_{n+1}^w \delta_\infty\big)\Big\}$$

satisfies P(Y_{n+1} ∈ Ĉ_n(X_{n+1})) ≥ 1 - α.

**Proof:** Follows directly from Lemma 3, applied with β = 1-α, in exactly the same way Theorem 1 follows from Lemma 1.

### Corollary 1: The Covariate Shift Result

**Statement:** Under the covariate shift model (training from P, test from P̃, with dP̃_X/dP_X = w), the weighted conformal band with weights:

$$p_i^w(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}$$

achieves P(Y_{n+1} ∈ Ĉ_n(X_{n+1})) ≥ 1 - α.

**Proof:** By Lemma 2, covariate shift data is weighted exchangeable. Apply Theorem 2.

### Important Remark 3: You Only Need w Up to a Constant

Since the weights appear as ratios in the formula for p_i^w:

$$p_i^w(x) = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}$$

if you replace w with c·w (multiply by any constant c), the c cancels:

$$\frac{c \cdot w(X_i)}{\sum_{j=1}^n c \cdot w(X_j) + c \cdot w(x)} = \frac{w(X_i)}{\sum_{j=1}^n w(X_j) + w(x)}$$

**This is very useful practically.** You don't need to know the exact density ratio — just something proportional to it. This means you can use the odds ratio p̂(x)/(1-p̂(x)) from a classifier, without worrying about normalization constants.

## Estimating the Likelihood Ratio in Practice

The paper proposes a clever trick for estimating w(x) when it's unknown.

### The Classifier Trick

**Step 1:** Label all training covariates as class C = 0 and all test covariates as class C = 1.

**Step 2:** Train any classifier (logistic regression, random forest, etc.) to predict P(C = 1 | X = x).

**Step 3:** Use Bayes' theorem:

$$\frac{P(C=1|X=x)}{P(C=0|X=x)} = \frac{P(C=1)}{P(C=0)} \cdot \frac{d\tilde{P}_X}{dP_X}(x)$$

The left side is the odds ratio (which we can estimate from the classifier). The right side has the likelihood ratio we want, times a constant P(C=1)/P(C=0). But by Remark 3, we only need the ratio up to a constant!

So:

$$\hat{w}(x) = \frac{\hat{p}(x)}{1 - \hat{p}(x)}$$

where p̂(x) is the classifier's estimated probability that x comes from the test set.

**Why this is clever:** You've turned a density estimation problem (estimate two densities and take their ratio) into a classification problem (train a classifier to distinguish two sets). Classification is much easier than density estimation, especially in moderate dimensions.

### Practical Note: Clipping

When using random forests, the estimated probability p̂(x) can sometimes be exactly 1 (the classifier is 100% sure x is from the test set). This makes ŵ(x) = infinity, causing numerical problems.

**Fix:** Clip p̂(x) to lie between 0.01 and 0.99. This bounds the weights between 0.01/0.99 ≈ 0.01 and 0.99/0.01 = 99.

## The Remark About No Upper Bound for Weighted Case

In standard conformal prediction, we have both:
- **Lower bound:** P(coverage) ≥ 1 - α
- **Upper bound:** P(coverage) ≤ 1 - α + 1/(n+1) (when no ties)

The upper bound means the coverage is tight — it's very close to exactly 1-α.

**For weighted conformal prediction, there's no meaningful upper bound.** The reason: the largest jump in the weighted CDF can be as large as max_i p_i^w, which can be much larger than 1/(n+1) if one weight dominates. This means the coverage could be much larger than 1-α (the prediction sets could be unnecessarily wide).

**Practical implication:** Weighted conformal prediction can be conservative (overcovering), especially when the effective sample size is small.

---

# DISCUSSION: TOWARDS LOCAL CONDITIONAL COVERAGE

One of the most interesting ideas in the paper is using weighted conformal for approximate conditional coverage. Here's how:

**Goal:** Instead of just marginal coverage (averaged over all test points), you want local coverage at a specific point x₀:

$$P\big(Y_{n+1} \in \hat{C}_n(x_0) \mid X_{n+1} = x_0\big) \geq 1 - \alpha$$

**Problem:** Vovk (2012) and Lei & Wasserman (2014) proved this is IMPOSSIBLE to achieve distribution-free with finite-length intervals. You'd need infinitely wide intervals.

**Relaxation:** Instead, ask for coverage in a neighborhood around x₀:

$$\frac{\int P(Y_{n+1} \in \hat{C}_n(x_0) | X_{n+1} = x) \cdot K\big(\frac{x - x_0}{h}\big) dP_X(x)}{\int K\big(\frac{x - x_0}{h}\big) dP_X(x)} \geq 1 - \alpha$$

where K is a kernel (a bump-shaped function that weights nearby points more) and h is a bandwidth (how "wide" the neighborhood is).

**The connection to covariate shift:** This local coverage problem is EXACTLY a covariate shift problem! The "test distribution" is the training distribution P_X reweighted by the kernel K((x - x₀)/h). The likelihood ratio is:

$$w(x) = K\Big(\frac{x - x_0}{h}\Big)$$

So you can directly apply weighted conformal prediction with these kernel weights!

**The catch:** The band depends on the center point x₀. If you change x₀, you need to recompute the entire band. This makes it computationally expensive for simultaneous coverage at many points.

---

# KEY CONCEPTS GLOSSARY

| Concept | Simple Explanation |
|---------|-------------------|
| **Conformal Prediction** | A framework for building prediction intervals with guaranteed coverage, without assuming anything about the data distribution |
| **Exchangeability** | Data points are "statistically equivalent" — their joint distribution doesn't change if you rearrange them |
| **Covariate Shift** | The distribution of inputs X changes between training and test, but the relationship Y\|X stays the same |
| **Likelihood Ratio** | How much more likely a point is under one distribution vs. another: w(x) = P̃(x)/P(x) |
| **Nonconformity Score** | A number measuring how "surprising" a data point is relative to the rest of the data |
| **Weighted Exchangeability** | A generalization of exchangeability where data points can have different "importance weights" |
| **Split Conformal** | A computationally efficient variant that splits the training data into a fitting set and a calibration set |
| **Effective Sample Size** | How many "equivalent" unweighted samples your weighted samples correspond to |
| **Exponential Tilting** | Creating a new distribution by multiplying the density by exp(x^T β), which shifts the distribution toward certain regions |
| **Point Mass (δ_a)** | A distribution that puts all its probability at a single value a |
| **Quantile** | The value below which a specified fraction of the distribution falls |
| **Absolute Continuity** | Distribution P̃ doesn't put probability where P doesn't — ensures the likelihood ratio is well-defined |
| **Importance Sampling** | Using samples from one distribution to estimate properties of another, by reweighting |

---

# CONNECTION TO YOUR DS-SGen PROJECT

## This Paper's Role in the Chain

```
This paper (2019)
    │  Introduces weighted conformal prediction for covariate shift
    │  Theory: coverage guarantee with known likelihood ratios
    │
    ├──► DS-CP (Lin et al., 2025)
    │      Applies weighted CP to LLMs
    │      Innovation: density ratios in embedding space
    │      Handles: conformal prediction sets under domain shift
    │
    ├──► Subpopulation Shift CP (Wang et al., 2025)
    │      Uses similarity-weighted calibration
    │      Handles: conformal prediction under subpopulation shift
    │
    └──► YOUR DS-SGen Project
           Extends SGen's FDR-E control to domain shift
           Uses: importance-weighted conformal for pseudo-labeling
           Uses: reweighted binomial bounds for PAC guarantees
```

## Specific Technical Bridges to Your Work

1. **Weighted conformal for pseudo-labeling:** SGen-Semi uses conformal prediction to pseudo-label whether model outputs are correct. Under domain shift, you need WEIGHTED conformal (from this paper) to ensure the pseudo-labels remain reliable.

2. **The classifier trick for density ratio estimation:** The idea of training a classifier to distinguish training vs. test data and using the odds ratio as the likelihood ratio — this is exactly what DS-CP does (in embedding space) and what you'll likely do in DS-SGen.

3. **Effective sample size considerations:** When your weights are extreme (old and new domains are very different), the effective sample size drops, making all guarantees looser. This directly affects how tight your FDR-E bounds can be.

4. **The proportionality constant doesn't matter (Remark 3):** This is crucial for practical implementations — you don't need perfectly calibrated density ratios, just something proportional.

---

# DEVIL'S ADVOCATE — THREE WEAKEST POINTS

1. **Single dataset evaluation:** The empirical evaluation uses only the Airfoil dataset (1503 points, 5 dimensions). This is very low-dimensional and small by modern standards. It's unclear how well the likelihood ratio estimation works in higher dimensions.

2. **No theory for estimated weights:** The formal guarantee (Corollary 1) assumes the EXACT likelihood ratio is known. When using estimated weights, there's no theorem bounding the coverage loss due to estimation error. The paper only shows empirically that it works.

3. **No discussion of how extreme weights degrade performance:** While the effective sample size is mentioned, there's no formal analysis of what happens when the test distribution is far from the training distribution. In extreme cases, one weight dominates and the method produces trivially wide intervals.

---

# OPEN QUESTIONS AND RESEARCH IDEAS

1. **Can we provide formal guarantees with estimated weights?** What coverage guarantee can we prove when using ŵ instead of w? How does the estimation error propagate to coverage loss?

2. **Can weighted conformal prediction be made adaptive?** Instead of precomputing weights, can the method adapt as it sees more test points?

3. **How to handle the case where the covariate shift assumption itself is wrong?** What if P(Y|X) also changes between training and test?

4. **For your DS-SGen:** Can you bound the FDR-E error term that arises from using estimated importance weights in the SGen framework? This would be a genuine theoretical contribution.