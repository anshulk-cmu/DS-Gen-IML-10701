# Selective Generation for Controllable Language Models — Complete Paper Analysis

**Paper:** Selective Generation for Controllable Language Models
**Authors:** Minjae Lee*, Kyungmin Kim* (POSTECH), Taesoo Kim (GaTech), Sangdon Park (POSTECH)
**Venue:** NeurIPS 2024 (top ML conference)
**Type:** Breakthrough / Theory paper with empirical validation
**Peer-Reviewed:** Yes (NeurIPS 2024)
**Reading Purpose:** Deep understanding — this is a foundational paper for your DS-SGen research project

---

# PHASE -1: PAPER CLASSIFICATION

This paper is a **Breakthrough / Theory** paper with significant empirical validation. It introduces a genuinely new framework (selective generation with entailment-based correctness) backed by formal mathematical guarantees (PAC bounds) and demonstrates it works in practice on real LLMs. This means we should apply the full Pass 1-2-3 framework.

The paper sits at the intersection of three areas: conformal prediction, selective prediction, and natural language generation. For your research project combining conformal prediction with LLM reliability under domain shift, this is THE foundational paper — it's the "vanilla SGen-Sup baseline" in your proposal.

---

# PHASE 0: PRE-READING CONTEXT

**Authors:**
- Minjae Lee and Kyungmin Kim: Graduate students at POSTECH (top Korean university), equal first authors
- Taesoo Kim: Professor at Georgia Tech, well-known in systems security and ML
- Sangdon Park: Professor at POSTECH, expert in conformal prediction and PAC learning — the senior author driving the theoretical contributions

**Venue quality:** NeurIPS is one of the top 3 ML conferences (alongside ICML and ICLR). Acceptance here means the theoretical contributions and experiments passed rigorous peer review.

**Code released:** Yes, at the GitHub link provided. This is a strong reproducibility signal.

**Why this matters for you:** Your research project proposes DS-SGen, which extends this paper's SGen framework to handle domain shift. Understanding this paper deeply is essential — it's the foundation you're building on.

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## The Real-World Problem (No Math Yet)

Imagine you're using ChatGPT to help with medical questions. You ask: "What are the symptoms of appendicitis?" and it gives you an answer. But how do you know if the answer is correct? Language models sometimes "hallucinate" — they generate confident-sounding answers that are completely wrong.

Now imagine this system is used in a hospital. Wrong answers could hurt patients. We need a way to make the model say **"I don't know"** when it's not confident, rather than giving a potentially wrong answer.

This is called **selective prediction** — the model selects when to answer and when to abstain.

## Q1: What Is the Problem Being Solved?

**In simple words:** How do we teach a language model to say "I don't know" at the right times, so that when it DOES give an answer, we have a mathematical guarantee that it's usually correct?

**More precisely:** The paper addresses how to build a "selective generator" — a system that wraps around any language model (like GPT-3.5 or Llama) and decides whether to return the generated answer or say "I don't know" (IDK). The goal is to control the **false discovery rate** (the fraction of returned answers that are wrong) below a user-specified level, with a formal probability guarantee.

**In one sentence:** "This paper studies how to learn a selection function for language models that controls the rate of hallucinated answers with a PAC (probably approximately correct) guarantee, using textual entailment as the correctness metric."

## Q2: Why Is This Problem Hard and Interesting?

There are three key difficulties:

### Difficulty 1: The "Metric Misalignment" Problem

In classification (like spam detection), checking if the model is correct is easy: you compare the predicted label to the true label. Either they match or they don't.

But in language generation, there are **many correct answers** to the same question. For example:

- Question: "Who wrote Romeo and Juliet?"
- True answer: "William Shakespeare wrote Romeo and Juliet."
- Model's answer: "The play Romeo and Juliet was authored by Shakespeare, an English playwright."

These two answers are different strings of text, but they're both correct! If you use "exact match" (do the strings match character by character?), you'd wrongly say the model is wrong. This gap between measured correctness and true correctness is what the authors call **metric misalignment**.

### Difficulty 2: The Label Space Is Infinite

In classification, there are a fixed number of categories (spam/not-spam, cat/dog/bird). In language generation, there are infinitely many possible text sequences. This makes conformal prediction (which constructs "prediction sets") much harder — the prediction set could be infinitely large and therefore useless.

### Difficulty 3: Confidence Calibration Is Terrible for Language Models

For selective prediction to work well, you need a good "confidence score" — a number that tells you how sure the model is. In classification, the softmax probability is a decent (though imperfect) confidence score. For language models, the token-by-token probability is known to be very poorly calibrated — a model can assign high probability to wrong answers and low probability to correct ones.

**In one sentence:** "This is nontrivial because exact-match metrics fail for open-ended text, the label space is infinite (breaking standard conformal prediction), and confidence calibration for language models is unreliable."

## Q3: What Is the Main Claim?

The paper proposes two algorithms:

1. **SGen-Sup** (supervised): Uses human-labeled entailment data to learn when to abstain
2. **SGen-Semi** (semi-supervised): Uses a mix of labeled and unlabeled data, with conformal prediction providing pseudo-labels

Both control the **FDR-E** (False Discovery Rate with respect to textual Entailment):

$$\text{FDR-E} = P(\text{answer is wrong} \mid \text{model chose to answer})$$

The main theoretical guarantee (Theorem 1) says:

$$P\{\text{FDR-E of learned system} \leq \hat{U}\} \geq 1 - \delta$$

**In one sentence:** "They show that SGen-Semi learns a selective generator that controls the entailment-based false discovery rate to a desired level with PAC guarantees, while achieving better selection efficiency than baselines, even with 75% of the labeled data."

---

# PASS 2: THE SCUBA DIVE — How Does It Work?

## Q1: What Was the Main Technical Hurdle Before This Paper?

### The State of the Art Before SGen

Before this paper, there were two approaches to handling hallucination with guarantees:

**Approach A: Conformal prediction for language models.** Papers like "Conformal Language Modeling" (Quach et al., 2024) tried to build prediction sets — sets of possible answers guaranteed to contain the correct one. But for open-ended generation, these sets become absurdly large (because there are infinitely many ways to say the same thing), making the guarantee practically useless.

**Approach B: Selective prediction using exact match.** Geifman & El-Yaniv (2017) proposed selective prediction for classification, controlling the FDR. You could apply this to language generation, but you'd be using exact match as the correctness metric. This is way too conservative — it marks correct answers as wrong just because the wording is different. The result: the system says "I don't know" almost all the time, even when it actually knows the answer.

### The Barrier

The fundamental barrier was: **there was no good way to measure "correctness" for generated text that was both (a) semantically meaningful (not just string matching) and (b) compatible with the mathematical machinery of selective prediction / conformal prediction.**

### How This Paper Overcomes It

The key insight is to use **textual entailment** as the correctness metric. Instead of asking "does the generated answer exactly match the reference answer?", they ask: "does the generated answer logically imply the reference answer?"

This is brilliant because:
- It handles multiple valid phrasings (entailment captures semantic equivalence)
- It's computable (NLI models like DeBERTa can estimate entailment scores)
- It's compatible with the selective prediction framework (you can replace exact match with entailment and the math still works)

## Q2: The Core Technical Machinery — Explained Step by Step

### Building Block 1: What Is Textual Entailment?

Think of it like a logic test between two sentences:

- **Premise (P):** "The Bible mentions Sodom and Gomorrah in the book of Genesis."
- **Hypothesis (H):** "The book of Genesis mentions Sodom and Gomorrah."

Does P imply H? Yes! If the Bible mentions them in Genesis, then Genesis mentions them. This is **entailment**.

The three possible relationships:
- **Entailment:** H is true if P is true (P implies H)
- **Contradiction:** H is false if P is true (P contradicts H)
- **Neutral:** H might or might not be true given P (can't tell)

In this paper:
- P = the model's generated answer G(x)
- H = the true reference answer y (in declarative form)

So the correctness check becomes: **"Does the model's answer entail (logically imply) the true answer?"**

### Building Block 2: The Selective Generator

A selective generator is a system with two parts:

1. **Generator G:** The language model (e.g., GPT-3.5) that produces an answer G(x) given question x
2. **Selection function ŝ:** A function that decides whether to show the answer or say "I don't know"

Mathematically:

$$\hat{S}(x) = \begin{cases} G(x) & \text{if } \hat{s}(x, G(x)) = 1 \quad \text{(show the answer)} \\ \text{IDK} & \text{otherwise} \quad \text{(say "I don't know")} \end{cases}$$

The selection function is typically a threshold on some confidence score:

$$\hat{s}(x, G(x)) = \mathbf{1}(f_M(x, G(x)) \geq \tau_S)$$

This says: "If the confidence score is at least τ_S, show the answer; otherwise, abstain."

### Building Block 3: FDR-E — The Metric We Want to Control

The **False Discovery Rate with respect to Entailment (FDR-E)** is:

$$\text{FDR-E}(\hat{S}) = P\{G(x) \notin E_{\text{true}}(y) \mid \hat{S}(x) \neq \text{IDK}\}$$

Let's unpack this carefully:

- $E_{\text{true}}(y)$ is the "true entailment set" — the set of ALL text sequences that logically entail y
- $G(x) \notin E_{\text{true}}(y)$ means "the model's answer does NOT entail the true answer" (i.e., it's wrong)
- $\hat{S}(x) \neq \text{IDK}$ means "the system decided to show the answer"
- So FDR-E = the probability that when the system shows an answer, that answer is wrong

**Goal:** Learn a selection function so that FDR-E ≤ ε (a user-chosen error tolerance, e.g., 0.25 = "at most 25% of shown answers should be wrong")

### Building Block 4: The PAC Guarantee

PAC stands for "Probably Approximately Correct." The guarantee is:

$$P\{\text{FDR-E}(\hat{S}) \leq \epsilon\} \geq 1 - \delta$$

This means: "With probability at least 1 − δ (over the randomness of the training data), the FDR-E of our learned system will be at most ε."

Think of it as a double safety net:
- **Approximately correct:** The error rate is at most ε (not zero, but bounded)
- **Probably:** This bound holds with probability at least 1 − δ (not always, but almost always)

Typical values: ε = 0.25 (at most 25% errors), δ = 0.02 (this holds 98% of the time)

### Algorithm 1: SGen-Sup (Supervised Version)

This is the simpler algorithm. Here's how it works:

**Input:** A labeled dataset where each example has (question x, true answer y, entailment label e), where e = 1 if G(x) entails y (correct) and e = 0 if not (incorrect).

**Step 1:** Sort all examples by their confidence score f_M(x, G(x))

**Step 2:** Do a binary search over threshold values τ_S. For each candidate threshold:
- Filter to only examples where the confidence score ≥ τ_S (these would be "selected" answers)
- Count how many of these selected answers have e = 0 (wrong answers)
- Use a **binomial tail bound** (a statistical tool) to compute an upper bound on the true FDR-E

**Step 3:** Find the lowest threshold τ_S such that the FDR-E upper bound ≤ ε

**Output:** The threshold τ_S and the FDR-E bound

**The binomial tail bound — what is it?** Imagine you flip a coin n times and get k heads. What's the maximum probability the coin could have for heads? The binomial tail bound gives you a statistically rigorous upper bound on the true probability, given what you observed. It's the same idea here: you observe k wrong answers out of n selected answers, and the binomial tail bound tells you the maximum true error rate could be, with confidence δ.

**Problem with SGen-Sup:** It needs human-annotated entailment labels, which are expensive. For every (question, model answer, true answer) triple, a human must judge whether the model's answer entails the true answer.

### Algorithm 2: SGen-Semi (Semi-Supervised Version) — The Main Contribution

This is the paper's crown jewel. The idea: **use conformal prediction to generate "pseudo-labels" for unlabeled data**, then use both real labels and pseudo-labels to learn the selection function.

Here's the step-by-step intuition:

#### Step 2a: Learn an Entailment Set via Conformal Prediction

We have a small labeled set Z_E (with entailment labels) and a large unlabeled set Z_U (without labels).

First, we learn an "estimated entailment set" Ê(y) using conformal prediction on Z_E:

$$\hat{E}(y) = \{y' \in \mathcal{Y} \mid f_E(y', y) \geq \tau_E\}$$

Here, f_E is an entailment scoring function (from a pre-trained NLI model like DeBERTa). The threshold τ_E is chosen by the conformal prediction algorithm to control the **False Entailment Rate (FER)** — the rate at which we incorrectly call something "entailment" when it's not.

**In plain English:** We build a function that, given a true answer y, tells us which generated answers are "close enough" to count as correct. We calibrate this function using the labeled data so it doesn't make too many false positives (calling wrong answers correct).

#### Step 2b: Pseudo-Label the Unlabeled Data

For each unlabeled example (x, y) in Z_U:
- Compute the entailment score f_E(G(x), y)
- If f_E(G(x), y) ≥ τ_E, pseudo-label as ê = 1 (correct)
- Otherwise, pseudo-label as ê = 0 (wrong)

#### Step 2c: The FDR-E Decomposition — Why This Works

Here's the clever mathematical trick. The FDR-E can be decomposed based on whether data is labeled (v=1) or unlabeled (v=0):

$$\text{FDR-E} = \underbrace{P(v=1|\text{selected})}_{(B)} \cdot \underbrace{P(e=0|v=1, \text{selected})}_{(C)} + \underbrace{P(v=0|\text{selected})}_{(D)} \cdot \underbrace{P(e=0|v=0, \text{selected})}_{(E)}$$

- Term (B) × (C): The contribution from labeled data (handled like SGen-Sup)
- Term (D) × (E): The contribution from unlabeled data (handled via pseudo-labels)

For term (E), the key decomposition (Lemma 1) is:

$$P(e=0|\text{selected, unlabeled}) = \underbrace{P(e=0, \hat{e}=1)}_{FER} - \underbrace{P(e=1, \hat{e}=0)}_{FNER} + \underbrace{P(\hat{e}=0)}_{NER}$$

Where:
- **FER (False Entailment Rate):** We said it was correct (ê=1) but it was actually wrong (e=0). Controlled by conformal prediction on Z_E.
- **FNER (False Non-Entailment Rate):** We said it was wrong (ê=0) but it was actually correct (e=1). Bounded from below using binomial bound on Z_E.
- **NER (Non-Entailment Rate):** Rate of pseudo-labeled "wrong" answers. Bounded from above using binomial bound on Z_U.

Each of these three terms can be bounded separately with high probability, and then combined using a **union bound** (a probability tool that says: if event A fails with probability δ₁ and event B fails with probability δ₂, then at least one failing has probability ≤ δ₁ + δ₂).

#### Step 2d: The Optimization

The algorithm optimizes over:
- The threshold τ_S for the selection function (which answers to show)
- The hyperparameter ε_E for the FER control (how tight to make the pseudo-labels)

It searches for the combination that achieves FDR-E ≤ ε with the highest "selection efficiency" (showing as many answers as possible).

### The Neuro-Selection Function — A Better Selection Function

Standard selective prediction uses a single threshold on a single confidence score:

$$\hat{s}(x) = \mathbf{1}(f_{M_1}(x, G(x)) \geq \tau_S)$$

But what if your confidence score is poorly calibrated? You'll end up being too conservative (saying "IDK" too much) or too aggressive (showing too many wrong answers).

The paper proposes **neuro-selection functions** — using multiple thresholds on multiple confidence scores:

$$\hat{s}(x) = \mathbf{1}(f_{M_1}(x, G(x)) \geq \tau_{S,1}) \wedge \mathbf{1}(f_{M_2}(x, G(x)) \geq \tau_{S,2})$$

Where:
- $f_{M_1}$: The conditional probability of the answer (standard LLM confidence)
- $f_{M_2}$: The self-consistency score (generate multiple answers, check how much they agree using entailment)

The system tries three configurations:
1. Single threshold on f_M1 alone
2. Single threshold on f_M2 alone
3. Double threshold using both f_M1 AND f_M2

It picks whichever configuration achieves the FDR-E guarantee with the highest efficiency.

### Lemma 4: Why Calibration Matters (The Theoretical Insight)

Lemma 4 proves that if the confidence score f_M is **perfectly calibrated** with respect to the entailment relation — meaning P(G(x) entails y | f_M(x,G(x)) = t) = t for all t — then raising the threshold τ_S always decreases the FDR-E.

**In plain English:** If your confidence score is honest (a score of 0.8 really means 80% chance of being correct), then you can always trade off between answering fewer questions (higher threshold) and having a lower error rate. This monotonicity is what makes the binary search in the algorithm work.

**Why this matters:** In practice, LLM confidence scores are NOT perfectly calibrated, which is exactly why the neuro-selection function (using multiple scores) helps — it can compensate for miscalibration by combining information from multiple sources.

## Q2: What Is the Simplest Baseline and How Much Better Is SGen?

### Baselines

1. **SGen-EM (exact match):** The unsupervised baseline from Geifman & El-Yaniv (2017), applied to language generation using exact match as correctness. This is way too conservative because EM marks most correct answers as wrong.

2. **SGen-H-Semi-PL (heuristic pseudo-labeling):** Uses a fixed threshold to pseudo-label, with no guarantee on pseudo-label quality.

3. **SGen-H-Semi-PFL (heuristic with filtering):** Same as above but filters out uncertain pseudo-labels.

### Results Summary (Table 1)

For GPT-3.5-Turbo with ε = 0.25 (desired FDR-E ≤ 25%):

| Method | FDR-E | Efficiency | Satisfies Guarantee? |
|--------|-------|------------|---------------------|
| SGen-EM | 0.134 | 0.549 | No (based on wrong metric) |
| SGen-Semi (theirs) | 0.159 | 0.733 | Yes |
| SGen-H-Semi-PL | 0.096 | 0.419 | No formal guarantee |

**Key observations:**
- SGen-Semi achieves 73.3% efficiency (shows 73.3% of answers) while keeping FDR-E at 15.9% — well below the 25% target
- SGen-EM only achieves 54.9% efficiency — it rejects way too many correct answers because exact match is too strict
- The heuristic methods have no formal guarantees — they might fail in ways you can't predict

### Why Entailment Labels Matter (Table 2)

The paper shows concrete examples where entailment helps:

- Question: "Who plays Draco Malfoy?"
- True answer: "Thomas Andrew Felton plays Draco Malfoy"
- Generated answer: "The actor who plays Draco Malfoy is Tom Felton"

SGen-EM rejects this because "Tom Felton" ≠ "Thomas Andrew Felton" (not exact match). But SGen-Semi accepts it because "Tom Felton" entails "Thomas Andrew Felton" (they refer to the same person).

### Semi-Supervised vs. Fully Supervised

Fully supervised methods (using 100% labeled data) achieve efficiency of 0.754 on GPT-3.5. SGen-Semi achieves 0.733 using only 75% of the labeled data plus cheap unlabeled data. This is remarkable — you get almost the same performance with 25% fewer expensive labels.

## Q3: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: The i.i.d. Assumption

The entire theoretical framework assumes that training and test data come from the same distribution (i.i.d.). If the distribution shifts — for example, you train on Natural Questions but deploy on medical QA — the guarantees may not hold.

**This is exactly the gap your research project targets!** DS-SGen would extend this framework to handle domain shift.

### Limitation 2: The FDR-E Bound Can Be Loose

On Alpaca-7B, the empirical FDR-E is much lower than ε = 0.25 (around 0.07), meaning the bound is quite conservative. The system says "I don't know" more than necessary because the mathematical bound is not tight.

### Limitation 3: Expensive Entailment Labels

Even though the semi-supervised method reduces the need for labels, it still requires some human-annotated entailment labels. The paper created a new dataset for this purpose.

### Limitation 4: Simple Selection Functions

The neuro-selection function only uses two scoring functions with a linear combination. More complex selection functions (e.g., neural networks) might achieve better efficiency, but the theory doesn't support them yet.

## Q4: Does This Insight Apply to Other Problems?

### Connection 1: Your DS-SGen Research Project

The SGen framework is the foundation for your proposed DS-SGen method. Your insight is to integrate importance reweighting (from domain shift literature) into the SGen-Semi framework. Specifically:
- SGen assumes i.i.d. data
- You want to handle domain shift (e.g., train on Natural Questions, test on SciQ)
- The DS-CP (domain-shift conformal prediction) literature provides importance weights
- You could potentially reweight the binomial tail bounds in SGen-Semi to account for distribution shift

### Connection 2: Any Structured Output Task

The entailment-based approach could extend to other structured output tasks beyond QA: summarization (does the summary entail key facts?), translation (does the translation entail the meaning?), code generation (does the code satisfy the specification?).

### Connection 3: Active Learning

The semi-supervised framework suggests a natural active learning extension: which unlabeled examples would benefit most from human entailment labels?

## Q5: Caveats and Takeaways

### Strengths
- Clean theoretical framework with rigorous PAC guarantees
- Practical and works on real LLMs (both open and closed source)
- The semi-supervised method is genuinely useful — reduces labeling costs
- The entailment-based correctness metric is a real conceptual advance

### Weaknesses
- Only tested on one dataset (Natural Questions) and two models
- The entailment model (DeBERTa) itself could make errors — the paper doesn't deeply analyze how NLI model quality affects the guarantees
- The i.i.d. assumption limits real-world applicability
- The FDR-E bounds are somewhat loose (especially on weaker models)

---

# PASS 3: THE SWAMP — Deep Dive into the Mathematical Machinery

## Proof Architecture Overview

The paper's theoretical contribution has a layered structure. Let me walk through each layer.

### Layer 1: The PAC Conformal Prediction Algorithm (Theorem 2)

This is the foundational building block. It says: given n i.i.d. samples, the algorithm A_Binom returns a conformal set Ĉ such that:

$$P\{R_{01}(\hat{C}) \leq \epsilon\} \geq 1 - \delta$$

**How the algorithm works:**

1. You have a scoring function f(x, y) and you want to build a prediction set C(x) = {y : f(x,y) ≥ τ}
2. For each candidate threshold τ, count how many calibration examples are "miscovered" (true label not in the set): $k_\tau = \sum_{i=1}^{n} \mathbf{1}(y_i \notin \hat{C}(x_i))$
3. Use the binomial tail bound to compute an upper bound on the true miscoverage rate: $U_{\text{Binom}}(k_\tau; n, \delta)$
4. Find the largest τ (tightest set) such that this bound ≤ ε

**Why the largest τ?** Larger τ means a smaller prediction set (fewer elements pass the threshold). Smaller sets are more informative. So we want the tightest set that still has the coverage guarantee.

**The binomial tail bound, explained:** If you observe k failures out of n trials, the upper binomial tail bound gives you the maximum true failure probability θ such that observing k or fewer failures would happen with probability at least δ.

Formally: $U_{\text{Binom}}(k; n, \delta) = \inf\{\theta \in [0,1] \mid F(k; n, \theta) \leq \delta\} \cup \{1\}$

where F(k; n, θ) is the CDF of the Binomial(n, θ) distribution.

**Intuition:** If you flip a coin 100 times and get 10 heads, the binomial tail bound with δ = 0.05 might tell you the true probability of heads is at most 0.16. This is a conservative but rigorous upper bound.

**The proof of Theorem 2 (Appendix F):**

The key proof idea is a "switching argument":

1. If the algorithm outputs τ̂, then by construction, $U_{\text{Binom}}(k_{\hat{\tau}}; n, \delta) \leq \epsilon$
2. We want to show the true risk $R_{01}(\hat{C})$ is also ≤ ε
3. Suppose for contradiction that $R_{01}(\hat{C}) > \epsilon$. Then there exists some τ* (the infimum of thresholds with risk > ε) such that $R_{01}(C_{\tau^*}) > \epsilon$
4. But the algorithm found $U_{\text{Binom}}(k_{\tau^*}; n, \delta) \leq \epsilon$, which means the bound says the risk should be ≤ ε
5. For $R_{01}(C_{\tau^*}) > U_{\text{Binom}}(k_{\tau^*}; n, \delta)$ to happen, the binomial tail bound must fail
6. By definition, the binomial tail bound fails with probability at most δ

This gives us $P\{R_{01}(\hat{C}) > \epsilon\} \leq \delta$, completing the proof.

**Key insight for the proof:** The monotonicity of the indicator loss with respect to τ is crucial. As τ increases, the prediction set shrinks, so the miscoverage can only increase. This monotonicity allows the "switching" from the algorithm's threshold to the boundary threshold τ*.

### Layer 2: The FDR-E Decomposition (Lemma 1)

This is the algebraic foundation for the semi-supervised method. It decomposes the unknown FDR-E into three terms that can each be bounded separately.

Starting from:

$$P_{D_{\hat{S}}}(e=0) = P_{D_{\hat{S}}}(e=0, \hat{e}=1) - P_{D_{\hat{S}}}(e=1, \hat{e}=0) + P_{D_{\hat{S}}}(\hat{e}=0)$$

**Why is this true?** It's just algebra using the law of total probability:

$$P(e=0) = P(e=0, \hat{e}=1) + P(e=0, \hat{e}=0)$$

And:

$$P(\hat{e}=0) = P(e=0, \hat{e}=0) + P(e=1, \hat{e}=0)$$

So:

$$P(e=0, \hat{e}=0) = P(\hat{e}=0) - P(e=1, \hat{e}=0)$$

Substituting:

$$P(e=0) = P(e=0, \hat{e}=1) + P(\hat{e}=0) - P(e=1, \hat{e}=0)$$

Which is exactly FER − FNER + NER.

**Why this decomposition is useful:** FER can be controlled by conformal prediction (using labeled data), FNER can be lower-bounded using binomial bounds (using labeled data), and NER can be upper-bounded using binomial bounds (using unlabeled data). This allows us to combine labeled and unlabeled data rigorously.

### Layer 3: The Semi-Supervised FDR-E Bound (Lemma 2)

This is where everything comes together. The bound states:

$$P_D\{e=0\} \leq \epsilon_E - L_{\text{Binom}}(\hat{k}; |\hat{Z}_E|, \delta'_E/2) + U_{\text{Binom}}(\hat{l}; |\hat{Z}_U|, \delta'_S) =: U_{\text{SSL}}$$

with probability at least $1 - \delta'_E - \delta'_S$.

**The three terms:**

1. **ε_E (FER bound):** The conformal prediction guarantees $P(e=0, \hat{e}=1) \leq \epsilon_E$ with high probability. This comes from applying A_Binom to learn the entailment set.

2. **$-L_{\text{Binom}}(\hat{k}; |\hat{Z}_E|, \delta'_E/2)$ (FNER lower bound):** We observe $\hat{k}$ examples in Z_E where e=1 but ê=0 (false non-entailments). The lower binomial tail bound gives us a lower bound on the true FNER. Since this term appears with a negative sign in the FDR-E decomposition, a lower bound on FNER gives us an upper bound on FDR-E. This is good for us.

3. **$U_{\text{Binom}}(\hat{l}; |\hat{Z}_U|, \delta'_S)$ (NER upper bound):** We observe $\hat{l}$ examples in Z_U where G(x) ∉ Ê(y) (pseudo-labeled as wrong). The upper binomial tail bound gives an upper bound on the true NER.

**The marginalization trick (key proof technique):**

A subtle but important technical point: the binomial tail bounds hold conditionally on the selected set Ẑ_E (the subset of Z_E where the model chose to answer). But the selected set depends on the selection function, which depends on the data. The proof handles this through marginalization:

$$P_{Z_E}\{R_{\text{FER}} \leq \epsilon_E\} = \sum_{m=1}^{n_E} P_{Z_E}\{R_{\text{FER}} \leq \epsilon_E \mid |\hat{Z}_E| = m\} \cdot P_{Z_E}\{|\hat{Z}_E| = m\}$$

Since the PAC guarantee holds for ANY number of samples (even m = 0, where the conformal set defaults to the entire space Y), each conditional probability is at least 1 − δ'_E/2. This gives the unconditional guarantee.

### Layer 4: The Union Bound Assembly (Theorem 1)

The final theorem assembles all the pieces. We need bounds on four terms from the FDR-E decomposition (Equation 2):

- (B): $P(v=1 | \text{selected})$ — proportion of labeled data among selected examples
- (C): $P(e=0 | v=1, \text{selected})$ — FDR-E on labeled data
- (D): $P(v=0 | \text{selected})$ — proportion of unlabeled data among selected examples
- (E): $P(e=0 | v=0, \text{selected})$ — FDR-E on unlabeled data

Each of (B), (C), (D) is bounded using simple binomial tail bounds. Term (E) is bounded using the SSL bound from Lemma 2.

The final bound is:

$$\text{FDR-E} \leq w_{\text{SL}} \cdot U_{\text{SL}} + w_{\text{SSL}} \cdot U_{\text{SSL}}^{\text{OPT}}$$

where w_SL and w_SSL are upper bounds on the proportions of labeled and unlabeled data, U_SL is the supervised FDR-E bound, and U_SSL^OPT is the optimized semi-supervised bound.

The union bound across all components gives us confidence δ = δ_W + δ_S + δ_E, where:
- δ_W: confidence for the proportion bounds (B) and (D)
- δ_S: confidence for the supervised FDR-E (C) and the NER
- δ_E: confidence for the FER and FNER

**The model selection union bound (Section I):** Since the neuro-selection function tries |H| different configurations, the proof takes a union bound over all of them:

$$P\{R_E(\hat{S}) > \hat{U}\} \leq \sum_{i=1}^{|H|} P\{R_E(S_i) > U_i\} \leq |H| \cdot \frac{\delta_E + \delta_S + \delta_W}{|H|} = \delta$$

This is why the individual confidence levels are divided by |H| in the algorithm.

### Layer 5: The Calibration Lemma (Lemma 4)

This is an elegant theoretical result. Under perfect calibration:

$$P\{G(x) \in E_{\text{true}}(y) \mid f_M(x, G(x)) = t\} = t$$

The proof shows that 1 − FDR-E = P(G(x) ∈ E_true(y) | f_M(x,G(x)) ≥ τ_S) is monotonically non-decreasing in τ_S.

**Proof idea:** Using the definition of conditional expectation and the calibration assumption:

$$P(\text{correct} \mid f_M \geq \tau_S) = \frac{\int_{\tau_S}^{1} t \cdot h(t) dt}{\int_{\tau_S}^{1} h(t) dt}$$

where h(t) = P(f_M = t). The derivative with respect to τ_S can be computed and shown to be ≥ 0 using the fact that the weighted average of t over [τ_S, 1] is always ≥ τ_S.

**Practical implication:** If your confidence score is well-calibrated, raising the threshold always helps. If it's poorly calibrated, you might get non-monotonic behavior, which is why the binary search might not find the optimal threshold. This motivates using multiple scoring functions (the neuro-selection approach).

## Techniques You Can Borrow for Your Research

### Technique 1: The Marginalization Trick for Conditional Guarantees

The trick of marginalizing over the size of the selected set to convert conditional guarantees to unconditional ones is directly applicable to your DS-SGen framework. When you apply importance weights under domain shift, you'll need similar marginalization arguments.

### Technique 2: The FDR-E Decomposition Pattern

The decomposition into FER, FNER, NER terms gives you a template for handling pseudo-labels under domain shift. In your DS-SGen, the importance weights would modify the binomial tail bounds, and you'd need an analogous decomposition.

### Technique 3: The ε_E Optimization (Algorithm 3)

The idea of optimizing over a hyperparameter grid (Q candidates of ε_E) and paying a union bound penalty is a general technique. In your work, you might optimize over the bandwidth of the importance weight estimator.

### Technique 4: Union Bound over Model Selection

The neuro-selection framework's approach of trying multiple selection function configurations and taking a union bound is directly applicable. In DS-SGen, you might want to try multiple reweighting strategies and select the best one while maintaining the overall PAC guarantee.

---

# KEY CONCEPTS GLOSSARY (for Grade 12 Level)

| Concept | Simple Explanation |
|---------|-------------------|
| **Hallucination** | When an AI generates confident-sounding text that is factually wrong |
| **Selective prediction** | A system that can say "I don't know" instead of always answering |
| **FDR (False Discovery Rate)** | The fraction of returned answers that turn out to be wrong |
| **Textual entailment** | Whether one sentence logically implies another |
| **Conformal prediction** | A statistical method to build prediction sets with guaranteed coverage |
| **PAC guarantee** | "Probably Approximately Correct" — the guarantee holds most of the time (probably) and the error is bounded (approximately correct) |
| **Selection efficiency** | The fraction of questions the system actually answers (doesn't say IDK) |
| **Pseudo-labeling** | Using a model to automatically label unlabeled data |
| **Binomial tail bound** | A statistical tool to bound the true probability of an event based on observed data |
| **Calibration** | Whether a model's confidence scores match its actual accuracy |
| **i.i.d. assumption** | Data points are independent and come from the same distribution |
| **Union bound** | If A happens with probability ≤ p₁ and B with ≤ p₂, then A or B happens with probability ≤ p₁ + p₂ |
| **Neuro-selection function** | A selection function that uses multiple confidence scores and multiple thresholds |

---

# CONNECTION TO YOUR DS-SGen RESEARCH PROJECT

## What SGen Does (This Paper)
- Controls FDR-E under the **i.i.d. assumption**
- Uses conformal prediction for pseudo-labeling entailment
- Proposes neuro-selection functions for better efficiency

## What SGen DOESN'T Handle (Your Research Gap)
- **Domain shift:** What happens when training data is from one domain (e.g., general knowledge QA) and test data is from another domain (e.g., medical QA)?
- Under domain shift, the i.i.d. assumption breaks, and ALL the PAC guarantees in this paper become invalid
- The binomial tail bounds assume i.i.d. samples — under domain shift, they could be arbitrarily wrong

## What Your DS-SGen Proposes
- Integrate importance reweighting into the SGen framework
- Use embedding-based domain shift detection to estimate importance weights
- Replace the standard binomial bounds with weighted versions that account for covariate shift
- Maintain formal guarantees under the covariate shift assumption (P(X) changes but P(Y|X) stays the same)

## Specific Technical Bridges

1. **Theorem 1 of SGen → Your modified theorem:** You need to replace each U_Binom bound with a weighted version (e.g., using Tibshirani et al.'s weighted conformal prediction)

2. **The FDR-E decomposition (Lemma 1) → Your decomposition:** The decomposition itself is distribution-agnostic (it's just algebra), so it still holds under domain shift. What changes is how you bound each term.

3. **The entailment set learning (Algorithm 1) → Your reweighted version:** The conformal set learning algorithm needs to use importance-weighted samples instead of uniform samples.

4. **The neuro-selection function → Your domain-aware selection:** You might add a third scoring function that captures domain similarity as an additional signal for the selection function.

---

# SUMMARY: ONE-PAGE CHEAT SHEET

**Problem:** LLMs hallucinate. We want them to say "I don't know" when uncertain, with guarantees.

**Key Innovation:** Use textual entailment (not exact match) to judge correctness → define FDR-E.

**Two Algorithms:**
- SGen-Sup: Supervised, needs entailment labels, direct modification of selective prediction
- SGen-Semi: Semi-supervised, uses conformal prediction to pseudo-label, needs fewer labels

**Theoretical Guarantee:** P{FDR-E ≤ ε} ≥ 1 − δ (PAC guarantee)

**How SGen-Semi Works:**
1. Learn entailment set via conformal prediction (controls pseudo-label quality)
2. Pseudo-label unlabeled data
3. Decompose FDR-E into FER, FNER, NER (each boundable separately)
4. Combine bounds via union bound
5. Optimize selection threshold to maximize efficiency while maintaining guarantee

**Neuro-Selection:** Use multiple confidence scores + multiple thresholds → better efficiency.

**Key Result:** SGen-Semi achieves 73.3% efficiency at FDR-E ≤ 25% on GPT-3.5, using only 75% of labels.

**Main Limitation:** Assumes i.i.d. data → your DS-SGen extends this to domain shift.