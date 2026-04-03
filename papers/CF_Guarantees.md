# Language Models with Conformal Factuality Guarantees — Complete Paper Analysis

**Paper:** Language Models with Conformal Factuality Guarantees
**Authors:** Christopher Mohri and Tatsunori Hashimoto (Stanford University, Department of Computer Science)
**Venue:** arXiv:2402.10978v1, February 2024
**Type:** Breakthrough / Theory paper with practical implementation and empirical validation
**Peer-Reviewed:** No (arXiv preprint), but from a top lab (Stanford NLP) with code released
**Reading Purpose:** Deep understanding — all three passes

---

# PHASE -1: PAPER CLASSIFICATION

This paper is a **Breakthrough / Theory** paper with a strong practical component. It introduces a genuinely new conceptual bridge — connecting conformal prediction (a statistical guarantee method) with language model factuality through entailment sets. It backs this up with a clean theorem (Theorem 4.1) and shows it works in practice on GPT-4 outputs across three different tasks.

What makes it a "breakthrough" rather than just "empirical" is the **conceptual innovation**: the idea that every language model output *implicitly defines* an uncertainty set through the entailment relation, and that this makes conformal prediction practical for language models for the first time. Before this paper, conformal prediction for open-ended language generation was considered essentially intractable.

This paper sits alongside the SGen paper (NeurIPS 2024) and the DS-CP paper as one of the key works connecting conformal prediction with LLM reliability. While SGen focuses on selective generation (deciding when to answer vs. abstain) and DS-CP focuses on domain shift, this paper focuses on **making outputs less specific** to guarantee they're correct.

---

# PHASE 0: PRE-READING CONTEXT

**Authors:**
- **Christopher Mohri:** PhD student at Stanford CS. Has previous work on learning to reject (deferral/abstention), which is a related topic. The name suggests a connection to Mehryar Mohri (a famous ML theorist at NYU/Google), which would indicate strong theoretical training.
- **Tatsunori Hashimoto:** Assistant Professor at Stanford CS, a rising star in the NLP/ML community. Known for work on language model evaluation, factuality, and distribution robustness. He leads a very productive lab — this is a strong credibility signal.

**Venue quality:** This is an arXiv preprint, so it hasn't gone through formal peer review at a conference. However, coming from Stanford NLP with Hashimoto as senior author provides significant credibility. The work has been presented at workshops and is well-cited.

**Code released:** Yes, at https://github.com/tatsu-lab/conformal-factual-lm. This is an excellent reproducibility signal — the authors are confident enough in their results to release code.

**Why this matters for your research:** This paper offers a *different* approach to the same high-level problem as SGen: making LLM outputs reliable with statistical guarantees. While SGen decides *whether* to answer (selective prediction), this paper decides *how much* to answer (backing off to less specific claims). Understanding both approaches gives you a complete picture of the landscape your DS-SGen project sits in.

---

# PASS 1: THE JIGSAW PUZZLE — What Does This Paper Do?

## The Real-World Problem (No Math Yet)

Imagine you ask ChatGPT: "Tell me about Abraham Lincoln." It responds:

> "Abraham Lincoln, born in Idaho, was the 16th President of the United States. He is best known for leading the country through the Civil War."

The problem? Lincoln was NOT born in Idaho — he was born in Kentucky. But the rest of the answer is correct! The model confidently stated something false alongside things that are true. This is called **hallucination**.

Now imagine you're using this AI in a hospital or a courtroom. A single wrong fact mixed in with correct ones could be dangerous. What do you do?

**Option 1 (The SGen approach):** Decide the whole answer might be unreliable, and say "I don't know" instead. This is safe but wasteful — you threw away the correct parts too.

**Option 2 (THIS paper's approach):** Keep the parts you're confident about and *remove* the parts you're uncertain about. So the answer becomes:

> "Abraham Lincoln was the 16th President of the United States."

This is less detailed, but **everything that remains is correct**. You've "backed off" to a less specific but more reliable answer. And the paper provides a mathematical guarantee that this process works.

**That's the core idea:** Instead of throwing away the whole answer, surgically remove the uncertain parts, and provide a mathematical guarantee that what remains is factually correct with high probability.

## Q1: What Is the Problem Being Solved?

**In the simplest words:** How do we take a language model's output (which might contain some false claims mixed with true ones) and automatically remove the uncertain parts, so that what's left is correct with a guaranteed probability — like 90% or 95%?

**More precisely:** Given any black-box language model (like GPT-4), the paper wants to build a system that:
1. Takes the model's original output
2. Breaks it into individual claims ("sub-claims")
3. Scores each claim by how confident the model is about it
4. Removes claims starting from the least confident ones
5. Chooses a cutoff point using a statistical method called "conformal prediction"
6. Guarantees that the remaining output is correct with probability at least 1 − α (where α is a user-chosen error tolerance)

**In one sentence:** "This paper studies how to use conformal prediction with entailment-based uncertainty sets to provide high-probability correctness guarantees on language model outputs, by progressively removing uncertain sub-claims from the output."

## Q2: Why Is This Problem Hard and Interesting?

There are three key difficulties that made this problem seem unsolvable before this paper:

### Difficulty 1: Conformal Prediction Seems Impossible for Language Models

Let's first understand what conformal prediction normally does. In a classification problem (like "is this email spam or not?"), conformal prediction builds a **confidence set** — a set of possible answers guaranteed to contain the true answer with high probability.

For example, instead of saying "this is spam," it might say "the answer is in the set {spam, not-spam}" — guaranteeing the true answer is in there at least 90% of the time. If it's more confident, it might say just {spam}.

Now try this with language models. The "output space" is ALL possible text sequences — infinitely many of them! Building a confidence set means listing out potentially billions or trillions of possible text sequences. This is completely impractical. You can't show a user a set containing millions of possible paragraphs and say "the correct answer is somewhere in here."

**In everyday terms:** Imagine trying to list every possible correct way to describe Abraham Lincoln. There are essentially infinite ways to phrase it. A "confidence set" containing all these phrasings would be useless.

### Difficulty 2: Previous Approaches Were Limited to Simple Settings

Before this paper, conformal prediction for language models only worked in very restricted settings:

- **Token-level approaches:** Build confidence sets for individual words/tokens, but a guarantee about individual words doesn't translate to a guarantee about the whole sentence being correct.
- **Multiple-choice approaches:** Only work when the answer is one of a small fixed set of options (like A, B, C, D). This doesn't help for open-ended generation where the model writes paragraphs.
- **Sequence-level approaches:** Tried to build confidence sets over ALL possible output sequences, but these sets were impossibly large and required approximations that broke the guarantees.

### Difficulty 3: You Need "Correctness" to Mean Something Precise

To give a guarantee like "the output is correct with probability 90%," you need a precise definition of "correct." But what does "correct" mean for a paragraph of text? Different phrasings of the same fact are all "correct." Slightly different levels of detail can all be "correct."

The paper needs a definition of correctness that is:
- Mathematically precise (so you can prove theorems about it)
- Practically meaningful (so the guarantee actually means something useful)
- Compatible with the conformal prediction framework

**In one sentence:** "This is nontrivial because standard conformal prediction requires enumerating output spaces (impossible for text), previous approaches only worked for multiple-choice or token-level settings, and connecting correctness of text to a formal statistical framework required a new conceptual bridge."

## Q3: What Is the Main Claim?

The paper makes two key claims:

### Claim 1: The Conceptual Bridge
Every language model output y defines an **entailment set** E(y) = {y' : y' ⇒ y}, which is the set of all statements that are "more specific" than y and logically imply y. The ground truth y* being in this entailment set (y* ∈ E(y)) is *exactly equivalent* to y being correct.

**What this means in plain English:** If the true answer is "Sinking Spring Farm, Hodgenville, Kentucky" (where Lincoln was actually born), then the less specific claim "Kentucky" is entailed by the truth — the truth implies Kentucky. But "Idaho" is NOT entailed. So checking if y is correct is the same as checking if the truth is in y's entailment set.

### Claim 2: The Mathematical Guarantee (Theorem 4.1)
The algorithm (Algorithm 1) produces an output that is "α-conformally factual":

$$P(Y^*_{n+1} \in E(\bar{L}(X_{n+1}))) \geq 1 - \alpha$$

**In the simplest English:** "The probability that the modified output is factually correct is at least 1 − α." If you set α = 0.1, you get a 90% guarantee of correctness.

The theorem also gives a matching upper bound (when the entailment sets are nested):

$$P(Y^*_{n+1} \in E(\bar{L}(X_{n+1}))) \leq 1 - \alpha + \frac{1}{n+1}$$

This means the method isn't too conservative — it's close to exactly hitting the target coverage.

### The Practical Results
On GPT-4:
- **FActScore (biography generation):** Correctness goes from ~30% → ~80% while keeping about half the sub-claims
- **Natural Questions:** Correctness goes from ~78% → ~93% by removing only ~25% of sub-claims
- **MATH (reasoning):** Correctness goes from ~75% → ~95% by removing only ~10% of steps

**In one sentence:** "They show that conformal factuality, by using entailment sets to implicitly represent confidence sets and a sub-claim removal back-off strategy, achieves any user-specified correctness target (e.g., 90%) on GPT-4 outputs across QA and reasoning tasks, while retaining the majority of the original content."

---

# PASS 2: THE SCUBA DIVE — How Does It Work?

## Q1: What Was the Main Technical Hurdle Before This Paper?

### The State of the Art Before Conformal Factuality

Before this paper, there was a fundamental impasse:

**The Conformal Prediction Community's Problem:** They had beautiful mathematical tools for building confidence sets with guarantees. These tools worked great for classification (small, finite output spaces) and regression (continuous but one-dimensional). But language generation has an output space that is both infinite AND discrete AND high-dimensional (sequences of tokens). Their existing tools simply couldn't handle it.

**The NLP Community's Problem:** They had various methods to detect and reduce hallucination — self-consistency checking, retrieval augmentation, uncertainty estimation — but none of these came with formal mathematical guarantees. They could say "this usually helps" but not "this works at least 90% of the time."

### The Specific Barrier

The barrier was conceptual, not just technical. Everyone was trying to do conformal prediction by:
1. Defining a scoring function over ALL possible output sequences
2. Building a set of sequences that scores above a threshold
3. Showing this set contains the correct answer with high probability

But Step 2 was the killer — the set of all possible text sequences is uncountably large. You can't enumerate it, store it, or show it to a user. Even approximations (like the Quach et al. 2023 paper on conformal language modeling) produced sets too large to be useful.

### How This Paper Overcomes the Barrier — THE KEY INSIGHT

Here is the brilliant conceptual move that makes everything work:

**Instead of building the confidence set explicitly, define it *implicitly* through the entailment relation.**

Let me explain this step by step:

**Step 1: Flip the direction of entailment.**

Normally in NLP, we think of entailment as: "Does the model's answer imply the true answer?" (premise → hypothesis). This paper flips it and asks: "Does the true answer imply the model's answer?" (truth → model output).

Why? Because if the truth is more specific than the model's output, then the model's output must be correct. For example:
- Truth: "Lincoln was born at Sinking Spring Farm, Hodgenville, Kentucky"
- Model output: "Lincoln was born in Kentucky"

The truth implies the model's output (Kentucky is correct because the more specific claim about Hodgenville, Kentucky is true). So the model's output is correct!

**Step 2: Define the entailment set.**

For any output y, define: E(y) = {y' ∈ Y : y' ⇒ y}

This is the set of ALL statements that are more specific than y and entail y. It's a huge set — potentially infinite — but we never need to list it out!

**Step 3: Connect correctness to set containment.**

y is correct ⟺ y* ∈ E(y)

The output y is correct if and only if the true reference y* is in the entailment set of y. This converts a "correctness" question into a "set containment" question — exactly what conformal prediction is designed for!

**Step 4: Use back-off to control the entailment set size.**

Here's the final piece. A very specific claim like "Lincoln was born at Sinking Spring Farm" has a *small* entailment set (only very specific statements entail it). A vaguer claim like "Lincoln was born in America" has a *huge* entailment set (lots of specific facts about Lincoln's birthplace entail "America").

By making the output less specific (removing uncertain sub-claims), you make the entailment set bigger. A bigger entailment set is more likely to contain the truth y*.

So the algorithm becomes: **Start with the full output, progressively remove uncertain claims, and stop when conformal prediction tells you the remaining output is correct with probability 1 − α.**

**Why this is brilliant:** The confidence set (the entailment set E(y)) is always implicitly defined — we never need to list its elements. We only need to check whether one specific element (the truth y*) is in it, which is a single entailment check. This completely sidesteps the intractability problem!

## Q2: The Core Technical Machinery — Explained Step by Step

### Building Block 1: What Is Conformal Prediction? (The Basics)

Imagine you're a teacher grading exams. You want to build a system that, for any new student, predicts a range of possible scores that's guaranteed to contain the student's actual score at least 90% of the time.

Here's how conformal prediction works, explained as simply as possible:

**Step 1: Get calibration data.** You have n past students with their actual scores. Think of these as your "reference group."

**Step 2: Define a "surprise score."** For each past student, compute a number that measures how "surprising" their actual score was. If the student scored about what you'd expect, the surprise is low. If they scored much higher or lower than expected, the surprise is high.

Formally, this surprise score is called a **nonconformity score** and is written as r(x, y*), where x is the input (e.g., the student's study habits) and y* is the actual outcome (their score).

**Step 3: Find the threshold.** Sort all n surprise scores from smallest to largest. Find the value that's bigger than roughly (1 − α) × 100% of them. This is called the **quantile** and is written as q̂_α.

For example, if you want 90% coverage (α = 0.10) and you have 100 past students, you find the surprise score such that 90 students had a surprise score at or below it. Let's say that value is 0.35.

**Step 4: Build the prediction set.** For a new student, include any possible score y in the prediction set if the "surprise" of that score — r(x_new, y) — is at most q̂_α (our threshold of 0.35).

**Why this works:** If the new student's data comes from the same distribution as the past students (the "exchangeability" assumption), then the new student's surprise score is equally likely to be at any rank among all n+1 scores. So the probability that it falls below the (1−α)-quantile is at least 1 − α.

**The mathematical statement:**

$$P(Y^*_{n+1} \in C(X_{n+1})) \geq 1 - \alpha$$

This says: with probability at least 1 − α, the true answer Y* is in the confidence set C.

### Building Block 2: The Back-Off Function F_t

Now we need to connect conformal prediction to language models. The key object is the **back-off function** F_t(x).

Think of F_t as a "specificity dial" for the language model's output:
- At t = 0 (lowest setting): Keep the full, original output with all its claims
- At t = 1: Remove the most uncertain claim
- At t = 2: Remove the two most uncertain claims
- ...
- At t = max: Remove ALL claims — the output becomes empty (the model says nothing)

As t increases, the output becomes less specific (fewer claims) but more likely to be correct (the remaining claims are the ones the model is most confident about).

**A concrete example from the paper (Figure 2):**

Question: "Who was Abe Lincoln?"

Original output F₁(x): "Abraham Lincoln, born in Idaho, was the 16th President of the United States. He is best known for leading the country through the Civil War."

This has three sub-claims:
1. Born in Idaho (WRONG)
2. 16th President of the United States (CORRECT)
3. Best known for leading the country through the Civil War (CORRECT)

After removing the least confident claim (claim 1):
F₂(x): "Abraham Lincoln was the 16th President of the United States. He is best known for leading the country through the Civil War."

After removing the next least confident claim (claim 3):
F₃(x): "Abraham Lincoln was the 16th President of the United States."

After removing everything:
F₄(x): ∅ (empty — says nothing)

The entailment sets get progressively bigger as we remove claims:
- E(F₁(x)) is small (must entail all three claims, including the wrong one)
- E(F₂(x)) is bigger (only needs to entail two claims)
- E(F₃(x)) is even bigger (only needs to entail one claim)
- E(F₄(x)) = E(∅) = all of Y (everything entails the empty claim)

The truth y* is NOT in E(F₁(x)) because y* doesn't entail "born in Idaho."
The truth y* IS in E(F₂(x)) because y* entails both remaining claims.

So the algorithm would choose t = 2 — it backs off just enough to make the output correct.

### Building Block 3: The Nonconformity Score for Language Models

In standard conformal prediction, the nonconformity score measures "how surprising is the true answer?" For language models, the paper defines:

$$r(x, y^*) := \inf\{t \in T : \forall j \geq t, \; y^* \in E(F_j(x))\}$$

**In plain English:** r(x, y*) is the *minimum* back-off level t such that, from that point onward, the output is always correct. It's the threshold where you've removed enough uncertain claims that the remaining output is guaranteed correct (verified by checking entailment).

**For our Lincoln example:**
- At t = 1: y* ∉ E(F₁(x)) because "born in Idaho" is wrong → not safe yet
- At t = 2: y* ∈ E(F₂(x)) because all remaining claims are correct → safe!
- At t = 3: y* ∈ E(F₃(x)) → still safe
- At t = 4: y* ∈ E(F₄(x)) → trivially safe (empty output)

So r(x, y*) = 2. The minimum safe threshold is 2.

### Building Block 4: The Full Algorithm (Algorithm 1)

Here's the complete algorithm, step by step:

**Inputs:**
- A base language model L (e.g., GPT-4)
- A target error rate α (e.g., 0.10 for 90% correctness)
- A calibration dataset of n examples: {(X₁, Y₁*), (X₂, Y₂*), ..., (Xₙ, Yₙ*)} where each pair is a question and its true answer
- A back-off mechanism F_t (how to progressively simplify outputs)

**Calibration Phase (done once):**
1. For each calibration example i = 1, ..., n:
   a. Generate the model's output L(Xᵢ)
   b. Break it into sub-claims
   c. Score each sub-claim by confidence
   d. For each possible threshold t, check if the truth Yᵢ* entails the remaining output
   e. Find r(Xᵢ, Yᵢ*) — the minimum safe threshold

2. Sort all n scores: r₁, r₂, ..., rₙ

3. Compute the quantile: q̂_α = the ⌈(n+1)(1−α)⌉/n-th quantile of these scores

**Deployment Phase (for each new question):**
1. Generate the model's output L(x_new)
2. Apply the back-off: output F_{q̂_α}(x_new) — remove all sub-claims with confidence score below q̂_α
3. Return this simplified output to the user

**That's it!** The beauty is in its simplicity. The hard work is in the calibration phase (computing the scores), and the deployment is just one threshold comparison.

### Building Block 5: How Sub-Claims Make Everything Practical

The back-off function F_t is implemented using **sub-claims** — individual factual statements extracted from the model's output.

There are three components:

**S (Separator):** Takes a full output and breaks it into individual claims.
- Input: "Abraham Lincoln was born in Idaho and was the 16th President"
- Output: ["Abraham Lincoln was born in Idaho", "Abraham Lincoln was the 16th President"]
- Implemented using GPT-4 with a prompt asking it to separate claims

**s (Scorer):** Assigns a confidence score to each sub-claim.
- High score = the model is very confident this claim is correct
- Low score = the model is uncertain
- Several scoring methods are explored (details below)

**M (Merger):** Takes a subset of sub-claims and combines them back into a coherent paragraph.
- Input: ["Abraham Lincoln was the 16th President", "He led the country through the Civil War"]
- Output: "Abraham Lincoln was the 16th President of the United States. He led the country through the Civil War."
- Also implemented using GPT-4 with a prompt

The back-off function then becomes:

$$F_t(x) = M(\{c \in (S \circ L)(x) : s((S \circ L)(x), c) \geq t\})$$

**In English:** "Take the model's output, break it into sub-claims, keep only the sub-claims with confidence score at least t, and merge them back into a paragraph."

### Building Block 6: Assumption 5.1 — Why Sub-Claims Make Entailment Checking Cheap

There's a crucial practical trick hidden in the paper. Normally, to compute r(x, y*), you'd need to check entailment for every possible threshold t — that means checking whether y* ∈ E(F_t(x)) for potentially many different merged outputs.

**Assumption 5.1** says:

$$y^* \Rightarrow M(\{c_i\}_{i=1}^n) \quad \Longleftrightarrow \quad \forall i \in [n], \; y^* \Rightarrow c_i$$

**In English:** The truth entails the merged paragraph if and only if the truth entails every individual sub-claim. This is a very natural assumption — it says that the merger function M doesn't add or remove any factual content; it just combines the claims into a readable paragraph.

**Why this matters enormously:** Instead of checking entailment on many different merged paragraphs (expensive!), you only need to check entailment once for each individual sub-claim (cheap!). Then you can compute r(x, y*) simply by looking at which sub-claims are entailed and which aren't.

Specifically, under Assumption 5.1, the score becomes (Proposition 5.2):

$$r(x, y^*) = \inf\{t \in T : \forall j \geq t, \forall c \in A_t(x), \; y^* \Rightarrow c\}$$

where A_t(x) is the set of sub-claims with score ≥ t. In practice, since removing claims only makes the set of accepted claims smaller, you just need to check: "What's the lowest-scored sub-claim that's still correct?" The score r is determined by the lowest-confidence *correct* sub-claim that's still included.

### Building Block 7: Sub-Claim Scoring Functions

The quality of the whole system depends critically on how well you can score sub-claims by their likelihood of being correct. The paper tests five scoring methods:

**1. Random scoring (baseline):**
- Assigns random numbers to each sub-claim
- This is the "do nothing useful" baseline — it removes claims in random order
- Obviously bad, but serves as a lower bound on performance

**2. Ordinal scoring (baseline):**
- Assigns scores based on the order claims appear: the first claim in the output gets the highest score, the last gets the lowest
- The intuition is that LLMs tend to be more confident about things they state first
- Simple but surprisingly not terrible

**3. GPT-4 confidence scoring (practical):**
- Directly asks GPT-4: "How confident are you in this claim, on a scale from 0 to 1?"
- Uses a prompt that defines 1 as "obvious facts like 'the earth is round'" and 0 as "very obscure claims"
- Leverages the model's own self-knowledge of its uncertainty

**4. Frequency scoring (practical — the best one):**
- Generate 5 alternative outputs for the same question (with temperature 1.0 for diversity)
- Count how many of the 5 alternatives support each sub-claim
- A claim that appears consistently across multiple samplings is more likely correct
- This is based on the "self-consistency" idea: if the model keeps saying the same thing across different random samples, it's probably right

**5. Oracle scoring (upper bound):**
- Uses the actual truth to score claims: correct claims get 1, incorrect get 0
- Obviously can't be used in practice (you'd need to already know the truth!)
- Serves as an upper bound — the best possible scoring function

**The key finding:** Frequency scoring is the best practical method. It exploits the insight that consistency across multiple samplings is a strong signal of correctness.

### Building Block 8: Partial Entailment (Corollary 5.3)

The paper also extends the framework to **partial correctness**. Instead of requiring ALL remaining sub-claims to be correct, you can require that at least a fraction a ∈ [0,1] of them are correct.

This is done by modifying the score:

$$r_a(x, y^*) = \inf\{t \in T : \forall j \geq t, \; T_{y^*}(A_t(x)) \geq a\}$$

where T_{y*}(A_t(x)) is the fraction of accepted sub-claims that are entailed by y*.

The guarantee becomes:

$$P(T_{Y^*_{n+1}}(A_{\hat{q}_\alpha}(X_{n+1})) \geq a) \geq 1 - \alpha$$

**In English:** "With probability at least 1 − α, at least fraction a of the remaining sub-claims are correct."

This is useful because requiring 100% correctness might force you to remove too many claims. If you're willing to tolerate, say, 20% incorrect claims, you can keep much more of the original output.

## Q2: What Is the Simplest Baseline and How Much Better Is This?

### The Baselines

The simplest meaningful baseline is **unmodified GPT-4** — just use the model's output directly without any modification. This has no correctness guarantee at all.

For scoring function baselines:
- **Random scoring:** Removes claims in random order — this represents "we know we should back off, but we have no idea which claims are wrong"
- **Ordinal scoring:** Removes claims from last-to-first in the output — this represents "a simple heuristic based on output position"

The meaningful comparison is between these baselines and the LM-based scoring functions (GPT-4 confidence and frequency scoring).

### Results Summary

#### FActScore (Biography Generation)
This is the hardest dataset. GPT-4 hallucinates aggressively when writing biographies of less-famous people.

- **Base GPT-4 correctness:** ~30% (meaning ~70% of biographies contain at least one false claim!)
- **With frequency scoring at α = 0.2 (target: 80% correctness):**
  - Achieved correctness: ~80% ✓
  - Claims removed: ~50%
  - So we keep about half the original content, but it's now correct 80% of the time instead of 30%

This is a dramatic improvement. Going from 30% to 80% correctness while keeping half the content is remarkably useful.

#### Natural Questions (Factual QA)
- **Base GPT-4 correctness:** ~78%
- **With frequency scoring at α = 0.1 (target: 90% correctness):**
  - Achieved correctness: ~93% ✓
  - Claims removed: ~25%
  - We keep 75% of the content and get a 15-percentage-point boost in correctness

#### MATH (Reasoning)
- **Base GPT-4 correctness:** ~75%
- **With frequency scoring at α = 0.05 (target: 95% correctness):**
  - Achieved correctness: ~95% ✓
  - Claims removed: ~10%
  - We keep 90% of the reasoning steps and get a 20-percentage-point boost

#### Comparing Scoring Functions
The paper shows clear ranking of scoring functions:
1. **Oracle** (best possible — but unusable in practice)
2. **Frequency scoring** (best practical method)
3. **GPT-4 confidence scoring** (good but not as good as frequency)
4. **Ordinal scoring** (surprisingly decent for how simple it is)
5. **Random scoring** (worst — removes too many correct claims)

The gap between frequency scoring and oracle scoring tells us there's still room for improvement in uncertainty estimation, but frequency scoring already provides very useful performance.

### Qualitative Examples

The paper includes compelling qualitative examples in Tables 1, 3, 4, and 5. Let me highlight one:

**Original GPT-4 output (about Zamfir Ralli-Arbore):**
> "Zamfir Ralli-Arbore (1848-1933) was a Romanian political activist and historian from Bessarabia, who spent much of his life in exile. As a member of the National Liberal Party, he campaigned for the union of his native region with the Kingdom of Romania, and was a prominent opponent of Russian and Soviet policies. He was also a noted historian, specializing in the history of the Moldavia and Wallachia during the Middle Ages."

Much of this is fabricated — GPT-4 is hallucinating specific details about an obscure historical figure.

**Conformal factuality output:**
> "Zamfir Ralli-Arbore, born in 1848, was a Romanian political activist from Bessarabia. He passed away in 1933."

The system correctly identified that only the basic biographical facts (birth year, nationality, region of origin, death year) were reliable, and removed all the hallucinated details about party membership, political activities, and academic specialization.

## Q3: What's the Theoretical Analysis? (Theorem 4.1 — Explained Simply)

### The Setup

We have n+1 exchangeable data points: (X₁, Y₁*), ..., (Xₙ, Yₙ*), (X_{n+1}, Y_{n+1}*).

"Exchangeable" means: any reordering of these n+1 data points is equally likely. This is slightly weaker than assuming they're independently drawn from the same distribution (i.i.d.), but for practical purposes, think of it as "they all come from the same distribution."

We use the first n points for calibration and want a guarantee on the (n+1)-th point.

### Theorem 4.1 Statement (In Simple Words)

**Lower bound (the guarantee):**

$$P(Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))) \geq 1 - \alpha$$

"The probability that the backed-off output is correct is at least 1 − α."

**Upper bound (it's not too conservative):** If the entailment sets are nested (removing claims only makes the entailment set bigger — never smaller), then:

$$P(Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))) \leq 1 - \alpha + \frac{1}{n+1}$$

"The probability of correctness is at most 1 − α + 1/(n+1)."

So with n = 99 calibration points, the coverage is between 1 − α and 1 − α + 0.01 — extremely tight!

### The Proof (Explained Like a Story)

**Step 1: Compute and sort the calibration scores.**

For each calibration example i, we compute r_i = r(X_i, Y_i*) — the minimum back-off level needed to make the output correct. We also compute r_test = r(X_{n+1}, Y_{n+1}*) for the test example.

Assume the scores are sorted: r₁ < r₂ < ... < rₙ. (The paper assumes distinct scores, which can be achieved by adding tiny random tiebreakers.)

The quantile is q̂_α = r_{⌈(1−α)(n+1)⌉}. This is the score at position ⌈(1−α)(n+1)⌉ in the sorted list.

**Step 2: Use exchangeability.**

Here's the key insight. Since all n+1 data points are exchangeable, the test score r_test is equally likely to land at any position in the sorted list of all n+1 scores. It's like shuffling a deck of n+1 cards — the test card is equally likely to be in any position.

So:

$$P(r_{test} \leq r_{\lceil(1-\alpha)(n+1)\rceil}) = \frac{\lceil(1-\alpha)(n+1)\rceil}{n+1} \geq 1 - \alpha$$

**In everyday terms:** If you randomly insert one card into a sorted deck of 100 cards, the probability it lands in the bottom 90 positions is 90/101 ≈ 89.1%. With the ceiling function, this rounds up to give us at least 1 − α.

**Step 3: Connect the score to correctness.**

The crucial observation is:

$$\{r_{test} \leq \hat{q}_\alpha\} \quad \text{implies} \quad \{Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))\}$$

**Why?** If r_test ≤ q̂_α, that means the minimum safe threshold for the test point is at or below q̂_α. Since q̂_α is safe (and any threshold above the minimum safe threshold is also safe), the output F_{q̂_α}(X_{n+1}) is correct.

**In everyday terms:** If you need to remove at least 3 claims to make the output correct (r_test = 3), and the algorithm removes 5 claims (q̂_α = 5), then the output is definitely correct — you've removed more than enough.

**Step 4: Combine.**

$$P(Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))) \geq P(r_{test} \leq \hat{q}_\alpha) \geq 1 - \alpha \quad \square$$

**For the upper bound:** When entailment sets are nested (removing a claim always makes the entailment set strictly bigger), then the implication in Step 3 becomes an *equivalence* — the output is correct *exactly when* r_test ≤ q̂_α. Then the probability equals ⌈(1−α)(n+1)⌉/(n+1), which is at most 1 − α + 1/(n+1).

### Why the Proof Is Elegant

The beauty of this proof is that it's incredibly short and clean — just a few lines. All the heavy lifting is done by the *definition* of the nonconformity score r and the entailment set E. Once you set up the right correspondence (correctness ↔ set containment ↔ score comparison), the standard conformal prediction machinery gives you the guarantee for free.

This is the hallmark of a good conceptual contribution: the right framework makes the proof almost trivial.

## Q4: What's Still Open? Where Does the Technique Break Down?

### Limitation 1: The Exchangeability Assumption (Distribution Shift)

The guarantee requires that calibration data and test data come from the same distribution (exchangeable). In the real world, this is often violated:
- You calibrate on general knowledge questions, but users ask about specialized topics
- The model gets updated, changing its behavior
- User distributions shift over time

When the distribution changes, the threshold q̂_α computed on old data may be too low (not removing enough claims) or too high (removing too many). **The guarantee breaks.**

This is explicitly acknowledged in the paper's Limitations section: "in real-world scenarios where distributions change, the threshold computed on past calibration data can fail to maintain the desired coverage."

**Connection to your research:** This limitation is exactly what your DS-SGen project addresses. Combining conformal factuality with domain-shift-aware methods (like those in the DS-CP paper) would extend the guarantee to settings where the distribution changes.

### Limitation 2: Marginal, Not Conditional Coverage

The guarantee is **marginal** — it holds on average over all possible inputs. It does NOT hold for every individual input. For example:
- On easy questions, the system might provide 99% correctness
- On hard questions, it might only provide 60% correctness
- On average, it's 90% — but any individual question might be below the target

This means the system could systematically fail on certain types of questions (e.g., questions about rare topics) while performing great on others, and still satisfy the marginal guarantee.

**In everyday terms:** It's like saying "this airline has 90% on-time arrivals." That's true on average, but if you always fly the problematic route, YOUR experience might be much worse.

### Limitation 3: Marginal Over the Calibration Set Too

The guarantee is also marginal over the *random draw of the calibration set*. If you're unlucky and your calibration set is not representative, the threshold could be off. The paper shows (Section 6.2.1) that the standard deviation of empirical factuality across different calibration splits is about 0.09, which is non-trivial for a 25-sample calibration set.

### Limitation 4: Quality of Sub-Claim Decomposition

The whole system depends on being able to cleanly separate a language model's output into independent, atomic claims. In practice:
- Some claims are dependent on each other (removing one makes another ambiguous)
- The separator (GPT-4 prompt) might split too coarsely or too finely
- The merger might introduce subtle meaning changes

These issues don't break the theoretical guarantee (which holds for ANY sound F_t), but they affect the practical utility — how much useful content you get to keep.

### Limitation 5: Small Calibration Sets (Only 50 Examples!)

The experiments use only 50 examples per dataset, split in half for calibration and testing. While the authors demonstrate the guarantee holds across 1000 random splits, the small sample size means:
- High variance in the threshold
- Limited ability to distinguish between scoring functions
- Unclear how results scale to more realistic deployment scenarios with thousands of calibration points

### Limitation 6: Only Tested on GPT-4

All experiments use GPT-4 as the base model AND as the sub-claim separator/scorer/merger. It's unclear how well this works with:
- Open-source models (Llama, Mistral)
- Smaller models that may have worse self-knowledge
- Models from different families where the separator prompts might work differently

## Q5: Caveats and Takeaways

### Strengths

1. **Beautiful conceptual contribution:** The entailment-set correspondence is genuinely novel and elegant. It turns an "impossible" problem (conformal prediction for language generation) into a tractable one with a single insight.

2. **Clean, short proof:** Theorem 4.1 has a proof that fits in half a page. This is a sign of a well-chosen framework — the right abstraction makes the math simple.

3. **Practical and useful:** The algorithm actually works on real LLM outputs and provides meaningful improvements (30% → 80% on FActScore is substantial).

4. **Modular design:** The framework works with ANY scoring function, ANY back-off mechanism, and ANY model. Better components can be swapped in as they're developed.

5. **Code released:** Full reproducibility, which is especially important for a non-peer-reviewed paper.

### Weaknesses / Devil's Advocate — Three Weakest Points

**Weakness 1: The sub-claim decomposition is a hidden assumption factory.**
The practical implementation relies heavily on GPT-4's ability to decompose outputs into clean, independent sub-claims and merge them back together. Assumption 5.1 (that merger preserves entailment relations) is stated but never verified empirically. If the merger adds subtle implications or changes meaning, the practical guarantee could fail even though the theoretical guarantee holds.

**Weakness 2: Using GPT-4 to evaluate GPT-4 is circular.**
The scoring functions (especially GPT-4 confidence scoring) ask GPT-4 to judge its own uncertainty. The frequency scoring generates multiple GPT-4 samples and uses GPT-4 to check consistency. This creates a potential echo chamber — the model's blind spots about its own uncertainty might be consistent across samplings.

**Weakness 3: Extremely small evaluation scale.**
Only 50 examples per dataset, with manual annotations done by the authors themselves (not independent annotators). The 1000 random splits help with statistical power, but the underlying data is still just 50 examples. Results on 50 biographies of specific people may not generalize to other biography subjects, let alone to other genres of text generation.

---

# PASS 3: THE SWAMP — Deep Dive into the Mathematical Machinery

## Proof Architecture Overview

The paper's theoretical contribution is remarkably clean. There's essentially one main theorem (Theorem 4.1) with a straightforward proof, plus a proposition (Proposition 5.2) that simplifies score computation, and a corollary (Corollary 5.3) for partial entailment. Let me walk through each in complete detail.

## Detailed Proof Walkthrough: Theorem 4.1

### The Full Setup

We have:
- n + 1 exchangeable pairs: (X₁, Y₁*), ..., (Xₙ, Yₙ*), (X_{n+1}, Y_{n+1}*)
- A language model L : X → Y
- A sound back-off mechanism {F_t}_{t∈T} where F_{sup T} = ∅ (at the maximum threshold, output is empty)
- The entailment operator E : Y → 2^Y where E(y) = {y' : y' ⇒ y}
- The convention that ∀y ∈ Y, y ⇒ ∅ (everything entails the empty claim — this is what makes F_t "sound")

The nonconformity score is:

$$r(x, y^*) = \inf\{t \in T : \forall j \geq t, \; y^* \in E(F_j(x))\}$$

Note the "∀j ≥ t" part — this asks for the threshold to be **strictly safe**, meaning it works for all thresholds above it too. In standard conformal prediction with nested sets, this is automatic (bigger sets always contain everything smaller sets contain). Here, without assuming nestedness, we need this extra condition.

**Why do we need strict safety?** Without it, a threshold t might be "accidentally safe" — the output happens to be correct at level t but not at t+1. The conformal quantile might land on such an accidental threshold, and the guarantee could fail. Requiring strict safety ensures monotonicity of the "safe event."

**Why is the score well-defined?** Because F_{sup T} = ∅ and ∀y, y ⇒ ∅. So at the maximum threshold, y* ∈ E(∅) always holds. There's always at least one safe threshold (the maximum), so the infimum exists.

### The Lower Bound Proof — Line by Line

Let rᵢ = r(Xᵢ, Yᵢ*) for i ∈ [n] and r_test = r(X_{n+1}, Y_{n+1}*).

**Step 1: Assume WLOG that scores are sorted.**

r₁ < r₂ < ... < rₙ (the paper says we can assume distinct scores without loss of generality — in practice, you add tiny random noise to break ties).

**Step 2: Identify the quantile.**

q̂_α = r_{⌈(1-α)(n+1)⌉} when α ≥ 1/(n+1).

This is the score at rank ⌈(1−α)(n+1)⌉ in the sorted list. For example, with n = 24 and α = 0.10, we'd take rank ⌈0.90 × 25⌉ = ⌈22.5⌉ = 23, so q̂_α = r₂₃ (the 23rd smallest score out of 24).

**Step 3: Use exchangeability to bound rank probability.**

By exchangeability, the test score r_test is equally likely to be at any rank among the n+1 scores {r₁, ..., rₙ, r_test}. So:

$$P(r_{test} \leq r_{\lceil(1-\alpha)(n+1)\rceil}) = \frac{\lceil(1-\alpha)(n+1)\rceil}{n+1}$$

**Why exactly?** Since all n+1 scores are exchangeable, the test score's rank is uniformly distributed over {1, 2, ..., n+1}. The event "r_test is at rank ≤ k" has probability k/(n+1). With k = ⌈(1−α)(n+1)⌉:

$$\frac{\lceil(1-\alpha)(n+1)\rceil}{n+1} \geq \frac{(1-\alpha)(n+1)}{n+1} = 1 - \alpha$$

The inequality holds because ⌈a⌉ ≥ a for any real number a.

**Step 4: Connect rank to correctness.**

**Claim:** {r_test ≤ q̂_α} implies {Y_{n+1}* ∈ E(F_{q̂_α}(X_{n+1}))}

**Proof of claim:** If r_test ≤ q̂_α, then by definition of r_test:

r_test = inf{t : ∀j ≥ t, Y_{n+1}* ∈ E(F_j(X_{n+1}))}

Since q̂_α ≥ r_test, we have q̂_α ≥ the infimum, so ∀j ≥ q̂_α ≥ r_test, Y_{n+1}* ∈ E(F_j(X_{n+1})).

In particular, taking j = q̂_α: Y_{n+1}* ∈ E(F_{q̂_α}(X_{n+1})). ✓

**Step 5: Combine.**

$$P(Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))) \geq P(r_{test} \leq \hat{q}_\alpha) = \frac{\lceil(1-\alpha)(n+1)\rceil}{n+1} \geq 1 - \alpha$$

**The lower bound is proved.** □

### The Upper Bound Proof — Under Nestedness

**Additional assumption:** E(F_t(·)) follows the nested property, meaning E(F_t(x)) ⊆ E(F_{t'}(x)) for t ≤ t'.

**In plain English:** Removing more claims (higher t) always makes the entailment set bigger. This is natural — a less specific statement is entailed by more things.

**Under Assumption 5.1:** This nestedness holds automatically because:

$$E(F_t(x)) = E(M(A_t(x))) = \bigcap_{c \in A_t(x)} E(c)$$

When you remove a claim from A_t(x) (going to A_{t'}(x) with fewer claims), you're removing one set from the intersection. Removing a set from an intersection can only make the result bigger. So E(F_{t'}(x)) ⊇ E(F_t(x)). ✓

**Under nestedness, the implication becomes an equivalence:**

$$\{r_{test} \leq \hat{q}_\alpha\} = \{Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))\}$$

**Why the equivalence holds:** The forward direction (⇒) is the same as before. For the reverse direction (⇐): if Y_{n+1}* ∈ E(F_{q̂_α}(X_{n+1})), then by nestedness, Y_{n+1}* ∈ E(F_j(X_{n+1})) for all j ≥ q̂_α. So q̂_α is a strictly safe threshold, meaning r_test ≤ q̂_α. ✓

**Now the upper bound:**

$$P(Y^*_{n+1} \in E(F_{\hat{q}_\alpha}(X_{n+1}))) = P(r_{test} \leq \hat{q}_\alpha) = \frac{\lceil(1-\alpha)(n+1)\rceil}{n+1}$$

Using the inequality ⌈a⌉ ≤ a + 1:

$$\frac{\lceil(1-\alpha)(n+1)\rceil}{n+1} \leq \frac{(1-\alpha)(n+1) + 1}{n+1} = 1 - \alpha + \frac{1}{n+1}$$

**The upper bound is proved.** □

### What the Upper Bound Tells Us

The gap between lower and upper bound is exactly 1/(n+1). With n = 24 (as in the paper's experiments), this is 1/25 = 0.04 = 4%. With n = 99, it's 1% — negligible.

This means the method is almost perfectly calibrated — it's not being overly conservative. The coverage is very close to exactly 1 − α.

## Detailed Walkthrough: Proposition 5.2 (Score Simplification)

### Statement

Under Assumption 5.1, the score can be computed as:

$$r(x, y^*) = \inf\{t \in T : \forall j \geq t, \forall c \in A_j(x), \; y^* \Rightarrow c\}$$

### Proof

The proof is a chain of equivalences:

**Step 1:** y* ∈ E(F_t(x))

**Step 2:** ⟺ y* ⇒ F_t(x) (by definition of E)

**Step 3:** ⟺ y* ⇒ M(A_t(x)) (by definition of F_t)

**Step 4:** ⟺ ∀c ∈ A_t(x), y* ⇒ c (by Assumption 5.1)

Substituting into the definition of r:

$$r(x, y^*) = \inf\{t \in T : \forall j \geq t, y^* \in E(F_j(x))\} = \inf\{t \in T : \forall j \geq t, \forall c \in A_j(x), y^* \Rightarrow c\}$$

**Practical implication:** You check entailment once per sub-claim, then compute r by looking at which sub-claims pass the entailment check. This reduces the annotation effort from "check entailment for every possible subset of claims" to "check entailment for each individual claim."

## Detailed Walkthrough: Corollary 5.3 (Partial Entailment)

### The Modified Score

Instead of requiring ALL accepted sub-claims to be entailed, require at least fraction a:

$$r_a(x, y^*) = \inf\{t \in T : \forall j \geq t, \; T_{y^*}(A_j(x)) \geq a\}$$

where $T_{y^*}(\{s_i\}_{i=1}^m) = \frac{1}{m}\sum_{i=1}^m \mathbf{1}_{y^* \Rightarrow s_i}$.

### The Guarantee

$$P(T_{Y^*_{n+1}}(A_{\hat{q}_\alpha}(X_{n+1})) \geq a) \geq 1 - \alpha$$

### The Proof

The proof is identical to the lower bound of Theorem 4.1. The key step is:

{r_{a,test} ≤ q̂_α} implies {T_{Y*_{n+1}}(A_{q̂_α}(X_{n+1})) ≥ a}

**Why?** If r_{a,test} ≤ q̂_α, then q̂_α is a safe threshold for the a-partial-entailment score, meaning at every threshold j ≥ q̂_α, at least fraction a of accepted claims are entailed. In particular at j = q̂_α itself.

**Important note:** This is NOT an equivalence (even with nestedness). The reason is that T_{y*}(A_t(x)) is not necessarily monotonically increasing in t. As you remove claims, you might remove correct ones (decreasing the fraction) or incorrect ones (increasing the fraction). So a threshold q̂_α that happens to give partial entailment ≥ a might not be above r_{a,test}.

**In everyday terms:** Imagine you have 10 claims, 7 correct and 3 wrong. Fraction correct = 70%. Now remove one correct claim: 6 correct out of 9 = 67% — it went down! So removing claims doesn't always improve the fraction of correct claims.

## Techniques You Can Borrow for Your DS-SGen Research

### Technique 1: The Entailment-Set Correspondence

The idea that LM outputs implicitly define confidence sets through entailment could be combined with your selective generation framework. Instead of just deciding "answer vs. IDK," you could have a spectrum: "full answer → partially backed-off answer → IDK."

### Technique 2: The Sub-Claim Decomposition Strategy

The decompose-score-remove pipeline is general and could be applied in your DS-SGen framework. Under domain shift, you could:
- Decompose the output into sub-claims
- Score each sub-claim using domain-aware confidence (incorporating importance weights from DS-CP)
- Remove claims that are likely to be wrong given the domain shift

### Technique 3: Frequency Scoring as a Domain-Robust Confidence Measure

Frequency scoring (self-consistency across multiple samplings) might be more robust to domain shift than single-sample confidence scores, because it measures the model's internal consistency rather than relying on calibrated probabilities.

### Technique 4: The Partial Entailment Extension

The partial entailment framework (Corollary 5.3) provides a useful middle ground between "everything must be correct" (too strict) and "anything goes" (too loose). This could complement your DS-SGen framework by allowing controllable levels of correctness.

---

# KEY CONCEPTS GLOSSARY (for Grade 12 Level)

| Concept | Simple Explanation |
|---------|-------------------|
| **Hallucination** | When an AI generates text that sounds confident but is factually wrong — like a student making up an answer on a test |
| **Conformal prediction** | A statistical method that builds "safety nets" around predictions, guaranteed to catch the right answer with a specified probability |
| **Entailment** | A logical relationship: sentence A "entails" sentence B if A being true guarantees B is true. Example: "It's raining heavily" entails "It's raining" |
| **Entailment set E(y)** | The collection of ALL statements that are more specific than y and logically imply y. Like a cloud of "more detailed versions" of a claim |
| **Back-off** | Making a statement less specific to make it more likely correct. "Born in Kentucky" is less specific than "Born in Hodgenville, Kentucky" |
| **Sub-claim** | An individual factual statement extracted from a longer piece of text. A paragraph might contain 5-10 sub-claims |
| **Nonconformity score r(x, y*)** | A number measuring "how much backing off is needed to make this output correct?" — higher means the model made more mistakes |
| **Quantile q̂_α** | A threshold computed from calibration data that determines how much to back off. Like setting the sensitivity dial on a metal detector |
| **α (alpha)** | The target error rate — e.g., α = 0.10 means you want at most 10% chance of the output being wrong |
| **Exchangeability** | The assumption that calibration data and test data come from the same process — like cards shuffled from the same deck |
| **Marginal coverage** | The guarantee holds "on average" across all possible inputs, not necessarily for every specific input |
| **Frequency scoring** | Generating multiple answers and checking how consistently each claim appears — consistent claims are probably correct |
| **Sound back-off mechanism** | One that eventually removes everything (at the highest threshold, the output is empty). This ensures a safe threshold always exists |
| **Nested sets** | Sets that get bigger as you go up the hierarchy — removing claims always makes the entailment set bigger (or at least not smaller) |
| **Calibration data** | A set of examples with known correct answers, used to compute the threshold q̂_α |
| **Black-box model** | A model you can use (give it input, get output) but can't look inside to see how it works — like a vending machine |

---

# HOW THIS PAPER CONNECTS TO THE OTHER PAPERS IN YOUR PROJECT

## The Three-Paper Landscape

Your DS-SGen research project sits at the intersection of three papers, each solving a different piece of the puzzle:

| Paper | What It Does | What It Guarantees | Key Limitation |
|-------|-------------|-------------------|----------------|
| **SGen** (Lee et al., NeurIPS 2024) | Decides whether to answer or say "I don't know" | P{FDR-E ≤ ε} ≥ 1 − δ (PAC guarantee on false discovery rate) | Assumes i.i.d. data (no domain shift) |
| **Conformal Factuality** (this paper) | Removes uncertain sub-claims to make outputs more reliable | P(output correct) ≥ 1 − α (coverage guarantee) | Assumes exchangeability (no domain shift), marginal guarantee only |
| **DS-CP** (Lin et al., arXiv 2025) | Adapts conformal prediction to handle domain shift | Coverage ≥ 1 − α − error term | Only works for multiple-choice QA, approximate guarantee |

## Comparing the Three Approaches

### SGen vs. Conformal Factuality

These two papers solve the SAME high-level problem (making LLM outputs reliable) but in fundamentally different ways:

**SGen's approach — Binary decision:**
- For each question, decide: "Answer normally" or "Say I don't know"
- When it answers, it gives the full original output (possibly with errors)
- Controls the False Discovery Rate among answered questions

**Conformal Factuality's approach — Graduated back-off:**
- For each question, keep some claims and remove others
- Always gives SOME output (unless everything is removed)
- Controls the probability that the remaining output is correct

**Which is better?** They serve different use cases:
- SGen is better when you need the complete answer or nothing (e.g., "What's the dosage for this medication?" — a partial answer might be dangerous)
- Conformal factuality is better when partial answers are useful (e.g., "Tell me about Abraham Lincoln" — some correct facts are better than none)

### What's Missing — Your DS-SGen Contribution

Neither SGen nor Conformal Factuality handles **domain shift**. Both assume the test data comes from the same distribution as the calibration data. Your DS-SGen project would:

1. Take SGen's selective generation framework (answer vs. abstain)
2. Add DS-CP's importance reweighting to handle domain shift
3. Potentially incorporate conformal factuality's partial back-off as a middle option
4. Provide PAC guarantees that hold even when the input distribution changes

This would create a unified framework with three possible outputs:
- "Here's the full answer" (high confidence, even under domain shift)
- "Here's a partial answer — the parts I'm confident about" (medium confidence)
- "I don't know" (low confidence)

All with formal guarantees that account for the fact that the test domain may differ from the calibration domain.

---

# SUMMARY: ONE-PAGE CHEAT SHEET

**Problem:** LLMs hallucinate — they mix correct and incorrect facts. We want to automatically identify and remove the unreliable parts, with a mathematical guarantee.

**Key Innovation:** Use entailment sets to create an implicit correspondence between conformal prediction confidence sets and LM outputs. Making the output less specific (removing sub-claims) is equivalent to enlarging the confidence set — and conformal prediction tells us exactly how much to enlarge it.

**Algorithm (Conformal Factuality):**
1. Break LM output into sub-claims using GPT-4
2. Score each sub-claim by confidence (best method: frequency scoring — check consistency across multiple samplings)
3. Calibrate a threshold using n labeled examples and conformal prediction
4. Remove all sub-claims below the threshold
5. Merge remaining sub-claims back into a coherent output

**Theoretical Guarantee (Theorem 4.1):**

$$1 - \alpha \leq P(\text{output is correct}) \leq 1 - \alpha + \frac{1}{n+1}$$

Requires: exchangeable data, sound back-off mechanism. Holds for ANY black-box LM.

**Key Results (GPT-4):**
- FActScore: 30% → 80% correctness, keeping ~50% of claims
- Natural Questions: 78% → 93% correctness, keeping ~75% of claims
- MATH: 75% → 95% correctness, keeping ~90% of steps

**Scoring Functions (ranked):** Oracle > Frequency > GPT-4 Confidence > Ordinal > Random

**Main Limitations:**
- Exchangeability assumption (breaks under domain shift)
- Marginal guarantee only (not per-input)
- Small evaluation scale (50 examples per dataset)
- Relies on GPT-4 for sub-claim decomposition/merging
- Only tested on one base model (GPT-4)

**For Your DS-SGen Project:** This paper offers a complementary approach to SGen. While SGen decides whether to answer at all, conformal factuality decides how much to answer. Both frameworks break under domain shift, creating the gap your DS-SGen project aims to fill. Borrowable techniques include the sub-claim decomposition strategy, frequency scoring as a domain-robust confidence measure, and the partial entailment extension.