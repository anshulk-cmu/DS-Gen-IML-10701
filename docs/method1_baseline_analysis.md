# Method 1: SGen-Semi Baseline — Complete Analysis

**DS-SGen: Domain-Shift-Aware Selective Generation for Reliable LLMs**
**Anshul Kumar, Justin Luan — Carnegie Mellon University, 10-701, Spring 2026**

This document records every decision, every number, every piece of math, and every
result from the Method 1 baseline implementation. It is the truth document for the
SGen-Semi baseline. Numbers marked as "actual" or "from cache" are validated against
cached data and log files. Numbers in the "Expected Results" and "Worked Example"
sections are predictions or illustrative — not measured outcomes. The pipeline has
not yet completed; statistics from partial caches may shift when the full run finishes.

---

## Table of Contents

1. [Purpose of This Method](#1-purpose-of-this-method)
2. [The Research Question](#2-the-research-question)
3. [What This Method Is and Is Not](#3-what-this-method-is-and-is-not)
4. [The FDR-E Metric: Why Not Accuracy](#4-the-fdr-e-metric-why-not-accuracy)
5. [The PAC Guarantee](#5-the-pac-guarantee)
6. [The Neuro-Selection Function](#6-the-neuro-selection-function)
7. [The Datasets](#7-the-datasets)
8. [The Generator Model: LLaMA-3.1-8B-Instruct](#8-the-generator-model-llama-31-8b-instruct)
9. [Response Generation: Greedy + Sampled](#9-response-generation-greedy--sampled)
10. [The Entailment Model: DeBERTa-v2-xxlarge-mnli](#10-the-entailment-model-deberta-v2-xxlarge-mnli)
11. [Scoring Function fM1: Mean Log-Probability](#11-scoring-function-fm1-mean-log-probability)
12. [Scoring Function fM2: Self-Consistency via Bidirectional NLI](#12-scoring-function-fm2-self-consistency-via-bidirectional-nli)
13. [Correctness Scoring: Unidirectional Entailment](#13-correctness-scoring-unidirectional-entailment)
14. [The SGen-Semi Algorithm: Full Mathematical Derivation](#14-the-sgen-semi-algorithm-full-mathematical-derivation)
15. [Step 1: Data Splitting](#15-step-1-data-splitting)
16. [Step 2: Conformal Pseudo-Labeling](#16-step-2-conformal-pseudo-labeling)
17. [Step 3: Threshold Grid Search with PAC Constraint](#17-step-3-threshold-grid-search-with-pac-constraint)
18. [Step 4: Evaluation on Test Sets](#18-step-4-evaluation-on-test-sets)
19. [The Clopper-Pearson Bound: Why This Specific Bound](#19-the-clopper-pearson-bound-why-this-specific-bound)
20. [Bonferroni Correction: Why and How](#20-bonferroni-correction-why-and-how)
21. [Hyperparameter Table and Justifications](#21-hyperparameter-table-and-justifications)
22. [The Pipeline: Four Stages](#22-the-pipeline-four-stages)
23. [Stage 1: Data Loading — Code and Decisions](#23-stage-1-data-loading--code-and-decisions)
24. [Stage 2: LLM Generation — Code and Decisions](#24-stage-2-llm-generation--code-and-decisions)
25. [Stage 3: Entailment Scoring — Code and Decisions](#25-stage-3-entailment-scoring--code-and-decisions)
26. [Stage 4: SGen-Semi Algorithm — Code and Decisions](#26-stage-4-sgen-semi-algorithm--code-and-decisions)
27. [Worked Example: One Complete Split](#27-worked-example-one-complete-split)
28. [Expected Results and What They Mean](#28-expected-results-and-what-they-mean)
29. [The Domain Shift Hypothesis](#29-the-domain-shift-hypothesis)
30. [Caching System and Preemption Safety](#30-caching-system-and-preemption-safety)
31. [Code Architecture](#31-code-architecture)
32. [Running the Pipeline](#32-running-the-pipeline)
33. [Runtime Estimates](#33-runtime-estimates)
34. [Current Status](#34-current-status)
35. [Preliminary Data: Generation Statistics](#35-preliminary-data-generation-statistics)
36. [What This Method Does NOT Do](#36-what-this-method-does-not-do)
37. [What This Method Already Tells Us](#37-what-this-method-already-tells-us)
38. [Connections to Methods 2 and 3](#38-connections-to-methods-2-and-3)

---

## 1. Purpose of This Method

Method 1 is the baseline. It implements the SGen-Semi algorithm from Lee et al.
(NeurIPS 2024) exactly as described in the paper, with no modifications for domain
shift. The purpose is to demonstrate the **motivating failure**: SGen's PAC guarantee
on false discovery rate holds in-domain (when calibration and test data come from the
same distribution) but **breaks under domain shift** (when the test distribution
differs from calibration).

This failure is the reason the project exists. Without demonstrating it clearly and
quantitatively, there is no motivation for Methods 2 and 3.

---

## 2. The Research Question

> Can a selective generation system for LLMs maintain provable PAC guarantees on its
> false discovery rate even when test queries come from a different domain than the
> calibration data?

Method 1 establishes one half of this question: **no, the vanilla method cannot**.
Specifically:

- Calibrate SGen-Semi on Natural Questions (NQ), a dataset of real Google search queries.
- Test on NQ itself (in-domain) and on TriviaQA (shifted domain — trivia-style factual questions).
- The PAC guarantee says: P{FDR-E ≤ ε} ≥ 1 - δ, i.e., with probability at least 98%, the false discovery rate is at most 25%.
- On NQ-test, this should hold (validity rate ≈ 98%).
- On TriviaQA, this should fail (validity rate drops well below 98%).

The gap between NQ validity and TQA validity is the domain shift effect. The larger
this gap, the stronger the motivation for DS-SGen.

---

## 3. What This Method Is and Is Not

### What it is

- A faithful reimplementation of SGen-Semi (Algorithm 2 from Lee et al., NeurIPS 2024)
- A complete 4-stage pipeline: data loading → LLM generation → entailment scoring → SGen-Semi algorithm
- A baseline establishing the domain shift failure
- A source of cached intermediate data (generations, entailment scores) consumed by Methods 2 and 3

### What it is not

- It is not an attempt to fix domain shift. That is Methods 2 and 3.
- It does not use importance reweighting, domain classifiers, or embedding-based methods.
- It does not produce the final figures for the paper. It produces the raw numbers that the final analysis script will consume.
- It is not a single run. The SGen-Semi algorithm runs 100 random calibration/test splits and reports aggregate statistics. This captures the randomness in the calibration split, which is essential for measuring validity rates.

---

## 4. The FDR-E Metric: Why Not Accuracy

### The problem with accuracy

Standard accuracy measures how often the model is correct across all questions. But in
a selective generation system, the model does not answer all questions. It answers some
and abstains on others. Accuracy of the answered subset is what matters.

Consider a model that answers 80 questions out of 100. If 10 of those 80 answers are
wrong, accuracy is 70/80 = 87.5%. But the user who received 80 answers got 10 wrong
ones. From the user's perspective, 12.5% of the information they received was wrong.

### False Discovery Rate with Entailment (FDR-E)

SGen defines:

```
FDR-E = (number of selected answers that are wrong) / (number of selected answers)
```

where "wrong" is determined by textual entailment, not exact string match.

If the model selects 80 answers and 10 are wrong (the greedy answer does NOT entail
the reference answer), then FDR-E = 10/80 = 0.125.

### Why entailment, not exact match

Exact string matching is too strict. Consider:

- Question: "When was the last time anyone was on the moon?"
- Reference: "14 December 1972 UTC"
- Model answer: "The last time humans visited the moon was during the Apollo 17 mission in December 1972"

Exact match: WRONG (strings differ).
Entailment: CORRECT (the model's answer logically implies the reference is true).

Using exact match would reject many correct answers, underestimating model quality and
making the FDR-E guarantee artificially easy to satisfy. Entailment-based evaluation
is strictly harder to satisfy (fewer false negatives) but more faithful to actual
correctness.

### The entailment direction

SGen uses **unidirectional** entailment for correctness:

```
NLI(greedy_answer → reference_answer)
```

meaning: does the greedy answer entail the reference answer? If yes, the answer is
marked correct. The direction matters:

- "December 1972" → "14 December 1972 UTC": does NOT entail (less specific)
- "The Apollo 17 mission on 14 December 1972" → "14 December 1972 UTC": ENTAILS

We check if the model's answer is specific enough to imply the reference, not the
other way around.

---

## 5. The PAC Guarantee

### The promise

SGen provides a Probably Approximately Correct (PAC) guarantee:

```
P{FDR-E(ĝ) ≤ ε} ≥ 1 - δ
```

In words: with probability at least (1 - δ), the false discovery rate of the learned
selection function ĝ is at most ε.

For our settings:
- ε = 0.25 (at most 25% of answered questions are wrong)
- δ = 0.02 (this holds with at least 98% probability)

The probability is over the randomness in the calibration data. If you drew a different
random calibration set from the same distribution, you would get different thresholds
and potentially different FDR-E on the test set. The guarantee says that for 98% of
such random draws, the FDR-E will be at most 25%.

### Why ε = 0.25

This is the main experimental setting in the SGen paper (Table 1). It represents a
practical tradeoff: 25% error rate is tolerable in many informational QA settings but
not in high-stakes domains. The paper also tests ε = 0.10 and ε = 0.50 as sensitivity
checks, but 0.25 is the headline number.

### Why δ = 0.02

This gives a 98% confidence level, matching the SGen paper's experiments. It means
that across 100 random calibration splits, we expect approximately 98 to satisfy
FDR-E ≤ ε. In practice, we measure the **validity rate**: the fraction of 100 splits
where FDR-E ≤ ε on the test set. The PAC guarantee predicts this should be ≥ 98%.

### The i.i.d. assumption

The PAC guarantee requires that calibration and test data are drawn i.i.d. from the
same distribution. Under this assumption, the binomial concentration inequalities
that underpin the Clopper-Pearson bound are valid.

Under domain shift, this assumption fails. The calibration data (NQ) and test data
(TQA) come from different distributions. The binomial tail bounds become invalid
because the calibration sample is not representative of the test distribution.
Specifically, the error rate estimated on NQ-calibration may underestimate (or
overestimate) the error rate on TQA, breaking the guarantee.

---

## 6. The Neuro-Selection Function

SGen uses a **neuro-selection function** that combines two confidence signals:

```
ĝ(x) = 1   if fM1(x) ≥ τ₁  AND  fM2(x) ≥ τ₂
ĝ(x) = 0   otherwise
```

where:
- fM1(x) = mean log-probability of the greedy answer (generation confidence)
- fM2(x) = self-consistency score among K=5 sampled answers (semantic agreement)
- τ₁, τ₂ = thresholds learned from calibration data
- ĝ(x) = 1 means "answer this question"
- ĝ(x) = 0 means "abstain (say I don't know)"

The intuition: a model should only answer when it is **both** confident in its
generation (high log-prob) **and** consistent across multiple samples (high
self-consistency). Either signal alone is insufficient:

- High log-prob but low consistency: the model confidently generates one answer but
  produces different answers when sampled — unreliable.
- High consistency but low log-prob: multiple samples agree, but the model assigns
  low probability — the agreement might be a systematic hallucination.

The AND combination requires both signals to clear their thresholds.

---

## 7. The Datasets

### Natural Questions (NQ-Open)

| Property | Value |
|----------|-------|
| HuggingFace ID | `google-research-datasets/nq_open` |
| Split | `validation` |
| Size | 3,610 questions |
| Fields | `question` (str), `answer` (list[str]) |
| Download size | ~2 MB (metadata only; the full NQ dataset with documents is 42 GB) |
| Character of questions | Real Google search queries submitted by users |

NQ-Open is the open-domain subset of Google's Natural Questions dataset. Each entry
is a real query that a user typed into Google search, paired with one or more valid
answers extracted from Wikipedia by human annotators.

**Example questions from our cached data:**

```
[0]    Q: "when was the last time anyone was on the moon"
       A: "14 December 1972 UTC"  (also: "December 1972")

[100]  Q: "who has won the most games in nfl 2017"
       A: "Dallas Cowboys"

[500]  Q: "who turned out to be the mother on how i met your mother"
       A: "Tracy McConnell"

[1000] Q: "what does g stand for in baseball stats"
       A: "Games"

[2000] Q: "the region that stretches between the black and caspian seas"
       A: "The Caucasus Mountains"

[3000] Q: "what is the name of the skin between your nostrils"
       A: "the nasal septum"
```

Questions are informal, varied in topic, and reflect how people naturally ask search
engines. They range from pop culture to geography to anatomy. Many are incomplete
sentences ("the region that stretches...") because that is how people actually search.

### Why NQ as the calibration domain

NQ is the same dataset used in the original SGen paper for calibration and in-domain
testing. Using the same dataset allows direct comparison with their reported results.
NQ's diversity (many topics, many question styles) makes it a realistic calibration
distribution.

### TriviaQA (unfiltered, no-context)

| Property | Value |
|----------|-------|
| HuggingFace ID | `mandarjoshi/trivia_qa` |
| Config | `unfiltered.nocontext` |
| Split | `validation` |
| Raw size | 11,313 questions |
| Downsampled to | 3,610 questions (to match NQ size) |
| Downsample seed | 42 |
| Fields | `question` (str), `answer.value` (str), `answer.aliases` (list[str]) |
| Download size | ~633 MB (not 29 GB — the `nocontext` config skips document downloads) |

TriviaQA consists of trivia-style factual questions written by trivia enthusiasts.
The `unfiltered.nocontext` config provides the same questions and answers as the full
`unfiltered` config but without the 29 GB of supporting documents (which we do not need
since we are doing closed-book QA).

**Example questions from our cached data:**

```
[0]    Q: "What type of dance shoe has a specially hardened sole or attached metal plates?"
       A: "Tap shoe"  (19 aliases)

[100]  Q: "Four score and seven years ago our fathers brought forth on this continent..."
       A: "The Gettysburg Address"

[500]  Q: "Which play is featured in the film The Producers?"
       A: "Springtime for Hitler"

[1000] Q: "Which ship survived Pearl Harbour but was sank in 1982"
       A: "General Belgrano"

[2000] Q: "The first Paralympic Games to officially tie in with the Summer Olympics were he..."
       A: "Rome"

[3000] Q: "The internal angles of a quadrilateral add up to how many degrees?"
       A: "360"
```

TriviaQA questions are more formally structured, often use trivia-style phrasing
("Which X is known for Y?"), and tend to be more specific. The answer alias system
is richer — "Tap shoe" has 19 valid aliases (e.g., "tap shoes", "tap dancing shoes",
"tap-dance shoes").

### Why TriviaQA as the shifted domain

The domain shift between NQ and TQA is **real and measurable**:

1. **Question style:** NQ = natural search queries ("when was the last time..."),
   TQA = trivia phrasing ("Which ship survived Pearl Harbour...").
2. **Topic distribution:** NQ covers whatever people search for (heavily weighted
   toward pop culture, current events, definitions). TQA covers trivia topics
   (history, geography, science, arts).
3. **Answer structure:** NQ answers are often short phrases or dates. TQA answers
   tend to be proper nouns with many aliases.

This is a **covariate shift**: the input distribution changes (P_test(X) ≠ P_cal(X))
but the relationship between question difficulty and model correctness is roughly
preserved (P_test(Y|X) ≈ P_cal(Y|X)). The model's ability to answer a question given
its difficulty does not change between domains — what changes is the distribution of
difficulty levels encountered.

### Why downsample TQA to 3,610

Matching sizes ensures that any differences in SGen-Semi behavior between NQ and TQA
are due to the domain shift, not to dataset size effects. The Clopper-Pearson bound
is sensitive to sample size (larger samples → tighter bounds → higher efficiency).
If TQA had 11,313 questions while NQ had 3,610, the algorithm would have a structural
advantage on TQA from the larger test set, confounding the domain shift comparison.

### Normalized schema

Both datasets are normalized to a common schema in `data_loading.py`:

```json
{
    "idx": 0,
    "question": "when was the last time anyone was on the moon",
    "reference_answer": "14 December 1972 UTC",
    "all_answers": ["14 December 1972 UTC", "December 1972"],
    "dataset": "nq"
}
```

For NQ, `reference_answer` is `answer[0]` and `all_answers` is the full `answer` list.
For TQA, `reference_answer` is `answer["value"]` (the primary answer) and `all_answers`
is the deduplicated union of `[answer["value"]] + answer["aliases"]`.

The `reference_answer` field is used for entailment scoring (greedy_answer → reference).
The `all_answers` field is available for future exact-match evaluation but is not used
in the current pipeline.

---

## 8. The Generator Model: LLaMA-3.1-8B-Instruct

| Property | Value |
|----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Parameters | 8 billion |
| Architecture | Transformer decoder, 32 layers, 4096 hidden dim |
| Inference dtype | float16 (~16 GB VRAM) |
| GPU | NVIDIA RTX A6000 (49,140 MiB / ~50.9 GB total) |
| CUDA version | 12.4 |
| PyTorch version | 2.6.0+cu124 |
| Transformers version | 5.5.0 |
| Loading method | `AutoModelForCausalLM.from_pretrained(device_map="auto", dtype=torch.float16)` (note: `dtype=` is the transformers 5.x parameter name; older versions used `torch_dtype=`) |
| Local path | `/data/user_data/anshulk/dsgen/model_cache/Llama-3.1-8B-Instruct` |
| Load time | ~14 seconds (from local disk) |

### Why LLaMA-3.1-8B-Instruct

Three reasons:

1. **Size vs. capability tradeoff.** At 8B parameters and ~16 GB VRAM in fp16, the
   model fits comfortably on a single A6000 alongside DeBERTa for entailment scoring
   (~6 GB). A 70B model would require multi-GPU or quantization, adding engineering
   complexity without changing the research question (which is about the statistical
   framework, not the base model).

2. **Instruct tuning provides chat template.** The instruct version responds to
   system prompts and follows instructions ("Answer concisely in one sentence"),
   producing focused answers rather than rambling completions. This matters because
   SGen's entailment scoring works best when answers are concise and factual. A base
   model would require more prompt engineering to elicit focused answers.

3. **Open weights and reproducibility.** Anyone with a HuggingFace account can
   download the same weights and reproduce our results. No API access, no rate
   limits, no versioning ambiguity.

### Why NOT base model

Unlike the arithmetic-geometry project where base model avoids RLHF confounds, here
the instruct model is preferred because:

- We are not studying the model's internal representations. We are studying the
  statistical properties of its answers.
- The chat template gives consistent, concise answers that are easier to evaluate
  via entailment.
- The SGen paper itself uses instruction-tuned models.

### Chat template

LLaMA-3.1-Instruct uses a specific chat template with `<|begin_of_text|>`,
`<|start_header_id|>`, and other special tokens. We use `tokenizer.apply_chat_template()`
rather than formatting prompts manually:

```python
messages = [
    {"role": "system", "content": "Answer the following question concisely in one sentence."},
    {"role": "user", "content": question},
]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
)
```

The `add_generation_prompt=True` flag appends the `<|start_header_id|>assistant` tokens
so the model knows to start generating its response.

### The apply_chat_template return type issue

In transformers 5.5.0, `apply_chat_template` with `return_tensors="pt"` can return
either a raw tensor (older behavior) or a `BatchEncoding` object (newer behavior).
Our code handles both:

```python
result = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
if hasattr(result, "input_ids"):
    return result.input_ids    # BatchEncoding → extract tensor
return result                  # Already a tensor
```

The first baseline run (job 6943087) crashed on this exact issue before the fix was
added. The second run (job 6943094) uses the fixed code and is running successfully.

---

## 9. Response Generation: Greedy + Sampled

For each question, the pipeline generates two types of responses:

### Pass 1: Greedy decoding with log-probabilities

```python
greedy_out = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    output_logits=True,
    return_dict_in_generate=True,
)
```

- `do_sample=False` → greedy decoding (deterministic, always picks the highest-probability token)
- `output_logits=True` → returns raw logits at each generation step (before temperature/top-p processing)
- `max_new_tokens=100` → generates up to 100 tokens

The greedy answer is the model's single best response. Its mean log-probability
becomes the fM1 score.

### Pass 2: Sampled responses

```python
sampled_out = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
    num_return_sequences=5,
)
```

- `do_sample=True` → stochastic sampling
- `temperature=0.7` → moderate randomness (lower than 1.0 = less random, but not greedy)
- `num_return_sequences=5` → generates K=5 independent samples in one call

The 5 sampled answers are used to compute the fM2 self-consistency score.

### Why two separate passes

The greedy pass requires `output_logits=True` to extract log-probabilities, which
uses more memory (stores the full vocabulary distribution at each step). The sampled
pass requires `num_return_sequences=5`, which also uses more memory (5x the sequence
storage). Combining both in a single call would require `output_logits=True` with
`num_return_sequences=6`, storing 6x the logits — unnecessary since we only need
logits from the greedy decoding.

### Output per question

```json
{
    "idx": 0,
    "question": "when was the last time anyone was on the moon",
    "greedy_answer": "The last time humans visited the moon was during the Apollo 17 mission in December 1972, when astron...",
    "mean_logprob": -0.0472,
    "token_logprobs": [-0.0012, -0.0034, -0.0089, ...],
    "sampled_answers": [
        "The last time humans visited the moon was during the Apollo 17 mission in Decemb...",
        "The last time humans visited the moon was during the Apollo 17 mission in Decemb...",
        "The last humans to walk on the moon were the crew of Apollo 17 in December 1972...",
        "The last time humans set foot on the moon was during the Apollo 17 mission...",
        "The last manned mission to the moon was Apollo 17, which returned on December 19..."
    ]
}
```

---

## 10. The Entailment Model: DeBERTa-v2-xxlarge-mnli

| Property | Value |
|----------|-------|
| Model | `microsoft/deberta-v2-xxlarge-mnli` |
| Parameters | 1.5 billion |
| Architecture | DeBERTa-v2 (enhanced attention with disentangled matrices) |
| Task | Natural Language Inference (NLI) — 3-class classification |
| Classes | {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT} |
| Inference dtype | float16 (~6 GB VRAM) |
| Batch size | 64 (fits comfortably with LLaMA already unloaded) |
| Max sequence length | 512 tokens |

### Why DeBERTa-v2-xxlarge-mnli

This is the same entailment model used in the SGen paper and its upstream code
repository ([ml-postech/selective-generation](https://github.com/ml-postech/selective-generation)).
Using the same model ensures our entailment scores are comparable to the paper's
results.

### CRITICAL: Label order

```
DeBERTa-v2-xxlarge-mnli:     {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
cross-encoder/nli-deberta-v3: {0: contradiction, 1: entailment, 2: neutral}
```

These two popular NLI models have **different label orders**. Getting this wrong would
swap NEUTRAL and ENTAILMENT, making every measurement meaningless. Our code uses
`ENTAILMENT_IDX = 2` (hardcoded in `entailment_scoring.py:28`), which is correct for
`deberta-v2-xxlarge-mnli`.

This is documented in the config file (`configs/default.yaml`) and in the module
docstring as a warning.

---

## 11. Scoring Function fM1: Mean Log-Probability

### Definition

For a greedy-decoded answer with tokens t₁, t₂, ..., tₙ:

```
fM1 = (1/n) * Σᵢ log P(tᵢ | t₁, ..., tᵢ₋₁, prompt)
```

This is the mean of the per-token log-probabilities under the model's own distribution.
Higher fM1 (closer to 0) means the model assigns high probability to each token in
its answer — it is "confident" in this answer.

### How log-probs are extracted

The `output_logits=True` flag in `model.generate()` returns the raw logits (before
softmax) at each generation step. We convert to log-probabilities:

```python
def _extract_logprobs_from_scores(scores, generated_ids):
    token_logprobs = []
    for step_idx, step_logits in enumerate(scores):
        logits = step_logits.squeeze(0).float()       # (vocab_size,)
        log_probs = F.log_softmax(logits, dim=-1)     # normalize to log-probs
        token_id = generated_ids[step_idx].item()      # the token that was chosen
        token_logprobs.append(log_probs[token_id].item())
    return token_logprobs
```

Each step's logits are cast to float32 before log_softmax to avoid numerical issues
with float16 softmax at the extreme tails of the distribution.

### Why `output_logits` not `output_scores`

In transformers 5.x, `output_logits=True` returns the **raw** logits before any
temperature or top-p processing. `output_scores=True` returns logits **after**
processing (e.g., after temperature scaling). Since we use greedy decoding
(`do_sample=False`), no processing is applied and both would be identical. But
`output_logits` is the semantically correct choice: we want raw model confidence,
not processed scores.

The returned attribute is `greedy_out.logits` (a tuple of tensors, one per generation
step).

### Preliminary statistics from cached data

From the first 2,900 NQ questions (partial cache, ~80% complete as of April 3).
These statistics will shift slightly when all 3,610 questions are generated.

| Statistic | Value |
|-----------|-------|
| Count | 2,900 questions (of 3,610) |
| Mean of fM1 | -0.2253 |
| Std of fM1 | 0.1382 |
| Min (least confident) | -0.8869 |
| Max (most confident) | -0.0005 |
| Median | -0.1971 |
| Mean token count per answer | 37.2 tokens |
| Min token count | 4 tokens |
| Max token count | 100 tokens (hit max_new_tokens limit) |

The distribution is right-skewed (median closer to 0 than the mean): most answers
have moderate confidence, with a long tail of low-confidence answers.

### Why mean, not sum

Sum log-prob penalizes longer answers (more tokens = more terms in the sum, each
≤ 0). A 100-token answer with per-token log-prob -0.05 would have sum -5.0, while a
10-token answer with per-token log-prob -0.10 would have sum -1.0. The short answer
looks "more confident" by sum even though each of its tokens is less confident.

Mean log-prob normalizes by length, measuring per-token confidence regardless of
answer length. This is the standard choice in the SGen paper.

### What fM1 captures and what it misses

fM1 captures the model's generation confidence — how likely it thinks each token is
given the preceding context. High fM1 means the model "flows naturally" through the
answer without hesitation.

fM1 does NOT capture semantic correctness. A model can confidently generate a wrong
answer (hallucination). This is why fM1 alone is insufficient for selection — it must
be combined with fM2 (self-consistency).

---

## 12. Scoring Function fM2: Self-Consistency via Bidirectional NLI

### Definition

Given K=5 sampled answers s₁, s₂, ..., s₅:

1. For all C(5,2) = 10 unordered pairs (i, j), check bidirectional entailment:
   - NLI(sᵢ → sⱼ) = ENTAILMENT?
   - NLI(sⱼ → sᵢ) = ENTAILMENT?
2. The pair "agrees" if BOTH directions have argmax = ENTAILMENT.
3. fM2 = (number of agreeing pairs) / C(K,2)

```
fM2 = (number of bidirectionally entailing pairs) / 10
```

fM2 ∈ {0/10, 1/10, 2/10, ..., 10/10} = {0.0, 0.1, 0.2, ..., 1.0}.

### Why bidirectional

Unidirectional entailment is asymmetric. "The capital of France is Paris, located on
the Seine" entails "Paris" but "Paris" does not entail the longer statement. Two
answers that are both correct but at different levels of specificity would not agree
unidirectionally.

Bidirectional entailment requires mutual implication — the answers must be semantically
equivalent, not just compatible. This is a stricter definition of "agreement."

### Why K=5

The SGen paper uses K=5, following Kuhn et al. (Nature 2024, "Semantic Uncertainty")
who found diminishing returns past K=5 for self-consistency metrics. With K=5:

- C(5,2) = 10 unordered pairs
- 2 × 10 = 20 directed NLI calls per question (both directions for each pair)
- Total NLI calls for self-consistency: 20 × 3,610 = 72,200 per dataset
- Total NLI calls for correctness: 1 × 3,610 = 3,610 per dataset
- Grand total: (72,200 + 3,610) × 2 datasets = 151,620 NLI calls

At batch size 64 with DeBERTa, this takes approximately 3-5 minutes of GPU time per
dataset.

### The pairwise entailment matrix

For debugging and analysis, we store the full K×K directed entailment matrix:

```
matrix[i][j] = True if NLI(sᵢ → sⱼ) has argmax = ENTAILMENT
```

This is a K×K boolean matrix (diagonal is ignored). The bidirectional agreement
count is computed from the upper triangle of the AND of the matrix with its transpose.

### What fM2 captures

fM2 measures **semantic agreement across independent samples**. If the model
generates the same answer (semantically) across 5 different random samples, the answer
is likely to be a stable output of the model, not a sampling artifact.

Low fM2 means the model gives different answers depending on the random seed — its
knowledge about this question is unreliable. High fM2 means the model consistently
produces the same semantic content regardless of sampling randomness.

### The connection to semantic uncertainty

Kuhn et al. (Nature 2024) formalized this intuition as "semantic uncertainty."
Their key insight: traditional uncertainty measures (entropy, token probabilities)
operate at the token level and are sensitive to paraphrase. Two semantically identical
answers with different word choices would appear uncertain by token-level metrics.
Semantic uncertainty uses entailment-based clustering to group semantically equivalent
outputs, then measures uncertainty over clusters. fM2 is a simplified version of
this: instead of forming clusters, it counts pairwise agreements.

---

## 13. Correctness Scoring: Unidirectional Entailment

For each question, we compute:

```python
entail_score, entail_label = score_correctness(model, tokenizer, greedy_answer, reference_answer, batch_size)
```

where:
- `entail_score` = P(ENTAILMENT) — the softmax probability of the ENTAILMENT class. Continuous in [0, 1].
- `entail_label` = 1 if argmax = ENTAILMENT, else 0. Binary.

### Two uses of the same NLI call

The same NLI evaluation serves two purposes:

1. **entail_score** (continuous) → used for conformal pseudo-labeling. The conformal
   threshold τ_CP operates on continuous scores to decide which Z_U examples are
   pseudo-labeled as "correct."

2. **entail_label** (binary) → used for FDR-E evaluation. When measuring the actual
   FDR-E on a test set, we need a binary correct/wrong label. argmax = ENTAILMENT
   is the binary decision.

### Output per question

```json
{
    "idx": 0,
    "entail_score": 0.873,
    "entail_label": 1,
    "fM2": 0.9,
    "pairwise_entailments": [[false, true, true, true, true], ...]
}
```

---

## 14. The SGen-Semi Algorithm: Full Mathematical Derivation

SGen-Semi (Algorithm 2 in Lee et al.) operates on calibration data from a single
source domain. It uses conformal prediction for pseudo-labeling and Clopper-Pearson
bounds with Bonferroni correction for PAC threshold selection.

The algorithm has three mathematical pillars:

1. **Conformal prediction** for pseudo-labeling unlabeled data
2. **Clopper-Pearson bounds** for binomial proportion confidence intervals
3. **Bonferroni correction** for multiple hypothesis testing across the threshold grid

Each is detailed in its own section below.

---

## 15. Step 1: Data Splitting

### The three-way split

NQ data (3,610 questions) is randomly split into:

```
NQ (3,610)
├── Calibration (70% = 2,527 questions)
│   ├── Z_U: unlabeled (75% of cal = 1,895 questions)
│   └── Z_E: labeled   (25% of cal = 632 questions)
└── Test (30% = 1,083 questions)
```

The split uses `np.random.RandomState(seed)` for reproducibility. Each of the 100
splits uses a different seed (base_seed + split_index).

### Why 70/30 calibration/test

More calibration data → tighter bounds → higher efficiency (more questions answered).
More test data → more reliable FDR-E estimates. The 70/30 split is a standard choice
that gives sufficient data for both purposes. With 2,527 calibration examples, the
Clopper-Pearson bounds are reasonably tight.

### Why 75/25 Z_U/Z_E within calibration

Z_E is the labeled set used to compute the conformal threshold. Z_E examples have
entailment scores that serve as calibration scores for the conformal prediction.
Z_U is the unlabeled set that gets pseudo-labeled and then used for threshold
selection.

More Z_U → more data for the grid search → more precise threshold selection.
More Z_E → more reliable conformal threshold → better pseudo-labels.

The 75/25 split gives Z_E = 632 questions, which is ample for a conformal quantile
(the quantile is the ⌈(n+1)(1-ε_e)⌉-th smallest score — with n=632, this is well-
determined). Z_U = 1,895 questions provides a good-sized dataset for the grid search.

### What "labeled" and "unlabeled" mean here

The terminology comes from the semi-supervised learning framing:

- **Z_E (labeled):** We use the entailment score as a continuous "label" for conformal
  calibration. These examples tell us the distribution of entailment scores for the
  population. From Z_E, we compute the conformal threshold that will be used to
  generate pseudo-labels.

- **Z_U (unlabeled):** We pretend we do not know the true correctness of these
  examples. Instead, we pseudo-label them using the conformal threshold from Z_E.
  Then we use these pseudo-labeled examples for the grid search.

In practice, we have entailment scores for ALL examples (we ran the NLI model on
everything). The "unlabeled" framing is a methodological choice: SGen-Semi is designed
for settings where you have a small labeled set and a larger unlabeled set. We simulate
this by withholding labels from Z_U and using conformal prediction to recover them.

---

## 16. Step 2: Conformal Pseudo-Labeling

### The conformal threshold

Given Z_E = {(x₁, s₁), ..., (xₙ, sₙ)} where sᵢ is the entailment score of example i:

```
τ_CP = sorted(s₁, ..., sₙ)[k - 1]

where k = ⌈(n + 1)(1 - ε_e)⌉
```

This is the **split conformal prediction quantile** from Vovk et al. (2005). Under
exchangeability, a new test point's score will exceed τ_CP with probability at least
(1 - ε_e).

With our settings:
- n = |Z_E| ≈ 632
- ε_e = 0.10 (conformal error rate)
- k = ⌈(633)(0.90)⌉ = ⌈569.7⌉ = 570
- τ_CP = the 570th smallest entailment score out of 632

### Pseudo-labeling Z_U

For each example in Z_U:

```
pseudo_label(xᵢ) = 1   if entail_score(xᵢ) ≥ τ_CP
                   0   otherwise
```

Examples with entailment score above the conformal threshold are pseudo-labeled as
"correct." This is a conservative pseudo-labeling: with probability at least (1 - ε_e),
a truly correct example will be pseudo-labeled as correct (assuming exchangeability).

### Why ε_e = 0.10

The conformal error rate controls the quality of pseudo-labels. Lower ε_e means the
threshold is higher, so fewer examples are pseudo-labeled as correct, but those that
are labeled are more likely to truly be correct. ε_e = 0.10 means the conformal
guarantee allows up to 10% false non-entailment rate in the pseudo-labels — a
moderate setting from the SGen paper.

### What happens if k > n

If ⌈(n + 1)(1 - ε_e)⌉ > n, the conformal threshold is set to infinity, meaning NO
examples are pseudo-labeled as correct. This happens when Z_E is too small or ε_e is
too low. With n ≈ 632 and ε_e = 0.10, k = 570 < 632, so this does not arise.

---

## 17. Step 3: Threshold Grid Search with PAC Constraint

### The grid

We build a 50×50 grid of candidate thresholds:

```
tau1_grid = 50 evenly-spaced percentiles of fM1 values in Z_U
tau2_grid = 50 evenly-spaced percentiles of fM2 values in Z_U
```

Each candidate is a (τ₁, τ₂) pair. The grid has |H| = |tau1_grid| × |tau2_grid|
candidates. After deduplication (np.unique removes identical percentile values), |H|
is at most 2,500 but often smaller because fM2 has a discrete distribution (only 11
possible values: 0.0, 0.1, ..., 1.0).

### For each candidate (τ₁, τ₂)

1. **Select:** S = {i ∈ Z_U : fM1(xᵢ) ≥ τ₁ AND fM2(xᵢ) ≥ τ₂}
2. **Count failures:** m = |S|, f = |{i ∈ S : pseudo_label(xᵢ) = 0}|
3. **Clopper-Pearson upper bound:** CP_upper = β⁻¹(1 - δ_adj; f + 1, m - f)
4. **PAC check:** Is CP_upper ≤ ε?
5. **Efficiency:** If yes, compute efficiency = m / |Z_U|

### The objective

Among all (τ₁, τ₂) pairs that pass the PAC check (CP_upper ≤ ε), select the pair
with **maximum efficiency** (highest fraction of Z_U examples selected). This
maximizes the number of questions the model answers while guaranteeing FDR-E ≤ ε.

```
(τ₁*, τ₂*) = argmax{m / |Z_U|} subject to CP_upper(f, m, δ_adj) ≤ ε
```

### What if no candidate passes

If no (τ₁, τ₂) pair satisfies the PAC constraint, the algorithm sets τ₁ = τ₂ = None,
meaning the model abstains on ALL questions. FDR-E = 0 (no errors because no answers),
efficiency = 0. This is a valid but degenerate outcome — the guarantee trivially holds
but the model is useless.

---

## 18. Step 4: Evaluation on Test Sets

The learned thresholds (τ₁*, τ₂*) are applied to two test sets:

### NQ-test (in-domain, 1,083 questions)

```
selected = {i : fM1(xᵢ) ≥ τ₁* AND fM2(xᵢ) ≥ τ₂*}
n_wrong = |{i ∈ selected : entail_label(xᵢ) = 0}|
FDR-E = n_wrong / |selected|
efficiency = |selected| / 1083
valid = (FDR-E ≤ ε)
```

### TriviaQA (shifted domain, all 3,610 questions)

Same computation but on the full TQA dataset. Note that TQA is never used for
calibration — it is entirely out-of-distribution from the algorithm's perspective.

### Why evaluate on ALL of TQA

The NQ test set is 30% of NQ (1,083 questions). TQA is evaluated on all 3,610
questions because none of TQA is used for calibration. Using the full TQA set gives
the most precise estimate of the domain-shifted FDR-E and efficiency.

---

## 19. The Clopper-Pearson Bound: Why This Specific Bound

### What it computes

Given f failures out of m trials (binomial data), the Clopper-Pearson upper bound
at confidence level (1 - α) is:

```
CP_upper(f, m, α) = β⁻¹(1 - α; f + 1, m - f)
```

where β⁻¹ is the inverse CDF (quantile function) of the Beta distribution.

In Python:

```python
from scipy.stats import beta as beta_dist
cp_upper = beta_dist.ppf(1 - alpha, failures + 1, total - failures)
```

This is an **exact** confidence interval — it does not use normal approximations or
asymptotic expansions. It is valid for any sample size, including small samples.

### Why Clopper-Pearson

Three reasons:

1. **Exactness.** Unlike Wald intervals (p̂ ± z√(p̂(1-p̂)/n)), Clopper-Pearson never
   has below-nominal coverage. It is conservative: the true coverage is at least
   (1 - α), never less.

2. **Small-sample validity.** When m (selected examples) is small (e.g., 50), normal
   approximations can fail badly. Clopper-Pearson works regardless of m.

3. **One-sided.** We need only an upper bound on the failure rate, not a two-sided
   interval. The one-sided Clopper-Pearson bound is the most efficient exact bound.

### Edge cases

```python
if total == 0:
    return 0.0    # No selected examples → failure rate is 0 by convention
if failures == total:
    return 1.0    # All selected examples failed → failure rate is 1
```

---

## 20. Bonferroni Correction: Why and How

### The multiple testing problem

We test |H| candidate threshold pairs. For each, we check whether CP_upper ≤ ε.
If we use confidence level δ for each test, the probability that ANY of the |H| tests
gives a false positive is up to |H| × δ — which can exceed 1, making the guarantee
meaningless.

### The Bonferroni correction

To control the overall failure probability at δ, we test each candidate at level
δ_adj = δ_CP / |H|:

```
δ_CP = δ - δ_p                  # reserve δ_p for pseudo-labeling error
δ_adj = δ_CP / |H|              # Bonferroni divide by number of candidates
```

With our settings:
- δ = 0.02
- δ_p = 1e-5 (negligible)
- δ_CP = 0.02 - 1e-5 ≈ 0.01999
- |H| ≤ 2,500 (50 × 50 grid)
- δ_adj ≈ 0.01999 / 2500 ≈ 8.0e-6

This is a very small per-test confidence level, which makes the Clopper-Pearson bounds
wider (more conservative). The bound says: "even accounting for testing 2,500
candidates, the probability that the best candidate has FDR-E > ε is at most δ."

### Why δ_p = 1e-5

δ_p is the probability that the conformal pseudo-labeling step fails. The conformal
guarantee (Section 16) holds with probability at least (1 - ε_e) under exchangeability.
δ_p accounts for the (extremely small) probability that the conformal calibration set
Z_E is pathological. The SGen paper sets this to 1e-5, which is so small it is
effectively zero.

### Why Bonferroni and not Bonferroni-Holm

Bonferroni is the simplest valid correction. Bonferroni-Holm is tighter (less
conservative) but more complex and harder to verify. Since the grid search already
considers efficiency as the objective, the slight conservatism of Bonferroni is
acceptable. The SGen paper uses Bonferroni.

---

## 21. Hyperparameter Table and Justifications

| Parameter | Symbol | Value | Source | Role |
|-----------|--------|-------|--------|------|
| FDR-E target | ε | 0.25 | SGen Table 1 | Max fraction of wrong answers among selected |
| PAC confidence | δ | 0.02 | SGen experiments | P{FDR-E ≤ ε} ≥ 1-δ = 98% |
| Pseudo-label failure prob | δ_p | 1e-5 | SGen default | Subtracted from δ before Bonferroni |
| Conformal error rate | ε_e | 0.10 | SGen default | Controls pseudo-label quality |
| Calibration fraction | cal_frac | 0.70 | SGen experimental design | 70% NQ for calibration, 30% for in-domain test |
| Unlabeled fraction | zu_frac | 0.75 | SGen experimental design | 75% of calibration as Z_U, 25% as Z_E |
| Sampled responses | K | 5 | SGen; Kuhn et al. | Number of samples for fM2 |
| Temperature | T | 0.7 | SGen default | Sampling temperature for K responses |
| Random splits | n_splits | 100 | SGen standard | Number of random calibration splits |
| Grid points per score | n_grid | 50 | Percentile-based | |H| ≤ 50² = 2,500 candidates |
| Max new tokens | — | 100 | Sufficient for 1-sentence answers | Generation length limit |
| NLI batch size | — | 64 | VRAM-constrained (~6 GB at batch 64) | DeBERTa inference batch |
| Gen save frequency | — | 50 | Incremental caching | Save every 50 questions during generation |
| Seed | — | 42 | Reproducibility | Base seed for all RNG |

Every parameter matches the SGen paper or the upstream code at
[ml-postech/selective-generation](https://github.com/ml-postech/selective-generation).
No parameters were tuned by us — this is a faithful reproduction.

---

## 22. The Pipeline: Four Stages

```
Stage 1: Data Loading
    NQ-Open (3,610) + TriviaQA (3,610, downsampled)
    ↓
Stage 2: LLM Generation (GPU-intensive, ~48 hours)
    For each question: greedy answer + fM1 + K=5 samples
    ↓
Stage 3: Entailment Scoring (GPU, ~10 minutes)
    For each question: correctness (entail_score, entail_label) + fM2
    ↓
Stage 4: SGen-Semi Algorithm (CPU only, ~30 seconds)
    100 random splits → threshold selection → evaluation
    ↓
Output: baseline_results.json
```

Each stage is independently cached. The pipeline can be restarted at any stage without
repeating previous stages. This is critical for SLURM preemption safety — if the job
is killed after completing generation for 2,000 questions, it resumes from question
2,001.

---

## 23. Stage 1: Data Loading — Code and Decisions

### Implementation: `ds_sgen/data_loading.py`

Two functions: `load_nq(cfg)` and `load_tqa(cfg)`, each following the pattern:

1. Check cache → if cache exists with correct count, return immediately
2. Download from HuggingFace
3. Normalize to common schema
4. Save to cache
5. Return

### NQ normalization

```python
records.append({
    "idx": i,
    "question": ex["question"],
    "reference_answer": ex["answer"][0],
    "all_answers": ex["answer"],
    "dataset": "nq",
})
```

NQ's `answer` field is a list of valid answers. We use the first as `reference_answer`.

### TQA normalization

```python
answer = ex["answer"]
primary = answer["value"]
aliases = answer.get("aliases", [])
all_ans = list(dict.fromkeys([primary] + aliases))
records.append({
    "idx": i,
    "question": ex["question"],
    "reference_answer": primary,
    "all_answers": all_ans,
    "dataset": "tqa",
})
```

TQA's answer is a dict with `value` (primary) and `aliases`. The `dict.fromkeys`
trick deduplicates while preserving order.

### TQA downsampling

```python
ds = ds.shuffle(seed=42).select(range(3610))
```

Deterministic shuffling with seed=42, then take the first 3,610. This is reproducible:
the same 3,610 questions are selected every time.

### Cache locations (actual)

```
/data/user_data/anshulk/dsgen/cache/nq_data.json     (792 KB, 3,610 records)
/data/user_data/anshulk/dsgen/cache/tqa_data.json     (2.0 MB, 3,610 records)
```

TQA is larger on disk because of the richer alias system (up to 19 aliases per answer
vs. typically 1-2 for NQ).

---

## 24. Stage 2: LLM Generation — Code and Decisions

### Implementation: `ds_sgen/generate_responses.py`

The function `generate_and_cache()` processes all questions in a dataset with
incremental caching:

1. Check cache → if complete, return
2. If partial cache exists, resume from where it left off
3. Load model + tokenizer (once)
4. For each question: call `generate_for_question()`
5. Save cache every 50 questions and at the end

### Incremental caching for preemption safety

```python
if (i + 1) % save_every == 0 or i == len(records) - 1:
    save_cache(results, cache_path)
```

If the SLURM job is preempted after processing question 2,050, the cache contains
2,050 results. On restart, `start_idx = len(results) = 2050`, and generation resumes
from question 2,050.

The `save_cache` function uses atomic writes:

```python
fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
with os.fdopen(fd, "w") as f:
    json.dump(data, f, indent=2)
os.replace(tmp_path, path)    # atomic on POSIX
```

This prevents a half-written JSON file if preemption happens during the write. The
`os.replace` is atomic on Linux (same filesystem), so the cache file is always either
the old version or the new version, never corrupted.

### System prompt

```
"Answer the following question concisely in one sentence."
```

This prompt was chosen to elicit focused, factual answers. Without it, the instruct
model tends to produce multi-paragraph responses that are harder to evaluate via
entailment and waste generation tokens.

### Generation progress logging

```python
if (i + 1) % 10 == 0 or i == len(records) - 1:
    print(f"    [{dataset_name.upper()}] {i+1}/{len(records)}: "
          f"logprob={gen['mean_logprob']:.3f}, "
          f"answer='{gen['greedy_answer'][:60]}...'")
```

Every 10 questions, the pipeline prints the question number, mean log-prob, and a
truncated answer. This provides visual progress feedback in the SLURM log.

---

## 25. Stage 3: Entailment Scoring — Code and Decisions

### Implementation: `ds_sgen/entailment_scoring.py`

The function `score_and_cache()` processes all questions with incremental caching
(every 200 questions).

### Batched NLI

The `_batch_nli()` function processes multiple (premise, hypothesis) pairs in batches
of 64:

```python
inputs = tokenizer(
    premises, hypotheses,
    padding=True, truncation=True, max_length=512,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    logits = model(**inputs).logits.float()
    probs = F.softmax(logits, dim=-1)
    argmaxes = logits.argmax(dim=-1)
```

Logits are cast to float32 before softmax (same rationale as log-prob extraction).

### NLI call count per question

For one question with K=5 sampled answers:

- Correctness: 1 NLI call (greedy → reference)
- Self-consistency: K(K-1) = 20 directed NLI calls (all ordered pairs excluding diagonal)
- Total: 21 NLI calls per question

For both datasets (3,610 questions each):

- Total NLI calls: 21 × 3,610 × 2 = 151,620

At batch size 64: ⌈151,620 / 64⌉ = 2,369 forward passes. DeBERTa-v2-xxlarge at fp16
processes a batch in ~50ms, so total time ≈ 2 minutes. In practice, the overhead of
data preparation and individual question processing loops makes this 3-5 minutes.

---

## 26. Stage 4: SGen-Semi Algorithm — Code and Decisions

### Implementation: `ds_sgen/sgen_semi.py`

The function `run_experiment()` orchestrates 100 random splits:

1. Merge records + generations + entailments into unified per-question dicts
2. For each split (seed = 42, 43, ..., 141):
   a. Call `_run_single_split()`
   b. Collect per-split metrics
3. Aggregate: mean FDR-E, mean efficiency, validity rate
4. Save to `baseline_results.json`

### Merging

```python
merged.append({
    "idx": rec["idx"],
    "question": rec["question"],
    "reference_answer": rec["reference_answer"],
    "greedy_answer": gen["greedy_answer"],
    "fM1": gen["mean_logprob"],
    "fM2": ent["fM2"],
    "entail_score": ent["entail_score"],
    "entail_label": ent["entail_label"],
    "dataset": rec["dataset"],
})
```

Each question now has all the fields needed for the algorithm: the two scoring
functions (fM1, fM2), the continuous entailment score (for conformal), and the binary
correctness label (for FDR-E evaluation).

### Vectorized grid search

The inner loop is implemented with numpy vectorization:

```python
zu_fM1 = np.array([r["fM1"] for r in z_u])
zu_fM2 = np.array([r["fM2"] for r in z_u])
zu_pseudo = np.array([r["pseudo_label"] for r in z_u])

for t1 in tau1_grid:
    selected = zu_fM1 >= t1    # boolean mask, vectorized
    for t2 in tau2_grid:
        sel = selected & (zu_fM2 >= t2)    # AND combination
        m = sel.sum()
        if m == 0:
            continue
        failures = int((sel & (zu_pseudo == 0)).sum())
        cp_upper = _clopper_pearson_upper(failures, int(m), delta_adj)
        if cp_upper <= epsilon:
            efficiency = m / len(z_u)
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_tau1 = t1
                best_tau2 = t2
```

The outer loop over t1 precomputes the fM1 mask once. The inner loop only needs the
AND with the fM2 mask. This is efficient: 2,500 iterations of fast numpy operations
complete in milliseconds.

---

## 27. Worked Example: One Complete Split

Let us walk through one complete split with concrete (illustrative) numbers to make the
algorithm tangible.

### Setup

Assume split_seed = 42, NQ has 3,610 questions.

**Step 1: Data split**

- Random permutation of indices [0, 1, ..., 3609]
- cal_size = floor(3610 × 0.70) = 2,527
- Calibration: indices [0:2527] → 2,527 questions
- NQ-test: indices [2527:3610] → 1,083 questions

**Step 2: Calibration sub-split**

- zu_size = floor(2527 × 0.75) = 1,895
- Z_U: first 1,895 calibration questions
- Z_E: remaining 632 calibration questions

**Step 3: Conformal threshold from Z_E**

- n = 632 entailment scores from Z_E
- k = ceil((633)(0.90)) = ceil(569.7) = 570
- Sort the 632 scores in ascending order
- τ_CP = the 570th smallest score

Say τ_CP = 0.72. This means: "to be pseudo-labeled as correct, a question's entailment
score must be at least 0.72."

**Step 4: Pseudo-label Z_U**

For each of the 1,895 Z_U questions:
- If entail_score ≥ 0.72 → pseudo_label = 1 (correct)
- If entail_score < 0.72 → pseudo_label = 0 (wrong)

Say 1,200 are pseudo-labeled correct and 695 are pseudo-labeled wrong.

**Step 5: Grid search**

Build tau1_grid from 50 percentiles of fM1 values in Z_U (e.g., [-0.85, -0.80, ..., -0.01]).
Build tau2_grid from 50 percentiles of fM2 values in Z_U (e.g., [0.0, 0.1, ..., 1.0]).
After np.unique, say |tau1_grid| = 48, |tau2_grid| = 11.
|H| = 48 × 11 = 528.
δ_adj = (0.02 - 1e-5) / 528 ≈ 3.79e-5.

Try (τ₁ = -0.20, τ₂ = 0.8):
- Selected: 850 out of 1,895 (fM1 ≥ -0.20 AND fM2 ≥ 0.8)
- Failures: 35 (selected AND pseudo_label = 0)
- CP_upper = beta.ppf(1 - 3.79e-5, 36, 815) = 0.063
- 0.063 ≤ 0.25? YES. Efficiency = 850/1895 = 0.449.

Try (τ₁ = -0.30, τ₂ = 0.6):
- Selected: 1,400 out of 1,895
- Failures: 220
- CP_upper = beta.ppf(1 - 3.79e-5, 221, 1180) = 0.183
- 0.183 ≤ 0.25? YES. Efficiency = 1400/1895 = 0.739.

The second candidate has higher efficiency (0.739 > 0.449) and still satisfies the
constraint. The grid search would keep (τ₁ = -0.30, τ₂ = 0.6) as the current best
and continue checking all remaining candidates.

**Step 6: Evaluate**

Suppose the best thresholds are (τ₁* = -0.30, τ₂* = 0.6).

On NQ-test (1,083 questions):
- Selected: 780 (fM1 ≥ -0.30 AND fM2 ≥ 0.6)
- Wrong: 150 (selected AND entail_label = 0)
- FDR-E = 150/780 = 0.192
- 0.192 ≤ 0.25? YES → valid = True
- Efficiency = 780/1083 = 0.720

On TriviaQA (3,610 questions):
- Selected: 2,100 (same thresholds applied)
- Wrong: 680
- FDR-E = 680/2100 = 0.324
- 0.324 ≤ 0.25? NO → valid = False
- Efficiency = 2100/3610 = 0.582

This single split shows the pattern: NQ-test is valid (0.192 ≤ 0.25), TQA is not
(0.324 > 0.25). Over 100 splits, we aggregate: NQ validity rate ≈ 98%, TQA validity
rate significantly lower.

---

## 28. Expected Results and What They Mean

Based on the SGen paper's reported results and our experimental design:

### NQ (in-domain)

| Metric | Expected Value | Meaning |
|--------|---------------|---------|
| Validity rate | ~98% (96-100%) | PAC guarantee holds |
| Mean FDR-E | ~0.15-0.20 | Well below ε = 0.25 |
| Mean efficiency | ~0.50-0.70 | Model answers 50-70% of questions |

### TriviaQA (shifted domain)

| Metric | Expected Value | Meaning |
|--------|---------------|---------|
| Validity rate | <90% (possibly 70-85%) | PAC guarantee FAILS |
| Mean FDR-E | ~0.25-0.35 | Around or above ε |
| Mean efficiency | ~0.40-0.60 | Possibly lower than NQ |

### The key comparison

```
NQ validity:  ~98%    ← guarantee holds
TQA validity: ~80%    ← guarantee fails (expected: 98%, actual: ≈80%)
Gap:          ~18 percentage points ← this IS the domain shift effect
```

If TQA validity ≈ NQ validity ≈ 98%, there is no domain shift problem and no need
for DS-SGen. Our hypothesis is that the gap will be large enough (>5 percentage points)
to be scientifically and practically significant.

---

## 29. The Domain Shift Hypothesis

### Why the guarantee should break

SGen's PAC guarantee relies on the binomial concentration inequality:

```
P{failure_rate > ε} ≤ δ
```

This requires that the calibration data (Z_U) is representative of the test data.
Specifically, the empirical failure rate on Z_U must be a good estimator of the true
failure rate on the test set.

Under domain shift:
- Z_U comes from NQ (calibration distribution)
- TQA has a different difficulty distribution (different question styles, topics)
- The model's error patterns on NQ may not predict its error patterns on TQA
- The threshold (τ₁*, τ₂*) optimized for NQ may be too lenient for TQA

### Two failure modes

1. **Under-selection (conservative failure):** If TQA questions are generally harder,
   the model's fM1/fM2 scores on TQA may be lower. The NQ-optimized thresholds select
   fewer TQA questions, reducing efficiency. But the selected questions may still have
   low FDR-E. This failure mode reduces efficiency without necessarily breaking validity.

2. **Over-selection (validity failure):** If TQA has a different relationship between
   confidence signals and correctness — e.g., the model is confidently wrong on certain
   trivia topics — then NQ-optimized thresholds may select TQA questions that look
   confident but are actually wrong. This breaks validity.

Both failure modes are interesting. Over-selection is the more concerning one because
it means the user receives wrong answers without warning. Our experiment measures which
failure mode dominates.

---

## 30. Caching System and Preemption Safety

### The problem

We run on the SLURM `preempt` partition, which provides GPUs but may kill our job at
any time to give the resources to a higher-priority job. A generation run for 3,610
questions takes ~1 hour. If preempted at minute 45, we must not lose 45 minutes of
work.

### The solution: incremental JSON caching with atomic writes

Every stage saves intermediate results to JSON files. Generation saves every 50
questions. Entailment saves every 200 questions. On restart, the pipeline loads the
partial cache and resumes.

The `save_cache` function (from `ds_sgen/utils.py`):

```python
def save_cache(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        os.unlink(tmp_path)
        raise
```

Key properties:
- `tempfile.mkstemp` creates a temporary file in the same directory (same filesystem)
- `json.dump` writes the full data to the temp file
- `os.replace` atomically renames temp → final (POSIX guarantee)
- If preemption happens during `json.dump`, only the temp file is corrupted; the
  previous cache file is intact
- If preemption happens after `os.replace`, the new cache is complete

### Cache files and their sizes

```
/data/user_data/anshulk/dsgen/cache/
├── nq_data.json              (792 KB)     Stage 1: 3,610 NQ records
├── tqa_data.json             (2.0 MB)     Stage 1: 3,610 TQA records
├── nq_generations.json       (8.0 MB)     Stage 2: NQ generation results (complete: 3,610)
├── tqa_generations.json      (~4.5 MB)    Stage 2: TQA generation results (partial: 2,550/3,610)
├── nq_entailment.json        (not yet)    Stage 3: NQ entailment scores
└── tqa_entailment.json       (not yet)    Stage 3: TQA entailment scores
```

---

## 31. Code Architecture

### Module dependency graph

```
configs/default.yaml
    ↓ (loaded by)
ds_sgen/utils.py              # load_config, set_seed, cache I/O
    ↑ (imported by all modules)

ds_sgen/data_loading.py       # load_nq, load_tqa, load_and_cache_datasets
ds_sgen/generate_responses.py # load_generator, generate_for_question, generate_and_cache
ds_sgen/entailment_scoring.py # load_entailment_model, score_correctness, score_self_consistency, score_and_cache
ds_sgen/sgen_semi.py          # _merge_records, _run_single_split, run_experiment

run_baseline.py               # orchestrates all 4 stages
```

### Design principles

1. **Config-driven.** All parameters (model paths, hyperparameters, save frequencies)
   live in `configs/default.yaml`. No magic numbers in code.

2. **Stage independence.** Each stage reads its inputs from cache and writes its
   outputs to cache. Stages can be run independently via `--stage` flag.

3. **Resume safety.** Every stage checks for existing cache before starting. Partial
   caches are detected by comparing cached count to expected count.

4. **Single model load.** Within a stage, the model is loaded once and reused for all
   questions. This avoids the 14-second load overhead per question.

---

## 32. Running the Pipeline

### Via SLURM

```bash
sbatch scripts/run_gpu.sh
```

The SLURM script:
- Requests 1x A6000 GPU, 48 GB RAM, 4 CPUs, 7-day wall time (preempt partition)
- Activates the `dsgen` conda environment
- Sets HF_HOME and TRANSFORMERS_CACHE to `/data/` (avoids filling home quota)
- Runs `python run_baseline.py --config configs/default.yaml`

### Stage-by-stage

```bash
python run_baseline.py --stage data         # Stage 1 only (CPU, ~2 min)
python run_baseline.py --stage generate     # Stages 1+2 (GPU, ~1-2 hours)
python run_baseline.py --stage entailment   # Stages 1-3 (GPU, ~10 min after gen)
python run_baseline.py --stage sgen         # All stages (GPU + CPU)
python run_baseline.py                      # Same as --stage all
```

### Method 2 (after baseline completes)

```bash
python run_conservative.py --config configs/default.yaml
```

This loads cached Stages 1-3 data and runs only the SGen-Semi algorithm with
conservative parameter sweeps. No GPU needed. Completes in minutes.

---

## 33. Runtime Estimates

| Stage | Time | VRAM | Bottleneck |
|-------|------|------|-----------|
| Data loading | ~2 min | CPU only | TriviaQA download (633 MB) — cached after first run |
| NQ generation (3,610 questions) | ~24 hours | ~16 GB | Autoregressive decoding, batch_size=1, ~24 sec/question |
| TQA generation (3,610 questions) | ~24 hours | ~16 GB | Same as NQ |
| NQ entailment scoring | ~3-5 min | ~6 GB | Batched NLI, batch_size=64 |
| TQA entailment scoring | ~3-5 min | ~6 GB | Same as NQ |
| SGen-Semi (100 splits) | ~30 sec | CPU only | Numpy grid search |
| **Total** | **~48 hours** | | Dominated by LLM generation |

Generation is the bottleneck because it uses batch_size=1 (each question is processed
individually due to variable input lengths from the chat template). Observed throughput
is approximately 24 seconds per question (greedy pass + 5 sampled passes). Each pass
generates up to 100 tokens autoregressively, and the 5 sampled passes cannot be batched
as a single `num_return_sequences=5` call effectively runs them sequentially through
the autoregressive loop. The SLURM job uses a 7-day wall time on the preempt partition
to accommodate the full pipeline including potential preemption and restarts.

---

## 34. Current Status

**As of April 4, 2026:**

| Component | Status | Details |
|-----------|--------|---------|
| NQ data cache | **Complete** | 3,610 records, 792 KB |
| TQA data cache | **Complete** | 3,610 records, 2.0 MB |
| NQ generation cache | **Complete** | 3,610 records, 8.0 MB |
| TQA generation cache | **In progress** | 2,550/3,610 (71%), job 6951565 running on babel-w9-20 |
| NQ entailment cache | Not started | Waiting on TQA generation (pipeline runs sequentially) |
| TQA entailment cache | Not started | Waiting on generation |
| SGen-Semi results | Not started | Waiting on entailment |
| Method 2 (conservative) | Code complete | Waiting on baseline Stages 1-3 caches |

**Job history:**

| Job ID | Partition | Duration | Outcome |
|--------|-----------|----------|---------|
| 6942461 | preempt | 5 sec | GPU check passed (A6000, CUDA 12.4, PyTorch 2.6) |
| 6943087 | preempt | ~2 min | Failed — crashed on `apply_chat_template` return type, fixed |
| 6943094 | preempt | ~4 hours | NQ complete (3,610), TQA reached 2,350/3,610 — **killed: 48h time limit** |
| 6951565 | preempt | **running** | Resumed from TQA 2,350, 7-day wall time |

**Note:** Job 6943094 proved the pipeline is functionally correct end-to-end through
generation but was killed by the 48-hour wall clock limit. The generation stage takes
~24 hours per dataset (not ~2 hours as originally estimated), so the 48-hour limit was
insufficient for both datasets. Job 6951565 uses a 7-day time limit.

---

## 35. Generation Statistics (from cached data)

### NQ: Complete (3,610 questions)

#### fM1 (mean log-probability) distribution

| Statistic | Value |
|-----------|-------|
| Count | 3,610 questions |
| Mean | -0.2261 |
| Std | 0.1371 |
| Min (least confident) | -0.8869 |
| Median | -0.1985 |
| Max (most confident) | -0.0005 |
| Mean answer length | 156 characters |

### TQA: Partial (2,350 of 3,610 questions)

#### fM1 (mean log-probability) distribution

| Statistic | Value |
|-----------|-------|
| Count | 2,550 questions (71% complete) |
| Mean | -0.1814 |
| Std | 0.1389 |
| Min (least confident) | -0.8960 |
| Median | -0.1445 |
| Max (most confident) | -0.0001 |
| Mean answer length | 113 characters |

### Cross-domain comparison: Early domain shift signal

| Metric | NQ (complete) | TQA (partial) | Difference |
|--------|---------------|---------------|------------|
| Mean fM1 | -0.2261 | -0.1814 | +0.0447 (TQA more confident) |
| Median fM1 | -0.1985 | -0.1445 | +0.0540 (TQA more confident) |
| fM1 range | [-0.887, -0.001] | [-0.896, -0.000] | Similar range |
| Mean answer length | 156 chars | 113 chars | TQA answers are 28% shorter |

**Key observation:** TQA answers have *higher* generation confidence (less negative
fM1) and are shorter. This is counterintuitive — one might expect the shifted domain
to be harder. Two possible explanations:

1. **TriviaQA questions have cleaner factual answers.** Trivia questions ("Who painted
   the Mona Lisa?") tend to have short, definitive answers ("Leonardo da Vinci").
   NQ questions are more diverse and often require longer explanatory answers.

2. **The model may be overconfident on TQA.** Higher fM1 does not mean higher accuracy.
   If the model confidently generates wrong trivia answers, this is exactly the failure
   mode that breaks SGen's PAC guarantee — the NQ-calibrated threshold τ₁ would be
   too permissive for TQA because the model's confidence-correctness calibration
   differs between domains. This will be tested in Stage 3 (entailment scoring).

### Example outputs

**High confidence (fM1 close to 0):**

```
Q: "when was the last time anyone was on the moon"
A: "The last time humans visited the moon was during the Apollo 17 mission
    in December 1972, when astronauts..."
fM1: -0.0472 (very confident)
Sampled answers: All 5 mention Apollo 17 and December 1972 (high fM2 expected)
```

**Low confidence (fM1 near -0.9):**

These tend to be questions where the model is uncertain — ambiguous questions,
questions requiring very specific knowledge, or questions with multiple plausible
answers.

### What we can already see

- fM1 has good dynamic range (0 to -0.89), providing meaningful variation for threshold selection
- The system prompt is effective: answers are focused and factual
- The cross-domain fM1 difference suggests the domains have different confidence profiles, which is exactly the signal that could cause SGen's PAC guarantee to break

---

## 36. What This Method Does NOT Do

1. **It does not handle domain shift.** That is the whole point — demonstrating the failure.
2. **It does not compute importance weights or train domain classifiers.** That is Method 3.
3. **It does not inflate thresholds conservatively.** That is Method 2.
4. **It does not produce figures.** It produces raw numbers (JSON) that a separate analysis script will visualize.
5. **It does not evaluate on more than two domains.** NQ (calibration) and TQA (shifted test) are sufficient for the baseline demonstration.
6. **It does not tune hyperparameters.** All settings match the SGen paper exactly.
7. **It does not use the `all_answers` field.** Only `reference_answer` is used for entailment. The aliases are available for future exact-match comparisons.

---

## 37. What This Method Already Tells Us

Even before the full pipeline completes:

1. **LLaMA-3.1-8B-Instruct generates focused, evaluable answers.** The system prompt works — answers are concise sentences, not multi-paragraph essays.

2. **fM1 has meaningful variation.** The range [-0.89, -0.0005] provides good discrimination between confident and uncertain answers.

3. **The chat template fix is necessary.** Transformers 5.5.0's `apply_chat_template` returns a `BatchEncoding` that must be unwrapped. Without the fix, the pipeline crashes.

4. **Incremental caching works.** Job 6943087 crashed at question 0 (before any caching). Job 6943094 has been running for 1.5 hours and the cache shows 2,450 results — if preempted, all work is preserved.

5. **Generation is the bottleneck.** At ~24 seconds per question (greedy + 5 sampled passes), 7,220 questions (NQ + TQA) takes ~48 hours. Entailment scoring and the SGen algorithm are fast by comparison.

---

## 38. Connections to Methods 2 and 3

### Method 2: Conservative Threshold (implemented)

Method 2 reuses ALL cached data from Stages 1-3. It only modifies Stage 4 (the
SGen-Semi algorithm) by injecting three conservative overrides:

- **Option A (safety factor):** τ₁ += log(γ), τ₂ *= γ after grid search
- **Option B (reduced epsilon):** Use ε/k in the grid constraint
- **Option C (delta budget):** Reserve part of δ for shift: δ_cp = δ - δ_p - δ_s

Method 2 runs in minutes on CPU because it only recomputes the grid search and
evaluation — no model inference needed. It is implemented in
`ds_sgen/conservative.py` and run via `run_conservative.py`.

### Method 3: DS-SGen with Importance Reweighting (planned)

Method 3 will add a new pre-processing step between Stages 1 and 4:

1. Embed all NQ and TQA prompts using sentence-transformers (all-MiniLM-L6-v2)
2. Train a domain classifier (logistic regression) on embeddings
3. Compute importance weights: w(x) = P(TQA|x) / (1 - P(TQA|x))
4. Clip weights at the 95th percentile
5. Use weighted conformal prediction and weighted Clopper-Pearson bounds

The cached data from Stages 1-3 is reused entirely. Method 3 adds new computations
but does not re-run generation or entailment scoring.

### The experimental narrative

```
Method 1 (baseline):        "Look, the guarantee breaks under domain shift."
Method 2 (conservative):    "You can fix it by being more cautious, but you pay a huge efficiency cost."
Method 3 (DS-SGen):         "Importance reweighting fixes it while maintaining high efficiency."
```

This three-method comparison is the core contribution of the project.

---

---

## 39. The Selection Direction Convention

A subtle but critical detail: in SGen, **higher scores = more confident = more likely
to be selected**. This determines whether the selection rule uses ≥ or ≤.

### fM1: higher is better

fM1 is a mean log-probability. Log-probabilities are ≤ 0. An fM1 of -0.05 means the
model assigns ~95% probability to each token on average. An fM1 of -0.80 means the
model assigns ~45% probability per token. Higher (less negative) fM1 = more confident.

Selection: fM1 ≥ τ₁ (select if confidence exceeds threshold).

### fM2: higher is better

fM2 is a self-consistency score in [0, 1]. fM2 = 1.0 means all 5 samples agree
bidirectionally. fM2 = 0.0 means no pairs agree. Higher fM2 = more consistent.

Selection: fM2 ≥ τ₂ (select if consistency exceeds threshold).

### The conformal threshold: higher is better

The entailment score (P(ENTAILMENT)) is in [0, 1]. Higher = more likely correct.
The conformal threshold τ_CP selects the (1 - ε_e) quantile of Z_E scores. Points
above τ_CP are pseudo-labeled as correct.

Pseudo-label: 1 if entail_score ≥ τ_CP, else 0.

### Why this matters

Getting the direction wrong (using ≤ instead of ≥, or vice versa) would invert the
entire algorithm: select the LEAST confident answers instead of the MOST confident.
The code consistently uses `>=` for all three score types. This was verified against
the upstream SGen implementation.

---

## 40. Why 100 Random Splits

### The statistical argument

One random calibration/test split gives one FDR-E measurement. This single measurement
has high variance — a lucky split might give FDR-E = 0.10 while an unlucky split gives
FDR-E = 0.30, even under the same ground-truth conditions.

100 splits give 100 FDR-E measurements. The **validity rate** — the fraction of splits
where FDR-E ≤ ε — is a much more stable estimator. With 100 splits, a validity rate
of 95% vs. 98% is meaningfully distinguishable (the standard error of a proportion at
p = 0.95 with n = 100 is √(0.95 × 0.05 / 100) = 0.0218, so a 95% CI is ±4.3%).

### The practical argument

100 splits is the standard in the SGen paper. Using the same number ensures
comparability. More splits (e.g., 1000) would give tighter estimates but the marginal
benefit is small given that each split's grid search takes only ~300 ms.

### Seed management

Each split uses seed = base_seed + split_index:
- Split 0: seed 42
- Split 1: seed 43
- ...
- Split 99: seed 141

This is deterministic. Running the experiment twice produces identical results.

---

## 41. Aggregation Across Splits

For each of the 100 splits, we record:

```python
{
    "split_seed": 42,
    "cal_size": 2527,
    "zu_size": 1895,
    "ze_size": 632,
    "tau_cp": 0.723,
    "tau1": -0.302,
    "tau2": 0.600,
    "grid_size_H": 528,
    "nq_test": {"fdr_e": 0.192, "efficiency": 0.720, "valid": True, "n_selected": 780, "n_total": 1083},
    "tqa":     {"fdr_e": 0.324, "efficiency": 0.582, "valid": False, "n_selected": 2100, "n_total": 3610},
}
```

The final aggregated result computes:

```python
summary = {
    "nq": {
        "validity_rate": mean(nq_valid across 100 splits),       # target: ≥ 0.98
        "mean_fdr_e":    mean(nq_fdr_e across 100 splits),       # target: ≤ 0.25
        "std_fdr_e":     std(nq_fdr_e across 100 splits),
        "mean_efficiency": mean(nq_efficiency across 100 splits),
        "std_efficiency":  std(nq_efficiency across 100 splits),
    },
    "tqa": {
        "validity_rate": mean(tqa_valid across 100 splits),      # expected: < 0.98
        "mean_fdr_e":    mean(tqa_fdr_e across 100 splits),      # expected: around ε
        "std_fdr_e":     std(tqa_fdr_e across 100 splits),
        "mean_efficiency": mean(tqa_efficiency across 100 splits),
        "std_efficiency":  std(tqa_efficiency across 100 splits),
    },
}
```

### What "mean FDR-E" means vs. "validity rate"

These measure different things:

- **Mean FDR-E** = average error rate across splits. Even if some splits have high
  FDR-E (>ε), the average might still be below ε. This measures typical performance.

- **Validity rate** = fraction of splits where FDR-E ≤ ε. This measures the PAC
  guarantee: the guarantee promises validity_rate ≥ (1 - δ) = 98%.

A method can have mean FDR-E < ε but validity_rate < 98% (some splits are barely
over ε, pulling the validity rate down). Or it can have mean FDR-E > ε but
validity_rate > 0% (many splits are fine, a few have very high FDR-E). The validity
rate is the primary metric because it directly measures the PAC guarantee.

---

## 42. The Print Summary Function

The final output of `run_baseline.py` is a formatted summary printed to stdout
(captured in the SLURM `.out` file):

```python
def print_summary(results: dict):
    for domain, label in [("nq", "NQ (in-domain)"), ("tqa", "TriviaQA (shifted)")]:
        r = results[domain]
        print(f"  {label}:")
        print(f"    Validity rate:   {r['validity_rate']:.2%}  (target: >= 98%)")
        print(f"    Mean FDR-E:      {r['mean_fdr_e']:.4f} +/- {r['std_fdr_e']:.4f}  (target: <= 0.25)")
        print(f"    Mean efficiency: {r['mean_efficiency']:.4f} +/- {r['std_efficiency']:.4f}")

    nq_val = results["nq"]["validity_rate"]
    tqa_val = results["tqa"]["validity_rate"]
    if tqa_val < nq_val - 0.05:
        print("  >>> Domain shift detected: TQA validity dropped significantly")
    else:
        print("  >>> No significant domain shift in validity")
```

The 5-percentage-point threshold for "significant" domain shift is a conservative
heuristic. If the validity rate drops by more than 5pp (e.g., from 98% to 92%), we
flag it as a domain shift effect. This threshold is for reporting purposes only — the
formal test is whether TQA validity is below (1 - δ) = 98%.

---

## 43. Environment and Reproducibility

### Conda environment

The full environment is exported in `environment.yml`. Key packages:

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.20 | Runtime |
| torch | 2.6.0+cu124 | Model inference |
| transformers | 5.5.0 | Model loading, tokenization, generation |
| datasets | 4.8.4 | HuggingFace dataset download |
| scipy | 1.15.3 | Clopper-Pearson (beta.ppf) |
| numpy | 2.2.6 | Array operations, random seeds |
| sentence-transformers | 5.3.0 | For Method 3 (not used in Method 1) |
| scikit-learn | 1.7.2 | For Method 3 (not used in Method 1) |
| matplotlib | 3.10.8 | For future visualization |
| pyyaml | 6.0.3 | Config loading |

### Seed setting

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

Called once at the start of `run_baseline.py` with seed = 42. This ensures:
- Same dataset sampling (TQA downsampling)
- Same random splits in SGen-Semi
- Same generation outputs (torch manual seed affects model sampling)

Note: greedy decoding (`do_sample=False`) is deterministic regardless of seed.
The seed affects only the K=5 sampled responses.

### GPU reproducibility

PyTorch's forward pass is deterministic for the same input on the same GPU hardware.
LLaMA's greedy decoding produces bit-identical outputs across runs. The sampled
responses depend on the torch random state, which is controlled by `set_seed`.

---

## 44. Known Limitations of Method 1

1. **Batch size 1 for generation.** Each question is generated individually because
   the chat template produces variable-length inputs. Batching with padding is possible
   but was not implemented to keep the code simple and avoid padding-related artifacts.

2. **Single reference answer for entailment.** We use only `reference_answer` (the
   first/primary answer), not `all_answers`. Some NQ questions have 2 valid answers
   and some TQA questions have up to 19 aliases. Using only one reference may
   undercount correctness (label a correct answer as wrong because it matches an alias
   but not the primary). This is a conservative choice that makes FDR-E estimates
   slightly pessimistic.

3. **No stratification of NQ questions.** The 70/30 calibration/test split is purely
   random. It does not stratify by topic, difficulty, or answer type. This means some
   splits might have disproportionately easy or hard calibration sets. Over 100 splits,
   this averages out.

4. **The conformal guarantee assumes exchangeability within NQ.** If NQ validation
   questions have internal structure (e.g., questions from different Wikipedia articles
   are not exchangeable), the conformal threshold may be slightly miscalibrated. This
   is a standard assumption in conformal prediction literature.

5. **Entailment model errors are not accounted for.** DeBERTa is not perfect. It may
   mislabel some correct answers as not entailing (false negatives) or some wrong
   answers as entailing (false positives). These errors affect both the pseudo-labels
   and the FDR-E evaluation, but are assumed to be small and unbiased.

---

*Document generated April 3, 2026. Updated April 4, 2026 with complete NQ generation
statistics (3,610 questions), partial TQA statistics (2,350/3,610), corrected runtime
estimates (~24h/dataset, not ~2h), updated cache file sizes, and cross-domain fM1
comparison. All numbers validated against cached data and SLURM logs. This document
will be updated when the full pipeline completes and final results are available.*

