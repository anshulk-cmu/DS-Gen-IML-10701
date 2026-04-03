"""Entailment scoring using DeBERTa-v2-xxlarge-mnli.

Two modes:
  1. Correctness (unidirectional argmax): NLI(greedy → reference), argmax == ENTAILMENT
  2. Self-consistency (bidirectional argmax): for all sampled answer pairs (i,j),
     both NLI(i→j) and NLI(j→i) must have argmax == ENTAILMENT

DeBERTa-v2-xxlarge-mnli label order: {0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT}
WARNING: cross-encoder/nli-deberta-v3-large has {0: contradiction, 1: entailment, 2: neutral} — different!

Output per question:
{
    "idx": int,
    "entail_score": float,       # P(entailment) for greedy→reference (continuous, for conformal)
    "entail_label": int,         # 1 if argmax=ENTAILMENT, else 0 (binary correctness)
    "fM2": float,                # self-consistency: fraction of bidirectionally entailing pairs
    "pairwise_entailments": [[bool]]  # KxK directed entailment matrix (debugging)
}
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ds_sgen.utils import get_cache_path, load_cache, save_cache


ENTAILMENT_IDX = 2  # For microsoft/deberta-v2-xxlarge-mnli


def load_entailment_model(cfg: dict):
    """Load DeBERTa-v2-xxlarge-mnli for NLI."""
    model_name = cfg["paths"]["entailment_model"]
    cache_dir = cfg["paths"]["hf_cache"]

    print(f"  Loading entailment tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    print(f"  Loading entailment model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=torch.float16
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    print(f"  Entailment model loaded on {next(model.parameters()).device}")
    return model, tokenizer


def _batch_nli(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> list[dict]:
    """Run NLI on a list of (premise, hypothesis) pairs in batches.

    Returns list of dicts with 'logits', 'probs', 'argmax' for each pair.
    """
    results = []
    device = next(model.parameters()).device

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        premises = [p for p, _ in batch_pairs]
        hypotheses = [h for _, h in batch_pairs]

        inputs = tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits.float()  # (batch, 3)
            probs = F.softmax(logits, dim=-1)
            argmaxes = logits.argmax(dim=-1)

        for j in range(len(batch_pairs)):
            results.append({
                "probs": probs[j].cpu().tolist(),
                "argmax": argmaxes[j].item(),
            })

    return results


def score_correctness(
    model, tokenizer, greedy_answer: str, reference_answer: str, batch_size: int
) -> tuple[float, int]:
    """Unidirectional NLI: does greedy answer entail the reference?

    Returns (entail_score, entail_label).
    entail_score: P(entailment) — continuous, used for conformal threshold.
    entail_label: 1 if argmax is ENTAILMENT, else 0.
    """
    results = _batch_nli(model, tokenizer, [(greedy_answer, reference_answer)], batch_size)
    r = results[0]
    entail_score = r["probs"][ENTAILMENT_IDX]
    entail_label = 1 if r["argmax"] == ENTAILMENT_IDX else 0
    return entail_score, entail_label


def score_self_consistency(
    model, tokenizer, sampled_answers: list[str], batch_size: int
) -> tuple[float, list[list[bool]]]:
    """Bidirectional NLI: compute fM2 self-consistency score.

    For K sampled answers, checks all C(K,2) unordered pairs.
    Two answers "agree" iff NLI(i→j)=ENTAILMENT AND NLI(j→i)=ENTAILMENT.
    fM2 = (number of agreeing pairs) / C(K,2).

    Returns (fM2, pairwise_matrix) where pairwise_matrix[i][j] = directed entailment.
    """
    K = len(sampled_answers)
    if K <= 1:
        return 1.0, [[True]]

    # Build all ordered pairs (i→j) for the full KxK matrix
    pairs = []
    pair_indices = []
    for i in range(K):
        for j in range(K):
            if i != j:
                pairs.append((sampled_answers[i], sampled_answers[j]))
                pair_indices.append((i, j))

    nli_results = _batch_nli(model, tokenizer, pairs, batch_size)

    # Build directed entailment matrix
    matrix = [[False] * K for _ in range(K)]
    for idx, (i, j) in enumerate(pair_indices):
        matrix[i][j] = nli_results[idx]["argmax"] == ENTAILMENT_IDX

    # Bidirectional agreement: count unordered pairs where both directions entail
    n_agree = 0
    n_pairs = 0
    for i in range(K):
        for j in range(i + 1, K):
            n_pairs += 1
            if matrix[i][j] and matrix[j][i]:
                n_agree += 1

    fM2 = n_agree / n_pairs if n_pairs > 0 else 1.0
    return fM2, matrix


def score_and_cache(
    cfg: dict, dataset_name: str, records: list[dict], generations: list[dict]
) -> list[dict]:
    """Score all questions for correctness and self-consistency, with caching.

    Args:
        cfg: Full config dict.
        dataset_name: "nq" or "tqa".
        records: Normalized question dicts (from data_loading).
        generations: Generation result dicts (from generate_responses).

    Returns:
        List of entailment score dicts (one per question).
    """
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], f"{dataset_name}_entailment")
    batch_size = cfg["entailment"]["batch_size"]

    cached = load_cache(cache_path)
    if cached is not None and len(cached) == len(records):
        print(f"  {dataset_name.upper()}: all {len(cached)} entailment scores cached, skipping")
        return cached

    results = cached if cached is not None else []
    start_idx = len(results)

    if start_idx > 0:
        print(f"  {dataset_name.upper()}: resuming entailment from question {start_idx}/{len(records)}")
    else:
        print(f"  {dataset_name.upper()}: scoring {len(records)} questions")

    model, tokenizer = load_entailment_model(cfg)

    for i in range(start_idx, len(records)):
        rec = records[i]
        gen = generations[i]

        # Correctness: greedy → reference
        entail_score, entail_label = score_correctness(
            model, tokenizer, gen["greedy_answer"], rec["reference_answer"], batch_size
        )

        # Self-consistency: pairwise among K sampled answers
        fM2, pairwise = score_self_consistency(
            model, tokenizer, gen["sampled_answers"], batch_size
        )

        results.append({
            "idx": rec["idx"],
            "entail_score": entail_score,
            "entail_label": entail_label,
            "fM2": fM2,
            "pairwise_entailments": pairwise,
        })

        if (i + 1) % 100 == 0 or i == len(records) - 1:
            print(f"    [{dataset_name.upper()}] {i+1}/{len(records)}: "
                  f"correct={entail_label}, fM2={fM2:.2f}, "
                  f"entail_p={entail_score:.3f}")

        # Save every 200 questions
        if (i + 1) % 200 == 0 or i == len(records) - 1:
            save_cache(results, cache_path)

    print(f"  {dataset_name.upper()}: entailment scoring complete ({len(results)} questions)")
    return results
