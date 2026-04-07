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

import logging
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ds_sgen.utils import get_cache_path, load_cache, save_cache

logger = logging.getLogger(__name__)

ENTAILMENT_IDX = 2  # For microsoft/deberta-v2-xxlarge-mnli


def _setup_file_logger(log_dir: str):
    """Add a file handler to the module logger if one doesn't exist yet."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "entailment_scoring.log"))
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def load_entailment_model(cfg: dict):
    """Load DeBERTa-v2-xxlarge-mnli for NLI."""
    model_name = cfg["paths"]["entailment_model"]
    cache_dir = cfg["paths"]["hf_cache"]

    logger.info("Loading entailment tokenizer: %s", model_name)
    print(f"  Loading entailment tokenizer: {model_name}...")
    # use_fast=False: transformers 5.x auto-conversion tries to parse spm.model
    # as tiktoken BPE, which crashes. The slow DebertaV2Tokenizer works correctly.
    # See: https://github.com/huggingface/transformers/issues/42583
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=False)

    logger.info("Loading entailment model: %s", model_name)
    print(f"  Loading entailment model: {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, cache_dir=cache_dir, dtype=torch.float16
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    logger.info("Entailment model loaded on %s", device)
    print(f"  Entailment model loaded on {device}")
    return model, tokenizer


def _batch_nli(model, tokenizer, pairs: list[tuple[str, str]], batch_size: int) -> list[dict]:
    """Run NLI on a list of (premise, hypothesis) pairs in batches.

    Returns list of dicts with 'probs', 'argmax' for each pair.
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
    if not greedy_answer or not reference_answer:
        logger.warning("Empty answer in correctness scoring: greedy='%s', ref='%s'",
                        greedy_answer[:50], reference_answer[:50])
        return 0.0, 0

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

    # Filter out empty answers
    non_empty = [(i, a) for i, a in enumerate(sampled_answers) if a.strip()]
    if len(non_empty) <= 1:
        logger.warning("Only %d non-empty sampled answers out of %d", len(non_empty), K)
        return 0.0, [[False] * K for _ in range(K)]

    # Build all ordered pairs (i→j) for the full KxK matrix
    pairs = []
    pair_indices = []
    for i in range(K):
        for j in range(K):
            if i != j and sampled_answers[i].strip() and sampled_answers[j].strip():
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
    _setup_file_logger(cfg.get("log_dir", "logs"))

    cache_path = get_cache_path(cfg["paths"]["cache_dir"], f"{dataset_name}_entailment")
    batch_size = cfg["entailment"]["batch_size"]
    save_every = 200

    logger.info("=" * 60)
    logger.info("[%s] Entailment scoring started — model=%s, questions=%d, batch_size=%d",
                dataset_name.upper(), cfg["paths"]["entailment_model"], len(records), batch_size)

    # Validate inputs
    if len(records) != len(generations):
        msg = (f"records ({len(records)}) and generations ({len(generations)}) length mismatch "
               f"for {dataset_name}")
        logger.error(msg)
        raise ValueError(msg)

    cached = load_cache(cache_path)
    if cached is not None and len(cached) == len(records):
        msg = f"{dataset_name.upper()}: all {len(cached)} entailment scores cached, skipping"
        logger.info(msg)
        print(f"  {msg}")
        return cached

    results = cached if cached is not None else []
    start_idx = len(results)

    if start_idx > 0:
        msg = f"{dataset_name.upper()}: resuming entailment from question {start_idx}/{len(records)}"
        logger.info(msg)
        print(f"  {msg}")
    else:
        msg = f"{dataset_name.upper()}: scoring {len(records)} questions"
        logger.info(msg)
        print(f"  {msg}")

    model, tokenizer = load_entailment_model(cfg)

    # Log GPU memory
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("[%s] GPU memory: %.1f / %.1f GB", dataset_name.upper(), mem_alloc, mem_total)

    t_start = time.time()
    total_nli_pairs = 0
    n_correct = 0

    for i in range(start_idx, len(records)):
        rec = records[i]
        gen = generations[i]
        t_q = time.time()

        # Correctness: greedy → reference
        entail_score, entail_label = score_correctness(
            model, tokenizer, gen["greedy_answer"], rec["reference_answer"], batch_size
        )

        # Self-consistency: pairwise among K sampled answers
        K = len(gen["sampled_answers"])
        fM2, pairwise = score_self_consistency(
            model, tokenizer, gen["sampled_answers"], batch_size
        )

        elapsed_q = time.time() - t_q
        nli_pairs_q = 1 + K * (K - 1)  # 1 correctness + K*(K-1) directed pairs
        total_nli_pairs += nli_pairs_q
        n_correct += entail_label

        results.append({
            "idx": rec["idx"],
            "entail_score": entail_score,
            "entail_label": entail_label,
            "fM2": fM2,
            "pairwise_entailments": pairwise,
        })

        # Log every question at DEBUG
        logger.debug("[%s] idx=%d  correct=%d  entail_p=%.3f  fM2=%.2f  "
                     "nli_pairs=%d  time=%.2fs  greedy='%s'  ref='%s'",
                     dataset_name.upper(), rec["idx"], entail_label, entail_score, fM2,
                     nli_pairs_q, elapsed_q,
                     gen["greedy_answer"][:60], rec["reference_answer"][:60])

        # Progress every 100 questions
        if (i + 1) % 100 == 0 or i == len(records) - 1:
            elapsed_total = time.time() - t_start
            done = i + 1 - start_idx
            rate = done / elapsed_total if elapsed_total > 0 else 0
            eta_min = (len(records) - i - 1) / rate / 60 if rate > 0 else 0
            acc_so_far = n_correct / (i + 1) * 100

            progress_msg = (f"[{dataset_name.upper()}] {i+1}/{len(records)}: "
                            f"correct={entail_label}, fM2={fM2:.2f}, "
                            f"entail_p={entail_score:.3f}")
            print(f"    {progress_msg}")

            logger.info("[%s] progress=%d/%d  accuracy=%.1f%%  rate=%.1f q/s  "
                        "eta=%.1f min  total_nli_pairs=%d",
                        dataset_name.upper(), i + 1, len(records), acc_so_far,
                        rate, eta_min, total_nli_pairs)

        # Save every 200 questions
        if (i + 1) % save_every == 0 or i == len(records) - 1:
            save_cache(results, cache_path)
            logger.info("[%s] checkpoint saved: %d/%d to %s",
                        dataset_name.upper(), len(results), len(records), cache_path)

    elapsed_total = time.time() - t_start
    final_accuracy = n_correct / len(records) * 100 if len(records) > 0 else 0

    logger.info("[%s] Entailment scoring complete: %d questions in %.1f min",
                dataset_name.upper(), len(results), elapsed_total / 60)
    logger.info("[%s] Accuracy (entailment): %d/%d = %.1f%%",
                dataset_name.upper(), n_correct, len(records), final_accuracy)
    logger.info("[%s] Total NLI pairs evaluated: %d", dataset_name.upper(), total_nli_pairs)

    print(f"  {dataset_name.upper()}: entailment scoring complete "
          f"({len(results)} questions, {elapsed_total/60:.1f} min, "
          f"accuracy={final_accuracy:.1f}%)")

    return results
