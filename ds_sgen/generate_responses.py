"""LLM response generation: greedy (with log-probs) + K sampled answers.

Uses LLaMA-3.1-8B-Instruct with proper chat template (not raw prompts).

Output per question:
{
    "idx": int,
    "question": str,
    "greedy_answer": str,
    "mean_logprob": float,          # fM1: average token log-prob of greedy answer
    "token_logprobs": [float],      # per-token log-probs (debugging)
    "sampled_answers": [str] * K    # K=5 sampled responses
}
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ds_sgen.utils import get_cache_path, load_cache, save_cache


def load_generator(cfg: dict):
    """Load LLaMA-3.1-8B-Instruct model and tokenizer."""
    model_path = cfg["paths"]["model"]
    cache_dir = cfg["paths"]["hf_cache"]

    print(f"  Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=cache_dir, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    print(f"  Model loaded on {model.device}")
    return model, tokenizer


def _build_chat_input(tokenizer, question: str, system_prompt: str):
    """Build chat-templated input_ids for a single question."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    return input_ids


def _extract_logprobs_from_scores(scores, generated_ids):
    """Extract per-token log-probs from generate() output scores.

    Args:
        scores: tuple of (vocab_size,) tensors, one per generated step.
                These are raw logits from output_logits=True.
        generated_ids: tensor of generated token IDs, shape (num_new_tokens,).

    Returns:
        List of per-token log-probabilities.
    """
    token_logprobs = []
    for step_idx, step_logits in enumerate(scores):
        # step_logits shape: (1, vocab_size) or (vocab_size,)
        logits = step_logits.squeeze(0).float()
        log_probs = F.log_softmax(logits, dim=-1)
        token_id = generated_ids[step_idx].item()
        token_logprobs.append(log_probs[token_id].item())
    return token_logprobs


def generate_for_question(
    model, tokenizer, question: str, cfg: dict
) -> dict:
    """Generate greedy answer (with log-probs) and K sampled answers for one question."""
    gen_cfg = cfg["generation"]
    system_prompt = gen_cfg["system_prompt"]
    max_new_tokens = gen_cfg["max_new_tokens"]
    num_samples = gen_cfg["num_samples"]
    temperature = gen_cfg["temperature"]

    input_ids = _build_chat_input(tokenizer, question, system_prompt)
    input_ids = input_ids.to(model.device)
    input_len = input_ids.shape[1]

    # === Pass 1: Greedy decoding with log-prob extraction ===
    with torch.no_grad():
        greedy_out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_logits=True,         # Raw logits (not processed by temperature/top_p)
            return_dict_in_generate=True,
        )

    greedy_ids = greedy_out.sequences[0, input_len:]
    greedy_answer = tokenizer.decode(greedy_ids, skip_special_tokens=True).strip()

    # Extract per-token log-probs from raw logits
    token_logprobs = _extract_logprobs_from_scores(greedy_out.logits, greedy_ids)
    mean_logprob = sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0

    # === Pass 2: Sampled responses (K=5) ===
    sampled_answers = []
    with torch.no_grad():
        sampled_out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=num_samples,
        )
    for seq in sampled_out:
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        sampled_answers.append(text)

    return {
        "greedy_answer": greedy_answer,
        "mean_logprob": mean_logprob,
        "token_logprobs": token_logprobs,
        "sampled_answers": sampled_answers,
    }


def generate_and_cache(cfg: dict, dataset_name: str, records: list[dict]) -> list[dict]:
    """Generate responses for all questions in a dataset, with incremental caching.

    Args:
        cfg: Full config dict.
        dataset_name: "nq" or "tqa" — used for cache file naming.
        records: List of normalized question dicts from data_loading.

    Returns:
        List of generation result dicts (one per question).
    """
    cache_path = get_cache_path(cfg["paths"]["cache_dir"], f"{dataset_name}_generations")
    save_every = cfg["generation"]["save_every"]

    # Resume from partial cache
    cached = load_cache(cache_path)
    if cached is not None and len(cached) == len(records):
        print(f"  {dataset_name.upper()}: all {len(cached)} generations cached, skipping")
        return cached
    results = cached if cached is not None else []
    start_idx = len(results)

    if start_idx > 0:
        print(f"  {dataset_name.upper()}: resuming from question {start_idx}/{len(records)}")
    else:
        print(f"  {dataset_name.upper()}: generating for {len(records)} questions")

    model, tokenizer = load_generator(cfg)

    for i in range(start_idx, len(records)):
        rec = records[i]
        gen = generate_for_question(model, tokenizer, rec["question"], cfg)
        gen["idx"] = rec["idx"]
        gen["question"] = rec["question"]
        results.append(gen)

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            print(f"    [{dataset_name.upper()}] {i+1}/{len(records)}: "
                  f"logprob={gen['mean_logprob']:.3f}, "
                  f"answer='{gen['greedy_answer'][:60]}...'")

        # Incremental save
        if (i + 1) % save_every == 0 or i == len(records) - 1:
            save_cache(results, cache_path)

    print(f"  {dataset_name.upper()}: generation complete ({len(results)} questions)")
    return results
