"""GPT-4o-mini response generation via OpenAI real-time API.

Two API calls per question (run sequentially):
  Call 1 (greedy + logprobs)  →  greedy_answer + mean_logprob
  Call 2 (n=5 sampled)        →  sampled_answers

Output cache: list of dicts per question:
[
    {
        "greedy_answer": str,
        "mean_logprob": float,
        "token_logprobs": [float],           # per-token logprobs for greedy
        "sampled_answers": [str] * 5,
        "sampled_mean_logprobs": [float] * 5, # per-sample mean logprob
        "sampled_token_logprobs": [[float]] * 5,  # per-token logprobs for each sample
        "idx": int,
        "question": str
    },
    ...
]
"""

import logging
import os
import time

from dotenv import load_dotenv

from ds_sgen.utils import get_cache_path, load_cache, save_cache

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "Answer the following question concisely in one sentence."
MODEL = "gpt-4o-mini"
MAX_TOKENS_GREEDY = 512
MAX_TOKENS_SAMPLED = 512
TEMPERATURE_SAMPLED = 0.7
N_SAMPLES = 5

_client = None


def _get_client():
    """Lazy-init OpenAI client. Only called when actually generating."""
    global _client
    if _client is None:
        load_dotenv()
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _setup_file_logger(log_dir: str):
    """Add a file handler to the module logger if one doesn't exist yet."""
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "generate_responses.log"))
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


MAX_RETRIES = 5
RETRY_BASE_WAIT = 10  # seconds


def _api_call_with_retry(**kwargs):
    """Make an OpenAI API call with exponential backoff on rate limits."""
    from openai import RateLimitError, APIError
    for attempt in range(MAX_RETRIES):
        try:
            return _get_client().chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = RETRY_BASE_WAIT * (2 ** attempt)
            logger.warning("Rate limited (attempt %d/%d), waiting %ds: %s",
                           attempt + 1, MAX_RETRIES, wait, e)
            time.sleep(wait)
        except APIError as e:
            wait = RETRY_BASE_WAIT * (2 ** attempt)
            logger.warning("API error (attempt %d/%d), waiting %ds: %s",
                           attempt + 1, MAX_RETRIES, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def _generate_for_question(question: str) -> dict:
    """Generate greedy answer (with logprobs) and 5 sampled answers for one question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # Call 1: Greedy + logprobs
    greedy_resp = _api_call_with_retry(
        model=MODEL,
        temperature=0,
        max_tokens=MAX_TOKENS_GREEDY,
        logprobs=True,
        messages=messages,
    )
    choice = greedy_resp.choices[0]
    greedy_answer = choice.message.content.strip()

    logprob_tokens = choice.logprobs.content
    if logprob_tokens:
        token_logprobs = [t.logprob for t in logprob_tokens]
        mean_logprob = sum(token_logprobs) / len(token_logprobs)
    else:
        token_logprobs = []
        mean_logprob = 0.0

    greedy_usage = greedy_resp.usage

    # Call 2: n=5 sampled with logprobs
    sampled_resp = _api_call_with_retry(
        model=MODEL,
        temperature=TEMPERATURE_SAMPLED,
        max_tokens=MAX_TOKENS_SAMPLED,
        n=N_SAMPLES,
        logprobs=True,
        messages=messages,
    )

    sampled_answers = []
    sampled_mean_logprobs = []
    sampled_token_logprobs = []
    for c in sampled_resp.choices:
        sampled_answers.append(c.message.content.strip())
        if c.logprobs and c.logprobs.content:
            s_token_lps = [t.logprob for t in c.logprobs.content]
            sampled_token_logprobs.append(s_token_lps)
            sampled_mean_logprobs.append(sum(s_token_lps) / len(s_token_lps))
        else:
            sampled_token_logprobs.append([])
            sampled_mean_logprobs.append(0.0)

    sampled_usage = sampled_resp.usage

    return {
        "greedy_answer": greedy_answer,
        "mean_logprob": mean_logprob,
        "token_logprobs": token_logprobs,
        "sampled_answers": sampled_answers,
        "sampled_mean_logprobs": sampled_mean_logprobs,
        "sampled_token_logprobs": sampled_token_logprobs,
        "_greedy_usage": {"prompt": greedy_usage.prompt_tokens, "completion": greedy_usage.completion_tokens},
        "_sampled_usage": {"prompt": sampled_usage.prompt_tokens, "completion": sampled_usage.completion_tokens},
    }


def generate_and_cache_openai(cfg: dict, dataset_name: str, records: list) -> list:
    """Generate responses for all questions, with incremental caching.

    Returns list of dicts (one per question). Resumes from partial cache.
    """
    _setup_file_logger(cfg.get("log_dir", "logs"))

    cache_path = get_cache_path(cfg["paths"]["cache_dir"], f"{dataset_name}_generations")
    save_every = cfg["generation"].get("save_every", 50)

    cached = load_cache(cache_path)
    if cached is not None and len(cached) == len(records):
        msg = f"{dataset_name.upper()}: all {len(cached)} generations cached, skipping"
        logger.info(msg)
        print(f"  {msg}")
        return cached

    results = cached if cached is not None else []
    start_idx = len(results)

    logger.info("=" * 60)
    logger.info("[%s] Generation started — model=%s, questions=%d, resume_from=%d",
                dataset_name.upper(), MODEL, len(records), start_idx)
    logger.info("[%s] Config: max_tokens_greedy=%d, max_tokens_sampled=%d, temp=%.1f, n_samples=%d",
                dataset_name.upper(), MAX_TOKENS_GREEDY, MAX_TOKENS_SAMPLED, TEMPERATURE_SAMPLED, N_SAMPLES)

    if start_idx > 0:
        msg = f"{dataset_name.upper()}: resuming from question {start_idx}/{len(records)}"
        logger.info(msg)
        print(f"  {msg}")
    else:
        msg = f"{dataset_name.upper()}: generating for {len(records)} questions"
        logger.info(msg)
        print(f"  {msg}")
    print(f"  Model: {MODEL}")

    total_prompt_tokens = 0
    total_completion_tokens = 0
    t_start = time.time()

    for i in range(start_idx, len(records)):
        rec = records[i]
        t_q = time.time()

        try:
            gen = _generate_for_question(rec["question"])
        except Exception as e:
            logger.error("[%s] FAILED idx=%d question='%s': %s",
                         dataset_name.upper(), rec["idx"], rec["question"][:80], e)
            raise

        elapsed_q = time.time() - t_q

        # Track usage
        prompt_tok = gen["_greedy_usage"]["prompt"] + gen["_sampled_usage"]["prompt"]
        completion_tok = gen["_greedy_usage"]["completion"] + gen["_sampled_usage"]["completion"]
        total_prompt_tokens += prompt_tok
        total_completion_tokens += completion_tok

        # Log every question at DEBUG, every 10 at INFO
        logger.debug("[%s] idx=%d  logprob=%.4f  greedy_tokens=%d  sampled_tokens=%d  "
                     "prompt_tok=%d  completion_tok=%d  time=%.1fs  answer='%s'",
                     dataset_name.upper(), rec["idx"], gen["mean_logprob"],
                     len(gen["token_logprobs"]),
                     sum(len(lp) for lp in gen["sampled_token_logprobs"]),
                     prompt_tok, completion_tok, elapsed_q,
                     gen["greedy_answer"][:80])

        gen["idx"] = rec["idx"]
        gen["question"] = rec["question"]
        results.append(gen)

        if (i + 1) % 10 == 0 or i == len(records) - 1:
            elapsed_total = time.time() - t_start
            rate = (i + 1 - start_idx) / elapsed_total if elapsed_total > 0 else 0
            eta_min = (len(records) - i - 1) / rate / 60 if rate > 0 else 0

            progress_msg = (f"[{dataset_name.upper()}] {i+1}/{len(records)}: "
                            f"logprob={gen['mean_logprob']:.3f}, "
                            f"answer='{gen['greedy_answer'][:60]}...'")
            print(f"    {progress_msg}")

            logger.info("[%s] progress=%d/%d  rate=%.1f q/s  eta=%.1f min  "
                        "cumulative_prompt_tok=%d  cumulative_completion_tok=%d",
                        dataset_name.upper(), i + 1, len(records), rate, eta_min,
                        total_prompt_tokens, total_completion_tokens)

        if (i + 1) % save_every == 0 or i == len(records) - 1:
            save_cache(results, cache_path)
            logger.info("[%s] checkpoint saved: %d/%d to %s",
                        dataset_name.upper(), len(results), len(records), cache_path)

    elapsed_total = time.time() - t_start
    est_cost = (total_prompt_tokens / 1e6 * 0.15) + (total_completion_tokens / 1e6 * 0.60)

    logger.info("[%s] Generation complete: %d questions in %.1f min",
                dataset_name.upper(), len(results), elapsed_total / 60)
    logger.info("[%s] Token usage: prompt=%d  completion=%d  total=%d  est_cost=$%.4f",
                dataset_name.upper(), total_prompt_tokens, total_completion_tokens,
                total_prompt_tokens + total_completion_tokens, est_cost)

    print(f"  {dataset_name.upper()}: generation complete ({len(results)} questions, "
          f"{elapsed_total/60:.1f} min, ~${est_cost:.2f})")

    return results
