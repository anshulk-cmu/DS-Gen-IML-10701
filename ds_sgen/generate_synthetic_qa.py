"""Generate synthetic factoid QA pairs via GPT-4o-mini for Design A.

Produces records in the same schema as data_loading.py outputs:
    {"idx": int, "question": str, "reference_answer": str,
     "all_answers": [str], "dataset": str,
     "topic": str, "tier": int}

The pool spans 10 topics x 3 difficulty tiers to create genuine variance in
P(Y|X) so that covariate-shift resampling can produce a real accuracy gap.

Reuses the same OpenAI client + retry pattern as generate_responses.py. Uses
JSON mode (response_format={"type": "json_object"}) to batch-generate items
reliably.
"""

import json
import logging
import os
import re
import time

from dotenv import load_dotenv

from ds_sgen.utils import get_cache_path, load_cache, save_cache

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"

TOPICS = [
    "geography",
    "world history",
    "biology and animals",
    "physics and chemistry",
    "astronomy and space",
    "literature and authors",
    "visual art and painters",
    "music and composers",
    "sports and athletics",
    "food and cooking",
]

TIERS = {
    1: "easy — facts any well-read adult high-school graduate would know (e.g., capital cities, common animals, famous books)",
    2: "medium — specific facts that a college-educated enthusiast in the subject would know (e.g., specific scientific discoveries, lesser-known historical figures)",
    3: "hard — specialist knowledge requiring dedicated study or domain expertise (e.g., obscure dates, lesser-known works, technical details)",
}

MAX_RETRIES = 5
RETRY_BASE_WAIT = 10


_client = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _setup_file_logger(log_dir: str):
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "generate_synthetic_qa.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def _api_call_with_retry(**kwargs):
    from openai import RateLimitError, APIError
    for attempt in range(MAX_RETRIES):
        try:
            return _get_client().chat.completions.create(**kwargs)
        except (RateLimitError, APIError) as e:
            wait = RETRY_BASE_WAIT * (2 ** attempt)
            logger.warning("OpenAI error (attempt %d/%d), waiting %ds: %s",
                           attempt + 1, MAX_RETRIES, wait, e)
            time.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def _build_prompt(topic: str, tier: int, n: int) -> list[dict]:
    tier_desc = TIERS[tier]
    system = (
        "You are generating factoid quiz questions with unambiguous short answers. "
        "Your output must be valid JSON."
    )
    user = f"""Generate exactly {n} factoid questions about {topic} at difficulty tier {tier}.

Difficulty tier {tier}: {tier_desc}

Requirements for every question:
- The question must have a single unambiguous correct answer that can be expressed in 1 to 5 words.
- Avoid subjective, opinion, or contested questions.
- Avoid dates before 500 BCE.
- Questions must be self-contained — no "this", "above", or references to earlier items.
- Vary the question style (who, what, when, where, how many) across the {n} items.
- Do not repeat the same answer across items in this batch.

Return ONLY a JSON object with a single key "items" whose value is a list of {n} objects,
each with keys "question" (string) and "answer" (string, 1-5 words)."""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _normalize_answer(ans: str) -> str:
    return re.sub(r"\s+", " ", ans).strip().rstrip(".")


def _is_valid_item(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    q = item.get("question")
    a = item.get("answer")
    if not isinstance(q, str) or not isinstance(a, str):
        return False
    q = q.strip()
    a = _normalize_answer(a)
    if len(q) < 10 or len(q) > 300:
        return False
    if not q.endswith("?"):
        return False
    word_count = len(a.split())
    if word_count < 1 or word_count > 6:
        return False
    if any(p in q.lower() for p in ["this ", "above", "previous", "earlier"]):
        return False
    return True


def _generate_batch(topic: str, tier: int, n: int) -> list[dict]:
    """One API call: generate n (question, answer) items for (topic, tier).

    Returns a list of validated dicts {'question', 'answer'}.
    """
    messages = _build_prompt(topic, tier, n)
    resp = _api_call_with_retry(
        model=MODEL,
        temperature=1.0,
        max_tokens=4000,
        response_format={"type": "json_object"},
        messages=messages,
    )
    content = resp.choices[0].message.content
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Bad JSON for (%s, tier %d): %s", topic, tier, e)
        return []
    raw_items = obj.get("items", [])
    if not isinstance(raw_items, list):
        logger.warning("Response missing 'items' list for (%s, tier %d)", topic, tier)
        return []

    valid = []
    for item in raw_items:
        if _is_valid_item(item):
            valid.append({
                "question": item["question"].strip(),
                "answer": _normalize_answer(item["answer"]),
            })
    logger.info("  batch (%s, tier %d): requested %d, got %d, kept %d",
                topic, tier, n, len(raw_items), len(valid))
    return valid


def generate_qa_pool(
    cfg: dict,
    dataset_name: str = "synth_qa",
    per_cell: int = 80,
    topics: list[str] = None,
    tiers: list[int] = None,
) -> list[dict]:
    """Generate the full synthetic pool across topics x tiers.

    Caches the final records list at <cache_dir>/<dataset_name>_data.json so
    downstream Stage 1 (generate_and_cache_openai) can consume it. If the cache
    exists and has the expected structure, returns it without regenerating.
    """
    _setup_file_logger(cfg.get("log_dir", "logs"))

    if topics is None:
        topics = TOPICS
    if tiers is None:
        tiers = [1, 2, 3]

    cache_path = get_cache_path(cfg["paths"]["cache_dir"], f"{dataset_name}_data")
    cached = load_cache(cache_path)
    if cached is not None and len(cached) > 0:
        logger.info("Loaded %d cached records from %s", len(cached), cache_path)
        print(f"  Loaded {len(cached)} cached QA records from {cache_path}")
        return cached

    logger.info("=" * 60)
    logger.info("Generating synthetic QA pool")
    logger.info("  topics=%d, tiers=%s, per_cell=%d, target_total=%d",
                len(topics), tiers, per_cell, len(topics) * len(tiers) * per_cell)

    records = []
    t_start = time.time()

    for topic in topics:
        for tier in tiers:
            batch_items = _generate_batch(topic, tier, per_cell)
            for item in batch_items:
                records.append({
                    "idx": len(records),
                    "question": item["question"],
                    "reference_answer": item["answer"],
                    "all_answers": [item["answer"]],
                    "dataset": dataset_name,
                    "topic": topic,
                    "tier": tier,
                })
            save_cache(records, cache_path)
            logger.info("  saved %d total records so far", len(records))

    dedup = _dedupe(records)
    if len(dedup) != len(records):
        logger.info("  deduped: %d -> %d", len(records), len(dedup))
        records = [dict(r, idx=i) for i, r in enumerate(dedup)]
        save_cache(records, cache_path)

    elapsed = time.time() - t_start
    logger.info("Generation complete: %d records in %.1f min", len(records), elapsed / 60)
    print(f"  Synthetic QA pool: {len(records)} records in {elapsed/60:.1f} min")

    return records


def _dedupe(records: list[dict]) -> list[dict]:
    seen_q = set()
    seen_a = {}
    out = []
    for r in records:
        qn = r["question"].lower().strip()
        an = r["reference_answer"].lower().strip()
        if qn in seen_q:
            continue
        seen_a[an] = seen_a.get(an, 0) + 1
        if seen_a[an] > 8:
            continue
        seen_q.add(qn)
        out.append(r)
    return out
