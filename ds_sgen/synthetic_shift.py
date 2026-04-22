"""Synthetic covariate-shift construction for DS-SGen validation.

Builds a (source, target) pair from a single underlying TriviaQA pool such that
P(X) differs (via topic-mixture weighting) but P(Y | X) is identical (the
underlying question and its cached label are the same deterministic entity in
both samples). Positive control for the screening protocol.

All data is read from the cached TQA Stage 1-3 outputs plus the cached MiniLM
embeddings from Method 3. No HuggingFace / OpenAI / DeBERTa calls.
"""

import logging
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


@dataclass
class SyntheticPair:
    alpha: float
    seed: int
    K: int
    fm1_cutoff: float
    n_pool: int
    source_idx: list[int]
    target_idx: list[int]
    A_topics: list[int]
    B_topics: list[int]
    source_topic_hist: list[int]
    target_topic_hist: list[int]
    source_acc: float
    target_acc: float


def filter_high_confidence(
    tqa_merged: list[dict],
    fm1_quantile: float = 0.40,
) -> tuple[list[int], float]:
    """Indices above the fm1_quantile-th percentile (0.40 keeps top 60%)."""
    fm1 = np.array([r["fM1"] for r in tqa_merged])
    cutoff = float(np.percentile(fm1, 100.0 * fm1_quantile))
    pool_idx = [i for i, v in enumerate(fm1) if v >= cutoff]
    logger.info(
        "  high-confidence filter: fm1 >= %.4f (quantile %.2f) -> %d / %d retained",
        cutoff, fm1_quantile, len(pool_idx), len(tqa_merged),
    )
    return pool_idx, cutoff


def cluster_topics(
    embeddings: np.ndarray,
    K: int = 20,
    seed: int = 42,
) -> np.ndarray:
    logger.info("  clustering %d points into K=%d topics (seed=%d)",
                len(embeddings), K, seed)
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)
    counts = np.bincount(labels, minlength=K)
    logger.info("  topic sizes: min=%d, median=%d, max=%d, std=%.1f",
                counts.min(), int(np.median(counts)), counts.max(), float(counts.std()))
    return labels


def build_synthetic_pair(
    tqa_merged: list[dict],
    tqa_embeddings: np.ndarray,
    alpha: float,
    K: int = 20,
    n_S: int = 1000,
    n_T: int = 1000,
    fm1_quantile: float = 0.40,
    seed: int = 42,
    partition_strategy: str = "random",
) -> SyntheticPair:
    """Build a (source, target) pair.

    partition_strategy:
      - "random": randomly split K clusters into A (source-heavy) and B (target-heavy). seed-controlled.
      - "accuracy_sorted": top-(K/2) clusters by mean entail_label go to A; bottom to B.
        Engineers a deterministic accuracy gradient between source and target, so T3/T6 are active
        by construction when the pool has real per-cluster accuracy variance.
    """
    if not (0.5 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0.5, 1.0); got {alpha}")

    pool_idx, fm1_cutoff = filter_high_confidence(tqa_merged, fm1_quantile)
    pool_emb = tqa_embeddings[pool_idx]
    n_pool = len(pool_idx)

    if n_pool < n_S + n_T:
        raise ValueError(
            f"pool size {n_pool} too small for n_S={n_S} + n_T={n_T}; "
            "lower fm1_quantile"
        )

    topic = cluster_topics(pool_emb, K=K, seed=seed)

    rng = np.random.RandomState(seed)
    if partition_strategy == "random":
        topic_perm = rng.permutation(K)
        A = sorted(int(t) for t in topic_perm[: K // 2])
        B = sorted(int(t) for t in topic_perm[K // 2 :])
        logger.info("  topic partition (random): |A|=%d, |B|=%d (seed=%d)",
                    len(A), len(B), seed)
    elif partition_strategy == "accuracy_sorted":
        pool_labels = np.array([tqa_merged[i]["entail_label"] for i in pool_idx])
        cluster_acc = np.array([
            float(pool_labels[topic == k].mean()) if (topic == k).sum() > 0 else 0.5
            for k in range(K)
        ])
        order = np.argsort(-cluster_acc)   # descending
        A = sorted(int(t) for t in order[: K // 2])
        B = sorted(int(t) for t in order[K // 2 :])
        logger.info("  topic partition (accuracy_sorted): A_mean_acc=%.3f, B_mean_acc=%.3f (seed=%d)",
                    cluster_acc[A].mean(), cluster_acc[B].mean(), seed)
    else:
        raise ValueError(f"unknown partition_strategy: {partition_strategy}")
    A_set = set(A)

    pS = np.where(np.isin(topic, list(A_set)), alpha, 1.0 - alpha)
    pS = pS / pS.sum()
    source_local = rng.choice(n_pool, size=n_S, replace=False, p=pS)
    used = set(int(j) for j in source_local)

    remaining_local = np.array([j for j in range(n_pool) if j not in used])
    if len(remaining_local) < n_T:
        raise RuntimeError(f"remaining pool {len(remaining_local)} < n_T={n_T}")
    remaining_topic = topic[remaining_local]
    pT = np.where(np.isin(remaining_topic, list(A_set)), 1.0 - alpha, alpha)
    pT = pT / pT.sum()
    target_pick = rng.choice(len(remaining_local), size=n_T, replace=False, p=pT)
    target_local = remaining_local[target_pick]

    source_idx = [int(pool_idx[j]) for j in source_local]
    target_idx = [int(pool_idx[j]) for j in target_local]

    source_topic_hist = np.bincount(topic[source_local], minlength=K).tolist()
    target_topic_hist = np.bincount(topic[target_local], minlength=K).tolist()

    source_acc = float(np.mean([tqa_merged[i]["entail_label"] for i in source_idx]))
    target_acc = float(np.mean([tqa_merged[i]["entail_label"] for i in target_idx]))
    logger.info("  source: n=%d, acc=%.3f; target: n=%d, acc=%.3f",
                n_S, source_acc, n_T, target_acc)

    return SyntheticPair(
        alpha=alpha,
        seed=seed,
        K=K,
        fm1_cutoff=fm1_cutoff,
        n_pool=n_pool,
        source_idx=source_idx,
        target_idx=target_idx,
        A_topics=A,
        B_topics=B,
        source_topic_hist=source_topic_hist,
        target_topic_hist=target_topic_hist,
        source_acc=source_acc,
        target_acc=target_acc,
    )


def sweep_alpha_with_screening(
    tqa_merged: list[dict],
    tqa_embeddings: np.ndarray,
    alphas: list[float],
    K: int = 20,
    n_S: int = 1000,
    n_T: int = 1000,
    fm1_quantile: float = 0.40,
    epsilon: float = 0.25,
    classifier_C: float = 1.0,
    seed: int = 42,
    partition_strategy: str = "random",
) -> list[dict]:
    from ds_sgen.screening import run_screening_tests

    sweep = []
    for a in alphas:
        logger.info("")
        logger.info("==== sweep: alpha = %.2f ====", a)
        pair = build_synthetic_pair(
            tqa_merged, tqa_embeddings,
            alpha=a, K=K, n_S=n_S, n_T=n_T,
            fm1_quantile=fm1_quantile, seed=seed,
            partition_strategy=partition_strategy,
        )

        y_S = np.array([tqa_merged[i]["entail_label"] for i in pair.source_idx])
        y_T = np.array([tqa_merged[i]["entail_label"] for i in pair.target_idx])
        fM_S = np.array([tqa_merged[i]["fM1"] for i in pair.source_idx])
        fM_T = np.array([tqa_merged[i]["fM1"] for i in pair.target_idx])
        emb_S = tqa_embeddings[pair.source_idx]
        emb_T = tqa_embeddings[pair.target_idx]

        scorecard = run_screening_tests(
            y_S=y_S, y_T=y_T,
            fM_S=fM_S, fM_T=fM_T,
            emb_S=emb_S, emb_T=emb_T,
            epsilon=epsilon,
            classifier_C=classifier_C,
        )

        n_pass = sum(bool(scorecard[f"pass_{k}"]) for k in
                     ["1", "2a", "2b", "3", "4", "5", "6"])

        sweep.append({
            "alpha": a,
            "n_pass": n_pass,
            "scorecard": _json_safe(scorecard),
            "pair": asdict(pair),
        })
        logger.info("  alpha=%.2f -> n_pass=%d/7, T3_gap=%.3f, T4_clf=%.3f, T5_ess=%.3f",
                    a, n_pass, scorecard["gap"], scorecard["acc_clf"],
                    scorecard["ess_ratio"])

    return sweep


def pick_best_alpha(sweep: list[dict], target_clf_acc: float = 0.66) -> dict:
    """Max n_pass; tie-break by |acc_clf - target_clf_acc|."""
    def _key(entry):
        return (-entry["n_pass"],
                abs(entry["scorecard"]["acc_clf"] - target_clf_acc))
    return sorted(sweep, key=_key)[0]


def _json_safe(d: dict) -> dict:
    """Convert numpy scalars/arrays to plain Python for JSON serialization."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x.item() if isinstance(x, (np.floating, np.integer)) else x
                      for x in v]
        elif isinstance(v, dict):
            out[k] = _json_safe(v)
        else:
            out[k] = v
    return out
