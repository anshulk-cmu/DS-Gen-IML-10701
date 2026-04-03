"""SGen-Semi algorithm: semantic clustering and scoring."""

import numpy as np


def cluster_responses(entailment_matrix: list[list[float]], threshold: float = 0.5):
    """Cluster responses based on bidirectional entailment.

    Two responses are in the same semantic cluster if they mutually
    entail each other above the threshold.
    """
    n = len(entailment_matrix)
    clusters = []
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if (entailment_matrix[i][j] >= threshold and
                    entailment_matrix[j][i] >= threshold):
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


def compute_sgen_score(clusters: list[list[int]], num_responses: int):
    """Compute the SGen semantic entropy score.

    Lower entropy = higher confidence (responses are more semantically consistent).
    """
    if num_responses == 0:
        return float("inf")

    probs = [len(c) / num_responses for c in clusters]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)
    return float(entropy)
