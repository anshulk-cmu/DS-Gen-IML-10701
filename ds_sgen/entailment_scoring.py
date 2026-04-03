"""Entailment-based semantic scoring."""

from sentence_transformers import CrossEncoder


def load_entailment_model(model_name: str, cache_dir: str = None):
    """Load an NLI cross-encoder model."""
    model = CrossEncoder(model_name, cache_folder=cache_dir)
    return model


def compute_entailment_matrix(model, responses: list[str]):
    """Compute pairwise entailment scores between all response pairs.

    Returns an NxN matrix where entry [i][j] is the entailment score
    from response i to response j.
    """
    n = len(responses)
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((responses[i], responses[j]))

    if not pairs:
        return [[1.0]]

    scores = model.predict(pairs)

    matrix = [[0.0] * n for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                # NLI models output [contradiction, neutral, entailment]
                matrix[i][j] = float(scores[idx][2]) if len(scores[idx]) > 2 else float(scores[idx])
                idx += 1

    return matrix
