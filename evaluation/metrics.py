"""Métricas de busca: nDCG@k, recall@k e MRR, com relevância graduada."""

from __future__ import annotations

import math


def ndcg_at_k(ranked: list[str], gains: dict[str, int], k: int = 10) -> float:
    dcg = sum(gains.get(t, 0) / math.log2(i + 2) for i, t in enumerate(ranked[:k]))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], gains: dict[str, int], k: int = 10) -> float:
    relevant = {t for t, g in gains.items() if g > 0}
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else 0.0


def reciprocal_rank(ranked: list[str], gains: dict[str, int]) -> float:
    for i, t in enumerate(ranked, 1):
        if gains.get(t, 0) > 0:
            return 1 / i
    return 0.0
