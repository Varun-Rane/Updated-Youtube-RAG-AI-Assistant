"""
metrics.py
--------
Retrieval evaluation metrics.

All functions take:
    retrieved_indices : list[int]  — chunk_index values returned by the retriever,
                                 in ranked order (best first).
    relevant_indices  : list[int]  — chunk_index values that are ground-truth
                                 relevant for this query (from benchmark.json).

All functions return a float in [0, 1] (or 0.0 if relevant_indices is empty
and the question is a negative example with expected_chunk_indices=[]).
"""

import math
from typing import List, Dict

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """Fraction of relevant chunks found in the top-k retrieved results.

    Recall@K = |relevant ∩ top-K retrieved| / |relevant|

    Returns 0.0 if relevant_indices is empty (negative question).
    """
    if not relevant_indices:
        return 0.0
    top_k = set(retrieved_indices[:k])
    relevant = set(relevant_indices)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant.

    Precision@K = |relevant ∩ top-K retrieved| / K

    Returns 1.0 if relevant_indices is empty and retrieved list is also empty
    (perfect negative query).  Returns 0.0 if something was retrieved for a
    negative query.
    """
    if not relevant_indices:
        # Negative question: precision is 1.0 only if nothing relevant retrieved.
        return 1.0 if not retrieved_indices else 0.0
    if k == 0:
        return 0.0
    top_k = set(retrieved_indices[:k])
    relevant = set(relevant_indices)
    return len(top_k & relevant) / k


def reciprocal_rank(retrieved_indices: List[int], relevant_indices: List[int]) -> float:
    """Reciprocal rank of the first relevant document in the retrieved list.

    RR = 1 / rank_of_first_relevant  (or 0.0 if no relevant document found)

    Returns 0.0 for negative questions (no expected relevant docs).
    """
    if not relevant_indices:
        return 0.0
    relevant = set(relevant_indices)
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def hit_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """Hit@K = 1 if any relevant document is found in top-k, else 0.

    Returns 0.0 for negative questions (no expected relevant docs).
    """
    if not relevant_indices:
        return 0.0
    top_k = set(retrieved_indices[:k])
    relevant = set(relevant_indices)
    return 1.0 if top_k & relevant else 0.0


def ndcg_at_k(retrieved_indices: List[int], relevant_indices: List[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K.

    Uses binary relevance (1 if chunk is relevant, 0 otherwise).

    NDCG@K = DCG@K / IDCG@K

    Returns 0.0 for negative questions.
    """
    if not relevant_indices:
        return 0.0
    relevant = set(relevant_indices)

    def _dcg(ranked: List[int], k: int) -> float:
        gain = 0.0
        for i, idx in enumerate(ranked[:k], start=1):
            if idx in relevant:
                gain += 1.0 / math.log2(i + 1)
        return gain

    dcg = _dcg(retrieved_indices, k)
    # Ideal: all relevant docs at the top.
    ideal_ranked = list(relevant_indices) + [x for x in retrieved_indices if x not in relevant]
    idcg = _dcg(ideal_ranked, k)

    return dcg / idcg if idcg > 0 else 0.0

# ---------------------------------------------------------------------------
# Aggregate over a benchmark run
# ---------------------------------------------------------------------------

def compute_aggregate_metrics(
    results: List[Dict],
    k_values: List[int] = (5, 10),
) -> Dict:
    """Compute mean metrics over a list of per-question result dicts.

    Each element of ``results`` must have:
        retrieved_indices : list[int]
        relevant_indices  : list[int]
        is_negative       : bool  (True if the question has no expected chunks)

    Returns a dict with keys like:
        recall@5, recall@10, precision@5, precision@10, mrr, ndcg@5, ndcg@10,
        hit@5, hit@10
    """
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    # Separate positive and negative questions for cleaner reporting.
    positive = [r for r in results if not r.get("is_negative", False)]
    negative = [r for r in results if r.get("is_negative", False)]

    def _add(key: str, value: float) -> None:
        totals[key] = totals.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1

    for r in positive:
        ri = r["retrieved_indices"]
        rel = r["relevant_indices"]
        for k in k_values:
            _add(f"recall@{k}", recall_at_k(ri, rel, k))
            _add(f"precision@{k}", precision_at_k(ri, rel, k))
            _add(f"ndcg@{k}", ndcg_at_k(ri, rel, k))
            _add(f"hit@{k}", hit_at_k(ri, rel, k))
        _add("mrr", reciprocal_rank(ri, rel))

    # For negative questions we only track precision (should retrieve nothing relevant).
    for r in negative:
        ri = r["retrieved_indices"]
        for k in k_values:
            _add(f"neg_precision@{k}", precision_at_k(ri, [], k))

    aggregated = {
        key: round(totals[key] / counts[key], 4)
        for key in totals
    }
    aggregated["n_positive"] = len(positive)
    aggregated["n_negative"] = len(negative)
    return aggregated


def print_metrics_report(metrics: Dict, title: str = "Retrieval Evaluation") -> None:
    """Pretty-print an aggregated metrics dict."""
    SEP = "=" * 60
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"  {key:<20} {val:.4f}")
        else:
            print(f"  {key:<20} {val}")
    print(SEP)
