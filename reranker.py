"""Cross-Encoder reranker for Phase 2.1.

Takes the (doc, scores_dict) pairs produced by HybridRetriever.retrieve_with_scores()
and re-scores them as (query, document) pairs using a CrossEncoder.

Returned objects are plain dataclasses that carry all four scores:
  bm25_score  — from BM25 Okapi
  dense_score — normalised FAISS similarity
  rrf_score   — Reciprocal Rank Fusion fused score
  rerank_score — CrossEncoder logit (unbounded; higher = more relevant)

The CrossEncoder is loaded once via a module-level lru_cache so it is never
re-instantiated between queries.

Typical usage::

    from reranker import Reranker

    reranker = Reranker(settings)

    # raw_docs is List[Tuple[Document, scores_dict]] from retrieve_with_scores()
    reranked = reranker.rerank(query, raw_docs)

    # reranked is List[RerankedDoc], sorted best-first, length <= rerank_top_n
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CrossEncoder singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load and cache a CrossEncoder by model name.

    Called at most once per unique model name per process lifetime.
    lru_cache(maxsize=4) allows switching between a small number of models
    (e.g. during benchmarking) without reloading unnecessarily.
    """
    logger.info("Loading CrossEncoder: %s", model_name)
    model = CrossEncoder(model_name, max_length=512)
    model.model.eval()   # explicit eval mode — deterministic inference, no dropout
    logger.info("CrossEncoder ready: %s", model_name)
    return model


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RerankedDoc:
    """A retrieved document after cross-encoder rescoring.

    Carries every score produced at each pipeline stage so that confidence
    gating (Phase 2.2) and debugging can inspect the full signal chain.

    Attributes
    ----------
    doc          : Original LangChain Document.
    bm25_score   : BM25 Okapi score. None if document only came from dense search.
    dense_score  : Normalised FAISS score in [0, 1]. None if only from BM25.
    rrf_score    : RRF fused score from HybridRetriever.
    rerank_score : CrossEncoder logit. Unbounded; higher = more relevant.

    Properties
    ----------
    text     : doc.page_content  (no copy stored)
    metadata : doc.metadata      (no copy stored)
    """
    doc:          Any
    bm25_score:   Optional[float] = None
    dense_score:  Optional[float] = None
    rrf_score:    float           = 0.0
    rerank_score: float           = 0.0

    @property
    def text(self) -> str:
        return self.doc.page_content

    @property
    def metadata(self) -> Dict:
        return self.doc.metadata


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Wraps a CrossEncoder and provides a single .rerank() method.

    Parameters
    ----------
    settings : Settings dataclass from config.py.
        Uses settings.rerank_model, settings.rerank_top_n,
        and settings.rerank_batch_size.
    """

    def __init__(self, settings: Any) -> None:
        self._top_n      = settings.rerank_top_n
        self._batch_size = settings.rerank_batch_size
        self._model      = _load_cross_encoder(settings.rerank_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        raw_docs: List[Tuple[Any, Dict]],
    ) -> List[RerankedDoc]:
        """Score every (query, document) pair and return top-N, best first.

        Parameters
        ----------
        query    : The original user question, unmodified.
        raw_docs : List[Tuple[Document, scores_dict]] as returned by
                   HybridRetriever.retrieve_with_scores().

        Returns
        -------
        List[RerankedDoc] sorted by rerank_score descending, length <= rerank_top_n.
        """
        if not raw_docs:
            return []

        # Build (query, text) pairs for batch scoring.
        pairs = [(query, doc.page_content) for doc, _ in raw_docs]

        # predict() returns a numpy array of floats; one per pair.
        # batch_size controls GPU/CPU throughput — tune via RERANK_BATCH_SIZE.
        scores = self._model.predict(pairs, batch_size=self._batch_size)

        reranked: List[RerankedDoc] = []
        for (doc, score_dict), ce_score in zip(raw_docs, scores):
            reranked.append(RerankedDoc(
                doc          = doc,
                bm25_score   = score_dict.get("bm25_score"),
                dense_score  = score_dict.get("dense_score"),
                rrf_score    = score_dict.get("rrf_score", 0.0),
                rerank_score = float(ce_score),
            ))

        reranked.sort(key=lambda r: r.rerank_score, reverse=True)

        top = reranked[: self._top_n]

        logger.debug(
            "Reranked %d docs → top %d | scores: %s",
            len(reranked),
            self._top_n,
            [f"{r.rerank_score:.3f}" for r in top],
        )

        return top