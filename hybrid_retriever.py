"""HybridRetriever implementation.

Combines BM25 lexical search with FAISS dense vector search and merges results
via Reciprocal Rank Fusion (RRF).

# FAISS score semantics
----------------------------------------------------------------------
LangChain's ``similarity_search_with_score`` returns different values depending
on how the FAISS index was constructed:

* ``FAISS.from_documents(...)`` with no ``distance_strategy`` ->
  **L2 distance** (lower = more similar, range [0, inf))
* ``FAISS.from_documents(..., distance_strategy=DistanceStrategy.COSINE)`` ->
  **cosine similarity** (higher = more similar, range [-1, 1])

The class detects which mode is in use at construction time and normalises
scores internally so that all downstream logic can always assume:
  ``higher normalised_score = more similar``

Dense candidates whose normalised score falls below ``dense_score_threshold``
are dropped before RRF so that weak matches do not pollute the ranked list.

Typical usage::

    from hybrid_retriever import HybridRetriever

    # Pass the FAISS vectorstore directly (not the retriever wrapper).
    hybrid = HybridRetriever(vectorstore, settings, verbose=True)
    hybrid.build_index(hybrid.get_all_documents())

    docs = hybrid.retrieve("What is IVF?")
    scored = hybrid.retrieve_with_scores("What is IVF?")
"""

import re
import logging
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Query normalisation
# ---------------------------------------------------------------------------
_FILLER_WORDS = {
    "what", "is", "the", "a", "an",
    "explain", "tell", "me", "about",
    "in", "this", "video", "from",
    "how", "why", "when", "where",
    "does", "do", "did", "can",
    "could", "would", "should",
    "please", "give", "show",
    "describe", "define", "and", "or",
    "of", "to", "for", "with", "are", "was", "were",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, collapse whitespace, return token list."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(cleaned.split()).split()


def _normalise_query(query: str) -> str:
    """Remove filler words for BM25; preserve technical terms."""
    tokens = [t for t in _tokenize(query) if t not in _FILLER_WORDS]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------
@dataclass
class RetrievedDoc:
    """A document paired with all retrieval scores.

    All scores are normalised so that higher = more similar / more relevant.

    Attributes
    ----------
    doc         : LangChain Document
    bm25_score  : Raw BM25 Okapi score (higher = more relevant).
                  None if the document only appeared in dense results.
    dense_score : Normalised FAISS score in [0, 1] (higher = more similar).
                  None if the document only appeared in BM25 results.
    rrf_score   : Fused Reciprocal Rank Fusion score (higher = better rank).
    """
    doc: Any
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: float = 0.0

    def scores_dict(self) -> Dict[str, Optional[float]]:
        return {
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "rrf_score": self.rrf_score,
        }


# ---------------------------------------------------------------------------
# FAISS score normalisation
# ---------------------------------------------------------------------------
def _detect_score_mode(vectorstore: Any) -> str:
    """Return 'cosine' or 'l2' based on the vectorstore configuration.

    LangChain stores the distance strategy on vectorstore.distance_strategy.
    Falls back to 'l2' — the default for FAISS.from_documents — which is the
    safe assumption when no strategy is explicitly configured.
    """
    strategy = getattr(vectorstore, "distance_strategy", None)
    if strategy is None:
        return "l2"
    strategy_str = str(strategy).lower()
    if "cosine" in strategy_str or "inner" in strategy_str or "dot" in strategy_str:
        return "cosine"
    return "l2"


def _normalise_dense_score(raw_score: float, mode: str) -> float:
    """Convert a raw FAISS score to a normalised similarity in [0, 1].

    For L2 distance  : 1 / (1 + distance)  maps [0, inf) -> (0, 1].
    For cosine / dot : clamp to [0, 1] (negative similarity becomes 0).

    Result is always: higher = more similar.
    """
    if mode == "l2":
        return 1.0 / (1.0 + raw_score)
    else:
        return max(0.0, min(1.0, float(raw_score)))


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------
class HybridRetriever:
    """BM25 + FAISS dense retrieval fused with Reciprocal Rank Fusion.

    Parameters
    ----------
    dense_retriever :
        A FAISS vectorstore (preferred) or a LangChain retriever whose
        .vectorstore attribute exposes the underlying store.
    settings :
        Settings dataclass from config.py.
    verbose :
        Print a structured per-query debug log when True.
    dense_score_threshold :
        Minimum *normalised* similarity [0, 1] for a dense candidate to enter
        RRF. Defaults to settings.dense_skip_threshold or 0.0 (no filter).
        Applied after normalisation, so it is independent of whether the FAISS
        index uses L2 or cosine distance.
    """

    def __init__(
        self,
        dense_retriever: Any,
        settings: Any,
        verbose: bool = False,
        dense_score_threshold: Optional[float] = None,
    ) -> None:
        # Unwrap a retriever wrapper to the underlying vectorstore.
        vs = (
            getattr(dense_retriever, "vectorstore", None)
            or getattr(dense_retriever, "_vectorstore", None)
        )
        self._vectorstore = vs if vs is not None else dense_retriever
        self._settings = settings
        self.verbose = verbose

        # Detect FAISS score direction once at construction time.
        self._score_mode: str = _detect_score_mode(self._vectorstore)

        # Tunable knobs.
        self.bm25_top_k: int = getattr(settings, "bm25_top_k", 50)
        self.dense_top_k: int = getattr(settings, "dense_top_k", 20)
        self.rrf_k: int = getattr(settings, "rrf_k", 60)
        self.final_top_k: int = getattr(settings, "final_top_k", 20)

        # Threshold applies to *normalised* scores (always higher = better).
        if dense_score_threshold is not None:
            self.dense_score_threshold: float = dense_score_threshold
        else:
            self.dense_score_threshold = getattr(settings, "dense_skip_threshold", 0.0)

        self._bm25: Optional[BM25Okapi] = None
        self._doc_lookup: Dict[Tuple[str, int], Any] = {}
        self._doc_list: List[Any] = []
        self._logger = logging.getLogger(__name__)

        if self.verbose:
            print(f"[HybridRetriever] FAISS score mode: {self._score_mode}")
            print(f"[HybridRetriever] Dense threshold (normalised): {self.dense_score_threshold}")

    # ------------------------------------------------------------------
    # Document access
    # ------------------------------------------------------------------
    def get_all_documents(self) -> List[Any]:
        """Return every Document stored in the vectorstore."""
        try:
            return list(self._vectorstore.docstore._dict.values())
        except AttributeError:
            self._logger.warning("Vectorstore does not expose docstore.")
            return []

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------
    def build_index(self, documents: List[Any]) -> None:
        """Build the BM25 index. Call once after loading transcripts."""
        if not documents:
            raise ValueError("No documents supplied to build_index().")

        tokenised: List[List[str]] = []
        lookup: Dict[Tuple[str, int], Any] = {}
        ordered: List[Any] = []

        for doc in documents:
            tokens = _tokenize(doc.page_content)
            tokenised.append(tokens)
            key = (
                doc.metadata.get("video_id", ""),
                doc.metadata.get("chunk_index", -1),
            )
            lookup[key] = doc
            ordered.append(doc)

        self._bm25 = BM25Okapi(tokenised)
        self._doc_lookup = lookup
        self._doc_list = ordered

        if self.verbose:
            print(f"[HybridRetriever] BM25 index built — {len(documents)} documents")

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------
    def _bm25_search(self, normalised_query: str) -> List[RetrievedDoc]:
        """BM25 search on a pre-normalised query string.

        Only documents with a positive BM25 score are returned. Zero-score
        documents carry no lexical signal and would corrupt RRF rankings by
        consuming rank slots that belong to genuinely matching chunks.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built — call build_index() first.")

        q_tokens = _tokenize(normalised_query)  # consistent with document tokenisation
        scores = self._bm25.get_scores(q_tokens)

        # Filter to positive scores only, then sort descending.
        ranked = [
            (i, float(scores[i]))
            for i in range(len(scores))
            if scores[i] > 0
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        ranked = ranked[: self.bm25_top_k]

        return [
            RetrievedDoc(doc=self._doc_list[i], bm25_score=score)
            for i, score in ranked
        ]

    # ------------------------------------------------------------------
    # Dense search
    # ------------------------------------------------------------------
    def _dense_search(self, raw_query: str) -> List[RetrievedDoc]:
        """Vector similarity search using the full un-normalised query.

        Scores are normalised to [0, 1] regardless of FAISS distance metric.
        Candidates below dense_score_threshold are filtered after normalisation.
        """
        try:
            pairs = self._vectorstore.similarity_search_with_score(
                raw_query, k=self.dense_top_k
            )
        except Exception as exc:
            self._logger.error("Dense retrieval failed: %s", exc)
            return []

        results: List[RetrievedDoc] = []

        if self.verbose:
            print("=" * 80)
            print(f"DENSE DEBUG | Query: {raw_query}")
            print("=" * 80)

        for i, (doc, raw_score) in enumerate(pairs, start=1):
            norm_score = _normalise_dense_score(float(raw_score), self._score_mode)

            if self.verbose:
                print(f"\nCandidate #{i}")
                print(f"Timestamp : {doc.metadata.get('timestamp_range', 'N/A')}")
                print(f"Raw Score : {raw_score}")
                print(f"Normalized: {norm_score:.4f}")
                print(f"Threshold : {self.dense_score_threshold}")
                print(f"Accepted  : {norm_score >= self.dense_score_threshold}")

            if norm_score < self.dense_score_threshold:
                continue

            results.append(
                RetrievedDoc(
                    doc=doc,
                    dense_score=norm_score,
                )
            )

        if self.verbose:
            print(f"\nTotal Accepted Dense Candidates: {len(results)}")
            print("=" * 80)

        return results

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------
    def _rrf_merge(
        self,
        bm25_results: List[RetrievedDoc],
        dense_results: List[RetrievedDoc],
    ) -> List[RetrievedDoc]:
        """Merge BM25 and dense ranked lists via Reciprocal Rank Fusion.

        All three scores (bm25, dense, rrf) are preserved on every returned
        RetrievedDoc for use in Phase 2 confidence gating.
        """
        def _key(rd: RetrievedDoc) -> Tuple[str, int]:
            return (
                rd.doc.metadata.get("video_id", ""),
                rd.doc.metadata.get("chunk_index", -1),
            )

        rrf_scores: Dict[Tuple[str, int], float] = {}
        score_store: Dict[Tuple[str, int], Dict[str, Optional[float]]] = {}

        for rank, rd in enumerate(bm25_results, start=1):
            k = _key(rd)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank)
            score_store.setdefault(k, {"bm25_score": None, "dense_score": None})
            score_store[k]["bm25_score"] = rd.bm25_score

        for rank, rd in enumerate(dense_results, start=1):
            k = _key(rd)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + 1.0 / (self.rrf_k + rank)
            score_store.setdefault(k, {"bm25_score": None, "dense_score": None})
            score_store[k]["dense_score"] = rd.dense_score

        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged: List[RetrievedDoc] = []
        seen: set = set()
        for key, fused_score in sorted_keys:
            if key in seen:
                continue
            seen.add(key)
            doc = self._doc_lookup.get(key)
            if doc is not None:
                s = score_store.get(key, {})
                merged.append(RetrievedDoc(
                    doc=doc,
                    bm25_score=s.get("bm25_score"),
                    dense_score=s.get("dense_score"),
                    rrf_score=fused_score,
                ))
            if len(merged) >= self.final_top_k:
                break

        return merged

    # ------------------------------------------------------------------
    # Debug logging
    # ------------------------------------------------------------------
    def _log_retrieval(
        self,
        query: str,
        norm_q: str,
        bm25_results: List[RetrievedDoc],
        dense_results: List[RetrievedDoc],
        merged: List[RetrievedDoc],
        top_k: int,
    ) -> None:
        SEP = "=" * 80
        DIV = "-" * 80
        print(SEP)
        print(f"QUERY         : {query}")
        print(f"NORMALISED    : {norm_q}")
        print(f"FAISS mode    : {self._score_mode}  (threshold={self.dense_score_threshold:.3f})")
        print(DIV)
        print(f"BM25  — {len(bm25_results)} candidates  (top 5)")
        for i, rd in enumerate(bm25_results[:5], 1):
            ts = rd.doc.metadata.get("timestamp_range", rd.doc.metadata.get("timestamp", "N/A"))
            print(f"  {i:2}. bm25={rd.bm25_score:8.4f}  {ts}")
        print(DIV)
        print(f"Dense — {len(dense_results)} candidates after threshold  (top 5, normalised)")
        for i, rd in enumerate(dense_results[:5], 1):
            ts = rd.doc.metadata.get("timestamp_range", rd.doc.metadata.get("timestamp", "N/A"))
            print(f"  {i:2}. dense={rd.dense_score:.4f}  {ts}")
        print(DIV)
        print(f"RRF   — top {top_k} returned")
        for i, rd in enumerate(merged[:top_k], 1):
            ts = rd.doc.metadata.get("timestamp_range", rd.doc.metadata.get("timestamp", "N/A"))
            bm = f"{rd.bm25_score:7.3f}" if rd.bm25_score is not None else "   N/A "
            dm = f"{rd.dense_score:.4f}" if rd.dense_score is not None else " N/A "
            print(f"  {i:2}. rrf={rd.rrf_score:.6f}  bm25={bm}  dense={dm}  {ts}")
        print(SEP)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Any]:
        """Hybrid retrieval — returns plain Document objects."""
        results = self._retrieve_internal(query, top_k)
        return [rd.doc for rd in results]

    def retrieve_with_scores(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Tuple[Any, Dict[str, Optional[float]]]]:
        """Returns (Document, scores_dict) pairs.

        scores_dict keys: bm25_score, dense_score, rrf_score.
        Designed for Phase 2 Cross-Encoder + Confidence Gate.
        """
        results = self._retrieve_internal(query, top_k)
        return [(rd.doc, rd.scores_dict()) for rd in results]

    def _retrieve_internal(
        self, query: str, top_k: Optional[int] = None
    ) -> List[RetrievedDoc]:
        if top_k is None:
            top_k = self.final_top_k

        norm_q = _normalise_query(query)
        bm25_results = self._bm25_search(norm_q)
        dense_results = self._dense_search(raw_query=query)
        merged = self._rrf_merge(bm25_results, dense_results)

        if self.verbose:
            self._log_retrieval(query, norm_q, bm25_results, dense_results, merged, top_k)

        return merged[:top_k]


# End of hybrid_retriever.py