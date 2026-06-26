"""HybridRetriever implementation.

Provides a lightweight, reusable retrieval component that combines BM25 lexical search (via the `rank-bm25` library) with dense vector search (the existing LangChain retriever). The class builds the BM25 index **once** and caches it for the lifetime of the object.

Typical usage::

    from hybrid_retriever import HybridRetriever
    from config import load_settings
    from rag import get_retriever  # function that returns a LangChain retriever

    settings = load_settings()
    dense = get_retriever(settings)
    hybrid = HybridRetriever(dense, settings, verbose=True)
    # Build the BM25 index from the same documents that the vectorstore holds
    hybrid.build_index(hybrid.get_all_documents())
    # At query time
    results = hybrid.retrieve("What is IVF?")

The public API mirrors the original `retriever.py` expectations (returning a list of LangChain `Document` objects) so the rest of the codebase can remain unchanged.
"""

import re
import logging
from typing import List, Tuple, Any

from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Query normalisation helpers
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
    """Consistent tokenisation for both documents and queries.

    * lower‑case
    * replace punctuation with spaces
    * collapse whitespace
    * optionally drop filler words (only for queries)
    """
    # Lower‑case and replace non‑alphanumeric characters with a space
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    # Collapse whitespace
    cleaned = " ".join(cleaned.split())
    return cleaned.split()

def _normalise_query(query: str) -> str:
    """Prepare a user query for retrieval.
    Removes filler words but keeps technical terms.
    """
    tokens = _tokenize(query)
    tokens = [tok for tok in tokens if tok not in _FILLER_WORDS]
    return " ".join(tokens)

# ---------------------------------------------------------------------------
# HybridRetriever class
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Combines BM25 lexical search with dense vector retrieval.

    Parameters
    ----------
    dense_retriever: Any
        A LangChain retriever instance that implements ``similarity_search`` or ``invoke``.
    settings: Any
        The global ``Settings`` object from ``config.py`` – used to pull configurable top‑k values.
    verbose: bool, optional
        When ``True`` prints short debug banners (mirroring the style used in the original ``retriever.py``).
    """

    def __init__(self, dense_retriever: Any, settings: Any, verbose: bool = False) -> None:
        self._dense = dense_retriever
        self._settings = settings
        self.verbose = verbose
        # Pull tunable limits from settings – fall back to sensible defaults if missing
        self.bm25_top_k = getattr(settings, "bm25_top_k", 50)
        self.dense_top_k = getattr(settings, "dense_top_k", 20)
        self.rrf_k = getattr(settings, "rrf_k", 60)
        self.final_top_k = getattr(settings, "final_top_k", 20)
        self._bm25: BM25Okapi | None = None
        self._doc_lookup: dict[Tuple[str, int], Any] = {}
        self._doc_list: List[Any] = []
        self._logger = logging.getLogger(__name__)

    # ---------------------------------------------------------------------
    # Helper – retrieve raw Document objects from the underlying vectorstore
    # ---------------------------------------------------------------------
    def get_all_documents(self) -> List[Any]:
        """Return the list of Document objects stored in the dense retriever's vectorstore.
        Mirrors the helper used in the legacy ``retriever.py``.
        """
        vectorstore = getattr(self._dense, "vectorstore", None) or getattr(self._dense, "_vectorstore", None)
        if vectorstore is None:
            self._logger.warning("Dense retriever does not expose a vectorstore – returning empty list.")
            return []
        return list(vectorstore.docstore._dict.values())

    # ---------------------------------------------------------------------
    # Index construction – should be called once after transcripts are loaded
    # ---------------------------------------------------------------------
    def build_index(self, documents: List[Any]) -> None:
        """Create the BM25 index from a list of Document objects.
        The method tokenises each document's ``page_content`` using a simple whitespace split.
        """
        if not documents:
            raise ValueError("No documents supplied to build BM25 index.")
        tokenised: List[List[str]] = []
        lookup: dict[Tuple[str, int], Any] = {}
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
            print("=" * 80)
            print("BM25 index built – documents:", len(documents))
            print("=" * 80)

    # ---------------------------------------------------------------------
    # BM25 lexical search (returns Document objects)
    # ---------------------------------------------------------------------
    def _bm25_search(self, query: str) -> List[Any]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built – call build_index() first.")
        q_tokens = query.lower().split()
        scores = self._bm25.get_scores(q_tokens)
        # Get top‑k indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.bm25_top_k]
        return [self._doc_list[i] for i in top_indices]

    # ---------------------------------------------------------------------
    # Dense vector search – uses the underlying LangChain retriever safely
    # ---------------------------------------------------------------------
    def _dense_search(self, query: str) -> List[Any]:
        # Prefer a ``similarity_search`` method that accepts a ``k`` argument.
        similarity_search = getattr(self._dense, "similarity_search", None)
        if callable(similarity_search):
            return similarity_search(query, k=self.dense_top_k)
        # Fallback to ``invoke`` – slice the result to the desired size.
        try:
            results = self._dense.invoke(query)
            return results[: self.dense_top_k]
        except Exception as exc:
            self._logger.error("Dense retrieval failed: %s", exc)
            return []

    # ---------------------------------------------------------------------
    # Reciprocal Rank Fusion (RRF) implementation
    # ---------------------------------------------------------------------
    @staticmethod
    def _rrf_score(rank: int, k: int) -> float:
        """Return the RRF contribution for a given rank (1‑based)."""
        return 1.0 / (k + rank)

    def _rrf_merge(self, bm25_docs: List[Any], dense_docs: List[Any]) -> List[Any]:
        """Merge two ranked lists using RRF and return the top ``final_top_k`` documents.
        If a document appears in both lists the scores are summed.
        """
        def _doc_key(doc: Any) -> Tuple[str, int]:
            return (
                doc.metadata.get("video_id", ""),
                doc.metadata.get("chunk_index", -1),
            )
        scores: dict[Tuple[str, int], float] = {}
        for rank, doc in enumerate(bm25_docs, start=1):
            scores[_doc_key(doc)] = scores.get(_doc_key(doc), 0.0) + self._rrf_score(rank, self.rrf_k)
        for rank, doc in enumerate(dense_docs, start=1):
            scores[_doc_key(doc)] = scores.get(_doc_key(doc), 0.0) + self._rrf_score(rank, self.rrf_k)
        # Order by descending fused score
        sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        merged: List[Any] = []
        seen: set[Tuple[str, int]] = set()
        for key, _ in sorted_items:
            if key in seen:
                continue
            seen.add(key)
            doc = self._doc_lookup.get(key)
            if doc is not None:
                merged.append(doc)
            if len(merged) >= self.final_top_k:
                break
        return merged

    # ---------------------------------------------------------------------
    # Public entry point – used by the existing codebase
    # ---------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Any]:
        """Run a hybrid retrieval round and return the top documents.

        Parameters
        ----------
        query: str
            The raw user question.
        top_k: int, optional
            Overrides ``self.final_top_k`` for this call only.
        """
        if top_k is None:
            top_k = self.final_top_k
        norm_q = _normalise_query(query)
        if self.verbose:
            print("=" * 80)
            print("HYBRID RETRIEVAL – raw query:", query)
            print("NORMALIZED    :", norm_q)
            print("=" * 80)
        bm25_docs = self._bm25_search(norm_q)
        dense_docs = self._dense_search(norm_q)
        merged = self._rrf_merge(bm25_docs, dense_docs)
        if self.verbose:
            print("=" * 80)
            print("BM25 candidates :", len(bm25_docs))
            print("Dense candidates :", len(dense_docs))
            print("Merged top {} :".format(top_k))
            for i, doc in enumerate(merged[:top_k], start=1):
                ts = doc.metadata.get("timestamp_range", doc.metadata.get("timestamp", "N/A"))
                print(f"{i}. {ts}")
            print("=" * 80)
        return merged[:top_k]

# End of hybrid_retriever.py
