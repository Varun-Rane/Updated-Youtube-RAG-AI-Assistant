"""retriever.py — orchestration layer for hybrid retrieval + cross-encoder reranking.

Pipeline
--------
    Question
        │
        ▼
    Timestamp routing  (exact match → return immediately)
        │
        ▼
    Comparison query split  (optional, preserves original behaviour)
        │
        ▼
    HybridRetriever  (BM25 + FAISS + RRF)  → top 20 candidates with scores
        │
        ▼
    Reranker  (CrossEncoder)  → top N docs, best first
        │
        ▼
    Deduplicate + return

Public surface used by rag.py (unchanged):
    retrieve_documents(retriever, question, settings) -> List[RerankedDoc]
    format_retrieved_context(documents, max_chars)   -> (context_str, chunks)
    unique_timestamps(chunks)
    unique_sources(chunks, videos)
"""

import re
import logging
from utils import trim_text

from hybrid_retriever import HybridRetriever
from reranker import Reranker, RerankedDoc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

_HYBRID_SINGLETON     = None
_HYBRID_DOC_SIGNATURE = None

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

TIMESTAMP_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")

_COMPARISON_WORDS = {
    "compare", "comparison", "difference", "different",
    "vs", "versus",
}


def is_comparison_query(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in _COMPARISON_WORDS)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def get_all_documents(retriever) -> list:
    """Return all Document objects from the underlying LangChain vectorstore."""
    vectorstore = (
        getattr(retriever, "vectorstore", None)
        or getattr(retriever, "_vectorstore", None)
    )
    if vectorstore is None:
        return []
    return list(vectorstore.docstore._dict.values())


def timestamp_search(retriever, timestamp_query: str):
    """Exact-match timestamp lookup — bypasses hybrid retrieval entirely."""
    all_docs = get_all_documents(retriever)
    matches = []
    for doc in all_docs:
        ts_range  = doc.metadata.get("timestamp_range", "")
        ts_single = doc.metadata.get("timestamp", "")
        if timestamp_query in ts_range or timestamp_query == ts_single:
            matches.append(doc)
    return matches[:5]


# ---------------------------------------------------------------------------
# Singleton factories
# ---------------------------------------------------------------------------

def _get_hybrid(retriever, settings) -> HybridRetriever:
    """Return a cached HybridRetriever; rebuild when the corpus changes."""
    global _HYBRID_SINGLETON, _HYBRID_DOC_SIGNATURE

    current_docs = get_all_documents(retriever)
    video_ids    = {doc.metadata.get("video_id") for doc in current_docs}
    signature    = (len(current_docs), tuple(sorted(video_ids)))

    if _HYBRID_SINGLETON is None or _HYBRID_DOC_SIGNATURE != signature:
        logger.info(
            "Building HybridRetriever index — %d documents, videos: %s",
            len(current_docs), sorted(video_ids),
        )
        _HYBRID_SINGLETON = HybridRetriever(retriever, settings, verbose=False)
        _HYBRID_SINGLETON.build_index(current_docs)
        _HYBRID_DOC_SIGNATURE = signature

    return _HYBRID_SINGLETON




# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve_documents(retriever, question, settings) -> list:
    """Full retrieval pipeline: hybrid → rerank → deduplicate.

    Returns
    -------
    List[RerankedDoc]  (or plain Documents for timestamp matches).
    Each item exposes .metadata and .page_content (via .doc) as before,
    plus .rerank_score, .rrf_score, .bm25_score, .dense_score.
    """
    if retriever is None:
        return []

    question = question.strip()

    # ------------------------------------------------------------------ 1
    # Timestamp routing — return immediately on exact timestamp match.
    # ------------------------------------------------------------------ 1
    timestamp_match = TIMESTAMP_PATTERN.search(question)
    if timestamp_match:
        docs = timestamp_search(retriever, timestamp_match.group())
        if docs:
            return docs

    # ------------------------------------------------------------------ 2
    # Comparison query split — preserves original behaviour.
    # ------------------------------------------------------------------ 2
    if is_comparison_query(question):
        terms = [
            kw for kw in re.findall(r"\w+", question.lower())
            if kw not in _COMPARISON_WORDS
        ]
    else:
        terms = [question]

    # ------------------------------------------------------------------ 3
    # Hybrid retrieval + reranking.
    # Both objects are constructed here — after all early-exit paths — so
    # timestamp queries and any future shortcuts never touch the reranker.
    # Reranker construction is cheap: lru_cache ensures the model is not
    # reloaded between calls.
    # ------------------------------------------------------------------ 3
    hybrid   = _get_hybrid(retriever, settings)
    reranker = Reranker(settings)

    raw_pairs: list = []
    seen      = set()
    fetch_k   = settings.hybrid_fetch_k

    for term in terms:
        term_pairs = hybrid.retrieve_with_scores(term, top_k=fetch_k)
        for doc, scores in term_pairs:
            key = (doc.metadata.get("video_id", ""), doc.metadata.get("chunk_index", ""))
            if key in seen:
                continue
            seen.add(key)
            raw_pairs.append((doc, scores))
            if len(raw_pairs) >= fetch_k:
                break
        if len(raw_pairs) >= fetch_k:
            break

    # ------------------------------------------------------------------ 4
    # Cross-encoder reranking.
    # ------------------------------------------------------------------ 4
    if settings.debug_reranker:
        SEP = "=" * 80
        DIV = "-" * 80
        print(SEP)
        print(f"HYBRID → RERANKER  |  query: {question}")
        print(DIV)
        print(f"Hybrid candidates ({len(raw_pairs)}):")
        for i, (doc, scores) in enumerate(raw_pairs, 1):
            ts  = doc.metadata.get("timestamp_range", doc.metadata.get("timestamp", "N/A"))
            rrf = scores.get("rrf_score", 0.0)
            bm  = scores.get("bm25_score")
            dm  = scores.get("dense_score")
            print(
                f"  {i:2d}.  rrf={rrf:.5f}"
                f"  bm25={'N/A   ' if bm is None else f'{bm:6.3f}'}"
                f"  dense={'N/A  ' if dm is None else f'{dm:.4f}'}"
                f"  {ts}"
            )

    reranked = reranker.rerank(question, raw_pairs)

    if settings.debug_reranker:
        print(DIV)
        print(f"After reranking (top {len(reranked)}):")
        for i, r in enumerate(reranked, 1):
            ts = r.metadata.get("timestamp_range", r.metadata.get("timestamp", "N/A"))
            print(
                f"  {i:2d}.  rerank={r.rerank_score:7.3f}"
                f"  rrf={r.rrf_score:.5f}"
                f"  bm25={'N/A   ' if r.bm25_score is None else f'{r.bm25_score:6.3f}'}"
                f"  dense={'N/A  ' if r.dense_score is None else f'{r.dense_score:.4f}'}"
                f"  {ts}"
            )
        print(SEP)

    return reranked


# ---------------------------------------------------------------------------
# Context formatting — compatible with both RerankedDoc and plain Document
# ---------------------------------------------------------------------------

def _get_doc_fields(document):
    """Extract text and metadata from either a RerankedDoc or a plain Document.

    Allows timestamp-routed plain Documents and reranked RerankedDocs to both
    flow through format_retrieved_context() without special-casing in rag.py.
    """
    if isinstance(document, RerankedDoc):
        return document.text, document.metadata
    # Plain LangChain Document (timestamp path).
    return document.page_content, document.metadata


def format_retrieved_context(documents, max_chars):
    """Build the RAG prompt context string from retrieved documents."""
    sections        = []
    retrieved_chunks = []
    current_chars   = 0

    for document in documents:
        text, metadata = _get_doc_fields(document)

        timestamp = metadata.get(
            "timestamp_range", metadata.get("timestamp", "00:00")
        )
        video_title  = metadata.get("video_title",  "Video")
        video_url    = metadata.get("video_url",    "")
        source_label = metadata.get("source_label", "Video")

        text = trim_text(text.strip(), 700)

        block = (
            f"\nSource Video: {source_label} - {video_title}\n"
            f"Video URL: {video_url}\n"
            f"Timestamp: {timestamp}\n\n"
            f"Transcript:\n{text}\n"
        )

        if current_chars + len(block) > max_chars:
            break

        current_chars += len(block)
        sections.append(block)

        retrieved_chunks.append({
            "video_id":     metadata.get("video_id",    ""),
            "video_title":  video_title,
            "video_url":    video_url,
            "source_label": source_label,
            "timestamp":    timestamp,
            "start":        metadata.get("start",       ""),
            "duration":     metadata.get("duration",    ""),
            "chunk_index":  metadata.get("chunk_index", ""),
            "text":         text,
        })

    context = "\n\n--------------------------\n\n".join(sections)
    context = trim_text(context, max_chars)
    return context, retrieved_chunks


# ---------------------------------------------------------------------------
# Deduplication helpers (unchanged public API)
# ---------------------------------------------------------------------------

def unique_timestamps(chunks):
    seen, result = set(), []
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if ts and ts not in seen:
            seen.add(ts)
            result.append(ts)
    return result


def unique_sources(chunks, videos=None):
    seen, result = set(), []
    for chunk in chunks:
        key = chunk.get("video_url", "")
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "label": chunk.get("source_label", "Video"),
            "title": chunk.get("video_title",  "Video"),
            "url":   key,
        })

    if result:
        return result

    if videos:
        for video in videos:
            result.append({
                "label": video.get("label",       "Video"),
                "title": video.get("video_title", "Video"),
                "url":   video.get("video_url",   ""),
            })
    return result