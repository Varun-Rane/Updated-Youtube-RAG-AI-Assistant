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
    Reranker  (CrossEncoder)  → top N docs, best first   [Phase 2.1]
        │
        ▼
    Dynamic Top-K selection                              [Phase 2.4]
        │
        ▼
    Adjacent chunk expansion                             [Phase 2.3]
        │
        ▼
    Return merged, ranked docs

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

# ---------------------------------------------------------------------------
# Phase 2.4 — Dynamic Top-K
# ---------------------------------------------------------------------------

_LARGE_K_WORDS  = {"compare", "comparison", "difference", "differences",
                   "advantages", "disadvantages", "pros", "cons",
                   "versus", "vs", "contrast", "between"}
_SMALL_K_WORDS  = {"what", "define", "definition", "who", "when", "where",
                   "which", "is", "are", "does"}


def compute_dynamic_top_k(question: str, settings) -> int:
    """Return a top-k value scaled to the breadth of the question.

    Rules (in priority order):
        broad / comparison keywords  →  top_k_large  (default 8)
        narrow / definition keywords →  top_k_small  (default 3)
        everything else              →  top_k_medium (default 5)

    All three values come from settings so they can be tuned via .env
    without touching code.
    """
    if not settings.dynamic_top_k:
        return settings.top_k_medium

    tokens = set(re.findall(r"\w+", question.lower()))

    if tokens & _LARGE_K_WORDS:
        return settings.top_k_large
    if tokens & _SMALL_K_WORDS:
        return settings.top_k_small
    return settings.top_k_medium


# ---------------------------------------------------------------------------
# Phase 2.3 — Adjacent Chunk Expansion
# ---------------------------------------------------------------------------

def _expand_adjacent_chunks(
    reranked: list,
    all_docs: list,
    settings,
) -> list:
    """Expand each reranked doc to include its neighbours from the same video.

    For every RerankedDoc at chunk_index N, we include chunks N-before … N+after
    from the same video.  The result preserves the reranker's priority order:
    the neighbours of the top-ranked chunk come first, then the neighbours of
    the second-ranked chunk, and so on.  Chunks already present are not
    duplicated.

    Parameters
    ----------
    reranked  : List[RerankedDoc] sorted best-first by reranker.
    all_docs  : All Document objects from the vectorstore (used as the chunk pool).
    settings  : Settings — reads merge_adjacent_chunks, adjacent_chunks_before,
                adjacent_chunks_after.

    Returns
    -------
    List[RerankedDoc | Document] in merged priority order.
    """
    if not settings.merge_adjacent_chunks or not reranked:
        return reranked

    n_before = settings.adjacent_chunks_before
    n_after  = settings.adjacent_chunks_after

    # Build a lookup: (video_id, chunk_index) → Document
    pool: dict = {}
    for doc in all_docs:
        vid = doc.metadata.get("video_id", "")
        idx = doc.metadata.get("chunk_index")
        if idx is not None:
            pool[(vid, int(idx))] = doc

    merged  = []
    seen    = set()   # (video_id, chunk_index) already added

    for ranked_doc in reranked:
        vid   = ranked_doc.metadata.get("video_id", "")
        pivot = ranked_doc.metadata.get("chunk_index")
        if pivot is None:
            # No chunk_index metadata — pass through as-is.
            key = id(ranked_doc)
            if key not in seen:
                seen.add(key)
                merged.append(ranked_doc)
            continue

        pivot = int(pivot)
        window = range(pivot - n_before, pivot + n_after + 1)

        for idx in window:
            key = (vid, idx)
            if key in seen:
                continue
            seen.add(key)
            if idx == pivot:
                merged.append(ranked_doc)          # keep RerankedDoc with scores
            elif key in pool:
                merged.append(pool[key])           # plain Document neighbour

    return merged


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
        print(f"Retrieved {len(term_pairs)} candidates")
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

    # ------------------------------------------------------------------ 5
    # Phase 2.4 — Dynamic Top-K fed directly into reranker.
    # One source of truth: reranker returns exactly top_n docs.
    # No post-rerank slice needed.
    # ------------------------------------------------------------------ 5
    top_n = min(compute_dynamic_top_k(question, settings), len(raw_pairs))

    if settings.debug_reranker:
        print(f"[dynamic_top_k] top_n={top_n}")

    reranked = reranker.rerank(question, raw_pairs, top_n=top_n)
    print(f"Reranked: {len(reranked)}")

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

    # ------------------------------------------------------------------ 6
    # Phase 2.3 — Adjacent chunk expansion.
    # ------------------------------------------------------------------ 6
    all_docs = get_all_documents(retriever)
    merged   = _expand_adjacent_chunks(reranked, all_docs, settings)

    if settings.debug_reranker:
        print(f"[adjacent_merge] {len(reranked)} reranked → {len(merged)} after expansion")
        for i, d in enumerate(merged, 1):
            ts  = d.metadata.get("timestamp_range", d.metadata.get("timestamp", "N/A"))
            idx = d.metadata.get("chunk_index", "?")
            print(f"  {i:2d}.  chunk={idx}  {ts}")
        print("=" * 80)

    return merged


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