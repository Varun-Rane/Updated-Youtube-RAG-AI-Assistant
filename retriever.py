import re
from utils import trim_text

# Import HybridRetriever at module level (no circular imports expected)
from hybrid_retriever import HybridRetriever

# Global singleton and document count cache
_HYBRID_SINGLETON = None
_HYBRID_DOC_SIGNATURE = None

# ---------------------------------------------------------------------------
# Helpers needed for the legacy interface (timestamp routing, comparison handling,
# and document extraction). These were present in the original retriever and are
# required here because ``retrieve_documents`` still calls them.
# ---------------------------------------------------------------------------
TIMESTAMP_PATTERN = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")

_COMPARISON_WORDS = {
    "compare",
    "comparison",
    "difference",
    "different",
    "vs",
    "versus",
}

def is_comparison_query(question: str) -> bool:
    q = question.lower()
    return any(word in q for word in _COMPARISON_WORDS)

# The original code used a simple keyword‑fallback search.  It is no longer
# needed because ``HybridRetriever`` already performs lexical BM25 search, but we
# keep ``timestamp_search`` and ``get_all_documents`` for timestamp routing and
# debugging output.

def get_all_documents(retriever) -> list:
    """Return all Document objects from the underlying LangChain vectorstore.
    Mirrors the helper used in the legacy ``retriever.py``.
    """
    vectorstore = getattr(retriever, "vectorstore", None) or getattr(retriever, "_vectorstore", None)
    if vectorstore is None:
        return []
    return list(vectorstore.docstore._dict.values())

def timestamp_search(retriever, timestamp_query: str):
    """Return documents whose ``timestamp`` or ``timestamp_range`` matches the query.
    This is a fast exact‑match lookup used before the hybrid retrieval step.
    """
    all_docs = get_all_documents(retriever)
    matches = []
    for doc in all_docs:
        ts_range = doc.metadata.get("timestamp_range", "")
        ts_single = doc.metadata.get("timestamp", "")
        if timestamp_query in ts_range or timestamp_query == ts_single:
            matches.append(doc)
    return matches[:5]


def retrieve_documents(
    retriever,
    question,
    settings,
):
    """Retrieve relevant transcript chunks.

    The flow is:
    1️⃣ Timestamp search (exact match).
    2️⃣ Comparison‑query handling – split into individual terms.
    3️⃣ Hybrid BM25 + dense retrieval via :class:`HybridRetriever`.
    4️⃣ Deduplicate by ``video_id`` + ``chunk_index``.
    5️⃣ Return up to ``MAX_RESULTS`` (default 8) chunks.
    """
    if retriever is None:
        return []

    question = question.strip()

    # ------------------------------------------------------------
    # 1️⃣ Timestamp routing – exact matches are returned immediately
    # ------------------------------------------------------------
    timestamp_match = TIMESTAMP_PATTERN.search(question)
    if timestamp_match:
        docs = timestamp_search(retriever, timestamp_match.group())
        if docs:
            return docs

    # ------------------------------------------------------------
    # 2️⃣ Comparison‑query handling – keep original behaviour
    # ------------------------------------------------------------
    if is_comparison_query(question):
        terms = [
            kw for kw in re.findall(r"\w+", question.lower())
            if kw not in _COMPARISON_WORDS
        ]
    else:
        terms = [question]

    # ------------------------------------------------------------
    # 3️⃣ Hybrid retrieval via a cached HybridRetriever singleton
    # ------------------------------------------------------------
    # HybridRetriever is already imported at module level.
    global _HYBRID_SINGLETON, _HYBRID_DOC_SIGNATURE
    # Determine a signature for the current corpus (set of video IDs).
    current_docs = get_all_documents(retriever)
    # Build a robust signature for the current corpus.
    # Use the total document count and the sorted set of video IDs present.
    # This changes whenever videos are added/removed, ensuring the BM25 index
    # is rebuilt when the underlying transcript collection changes.
    video_ids = {doc.metadata.get("video_id") for doc in current_docs}
    current_signature = (
        len(current_docs),
        tuple(sorted(video_ids))
    )
    if (
        _HYBRID_SINGLETON is None
        or _HYBRID_DOC_SIGNATURE != current_signature
    ):
        # (Re)create the hybrid retriever with the real settings object.
        _HYBRID_SINGLETON = HybridRetriever(
            retriever,
            settings,
            verbose=False,
        )
        _HYBRID_SINGLETON.build_index(current_docs)
        _HYBRID_DOC_SIGNATURE = current_signature

    hybrid = _HYBRID_SINGLETON

    docs: list = []
    seen = set()
    MAX_RESULTS = settings.final_top_k
    for term in terms:
        term_docs = hybrid.retrieve(term, top_k=MAX_RESULTS)
        for doc in term_docs:
            key = (doc.metadata.get("video_id", ""), doc.metadata.get("chunk_index", ""))
            if key in seen:
                continue
            seen.add(key)
            docs.append(doc)
            if len(docs) >= MAX_RESULTS:
                break
        if len(docs) >= MAX_RESULTS:
            break

    # ------------------------------------------------------------
    # 4️⃣ Debug output (kept for parity with the original file)
    # ------------------------------------------------------------
    print("=" * 80)
    print("IVF DEBUG")
    all_docs = get_all_documents(retriever)
    ivf_count = 0
    for doc in all_docs:
        text = doc.page_content.lower()
        if any(tok in text for tok in ("ivf", "inverted", "file index", "आईबीएफ")):
            ivf_count += 1
            print(doc.metadata.get("timestamp_range", "N/A"))
    print("TOTAL IVF DOCS:", ivf_count)
    print("=" * 80)
    print("FINAL DOCS")
    for doc in docs[:MAX_RESULTS]:
        print(doc.metadata.get("timestamp_range", "N/A"))
    print("=" * 80)

    return docs[:MAX_RESULTS]

def format_retrieved_context(
    documents,
    max_chars,
):
    """
    Build context for the RAG prompt.
    """

    sections = []

    retrieved_chunks = []

    current_chars = 0

    for document in documents:

        metadata = document.metadata

        timestamp = metadata.get(
            "timestamp_range",
            metadata.get(
                "timestamp",
                "00:00",
            ),
        )

        video_title = metadata.get(
            "video_title",
            "Video",
        )

        video_url = metadata.get(
            "video_url",
            "",
        )

        source_label = metadata.get(
            "source_label",
            "Video",
        )

        text = trim_text(
            document.page_content.strip(),
            700,
        )

        block = f"""
Source Video: {source_label} - {video_title}
Video URL: {video_url}
Timestamp: {timestamp}

Transcript:
{text}
"""

        if current_chars + len(block) > max_chars:
            break

        current_chars += len(block)

        sections.append(block)

        retrieved_chunks.append(
            {
                "video_id": metadata.get(
                    "video_id",
                    "",
                ),
                "video_title": video_title,
                "video_url": video_url,
                "source_label": source_label,
                "timestamp": timestamp,
                "start": metadata.get(
                    "start",
                    "",
                ),
                "duration": metadata.get(
                    "duration",
                    "",
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    "",
                ),
                "text": text,
            }
        )

    context = "\n\n--------------------------\n\n".join(
        sections
    )

    context = trim_text(
        context,
        max_chars,
    )

    return context, retrieved_chunks


def unique_timestamps(
    chunks,
):
    """
    Remove duplicate timestamps.
    """

    seen = set()

    result = []

    for chunk in chunks:

        ts = chunk.get(
            "timestamp"
        )

        if ts and ts not in seen:

            seen.add(ts)

            result.append(ts)

    return result


def unique_sources(
    chunks,
    videos=None,
):
    """
    Remove duplicate source videos.
    """

    seen = set()

    result = []

    for chunk in chunks:

        key = chunk.get(
            "video_url",
            "",
        )

        if key in seen:
            continue

        seen.add(key)

        result.append(
            {
                "label": chunk.get(
                    "source_label",
                    "Video",
                ),
                "title": chunk.get(
                    "video_title",
                    "Video",
                ),
                "url": chunk.get(
                    "video_url",
                    "",
                ),
            }
        )

    if result:
        return result

    if videos:

        for video in videos:

            result.append(
                {
                    "label": video.get(
                        "label",
                        "Video",
                    ),
                    "title": video.get(
                        "video_title",
                        "Video",
                    ),
                    "url": video.get(
                        "video_url",
                        "",
                    ),
                }
            )

    return result