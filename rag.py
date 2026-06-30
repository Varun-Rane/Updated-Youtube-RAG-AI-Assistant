from llm import invoke_text
from prompts import RAG_PROMPT, VERIFY_PROMPT
from retriever import (
    retrieve_documents,
    unique_sources,
    unique_timestamps,
)


def _chunk_position_key(doc):
    """Best-effort ordering key for a document's position within its video."""
    meta = doc.metadata
    start = meta.get("start", meta.get("start_snippet"))
    if start is None:
        start = meta.get("chunk_index", 0)
    return (meta.get("video_id", ""), start)


def _are_adjacent(doc_a, doc_b):
    """True if doc_a and doc_b are consecutive chunks from the same video.

    Adjacency is judged by chunk_index (preferred, since it's a stable
    integer position) and falls back to comparing start timestamps if
    chunk_index is missing.
    """
    meta_a, meta_b = doc_a.metadata, doc_b.metadata
    if meta_a.get("video_id", "") != meta_b.get("video_id", ""):
        return False

    idx_a = meta_a.get("chunk_index")
    idx_b = meta_b.get("chunk_index")
    if idx_a is not None and idx_b is not None:
        return abs(idx_a - idx_b) == 1

    # Fallback: treat as adjacent if their timestamp ranges touch closely.
    start_a = meta_a.get("start", meta_a.get("start_snippet"))
    end_a = meta_a.get("end_snippet", start_a)
    start_b = meta_b.get("start", meta_b.get("start_snippet"))
    if start_a is None or start_b is None or end_a is None:
        return False
    return abs(start_b - end_a) < 5  # within 5s = effectively adjacent


def _group_adjacent_runs(documents):
    """Group consecutive *reranker-ordered* documents into runs of adjacent chunks.

    Does NOT reorder documents — only merges a chunk into the previous run if
    it is immediately adjacent to it in the source video. This preserves the
    reranker's relevance ordering (e.g. [20, 21, 19] stays as two runs:
    [20, 21] then [19]) while still giving the LLM a clean, merged view of
    consecutive chunks instead of presenting them as separate blocks.

    Returns a list of runs, where each run is a list of one or more documents
    in their original (reranker) relative order.
    """
    runs = []
    for doc in documents:
        if runs and _are_adjacent(runs[-1][-1], doc):
            runs[-1].append(doc)
        else:
            runs.append([doc])

    # Within a run, sort by position so e.g. [21, 20] (out-of-order adjacent
    # hits) read in natural chunk order rather than reranker order.
    for run in runs:
        run.sort(key=_chunk_position_key)

    return runs


def _build_structured_context(documents, max_context_chars):
    """Format retrieved chunks into a structured context block.

    Preserves the reranker's relevance ordering across distinct topics —
    documents are NOT globally re-sorted by timestamp. Only chunks that are
    immediately adjacent in the source video are merged into a single block
    (in their natural order), since merging adjacent text genuinely improves
    readability without disturbing relevance ranking.

    Example: reranker order [chunk20, chunk21, chunk19, chunk55] becomes:

        Run 1 (chunk20, chunk21 — adjacent, merged in chunk order)
        Run 2 (chunk19 — not adjacent to chunk21, separate run)
        Run 3 (chunk55 — not adjacent to chunk19, separate run)

        Video: ...
        Timestamp: ...
        <chunk20 + chunk21 merged>

        --------------------

        Video: ...
        Timestamp: ...
        <chunk19>

        --------------------

        Video: ...
        Timestamp: ...
        <chunk55>

    Returns (context_string, retrieved_chunks) — retrieved_chunks is the
    flattened list of documents actually included, in the order they appear
    in the context, so downstream timestamp/source extraction stays
    consistent with what the LLM actually saw.
    """
    runs = _group_adjacent_runs(documents)

    blocks = []
    included = []
    used_chars = 0
    separator = "\n\n" + ("-" * 20) + "\n\n"

    for run in runs:
        first_meta = run[0].metadata
        video_title = first_meta.get("video_title", first_meta.get("source_label", "Unknown Video"))

        if len(run) == 1:
            timestamp = first_meta.get("timestamp_range", first_meta.get("timestamp", ""))
        else:
            start_ts = first_meta.get("timestamp_range", first_meta.get("timestamp", "")).split(" - ")[0]
            last_meta = run[-1].metadata
            end_ts = last_meta.get("timestamp_range", last_meta.get("timestamp", ""))
            end_ts = end_ts.split(" - ")[-1] if end_ts else ""
            timestamp = f"{start_ts} - {end_ts}" if start_ts and end_ts else start_ts or end_ts

        merged_text = " ".join(doc.page_content.strip() for doc in run)
        block = f"Video: {video_title}\nTimestamp: {timestamp}\n\n{merged_text}"

        added_len = len(block) + (len(separator) if blocks else 0)
        if used_chars + added_len > max_context_chars and blocks:
            # Stop once the budget is hit, but always include at least one block.
            break

        blocks.append(block)
        included.extend(run)
        used_chars += added_len

    context = separator.join(blocks)
    return context, included


def unavailable_bundle(message="I couldn't find this in the loaded video."):
    return {
        "mode": "VIDEO_QA",
        "answer": message,
        "timestamps": [],
        "source_videos": [],
        "retrieved_chunks": [],
    }


def _parse_verify_response(response: str) -> tuple:
    """Parse SUPPORTED / TOTAL / VERDICT lines from the verifier LLM response.

    Returns (supported: int, total: int, verdict: str).
    Falls back to (0, 1, 'FAIL') if parsing fails so the caller always gets
    a safe result rather than an exception.
    """
    supported, total, verdict = 0, 1, "FAIL"
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("SUPPORTED:"):
            try:
                supported = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("TOTAL:"):
            try:
                total = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().upper()
    return supported, total, verdict


def verify_answer(answer, question, context, chat_model, settings) -> bool:
    """Run a second LLM call to verify every claim in `answer` is in `context`.

    Returns True (keep the answer) or False (reject it).

    Skipped when settings.verify_answer is False so there is zero latency
    cost in production until the feature is explicitly enabled.
    """
    if not settings.verify_answer:
        return True

    prompt   = VERIFY_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    response = invoke_text(chat_model, prompt)

    supported, total, verdict = _parse_verify_response(response)

    if settings.debug_reranker:
        print("=" * 80)
        print("ANSWER VERIFICATION")
        print(f"  Supported claims : {supported} / {total}")
        print(f"  Verdict          : {verdict}")
        print("=" * 80)

    # Accept if the verifier says PASS and at least min_support claims are backed.
    return (
        verdict == "PASS"
        and supported >= settings.verification_min_support
    )


def confidence_gate(documents, settings):
    """Decide whether reranked documents are confident enough to send to the LLM.

    Returns
    -------
    passed : bool   — True if all three thresholds are met.
    metrics : dict  — top_score, avg_score, count for debug logging.

    Thresholds (all from settings / .env):
        rerank_top_score    — minimum score of the best document
        rerank_avg_score    — minimum average score across all documents
        relevant_count_min  — minimum number of documents that must be present

    Documents that lack a rerank_score (e.g. from the timestamp-routing path)
    are excluded from scoring; if none have a score, the gate passes unconditionally
    so timestamp-routed queries are never incorrectly rejected.
    """
    # If no document carries a rerank_score, reranking was not used (e.g. the
    # query was handled by timestamp routing).  Pass unconditionally so those
    # queries are never silently rejected by a gate that has no signal to act on.
    scores = [getattr(doc, "rerank_score", None) for doc in documents]
    scores = [s for s in scores if s is not None]

    if not scores:
        return True, {"top_score": None, "avg_score": None, "count": len(documents)}

    top_score = scores[0]                    # list is sorted best-first by reranker
    avg_score = sum(scores) / len(scores)
    count     = len(scores)

    passed = (
        top_score >= settings.rerank_top_score
        and avg_score >= settings.rerank_avg_score
        and count    >= settings.relevant_count_min
    )

    return passed, {"top_score": top_score, "avg_score": avg_score, "count": count}


def run_rag(question, retriever, videos, history, chat_model, settings):
    if retriever is None:
        return unavailable_bundle("Load at least one transcript first.")

    documents = retrieve_documents(
        retriever,
        question,
        settings,
    )

    passed, gate = confidence_gate(documents, settings)

    if settings.debug_reranker:
        _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A (timestamp route)"
        print("=" * 80)
        print("CONFIDENCE GATE")
        print(f"  Top Score     : {_fmt(gate['top_score'])}")
        print(f"  Average Score : {_fmt(gate['avg_score'])}")
        print(f"  Relevant Docs : {gate['count']}")
        print(f"  Decision      : {'PASS' if passed else 'FAIL'}")
        print("=" * 80)

    if not passed:
        # Preserve retrieved chunks so benchmarking and Phase 2.5 answer
        # verification can inspect why the gate rejected this query.
        return {
            "mode":             "VIDEO_QA",
            "answer":           settings.not_found_message,
            "timestamps":       unique_timestamps(
                                    [{"timestamp": d.metadata.get("timestamp_range",
                                                    d.metadata.get("timestamp", ""))}
                                     for d in documents]
                                ),
            "source_videos":    unique_sources(
                                    [{"video_url":    d.metadata.get("video_url", ""),
                                      "video_title":  d.metadata.get("video_title", ""),
                                      "source_label": d.metadata.get("source_label", "")}
                                     for d in documents],
                                    videos,
                                ),
            "retrieved_chunks": documents,
        }

    context, retrieved_chunks = _build_structured_context(
        documents, settings.max_context_chars
    )

    # If we have no context at all, return the unavailable bundle early
    if not context.strip():
        return unavailable_bundle()

    prompt = RAG_PROMPT.format(
        history=history,
        context=context,
        question=question,
    )
    print("=" * 80)
    print("RAG CONTEXT")
    print(context[:3000])
    print("=" * 80)

    answer = invoke_text(chat_model, prompt)

    # ------------------------------------------------------------------ 5
    # Phase 2.5 — Answer verification.
    # A second LLM call checks every claim in the answer against the context.
    # Disabled by default (VERIFY_ANSWER=false) to avoid extra latency.
    # ------------------------------------------------------------------ 5
    if not verify_answer(answer, question, context, chat_model, settings):
        return unavailable_bundle(
            "I found relevant transcript sections but could not verify a "
            "fully supported answer. Please try rephrasing your question."
        )

    return {
        "mode": "VIDEO_QA",
        "answer": answer,
        "timestamps": unique_timestamps(retrieved_chunks),
        "source_videos": unique_sources(retrieved_chunks, videos),
        "retrieved_chunks": retrieved_chunks,
    }