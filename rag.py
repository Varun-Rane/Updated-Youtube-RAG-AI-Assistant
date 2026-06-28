from llm import invoke_text
from prompts import RAG_PROMPT, VERIFY_PROMPT
from retriever import (
    retrieve_documents,
    format_retrieved_context,
    unique_sources,
    unique_timestamps,
)


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

    context, retrieved_chunks = format_retrieved_context(
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