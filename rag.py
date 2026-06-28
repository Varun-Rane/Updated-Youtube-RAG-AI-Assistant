from llm import invoke_text
from prompts import RAG_PROMPT
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

    return {
        "mode": "VIDEO_QA",
        "answer": answer,
        "timestamps": unique_timestamps(retrieved_chunks),
        "source_videos": unique_sources(retrieved_chunks, videos),
        "retrieved_chunks": retrieved_chunks,
    }