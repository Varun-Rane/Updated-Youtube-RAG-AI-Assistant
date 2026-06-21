"""
overview.py

VIDEO_OVERVIEW pipeline.
"""

from llm import invoke_text
from prompts import VIDEO_OVERVIEW_PROMPT


def get_representative_chunks(docs, n=15):
    if not docs:
        return []

    docs = sorted(
        docs,
        key=lambda d: d.metadata.get(
            "chunk_index",
            0,
        ),
    )

    if len(docs) <= n:
        return docs

    step = len(docs) / n

    selected = []

    for i in range(n):
        idx = int(i * step)

        if idx >= len(docs):
            idx = len(docs) - 1

        selected.append(docs[idx])

    return selected


def build_context(docs):
    contexts = []

    for doc in docs:

        metadata = doc.metadata

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

        text = doc.page_content.strip()

        contexts.append(
            f"""
Video: {video_title}

Timestamp: {timestamp}

Transcript:
{text}
"""
        )

    return "\n\n----------------------\n\n".join(
        contexts
    )


def run_video_overview(
    question,
    vector_store,
    videos,
    chat_model,
    settings,
):
    try:

        docs = list(
            vector_store.docstore._dict.values()
        )

    except Exception:

        return {
            "mode": "VIDEO_OVERVIEW",
            "answer": (
                "I couldn't access transcript data."
            ),
            "timestamps": [],
            "source_videos": [],
            "retrieved_chunks": [],
        }

    if not docs:

        return {
            "mode": "VIDEO_OVERVIEW",
            "answer": (
                "I couldn't find transcript data "
                "for the loaded videos."
            ),
            "timestamps": [],
            "source_videos": [],
            "retrieved_chunks": [],
        }

    representative_docs = (
        get_representative_chunks(
            docs,
            n=15,
        )
    )

    context = build_context(
        representative_docs
    )

    prompt = VIDEO_OVERVIEW_PROMPT.format(
        context=context,
        question=question,
    )

    answer = invoke_text(
        chat_model,
        prompt,
    )

    timestamps = []

    for doc in representative_docs:

        ts = doc.metadata.get(
            "timestamp_range",
            doc.metadata.get(
                "timestamp",
                "",
            ),
        )

        if ts and ts not in timestamps:
            timestamps.append(ts)

    source_videos = []

    seen_urls = set()

    for video in videos:

        url = video.get(
            "video_url",
            "",
        )

        if url in seen_urls:
            continue

        seen_urls.add(url)

        source_videos.append(
            {
                "label": video.get(
                    "label",
                    "Video",
                ),
                "title": video.get(
                    "video_title",
                    "Video",
                ),
                "url": url,
            }
        )

    return {
        "mode": "VIDEO_OVERVIEW",
        "answer": answer,
        "timestamps": timestamps,
        "source_videos": source_videos,
        "retrieved_chunks": [],
    }