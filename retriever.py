from utils import trim_text


def retrieve_documents(
    retriever,
    question,
):
    """
    Retrieve relevant transcript chunks.
    """

    if retriever is None:
        return []

    question = question.strip()

    return retriever.invoke(question)


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

        key = (
            chunk.get(
                "video_url",
                "",
            )
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