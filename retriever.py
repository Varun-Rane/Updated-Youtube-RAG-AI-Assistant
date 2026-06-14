from utils import trim_text


def retrieve_documents(retriever, question):
    if retriever is None:
        return []
    return retriever.invoke(question)


def format_retrieved_context(documents, max_chars):
    sections = []
    retrieved_chunks = []

    for document in documents:
        metadata = document.metadata
        timestamp = metadata.get("timestamp", "00:00")
        video_title = metadata.get("video_title", metadata.get("source_label", "Video"))
        video_url = metadata.get("video_url", "")
        source_label = metadata.get("source_label", "Video")
        text = document.page_content.strip()

        sections.append(
            f"[{timestamp}] {text}"
        )
        retrieved_chunks.append(
            {
                "video_id": metadata.get("video_id", ""),
                "video_title": video_title,
                "video_url": video_url,
                "source_label": source_label,
                "timestamp": timestamp,
                "start": metadata.get("start", ""),
                "duration": metadata.get("duration", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "text": text,
            }
        )

    context = "\n\n".join(sections)

    context = trim_text(context, 2500)

    return context, retrieved_chunks


def build_global_context(videos, max_chars):
    if not videos:
        return ""

    per_video_limit = max(1200, max_chars // max(1, len(videos)))
    sections = []

    for video in videos:
        sections.append(
            "\n".join(
                [
                    f"Source Video: {video['label']} - {video['video_title']}",
                    f"Video URL: {video['video_url']}",
                    "Transcript:",
                    trim_text(video["transcript"], per_video_limit),
                ]
            )
        )

    return trim_text("\n\n".join(sections), max_chars)


def unique_timestamps(chunks):
    timestamps = []
    seen = set()

    for chunk in chunks:
        timestamp = chunk.get("timestamp")
        if timestamp and timestamp not in seen:
            seen.add(timestamp)
            timestamps.append(timestamp)

    return timestamps


def unique_sources(chunks, videos=None):
    sources = []
    seen = set()

    for chunk in chunks:
        key = (chunk.get("source_label"), chunk.get("video_url"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "label": chunk.get("source_label", "Video"),
                "title": chunk.get("video_title", "Video"),
                "url": chunk.get("video_url", ""),
            }
        )

    if sources or not videos:
        return sources

    for video in videos:
        sources.append(
            {
                "label": video.get("label", "Video"),
                "title": video.get("video_title", "Video"),
                "url": video.get("video_url", ""),
            }
        )

    return sources
