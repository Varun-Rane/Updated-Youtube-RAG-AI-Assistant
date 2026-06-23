import hashlib
import time

from llm import invoke_text
from summary_cache import get_master_summary, save_summary, summary_exists

MAX_REPRESENTATIVE_CHUNKS = 20
MAX_SUMMARY_CONTEXT_CHARS = 8000
MAX_SUMMARY_PROMPT_CHARS = 10000
MAX_QUESTION_CHARS = 500
SUMMARY_CACHE_VERSION = "quality_v2"

SINGLE_PASS_SUMMARY_PROMPT = """
You are creating accurate notes from representative excerpts of a YouTube
transcript. The excerpts are ordered chronologically across the video.

STRICT RULES:
1. Use ONLY information explicitly present in the transcript excerpts.
2. Write in English; accurately translate Hindi or regional-language content.
3. Do not add outside facts, examples, definitions, conclusions, or context.
4. Do not invent topics, sections, timestamps, Q&A, recaps, or future directions.
5. If a topic is not discussed in the excerpts, do not mention it.
6. Preserve the speaker's lecture flow and consolidate only genuine repetition.
7. Use timestamp ranges exactly as supplied.
8. Do not mention exams, interviews, students, preparation, or learning goals
   unless the speaker explicitly discusses them.

User request:
{question}

Representative transcript excerpts:
{context}

Use only the following headings. Omit a heading when the transcript has no
supporting content for it. Do not create any additional headings.

# Executive Summary
<what the speaker actually teaches>

# Main Concepts
<the main concepts, grounded in the transcript>

# Important Technical Details
<specific technical details explicitly discussed>

# Algorithms / Processes
<only the steps, algorithms, or workflows actually taught>

# Key Takeaways
<the speaker's most important points, without generic advice>
""".strip()


def _chunk_text(chunk):
    if isinstance(chunk, dict):
        return str(chunk.get("text", "")).strip()

    return str(getattr(chunk, "page_content", "")).strip()


def _chunk_metadata(chunk):
    if isinstance(chunk, dict):
        metadata = chunk.get("metadata", {})
    else:
        metadata = getattr(chunk, "metadata", {})

    return metadata if isinstance(metadata, dict) else {}


def _chunk_order(chunk):
    metadata = _chunk_metadata(chunk)

    try:
        start = float(metadata.get("start", 0))
    except (TypeError, ValueError):
        start = 0

    try:
        chunk_index = int(metadata.get("chunk_index", 0))
    except (TypeError, ValueError):
        chunk_index = 0

    return start, chunk_index


def _video_key(chunk):
    metadata = _chunk_metadata(chunk)
    return (
        metadata.get("video_id")
        or metadata.get("video_url")
        or metadata.get("source_label")
        or metadata.get("video_title")
        or "video"
    )


def _evenly_spaced(items, count):
    if count <= 0:
        return []

    if len(items) <= count:
        return list(items)

    if count == 1:
        return [items[len(items) // 2]]

    indices = [
        round(position * (len(items) - 1) / (count - 1))
        for position in range(count)
    ]
    return [items[index] for index in indices]


def _representative_quotas(group_sizes, limit):
    quotas = [0] * len(group_sizes)
    if not group_sizes or limit <= 0:
        return quotas

    # Give every video coverage when the prompt budget allows it.
    for index in range(min(len(group_sizes), limit)):
        quotas[index] = 1

    remaining = limit - sum(quotas)

    # Add samples to the most under-represented video until the limit is met.
    while remaining > 0:
        candidates = [
            index
            for index, size in enumerate(group_sizes)
            if quotas[index] < size
        ]
        if not candidates:
            break

        selected = max(
            candidates,
            key=lambda index: group_sizes[index] / (quotas[index] + 1),
        )
        quotas[selected] += 1
        remaining -= 1

    return quotas


def get_representative_chunks(
    transcript_chunks,
    max_chunks=MAX_REPRESENTATIVE_CHUNKS,
):
    """Sample chronological chunks across every loaded video."""
    usable_chunks = [
        chunk for chunk in transcript_chunks if _chunk_text(chunk)
    ]
    if len(usable_chunks) <= max_chunks:
        return usable_chunks

    grouped = {}
    for chunk in usable_chunks:
        grouped.setdefault(_video_key(chunk), []).append(chunk)

    groups = [
        sorted(chunks, key=_chunk_order)
        for chunks in grouped.values()
    ]
    quotas = _representative_quotas(
        [len(group) for group in groups],
        max_chunks,
    )

    selected = []
    for group, quota in zip(groups, quotas):
        selected.extend(_evenly_spaced(group, quota))

    return selected


def _trim_to_budget(text, max_chars):
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or text[:max_chars].strip()


def _text_budgets(texts, total_budget):
    """Share the text budget fairly without wasting space on short chunks."""
    if not texts or total_budget <= 0:
        return [0] * len(texts)

    low = 0
    high = max(len(text) for text in texts)

    while low < high:
        candidate = (low + high + 1) // 2
        required = sum(min(len(text), candidate) for text in texts)
        if required <= total_budget:
            low = candidate
        else:
            high = candidate - 1

    budgets = [min(len(text), low) for text in texts]
    remaining = total_budget - sum(budgets)

    for index, text in enumerate(texts):
        if remaining <= 0:
            break
        if budgets[index] < len(text):
            budgets[index] += 1
            remaining -= 1

    return budgets


def build_summary_context(
    representative_chunks,
    max_chars=MAX_SUMMARY_CONTEXT_CHARS,
):
    """Format sampled chunks while keeping the final prompt safely bounded."""
    if not representative_chunks:
        return ""

    headers = []
    texts = []

    previous_video = None

    for index, chunk in enumerate(representative_chunks, start=1):
        metadata = _chunk_metadata(chunk)
        video_title = _trim_to_budget(
            str(metadata.get("video_title", "Video")),
            120,
        )
        video_key = _video_key(chunk)
        timestamp = metadata.get(
            "timestamp_range",
            metadata.get("timestamp", "00:00"),
        )

        video_header = ""
        if video_key != previous_video:
            video_header = f"Video: {video_title}\n"
            previous_video = video_key

        headers.append(
            f"{video_header}Excerpt {index} | Timestamp: {timestamp}\n"
        )
        texts.append(_chunk_text(chunk))

    separator = "\n\n---\n\n"
    overhead = sum(len(header) for header in headers)
    overhead += len(separator) * (len(headers) - 1)
    available_for_text = max(0, max_chars - overhead)
    budgets = _text_budgets(texts, available_for_text)

    sections = [
        header + _trim_to_budget(text, budget)
        for header, text, budget in zip(headers, texts, budgets)
    ]
    return separator.join(sections)[:max_chars]


def _cache_key(videos):
    identities = [
        str(
            video.get("video_id")
            or video.get("video_url")
            or video.get("video_title")
            or index
        )
        for index, video in enumerate(videos)
    ]

    if len(identities) == 1:
        return f"{identities[0]}_{SUMMARY_CACHE_VERSION}"

    cache_identity = "|".join(identities + [SUMMARY_CACHE_VERSION])
    digest = hashlib.sha256(cache_identity.encode("utf-8")).hexdigest()
    return f"multi_{digest[:20]}_{SUMMARY_CACHE_VERSION}"


def _source_videos(videos):
    return [
        {
            "label": video.get("label", "Video"),
            "title": video.get("video_title", "Video"),
            "url": video.get("video_url", ""),
        }
        for video in videos
    ]


def _display_chunks(chunks):
    result = []

    for chunk in chunks:
        metadata = _chunk_metadata(chunk)
        result.append(
            {
                "video_id": metadata.get("video_id", ""),
                "video_title": metadata.get("video_title", "Video"),
                "video_url": metadata.get("video_url", ""),
                "source_label": metadata.get("source_label", "Video"),
                "timestamp": metadata.get(
                    "timestamp_range",
                    metadata.get("timestamp", "00:00"),
                ),
                "start": metadata.get("start", ""),
                "duration": metadata.get("duration", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "text": _chunk_text(chunk),
            }
        )

    return result


def _timestamps(chunks):
    result = []

    for chunk in chunks:
        metadata = _chunk_metadata(chunk)
        timestamp = metadata.get(
            "timestamp_range",
            metadata.get("timestamp", ""),
        )
        if timestamp and timestamp not in result:
            result.append(timestamp)

    return result


def _answer_bundle(answer, videos, representative_chunks):
    return {
        "mode": "VIDEO_SUMMARY",
        "answer": answer,
        "timestamps": _timestamps(representative_chunks),
        "source_videos": _source_videos(videos),
        "retrieved_chunks": _display_chunks(representative_chunks),
    }


def run_summary(
    question,
    videos,
    transcript_chunks,
    chat_model,
    settings,
):
    print(
        f"QA CHUNKS AVAILABLE: "
        f"{len(transcript_chunks)}"
    )

    if not videos:
        return _answer_bundle(
            "Load at least one transcript first.",
            [],
            [],
        )

    representative_chunks = get_representative_chunks(transcript_chunks)
    cache_key = _cache_key(videos)
    cached_summary = (
        get_master_summary(cache_key)
        if summary_exists(cache_key)
        else None
    )

    if cached_summary:
        print(f"SUMMARY CACHE HIT -> {cache_key}")
        return _answer_bundle(
            cached_summary,
            videos,
            representative_chunks,
        )

    if not representative_chunks:
        return _answer_bundle(
            "I couldn't find transcript chunks for the loaded video.",
            videos,
            [],
        )

    configured_context_chars = int(
        getattr(
            settings,
            "summary_context_chars",
            MAX_SUMMARY_CONTEXT_CHARS,
        )
    )
    context_limit = min(
        MAX_SUMMARY_CONTEXT_CHARS,
        max(1, configured_context_chars),
    )
    context = build_summary_context(
        representative_chunks,
        max_chars=context_limit,
    )
    prompt = SINGLE_PASS_SUMMARY_PROMPT.format(
        question=_trim_to_budget(question.strip(), MAX_QUESTION_CHARS),
        context=context,
    )

    print(
        "SUMMARY CACHE MISS -> "
        f"{len(representative_chunks)} representative chunks, "
        f"{len(context)} context chars, "
        f"{len(prompt)} prompt chars"
    )

    start = time.perf_counter()
    answer = invoke_text(
        chat_model,
        prompt,
        max_prompt_chars=MAX_SUMMARY_PROMPT_CHARS,
    )
    print("=" * 80)
    print("SUMMARY LENGTH")
    print(len(answer))
    print("=" * 80)

    title = (
        videos[0].get("video_title", "Video")
        if len(videos) == 1
        else f"Summary of {len(videos)} videos"
    )
    save_summary(cache_key, answer, title)

    print(f"SUMMARY SAVED -> {cache_key}")
    print(f"TOTAL SUMMARY TIME: {time.perf_counter() - start:.2f}s")

    return _answer_bundle(
        answer,
        videos,
        representative_chunks,
    )
