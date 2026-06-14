from youtube_transcript_api import YouTubeTranscriptApi

from utils import format_timestamp, get_video_id, trim_text


def _snippet_value(snippet, key, default=None):
    if isinstance(snippet, dict):
        return snippet.get(key, default)
    return getattr(snippet, key, default)


def fetch_transcript(video_id, languages):
    transcript_api = YouTubeTranscriptApi()
    last_error = None

    for language in languages:
        try:
            transcript = transcript_api.fetch(video_id, languages=[language])
            transcript_items = list(transcript)
            if transcript_items:
                return transcript_items, None
        except Exception as exc:
            last_error = exc

    return None, last_error


def fetch_video_title(video_url):
    try:
        from yt_dlp import YoutubeDL

        options = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(video_url, download=False)
        return info.get("title")
    except Exception:
        return None


def normalize_snippets(transcript_items):
    snippets = []

    for item in transcript_items:
        text = str(_snippet_value(item, "text", "")).strip()
        if not text:
            continue

        start = float(_snippet_value(item, "start", 0) or 0)
        duration = float(_snippet_value(item, "duration", 0) or 0)
        snippets.append(
            {
                "text": text,
                "start": start,
                "duration": duration,
                "timestamp": format_timestamp(start),
                "end": start + duration,
            }
        )

    return snippets


def _chunk_from_lines(video, lines, chunk_index):
    start = lines[0]["start"]
    end = max(line["end"] for line in lines)
    timestamp = lines[0]["timestamp"]
    text = "\n".join(line["line"] for line in lines)

    return {
        "text": text,
        "metadata": {
            "video_id": video["video_id"],
            "video_title": video["video_title"],
            "video_url": video["video_url"],
            "timestamp": timestamp,
            "start": start,
            "duration": max(0, end - start),
            "chunk_index": chunk_index,
            "source_label": video["label"],
        },
    }


def build_transcript_chunks(video, chunk_size=800, chunk_overlap=200):
    chunks = []
    current_lines = []
    current_length = 0

    for snippet in video["snippets"]:
        line = f"[{snippet['timestamp']}] {snippet['text']}"
        line_entry = {
            "line": line,
            "start": snippet["start"],
            "end": snippet["end"],
            "timestamp": snippet["timestamp"],
        }
        next_length = current_length + len(line) + 1

        if current_lines and next_length > chunk_size:
            chunks.append(_chunk_from_lines(video, current_lines, len(chunks) + 1))

            overlap_lines = []
            overlap_length = 0
            for existing_line in reversed(current_lines):
                overlap_lines.insert(0, existing_line)
                overlap_length += len(existing_line["line"]) + 1
                if overlap_length >= chunk_overlap:
                    break

            current_lines = overlap_lines
            current_length = sum(len(item["line"]) + 1 for item in current_lines)

        current_lines.append(line_entry)
        current_length += len(line) + 1

    if current_lines:
        chunks.append(_chunk_from_lines(video, current_lines, len(chunks) + 1))

    return chunks


def load_transcripts(video_urls, settings):
    videos = []
    all_chunks = []
    warnings = []

    for index, video_url in enumerate(video_urls, start=1):
        video_id = get_video_id(video_url)
        transcript_items, transcript_error = fetch_transcript(
            video_id,
            settings.transcript_languages,
        )

        if not transcript_items:
            warnings.append(
                f"Skipped Video {index} ({video_id}): "
                f"{transcript_error or 'transcript unavailable'}"
            )
            continue

        snippets = normalize_snippets(transcript_items)
        if not snippets:
            warnings.append(f"Skipped Video {index} ({video_id}): transcript was empty.")
            continue

        title = None
        if settings.fetch_video_titles:
            title = fetch_video_title(video_url)

        video_title = title or f"YouTube Video {index}"
        video = {
            "label": f"Video {index}",
            "video_id": video_id,
            "video_title": video_title,
            "video_url": video_url,
            "snippets": snippets,
            "transcript": "\n".join(
                f"[{snippet['timestamp']}] {snippet['text']}" for snippet in snippets
            ),
            "preview": trim_text(
                "\n".join(
                    f"[{snippet['timestamp']}] {snippet['text']}"
                    for snippet in snippets[:12]
                ),
                1600,
            ),
        }
        videos.append(video)
        all_chunks.extend(
            build_transcript_chunks(
                video,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    if not videos or not all_chunks:
        raise ValueError("No transcripts could be loaded from the provided URLs.")

    return videos, all_chunks, warnings
