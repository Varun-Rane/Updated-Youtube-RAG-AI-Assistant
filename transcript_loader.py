from youtube_transcript_api import YouTubeTranscriptApi

from utils import format_timestamp, get_video_id, trim_text


def _snippet_value(snippet, key, default=None):
    if isinstance(snippet, dict):
        return snippet.get(key, default)
    return getattr(snippet, key, default)


def fetch_transcript(video_id, languages):
    transcript_api = YouTubeTranscriptApi()

    print("=" * 80)
    print("TRANSCRIPT DEBUG")
    print("VIDEO ID:", video_id)
    print("LANGUAGES:", languages)
    print("=" * 80)

    last_error = None

    for language in languages:
        try:
            print(f"Trying language -> {language}")

            transcript = transcript_api.fetch(
                video_id,
                languages=[language],
            )

            transcript_items = list(transcript)

            print(
                f"SUCCESS -> {language} -> "
                f"{len(transcript_items)} entries"
            )

            if transcript_items:
                return transcript_items, None

        except Exception as exc:
            print(f"FAILED -> {language}")
            print("TYPE:", type(exc).__name__)
            print("ERROR:", repr(exc))

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
            info = ydl.extract_info(
                video_url,
                download=False,
            )

        return info.get("title")

    except Exception:
        return None


def normalize_snippets(transcript_items):

    snippets = []

    for item in transcript_items:

        text = str(
            _snippet_value(
                item,
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        start = float(
            _snippet_value(
                item,
                "start",
                0,
            )
            or 0
        )

        duration = float(
            _snippet_value(
                item,
                "duration",
                0,
            )
            or 0
        )

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
    end = lines[-1]["end"]

    start_ts = lines[0]["timestamp"]
    end_ts = lines[-1]["timestamp"]

    text = " ".join(
        line["line"]
        for line in lines
    )

    return {

        "text": text,

        "metadata": {

            "video_id": video["video_id"],

            "video_title": video["video_title"],

            "video_url": video["video_url"],

            "timestamp": start_ts,

            "timestamp_range": f"{start_ts} - {end_ts}",

            "start": start,

            "duration": end - start,

            "chunk_index": chunk_index,

            "source_label": video["label"],

        },

    }


def build_transcript_chunks(
    video,
    chunk_size=500,
    chunk_overlap=100,
):

    chunks = []

    current_lines = []

    current_length = 0

    for snippet in video["snippets"]:

        line = snippet["text"]

        line_entry = {

            "line": line,

            "start": snippet["start"],

            "end": snippet["end"],

            "timestamp": snippet["timestamp"],

        }

        line_length = len(line) + 1

        if (
            current_lines
            and current_length + line_length > chunk_size
        ):

            chunks.append(

                _chunk_from_lines(

                    video,

                    current_lines,

                    len(chunks) + 1,

                )

            )

            overlap = []

            overlap_chars = 0

            for old in reversed(current_lines):

                overlap.insert(0, old)

                overlap_chars += len(old["line"])

                if overlap_chars >= chunk_overlap:
                    break

            current_lines = overlap

            current_length = sum(
                len(x["line"])
                for x in current_lines
            )

        current_lines.append(line_entry)

        current_length += line_length

    if current_lines:

        chunks.append(

            _chunk_from_lines(

                video,

                current_lines,

                len(chunks) + 1,

            )

        )

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

                f"Skipped Video {index}: "

                f"{transcript_error}"

            )

            continue

        snippets = normalize_snippets(
            transcript_items
        )

        if not snippets:

            continue

        title = None

        if settings.fetch_video_titles:

            title = fetch_video_title(
                video_url
            )

        video_title = (
            title
            or f"YouTube Video {index}"
        )

        transcript_text = " ".join(
            snippet["text"]
            for snippet in snippets
        )
        print("=" * 80)
        print("HNSW CHECK")
        print("HNSW" in transcript_text.upper())
        print("IVF" in transcript_text.upper())
        print("PRODUCT QUANTIZATION" in transcript_text.upper())
        print("=" * 80)
        print("=" * 80)
        print("IVF CHECK")

        print("IVF" in transcript_text.upper())
        print("INVERTED FILE" in transcript_text.upper())
        print("आईबीएफ" in transcript_text)
        print("=" * 80)
        
        video = {

            "label": f"Video {index}",

            "video_id": video_id,

            "video_title": video_title,

            "video_url": video_url,

            "snippets": snippets,

            "transcript": transcript_text,

            "preview": trim_text(
                transcript_text,
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

    if not videos:
        raise ValueError(
            "No transcripts found."
        )   
    return videos, all_chunks, warnings