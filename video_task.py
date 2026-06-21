from llm import invoke_text
from prompts import (
    FINAL_VIDEO_TASK_PROMPT,
    VIDEO_TASK_CHUNK_PROMPT,
    VIDEO_TASK_MERGE_PROMPT,
)
from summary import split_transcript
from utils import format_timestamp, trim_text

# Safe chunk size for Groq free tier
TASK_CHUNK_CHARS = 2500
MAX_MATERIAL_CHARS = 5000


def _unavailable_bundle():
    return {
        "mode": "VIDEO_TASK",
        "answer": "Load at least one transcript first.",
        "timestamps": [],
        "source_videos": [],
        "retrieved_chunks": [],
    }


def _video_transcript_chunks(video, chunk_size=TASK_CHUNK_CHARS):
    snippets = video.get("snippets") or []

    if not snippets:
        return [
            {"text": text, "timestamp_range": "Not available"}
            for text in split_transcript(video.get("transcript", ""), chunk_size)
            if text.strip()
        ]

    chunks = []
    current_lines = []
    current_chars = 0

    def finish_chunk():
        if not current_lines:
            return
        first = current_lines[0]
        last = current_lines[-1]
        chunks.append({
            "text": "\n".join(line["text"] for line in current_lines),
            "timestamp_range": f"{first['timestamp']} - {format_timestamp(last['end'])}",
        })

    for snippet in snippets:
        text = str(snippet.get("text", "")).strip()
        if not text:
            continue
        timestamp = snippet.get("timestamp") or format_timestamp(snippet.get("start"))
        line = f"[{timestamp}] {text}"

        if current_lines and current_chars + len(line) + 1 > chunk_size:
            finish_chunk()
            current_lines = []
            current_chars = 0

        current_lines.append({
            "text": line,
            "timestamp": timestamp,
            "end": snippet.get("end", snippet.get("start", 0)),
        })
        current_chars += len(line) + 1

    finish_chunk()
    return chunks


def _material_batches(items, max_chars):
    batches, current, current_chars = [], [], 0
    for item in items:
        bounded = trim_text(item, max_chars)
        added = len(bounded) + (2 if current else 0)
        if current and current_chars + added > max_chars:
            batches.append("\n\n".join(current))
            current, current_chars = [], 0
        current.append(bounded)
        current_chars += len(bounded) + (2 if len(current) > 1 else 0)
    if current:
        batches.append("\n\n".join(current))
    return batches


def _merge_material(items, question, chat_model, max_chars):
    material = "\n\n".join(items)
    for _ in range(3):
        if len(material) <= max_chars:
            return material
        merged_items = []
        for batch in _material_batches(items, max_chars):
            prompt = VIDEO_TASK_MERGE_PROMPT.format(question=question, material=batch)
            merged_items.append(invoke_text(chat_model, prompt))
        items = merged_items
        material = "\n\n".join(items)
    return trim_text(material, max_chars)


def run_video_task(question, videos, chat_model, settings):
    if not videos:
        return _unavailable_bundle()

    extracted_material = []
    timestamp_ranges = []

    for video in videos:
        title = video.get("video_title", "YouTube Video")
        for chunk in _video_transcript_chunks(video):
            transcript_chunk = chunk["text"]
            timestamp_range = chunk["timestamp_range"]
            prompt = VIDEO_TASK_CHUNK_PROMPT.format(
                question=question,
                video_title=title,
                timestamp_range=timestamp_range,
                transcript=transcript_chunk,
            )
            extraction = invoke_text(chat_model, prompt)
            extracted_material.append(
                "\n".join([
                    f"Video: {title}",
                    f"Timestamp Range: {timestamp_range}",
                    "",
                    "Extracted:",
                    extraction,
                ])
            )
            timestamp_ranges.append(timestamp_range)

    if not extracted_material:
        return _unavailable_bundle()

    material = _merge_material(
        extracted_material, question, chat_model, MAX_MATERIAL_CHARS
    )
    material = trim_text(material, MAX_MATERIAL_CHARS)

    prompt = FINAL_VIDEO_TASK_PROMPT.format(question=question, material=material)
    answer = invoke_text(chat_model, prompt)

    return {
        "mode": "VIDEO_TASK",
        "answer": answer,
        "timestamps": list(dict.fromkeys(timestamp_ranges)),
        "source_videos": [
            {
                "label": v.get("label", "Video"),
                "title": v.get("video_title", "YouTube Video"),
                "url": v.get("video_url", ""),
            }
            for v in videos
        ],
        "retrieved_chunks": [],
    }