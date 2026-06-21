from llm import invoke_text
from prompts import SUMMARY_CHUNK_PROMPT, FINAL_SUMMARY_PROMPT

# Safe limits for Groq free tier (6000 TPM)
MAX_CHUNK_CHARS = 2500   # ~625 tokens per chunk
MAX_SUMMARY_CHARS = 600  # trim each partial before merging


def split_transcript(transcript, chunk_size=MAX_CHUNK_CHARS):
    words = transcript.split()
    chunks, current, length = [], [], 0
    for word in words:
        current.append(word)
        length += len(word) + 1
        if length >= chunk_size:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_summary_context(video):
    return split_transcript(video["transcript"])


def summarize_chunk(chunk, chat_model):
    chunk = chunk[:MAX_CHUNK_CHARS]  # hard cap before sending
    prompt = SUMMARY_CHUNK_PROMPT.format(transcript=chunk)
    return invoke_text(chat_model, prompt)


def merge_batch(summaries, chat_model, batch_size=3):
    merged = []
    for i in range(0, len(summaries), batch_size):
        batch = summaries[i:i + batch_size]
        trimmed = [s[:MAX_SUMMARY_CHARS] for s in batch]  # trim before merging
        prompt = FINAL_SUMMARY_PROMPT.format(
            summaries="\n\n".join(trimmed),
            question="Merge these notes concisely, preserving chronological order.",
        )
        merged.append(invoke_text(chat_model, prompt))
    return merged


def hierarchical_merge(summaries, chat_model):
    if len(summaries) == 1:
        return summaries[0]  # skip merge if only one chunk
    while len(summaries) > 1:
        summaries = merge_batch(summaries, chat_model, batch_size=3)
    return summaries[0]


def run_summary(question, videos, chat_model, settings):
    if not videos:
        return {
            "mode": "VIDEO_SUMMARY",
            "answer": "Load at least one transcript first.",
            "timestamps": [],
            "source_videos": [],
            "retrieved_chunks": [],
        }

    partial_summaries = []
    for video in videos:
        chunks = build_summary_context(video)
        for chunk in chunks:
            partial = summarize_chunk(chunk, chat_model)
            partial_summaries.append(partial)

    answer = hierarchical_merge(partial_summaries, chat_model)

    return {
        "mode": "VIDEO_SUMMARY",
        "answer": answer,
        "timestamps": [],
        "source_videos": [
            {"title": v["video_title"], "url": v["video_url"]}
            for v in videos
        ],
        "retrieved_chunks": [],
    }