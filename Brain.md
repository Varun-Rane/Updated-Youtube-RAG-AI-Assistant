# Brain 📚 – Project Knowledge Base

## 1️⃣ Project Overview

- **Repository**: `youtube-rag-ai-assistant`
- **Purpose**: A LangChain‑based Retrieval‑Augmented Generation (RAG) assistant that answers user questions **exclusively** from YouTube video transcripts.  It can:
  1. Load one or many videos.
  2. Build a high‑quality *summary* for each video (cached).
  3. Answer ad‑hoc questions using the cached summaries (or raw chunks).
  4. Generate study‑oriented artefacts (quiz, flashcards, cheat‑sheet, etc.).
- **Key design goals**:
  - Zero hallucination – never use knowledge outside the transcript.
  - Cache versioning to invalidate stale summaries.
  - Transparent source metadata (timestamps, video titles, URLs).

---

## 2️⃣ Code Structure & Core Modules

| File | Important Functions / Classes | Responsibility |
|------|------------------------------|----------------|
| **`app.py`** (entry point) | `main`, route handlers | CLI / FastAPI glue – parses arguments, loads transcripts, dispatches to `run_summary` or `run_video_task`. |
| **`summary.py`** | `run_summary`, `_cache_key`, `build_summary_context`, `get_representative_chunks` | Generates a concise, quality‑focused summary for a set of videos, builds prompt, calls LLM, saves to cache. |
| **`summary_cache.py`** | `summary_exists` (now rarely used), `save_summary`, `load_summary`, `get_master_summary`, `clear_summary`, `clear_all_summaries` | Simple JSON‑file cache under `.summary_cache/`.  Filename is the **cache key**. |
| **`video_task.py`** | `run_video_task` | Retrieves cached summary(ies) using the *same* cache key as `run_summary`, builds the final answer prompt (`FINAL_VIDEO_TASK_PROMPT`), returns a structured bundle for the UI. |
| **`prompts.py`** | `RAG_PROMPT`, `FINAL_VIDEO_TASK_PROMPT`, `SINGLE_PASS_SUMMARY_PROMPT`, `SUMMARY_CHUNK_PROMPT`, `FINAL_SUMMARY_PROMPT`, … | Prompt templates; the most critical for hallucination control. |
| **`llm.py`** | `invoke_text` | Thin wrapper around Claude (`chat_model`) – sends a prompt, returns the model response. |
| **`utils/…`** (if present) | Helper utilities – e.g. video download, transcript extraction. |

---

## 3️⃣ Execution Flow (Simplified)

```
User selects video(s) → app loads transcript chunks
        │
        ├─► run_summary(question, videos, chunks, model, settings)
        │       ├─> cache_key = _cache_key(videos)
        │       ├─> if cached → return saved summary
        │       └─> else → build context → LLM → save_summary(cache_key)
        │
        └─► run_video_task(question, videos, model, settings)
                ├─> cache_key = _cache_key(videos) (single) or _cache_key([v]) (each video)
                ├─> material = get_master_summary(cache_key)
                ├─> if missing → friendly error
                └─> prompt = FINAL_VIDEO_TASK_PROMPT.format(...)
                    → LLM → answer bundle
```

---

## 4️⃣ Issues & Debugging History

### ✅ Bug #1 – Cache‑Key Mismatch
- **Symptom**: After summarising a video, `run_video_task` could not find the cached file and returned *"Summary not found"*.
- **Cause**: `run_summary` saved using `_cache_key(videos)` (e.g. `lvH_zPj2o04_quality_v2.json`).  `run_video_task` looked up using the raw `video_id` (`lvH_zPj2o04.json`).
- **Fix**:
  1. Imported `_cache_key` into `video_task.py`.
  2. Replaced raw `video_id` look‑ups with the same cache‑key logic for both single‑video and multi‑video branches.
  3. Updated imports accordingly.
- **Result**: Summaries are now correctly retrieved for any number of videos.

### ✅ Bug #2 – Robust Cache Retrieval
- **Symptom**: A corrupted cache file (or a file missing `master_summary`) caused `len(material)` to raise a `TypeError` because `material` was `None`.
- **Fix**:
  - Removed the redundant `summary_exists` guard.
  - Directly called `material = get_master_summary(cache_key)` and checked `if not material:`.
  - Applied the same pattern inside the multi‑video loop.
- **Result**: Graceful error handling; the function now returns a user‑friendly message instead of crashing.

### ⚠️ Bug #4 – Hallucination / Prompt Weakness (still open)
- **Symptom**: The assistant sometimes answered with knowledge not present in the transcript (e.g. talking about *Quantum Computing* when the video never mentioned it) and used phrases like *"However…"*.
- **Root Cause**: `RAG_PROMPT` only said *“If transcript contains relevant information, answer from it”* but did **not** explicitly forbid the model from improvising when evidence is absent.
- **Proposed Fix** (to be applied in `prompts.py`):
  1. Replace the existing **STRICT RULES** block with a stricter list (see user instructions).
  2. Ensure the answer template forces an exact *"I couldn't find this in the loaded video."* response when no evidence exists.
  3. Add a guard in the RAG pipeline before the LLM call:
  ```python
  if not context.strip():
      return "I couldn't find this in the loaded video."
  ```
  4. Update the final answer format to omit *Evidence*/*Source* sections when the answer is the “not‑found” string.
- **Current Status**: Prompt changes have been documented but not yet committed to the codebase.

---

## 5️⃣ Prompt Changes (Exact Text to Apply)

Replace the **STRICT RULES** section in `RAG_PROMPT` (lines 15‑28) with:
```
STRICT RULES:

1. Answer ONLY from the retrieved transcript.
2. Never use external knowledge.
3. Never hallucinate.
4. Never infer.
5. Never guess.
6. Never complete missing information.
7. Never use prior knowledge.
8. Never explain concepts that are absent from the transcript.
9. If the retrieved transcript does NOT explicitly answer the user's question, reply with exactly:

   I couldn't find this in the loaded video.
10. Do NOT write phrases such as:
    - However…
    - Generally…
    - Usually…
    - In practice…
    - Outside the transcript…
    - Although…
    - Typically…
    - According to my knowledge…
11. If transcript is Hindi/Hinglish, translate it internally and answer in English.
12. Use ONLY information explicitly supported by the retrieved transcript.
```

Then adjust the answer template (around line 39) to:
```
Return in EXACTLY this format:

## Answer
<answer derived only from transcript>

## Evidence
- Timestamp: <timestamp from transcript>
- Transcript Fact: <fact directly supported by transcript>

## Source Video
<video title>
```

Finally, add the *early‑exit guard* in the RAG code path (e.g., in `app.py` or the function that builds `context` before calling the LLM):
```python
if not context.strip():
    return "I couldn't find this in the loaded video."
```

---

## 6️⃣ Future Work & Recommendations

1. **Apply Prompt Fix** – edit `prompts.py` as described and run the test suite to confirm hallucinations are eliminated.
2. **Add Unit Tests** – for a set of transcripts, assert that questions with no evidence return the exact *"I couldn't find this in the loaded video."* string.
3. **Cache Version Bump** – if the summary generation algorithm changes, bump `SUMMARY_CACHE_VERSION` in `summary.py` to force regeneration.
4. **Monitoring** – log cache hits/misses and any fallback “no‑evidence” responses to detect regressions.
5. **Documentation** – keep this `Brain.md` updated whenever a new bug is fixed or a major feature is added.

---

## 7️⃣ References (Files)
- `summary.py` – summary generation logic.
- `summary_cache.py` – cache helper.
- `video_task.py` – answer orchestration (now uses unified cache keys).
- `prompts.py` – prompt templates (needs the stricter RAG rules).
- `llm.py` – Claude wrapper.
- `app.py` – CLI / API entry point.

*Generated on 2026‑06‑25 by Claude Code.*