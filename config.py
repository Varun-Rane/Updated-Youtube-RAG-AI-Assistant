import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent
LOCAL_ENV = dotenv_values(BASE_DIR / ".env")
ROOT_ENV = dotenv_values(BASE_DIR.parent / ".env")
PLACEHOLDER_VALUES = {
    "your_api_key_here",
    "your_groq_api_key_here",
    "your_huggingface_api_key_here",
}


def _env(name, default=""):
    candidates = [
        os.getenv(name),
        LOCAL_ENV.get(name),
        ROOT_ENV.get(name),
        default,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        value = str(candidate).strip()
        if value and value not in PLACEHOLDER_VALUES:
            return value
    return default


def _int_env(name, default):
    raw_value = _env(name, "")
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _float_env(name, default):
    raw_value = _env(name, "")
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _bool_env(name, default):
    raw_value = _env(name, "")
    if not raw_value:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    hf_api_key: str
    groq_api_key: str
    groq_model: str
    llm_repo_id: str
    embedding_provider: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int
    summary_context_chars: int
    max_history_messages: int
    max_history_chars: int
    max_new_tokens: int
    temperature: float
    transcript_languages: tuple[str, ...]
    fetch_video_titles: bool
    memory_file: Path


def load_settings():
    languages = tuple(
        language.strip()
        for language in _env("TRANSCRIPT_LANGUAGES", "en,hi").split(",")
        if language.strip()
    )

    return Settings(
        llm_provider=_env("LLM_PROVIDER", "auto").lower(),
        hf_api_key=_env("HF_API_KEY", ""),
        groq_api_key=_env("GROQ_API_KEY", ""),
        groq_model=_env("GROQ_MODEL", "llama-3.1-8b-instant"),
        llm_repo_id=_env("LLM_REPO_ID", "Qwen/Qwen2.5-7B-Instruct"),
        embedding_provider=_env("EMBEDDING_PROVIDER", "local").lower(),
        embedding_model=_env(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        chunk_size=_int_env("CHUNK_SIZE", 1000),       # smaller = more precise retrieval
        chunk_overlap=_int_env("CHUNK_OVERLAP", 200),
        top_k=_int_env("TOP_K", 6),                    # fetch more chunks for detail
        max_context_chars=_int_env("MAX_CONTEXT_CHARS", 3500),  # more context = better answers
        summary_context_chars=_int_env("SUMMARY_CONTEXT_CHARS", 8000),
        max_history_messages=_int_env("MAX_HISTORY_MESSAGES", 6),
        max_history_chars=_int_env("MAX_HISTORY_CHARS", 2000),
        max_new_tokens=_int_env("MAX_NEW_TOKENS", 900),  # longer answers for notes
        temperature=_float_env("TEMPERATURE", 0.2),      # lower = more factual
        transcript_languages=languages or ("en",),
        fetch_video_titles=_bool_env("FETCH_VIDEO_TITLES", True),
        memory_file=BASE_DIR / ".youtube_rag_memory.json",
    )
