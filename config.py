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
    # Hybrid retrieval configuration
    bm25_top_k: int
    dense_top_k: int
    rrf_k: int
    final_top_k: int
    # Retrieval / reranking settings
    rerank_model: str
    rerank_top_n: int
    rerank_batch_size: int       # predict() batch size; increase for GPU inference
    hybrid_fetch_k: int          # candidates sent from HybridRetriever to CrossEncoder
    debug_reranker: bool         # print pre/post rerank rankings on every query
    dense_skip_threshold: float
    rerank_top_score: float
    rerank_avg_score: float
    relevant_count_min: int
    dedup_similarity: float
    not_found_message: str
    # Phase 1 acceptance thresholds
    phase1_hit_target: float
    phase1_recall_target: float
    phase1_mrr_target: float
    phase1_neg_precision_target: float


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
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        chunk_size=_int_env("CHUNK_SIZE", 1000),       # smaller = more precise retrieval
        chunk_overlap=_int_env("CHUNK_OVERLAP", 200),
        top_k=_int_env("TOP_K", 10),                    # fetch more chunks for detail
        max_context_chars=_int_env("MAX_CONTEXT_CHARS", 3500),  # more context = better answers
        summary_context_chars=_int_env("SUMMARY_CONTEXT_CHARS", 8000),
        max_history_messages=_int_env("MAX_HISTORY_MESSAGES", 6),
        max_history_chars=_int_env("MAX_HISTORY_CHARS", 2000),
        max_new_tokens=_int_env("MAX_NEW_TOKENS", 900),  # longer answers for notes
        temperature=_float_env("TEMPERATURE", 0.2),      # lower = more factual
        transcript_languages=languages or ("en",),
        fetch_video_titles=_bool_env("FETCH_VIDEO_TITLES", True),
        # Retrieval / reranking settings (with sensible defaults)
        rerank_model=_env("RERANK_MODEL", "BAAI/bge-reranker-base"),
        rerank_top_n=_int_env("RERANK_TOP_N", 5),
        rerank_batch_size=_int_env("RERANK_BATCH_SIZE", 16),
        hybrid_fetch_k=_int_env("HYBRID_FETCH_K", 20),
        debug_reranker=_bool_env("DEBUG_RERANKER", False),
        dense_skip_threshold=_float_env("DENSE_SKIP_THRESHOLD", 0.88),
        rerank_top_score=_float_env("RERANK_TOP_SCORE", 0.35),
        rerank_avg_score=_float_env("RERANK_AVG_SCORE", 0.15),
        relevant_count_min=_int_env("RELEVANT_COUNT_MIN", 2),
        dedup_similarity=_float_env("DEDUP_SIMILARITY", 0.85),
        not_found_message=_env("NOT_FOUND_MESSAGE", "I couldn't find this in the loaded video."),
        # Phase 1 acceptance thresholds
        phase1_hit_target=_float_env("PHASE1_HIT_TARGET", 0.90),
        phase1_recall_target=_float_env("PHASE1_RECALL_TARGET", 0.50),
        phase1_mrr_target=_float_env("PHASE1_MRR_TARGET", 0.50),
        phase1_neg_precision_target=_float_env("PHASE1_NEG_PRECISION_TARGET", 0.95),
        bm25_top_k=_int_env("BM25_TOP_K", 50),
        dense_top_k=_int_env("DENSE_TOP_K", 20),
        rrf_k=_int_env("RRF_K", 60),
        final_top_k=_int_env("FINAL_TOP_K", 8),
        memory_file=BASE_DIR / ".youtube_rag_memory.json",
    )