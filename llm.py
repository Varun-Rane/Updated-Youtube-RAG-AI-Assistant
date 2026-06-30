import time
from functools import lru_cache

from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    HuggingFaceEndpointEmbeddings,
)

from utils import extract_llm_text

# Hard cap on prompt size — prevents Groq 413 token errors
MAX_PROMPT_CHARS = 10000  # ~2500 tokens, safe for Groq 6k TPM


class LLMProviderError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def _cached_hf_chat_model(api_key, repo_id, max_new_tokens, temperature):
    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=api_key,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return ChatHuggingFace(llm=endpoint)


@lru_cache(maxsize=8)
def _cached_groq_chat_model(api_key, model, max_new_tokens, temperature):
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=api_key,
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )


@lru_cache(maxsize=8)
def _cached_endpoint_embeddings(api_key, embedding_model):
    try:
        return HuggingFaceEndpointEmbeddings(
            model=embedding_model,
            huggingfacehub_api_token=api_key,
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )
    except TypeError:
        return HuggingFaceEndpointEmbeddings(
            model=embedding_model,
            huggingfacehub_api_token=api_key,
        )


@lru_cache(maxsize=8)
def _cached_local_embeddings(embedding_model):
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

def _chat_provider(settings):
    if settings.llm_provider == "auto":
        return "groq" if settings.groq_api_key else "huggingface"
    return settings.llm_provider


def _hf_credit_message():
    return (
        "HuggingFace Inference credits are depleted. Add prepaid HuggingFace credits, "
        "or set LLM_PROVIDER=groq and GROQ_API_KEY in .env."
    )


def _is_hf_credit_error(exc):
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    message = str(exc).lower()
    return (
        status_code == 402
        or "402 payment required" in message
        or "depleted your monthly included credits" in message
    )


def _is_rate_limit_error(exc):
    message = str(exc).lower()
    return (
        "429" in message
        or "rate limit" in message
        or "rate_limit_exceeded" in message
        or "tokens per minute" in message
    )


def _extract_retry_seconds(exc):
    """Try to parse 'Please try again in Xs' from the error message."""
    import re
    message = str(exc)
    match = re.search(r"try again in ([0-9.]+)s", message)
    if match:
        return min(float(match.group(1)) + 2, 60)  # add 2s buffer, cap at 60s
    return 20  # default wait


def get_chat_model(settings):
    provider = _chat_provider(settings)

    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is missing. Add it to .env or set LLM_PROVIDER=huggingface."
            )
        return _cached_groq_chat_model(
            settings.groq_api_key,
            settings.groq_model,
            settings.max_new_tokens,
            settings.temperature,
        )

    if provider != "huggingface":
        raise LLMProviderError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use auto, groq, or huggingface."
        )

    if not settings.hf_api_key:
        raise LLMProviderError(
            "HF_API_KEY is missing. Add it to .env or set LLM_PROVIDER=groq with GROQ_API_KEY."
        )

    return _cached_hf_chat_model(
        settings.hf_api_key,
        settings.llm_repo_id,
        settings.max_new_tokens,
        settings.temperature,
    )


def get_embeddings(settings):
    if settings.embedding_provider == "local":
        return _cached_local_embeddings(settings.embedding_model)

    if settings.embedding_provider != "huggingface":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            "Use local or huggingface."
        )

    if not settings.hf_api_key:
        raise ValueError("HF_API_KEY is missing for HuggingFace endpoint embeddings.")

    return _cached_endpoint_embeddings(settings.hf_api_key, settings.embedding_model)


def invoke_text(
    chat_model,
    prompt,
    max_retries=3,
    max_prompt_chars=MAX_PROMPT_CHARS,
):
    # Hard truncate oversized prompts before sending
    if len(prompt) > max_prompt_chars:
        prompt = prompt[:max_prompt_chars]

    last_exc = None
    for attempt in range(max_retries):
        try:
            response = chat_model.invoke(prompt)
            return extract_llm_text(response).strip()

        except Exception as exc:
            if _is_hf_credit_error(exc):
                raise LLMProviderError(_hf_credit_message()) from exc

            if _is_rate_limit_error(exc):
                wait = _extract_retry_seconds(exc)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise LLMProviderError(
                    f"Groq rate limit hit. Please wait ~{int(wait)}s and try again. "
                    "Consider upgrading at https://console.groq.com/settings/billing"
                ) from exc

            raise

    raise last_exc
