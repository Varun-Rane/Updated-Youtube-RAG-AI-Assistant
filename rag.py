from llm import invoke_text
from prompts import GENERAL_PROMPT, RAG_PROMPT
from retriever import (
    build_global_context,
    format_retrieved_context,
    retrieve_documents,
    unique_sources,
    unique_timestamps,
)


GLOBAL_RAG_HINTS = (
    "summarize",
    "summary",
    "what is the video about",
    "what is this video about",
    "generate notes",
    "detailed notes",
    "key moments",
    "main points",
    "chapters",
    "timeline",
    "list all questions",
    "questions discussed",
)


def is_global_rag_question(question):
    lowered = question.lower()
    return any(hint in lowered for hint in GLOBAL_RAG_HINTS)


def unavailable_bundle(message="I couldn't find this in the loaded video."):
    return {
        "mode": "RAG",
        "answer": message,
        "timestamps": [],
        "source_videos": [],
        "retrieved_chunks": [],
    }



def run_rag(question, retriever, videos, history, chat_model, settings):

    if retriever is None:
        return unavailable_bundle(
            "Load at least one YouTube transcript first."
        )

    # Always retrieve only relevant chunks
    documents = retrieve_documents(
        retriever,
        question
    )

    context, retrieved_chunks = format_retrieved_context(
        documents,
        settings.max_context_chars,
    )

    if not context.strip():
        return unavailable_bundle()

    # RAG should NOT use conversation history
    prompt = RAG_PROMPT.format(
        context=context,
        question=question,
    )

    answer = invoke_text(
        chat_model,
        prompt
    )

    return {
        "mode": "RAG",
        "answer": answer or "I couldn't find this in the loaded video.",
        "timestamps": unique_timestamps(retrieved_chunks),
        "source_videos": unique_sources(
            retrieved_chunks,
            videos=videos,
        ),
        "retrieved_chunks": retrieved_chunks,
    }

def run_general(question, history, chat_model):
    prompt = GENERAL_PROMPT.format(history=history, question=question)
    answer = invoke_text(chat_model, prompt)

    return {
        "mode": "GENERAL",
        "answer": answer,
        "timestamps": [],
        "source_videos": [],
        "retrieved_chunks": [],
    }
