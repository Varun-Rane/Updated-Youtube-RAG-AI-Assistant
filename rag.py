from llm import invoke_text
from prompts import RAG_PROMPT
from retriever import (
    retrieve_documents,
    format_retrieved_context,
    unique_sources,
    unique_timestamps,
)


def unavailable_bundle(
    message="I couldn't find this in the loaded video."
):

    return {

        "mode":"VIDEO_QA",

        "answer":message,

        "timestamps":[],

        "source_videos":[],

        "retrieved_chunks":[],

    }


def run_rag(

    question,

    retriever,

    videos,

    history,

    chat_model,

    settings,

):

    if retriever is None:

        return unavailable_bundle(

            "Load at least one transcript first."

        )

    documents = retrieve_documents(

        retriever,

        question,

    )

    context,retrieved_chunks = format_retrieved_context(

        documents,

        settings.max_context_chars,

    )

    if not context.strip():

        return unavailable_bundle()

    prompt = RAG_PROMPT.format(

        context=context,

        question=question,

    )

    answer = invoke_text(

        chat_model,

        prompt,

    )

    return {

        "mode":"VIDEO_QA",

        "answer":answer,

        "timestamps":unique_timestamps(

            retrieved_chunks

        ),

        "source_videos":unique_sources(

            retrieved_chunks,

            videos,

        ),

        "retrieved_chunks":retrieved_chunks,

    }