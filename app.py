import streamlit as st
from config import load_settings
from llm import get_chat_model, get_embeddings
from memory import (
    answer_memory_question,
    clear_memory,
    history_text,
    init_session_state,
    is_memory_query,
    save_messages,
)
from rag import run_rag
from router import classify_query
from transcript_loader import load_transcripts
from ui import (
    configure_page,
    inject_styles,
    render_answer_bundle,
    render_chat_history,
    render_loader_panel,
    render_loaded_videos,
    render_quick_prompts,
)
from utils import parse_video_urls
from vector_store import build_vector_store


def load_video_index(raw_urls, settings):
    parsed_urls = parse_video_urls(raw_urls)

    if not parsed_urls:
        st.warning("Enter at least one YouTube URL first.")
        return

    if not settings.hf_api_key:
        st.error("HF_API_KEY is missing. Add it to youtube-rag-ai-assistant/.env.")
        return

    with st.spinner("Fetching transcripts and building the FAISS index..."):
        videos, chunks, warnings = load_transcripts(parsed_urls, settings)
        embeddings = get_embeddings(settings)
        vectorstore, retriever = build_vector_store(chunks, embeddings, settings)

    st.session_state.videos = videos
    st.session_state.transcript_chunks = chunks
    st.session_state.vectorstore = vectorstore
    st.session_state.retriever = retriever
    st.session_state.video_loaded = True

    st.success(f"Loaded {len(videos)} video(s) and indexed {len(chunks)} chunks.")
    for warning in warnings:
        st.warning(warning)


def answer_question(question, settings, history_before):

    if not settings.hf_api_key and not settings.groq_api_key:
        return {
            "mode": "ERROR",
            "answer": "No LLM API key found.",
            "timestamps": [],
            "source_videos": [],
            "retrieved_chunks": [],
        }

    chat_model = get_chat_model(settings)

    history = history_text(
        history_before,
        max_messages=settings.max_history_messages,
        max_chars=settings.max_history_chars,
    )

    route = classify_query(
        question=question,
        has_loaded_videos=bool(
            st.session_state.get("retriever")
        ),
    )

    print(f"ROUTE -> {route}")

    # -------------------------
    # MEMORY
    # -------------------------

    if route == "MEMORY":

        return {

            "mode": "MEMORY",

            "answer": answer_memory_question(

                question,

                history_before,

                chat_model,

                settings,

            ),

            "timestamps": [],

            "source_videos": [],

            "retrieved_chunks": [],

        }

    # -------------------------
    # VIDEO SUMMARY
    # -------------------------

    if route == "VIDEO_SUMMARY":

        from summary import run_summary

        return run_summary(

            question=question,

            videos=st.session_state.get(

                "videos",

                [],

            ),

            chat_model=chat_model,

            settings=settings,

        )

    # -------------------------
    # GENERAL
    # -------------------------

    if route == "GENERAL":

        from general import run_general

        return run_general(

            question=question,

            history=history,

            chat_model=chat_model,

        )

    # -------------------------
    # VIDEO QA
    # -------------------------

    return run_rag(

        question=question,

        retriever=st.session_state.get(

            "retriever"

        ),

        videos=st.session_state.get(

            "videos",

            [],

        ),

        history=history,

        chat_model=chat_model,

        settings=settings,

    )

def handle_question(question, settings):
    history_before = list(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_bundle = answer_question(question, settings, history_before)
            render_answer_bundle(answer_bundle)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer_bundle["answer"]}
    )
    save_messages(settings.memory_file, st.session_state.messages)


def main():
    settings = load_settings()
    configure_page()
    inject_styles()
    init_session_state(settings)

    st.title("YouTube RAG AI Assistant")
    st.caption(
        "Multi-video transcript RAG with hybrid routing, timestamps, source citations, "
        "retrieved chunks, persistent chat memory, and general chat mode."
    )

    raw_urls, load_clicked, clear_clicked = render_loader_panel(settings)

    if clear_clicked:
        clear_memory(settings.memory_file)
        st.session_state.messages = []
        st.session_state.videos = []
        st.session_state.transcript_chunks = []
        st.session_state.vectorstore = None
        st.session_state.retriever = None
        st.session_state.video_loaded = False
        st.rerun()

    if load_clicked:
        try:
            load_video_index(raw_urls, settings)
        except Exception as exc:
            st.error(f"Could not load transcripts: {exc}")

    render_loaded_videos(st.session_state.get("videos", []))
    selected_prompt = render_quick_prompts()
    render_chat_history(st.session_state.messages)

    typed_question = st.chat_input("Ask about the loaded videos or ask a general question")
    question = selected_prompt or typed_question

    if question:
        handle_question(question, settings)


if __name__ == "__main__":
    main()
