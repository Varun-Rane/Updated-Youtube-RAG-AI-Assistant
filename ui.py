import streamlit as st


def configure_page():
    st.set_page_config(
        page_title="YouTube RAG AI Assistant",
        page_icon="🎬",
        layout="wide",
    )


def inject_styles():
    st.markdown(
        """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.source-chip {
    display: inline-block;
    margin: 0.12rem 0.3rem 0.12rem 0;
    padding: 0.25rem 0.62rem;
    border-radius: 999px;
    border: 1px solid rgba(56, 189, 248, 0.32);
    background: rgba(56, 189, 248, 0.12);
    font-size: 0.88rem;
}

.section-rule {
    border-top: 1px solid rgba(148, 163, 184, 0.25);
    margin: 0.8rem 0;
}
</style>
""",
        unsafe_allow_html=True,
    )


def render_loader_panel(settings):

    st.subheader("📥 Load YouTube Videos")

    raw_urls = st.text_area(

        "Paste one or more YouTube URLs",

        height=120,

        placeholder="""
https://youtu.be/xxxx

https://youtu.be/yyyy
""",

    )

    c1,c2 = st.columns(2)

    with c1:

        load_clicked = st.button(

            "🚀 Load Videos",

            use_container_width=True,

        )

    with c2:

        clear_clicked = st.button(

            "🗑 Clear",

            use_container_width=True,

        )

    st.caption(

        f"""

LLM : {settings.llm_repo_id}

Embeddings : {settings.embedding_model}

Top-K : {settings.top_k}

"""

    )

    return raw_urls,load_clicked,clear_clicked

def render_loaded_videos(videos):

    if not videos:

        return

    st.divider()

    st.subheader("🎬 Loaded Videos")

    for video in videos:

        st.markdown(

            f"""

**{video["label"]}**

{video["video_title"]}

{video["video_url"]}

"""

        )

def render_quick_prompts():
    prompts = [
        "What is the video about?",
        "Generate notes.",
        "What are the key moments?",
        "Write a LinkedIn post for AI Engineer preparation.",
    ]

    st.subheader("Quick Prompts")
    columns = st.columns(len(prompts))
    selected_prompt = None

    for index, prompt in enumerate(prompts):
        with columns[index]:
            if st.button(prompt, use_container_width=True, key=f"quick_prompt_{index}"):
                selected_prompt = prompt

    return selected_prompt


def render_chat_history(messages):
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _render_source_videos(source_videos):
    if not source_videos:
        return

    st.markdown("#### 📹 Source")
    for source in source_videos:
        title = source.get("title") or source.get("label") or "Video"
        label = source.get("label") or "Video"
        url = source.get("url")
        if url:
            st.markdown(f"- **{label}:** [{title}]({url})")
        else:
            st.markdown(f"- **{label}:** {title}")


def _render_timestamps(timestamps):
    if not timestamps:
        return

    st.markdown("#### 📍 Mentioned Around")
    timestamp_markup = " ".join(
        f"<span class='source-chip'>{timestamp}</span>" for timestamp in timestamps
    )
    st.markdown(timestamp_markup, unsafe_allow_html=True)


def _render_chunks(retrieved_chunks):
    if not retrieved_chunks:
        return

    st.markdown("#### 📄 Retrieved Transcript")
    for index, chunk in enumerate(retrieved_chunks, start=1):
        title = chunk.get("video_title") or chunk.get("source_label") or "Video"
        timestamp = chunk.get("timestamp", "00:00")
        with st.expander(f"Chunk {index} - {timestamp} - {title}", expanded=False):
            st.markdown(f"**Source:** {chunk.get('source_label', 'Video')} - {title}")
            st.markdown(f"**Timestamp:** {timestamp}")
            if chunk.get("video_url"):
                st.markdown(f"**URL:** {chunk['video_url']}")
            st.write(chunk.get("text", ""))


def render_answer_bundle(answer_bundle):
    with st.container(border=True):
        st.markdown("### 🤖 Answer")
        st.markdown(answer_bundle.get("answer", ""))

        st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
        _render_timestamps(answer_bundle.get("timestamps", []))
        _render_source_videos(answer_bundle.get("source_videos", []))
        _render_chunks(answer_bundle.get("retrieved_chunks", []))
