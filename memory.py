import json
from pathlib import Path

import streamlit as st

from llm import invoke_text
from prompts import MEMORY_PROMPT
from utils import trim_text


MEMORY_HINTS = (
    "what was my first question",
    "what was my second question",
    "what was my last question",
    "what did i ask",
    "previous question",
    "last question",
    "chat history",
    "our conversation",
    "questions i asked",
    "summarize our conversation",
    "summarize our chat",
    "what did i ask before",
)


def load_messages(memory_file):
    path = Path(memory_file)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    return [
        item
        for item in data
        if isinstance(item, dict)
        and item.get("role") in {"user", "assistant"}
        and isinstance(item.get("content"), str)
    ]


def save_messages(memory_file, messages):
    path = Path(memory_file)
    path.write_text(json.dumps(messages, indent=2), encoding="utf-8")


def clear_memory(memory_file):
    save_messages(memory_file, [])


def init_session_state(settings):
    if "messages" not in st.session_state:
        st.session_state.messages = load_messages(settings.memory_file)
    if "videos" not in st.session_state:
        st.session_state.videos = []
    if "transcript_chunks" not in st.session_state:
        st.session_state.transcript_chunks = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "video_loaded" not in st.session_state:
        st.session_state.video_loaded = False


def history_text(messages, max_messages=5, max_chars=3000):
    recent_messages = messages[-max_messages:]
    text = "\n".join(
        f"{message['role']}: {message['content']}" for message in recent_messages
    )
    return trim_text(text, max_chars)


def is_memory_query(question):
    lowered = question.lower()
    return any(hint in lowered for hint in MEMORY_HINTS)


def _user_questions(messages):
    return [message["content"] for message in messages if message.get("role") == "user"]


def answer_memory_question(question, messages, chat_model=None, settings=None):
    user_questions = _user_questions(messages)
    lowered = question.lower()

    if "first question" in lowered:
        if user_questions:
            return f"Your first question was:\n\n{user_questions[0]}"
        return "You have not asked a previous question yet."

    if "second question" in lowered:
        if len(user_questions) >= 2:
            return f"Your second question was:\n\n{user_questions[1]}"
        return "You have not asked a second question yet."

    if "last question" in lowered or "previous question" in lowered:
        if user_questions:
            return f"Your previous question was:\n\n{user_questions[-1]}"
        return "You have not asked a previous question yet."

    if "questions i asked" in lowered or "what did i ask" in lowered:
        if not user_questions:
            return "You have not asked any previous questions yet."
        return "Questions you have asked:\n\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(user_questions, start=1)
        )

    if chat_model and settings:
        prompt = MEMORY_PROMPT.format(
            history=history_text(
                messages,
                max_messages=settings.max_history_messages,
                max_chars=settings.max_history_chars,
            ),
            question=question,
        )
        return invoke_text(chat_model, prompt)

    return "I could not find that in our previous conversation."
