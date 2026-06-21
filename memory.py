import json
from pathlib import Path

import streamlit as st

from llm import invoke_text
from prompts import MEMORY_PROMPT
from utils import trim_text


MEMORY_HINTS = (
    "what is my name",
    "what's my name",
    "who am i",
    "remember my name",
    "what was my first question",
    "what was my second question",
    "what was my third question",
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
    "last answer",
    "previous answer",
    "what did you answer",
    "your last answer",
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
    q = question.lower()
    return any(hint in q for hint in MEMORY_HINTS)


def _user_questions(messages):
    questions = []

    for message in messages:
        if message.get("role") != "user":
            continue

        content = message.get("content", "").strip()
        if not content:
            continue

        if content.lower().startswith(
            ("my name is", "i am", "i'm", "call me", "you can call me")
        ):
            continue

        questions.append(content)

    return questions


def _remembered_name(messages):
    name = None

    for message in messages:
        if message.get("role") != "user" or not message.get("content"):
            continue

        content = message["content"].strip()
        lowered = content.lower()

        for prefix in ("my name is", "i am", "i'm", "call me", "you can call me"):
            if lowered.startswith(prefix):
                candidate = content[len(prefix) :].strip().strip(".,!?")
                if candidate:
                    name = candidate
                break

    return name


def answer_memory_question(
    question,
    messages,
    chat_model=None,
    settings=None,
    history=None,
):
    q = question.lower()
    user_questions = _user_questions(messages)

    if (
        "what is my name" in q
        or "what's my name" in q
        or "who am i" in q
    ):
        name = _remembered_name(messages)

        if name:
            return f"Your name is {name}."

        return "I don't know your name yet."

    if (
        "last answer" in q
        or "previous answer" in q
        or "what did you answer" in q
        or "your last answer" in q
    ):
        assistant_answers = [
            message["content"]
            for message in messages
            if message.get("role") == "assistant" and message.get("content")
        ]

        if assistant_answers:
            return assistant_answers[-1]

        return "No previous answer."

    if "first question" in q:
        if user_questions:
            return user_questions[0]

        return "No previous question."

    if "second question" in q:
        if len(user_questions) >= 2:
            return user_questions[1]

        return "Second question not found."

    if "third question" in q:
        if len(user_questions) >= 3:
            return user_questions[2]

        return "Third question not found."

    if "last question" in q or "previous question" in q:
        if user_questions:
            return user_questions[-1]

        return "No previous question."

    if "questions i asked" in q:
        if not user_questions:
            return "No questions found."

        return "\n".join(
            f"{index}. {user_question}"
            for index, user_question in enumerate(user_questions, start=1)
        )

    if chat_model and settings:
        if history is None:
            history = history_text(
                messages,
                max_messages=settings.max_history_messages,
                max_chars=settings.max_history_chars,
            )

        prompt = MEMORY_PROMPT.format(
            history=history,
            question=question,
            total_questions=len(user_questions),
        )

        return invoke_text(chat_model, prompt)

    return "I don't remember that."