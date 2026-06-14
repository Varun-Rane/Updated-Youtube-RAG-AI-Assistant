from llm import invoke_text
from prompts import ROUTER_PROMPT


GENERAL_HINTS = (
    "write a resume",
    "write resume",
    "linkedin post",
    "cover letter",
    "tell me a joke",
    "write python code",
    "write code",
    "generate code",
    "prime minister",
    "current president",
    "latest news",
)

RAG_HINTS = (
    "video",
    "lecture",
    "transcript",
    "from the lecture",
    "from this video",
    "in the video",
    "summarize",
    "summary",
    "generate notes",
    "key moments",
    "mentioned",
    "timestamp",
)


def heuristic_route(question, has_loaded_videos):
    lowered = question.lower()

    if any(hint in lowered for hint in GENERAL_HINTS):
        return "GENERAL"

    if has_loaded_videos and any(hint in lowered for hint in RAG_HINTS):
        return "RAG"

    return "RAG" if has_loaded_videos else "GENERAL"


def parse_route(raw_response):
    first_line = raw_response.strip().upper().splitlines()[0] if raw_response else ""
    cleaned = first_line.replace("`", "").replace(".", "").replace(":", "").strip()

    if cleaned == "RAG":
        return "RAG"
    if cleaned == "GENERAL":
        return "GENERAL"
    if "GENERAL" in cleaned:
        return "GENERAL"
    if "RAG" in cleaned:
        return "RAG"
    return None


def classify_query(question, chat_model, has_loaded_videos):
    if not has_loaded_videos:
        return "GENERAL"

    try:
        raw_response = invoke_text(
            chat_model,
            ROUTER_PROMPT.format(question=question),
        )
        route = parse_route(raw_response)
        if route:
            return route
    except Exception:
        pass

    return heuristic_route(question, has_loaded_videos)
