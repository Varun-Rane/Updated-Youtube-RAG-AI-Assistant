MEMORY_HINTS = (
    "my name",
    "who am i",
    "what is my name",
    "remember",
    "previous question",
    "last question",
    "first question",
    "second question",
    "history",
    "conversation",
    "did i ask",
    "what did i ask",
    "questions i asked",
)

SUMMARY_HINTS = (
    "summary",
    "summarize",
    "video about",
    "what is the video about",
    "overview",
    "generate notes",
    "notes",
    "timeline",
    "chapters",
    "key moments",
    "main points",
    "questions discussed",
    "list questions",
)

GENERAL_HINTS = (
    "python",
    "java",
    "c++",
    "cpp",
    "javascript",
    "html",
    "css",
    "sql",
    "react",
    "node",
    "django",
    "flask",
    "fastapi",
    "leetcode",
    "algorithm",
    "binary search",
    "linked list",
    "tree",
    "graph",
    "stack",
    "queue",
    "dynamic programming",
    "resume",
    "linkedin",
    "cover letter",
    "email",
    "essay",
    "joke",
    "code",
    "program",
    "fibonacci",
)


def classify_query(
    question,
    chat_model=None,
    has_loaded_videos=False,
):

    q = question.lower()

    if any(x in q for x in MEMORY_HINTS):
        return "MEMORY"

    if any(x in q for x in SUMMARY_HINTS):
        return "VIDEO_SUMMARY"

    if any(x in q for x in GENERAL_HINTS):
        return "GENERAL"

    if has_loaded_videos:
        return "VIDEO_QA"

    return "GENERAL"