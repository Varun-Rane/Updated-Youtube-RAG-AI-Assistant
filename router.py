from llm import invoke_text

from prompts import (
    DOMAIN_CLASSIFIER_PROMPT,
    VIDEO_INTENT_PROMPT,
)

DOMAINS = {
    "MEMORY",
    "GENERAL",
    "VIDEO",
}

VIDEO_ROUTES = {
    "VIDEO_QA",
    "VIDEO_OVERVIEW",
    "VIDEO_SUMMARY",
    "VIDEO_TASK",
}


VIDEO_KEYWORDS = {
    "hnsw",
    "ivf",
    "product quantization",
    "pq",
    "metadata filtering",
    "vector database",
    "embedding",
    "embeddings",
    "ann",
    "approximate nearest neighbor",
    "centroid",
    "centroids",
    "cluster",
    "clustering",
    "codebook",
    "faiss",
    "similarity search",
    "indexing",
    "vector",
}


SUMMARY_KEYWORDS = {
    "summarize",
    "summary",
    "study notes",
    "notes",
    "revision notes",
    "study guide",
}
OVERVIEW_KEYWORDS = {
    "topics covered",
    "main topics",
    "key concepts",
    "overview",
    "video overview",
    "what is this video about",
    "main idea",
    "key takeaways",
    "overall video",
}

TASK_KEYWORDS = {
    "mcq",
    "quiz",
    "flashcards",
    "interview questions",
    "practice questions",
    "cheat sheet",
}


def classify_domain(
    question,
    chat_model,
):
    prompt = DOMAIN_CLASSIFIER_PROMPT.format(
        question=question
    )

    try:
        result = (
            invoke_text(
                chat_model,
                prompt,
            )
            .strip()
            .upper()
            .split()[0]
        )

        if result in DOMAINS:
            return result

    except Exception as exc:
        print("Domain Classifier Error:", exc)

    return "GENERAL"


def classify_video_intent(
    question,
    chat_model,
):
    prompt = VIDEO_INTENT_PROMPT.format(
        question=question
    )

    try:
        result = (
            invoke_text(
                chat_model,
                prompt,
            )
            .strip()
            .upper()
            .split()[0]
        )

        if result in VIDEO_ROUTES:
            return result

    except Exception as exc:
        print("Video Intent Error:", exc)

    return "VIDEO_QA"


def classify_query(
    question,
    chat_model,
    has_loaded_videos=False,
):
    q = question.lower().strip()

    # =====================================
    # MEMORY
    # =====================================

    memory_phrases = [
        "my first question",
        "previous question",
        "what did i ask",
        "conversation history",
        "chat history",
        "remember",
        "what was my",
    ]

    if any(x in q for x in memory_phrases):
        return "MEMORY"

    # =====================================
    # VIDEO OVERRIDES
    # =====================================

    if has_loaded_videos:

        # Timestamp queries
        if ":" in q:
            print("ROUTER OVERRIDE -> VIDEO_QA (timestamp)")
            return "VIDEO_QA"

        # Technical concepts from loaded video
        if any(keyword in q for keyword in VIDEO_KEYWORDS):
            print("ROUTER OVERRIDE -> VIDEO_QA (keyword)")
            return "VIDEO_QA"

        # Task requests
        if any(keyword in q for keyword in TASK_KEYWORDS):
            print("ROUTER OVERRIDE -> VIDEO_TASK")
            return "VIDEO_TASK"

        # Summary requests
        if any(keyword in q for keyword in SUMMARY_KEYWORDS):
            print("ROUTER OVERRIDE -> VIDEO_SUMMARY")
            return "VIDEO_SUMMARY"
        
        # Overview requests
        if any(keyword in q for keyword in OVERVIEW_KEYWORDS):
            print("ROUTER OVERRIDE -> VIDEO_OVERVIEW")
            return "VIDEO_OVERVIEW"

    # =====================================
    # LLM DOMAIN CLASSIFIER
    # =====================================

    domain = classify_domain(
        question,
        chat_model,
    )

    print(f"DOMAIN: {domain}")

    if domain == "MEMORY":
        return "MEMORY"

    if domain == "GENERAL":
        return "GENERAL"

    if domain == "VIDEO":

        if not has_loaded_videos:
            return "GENERAL"

        route = classify_video_intent(
            question,
            chat_model,
        )

        print(f"VIDEO ROUTE: {route}")

        return route

    return "GENERAL"