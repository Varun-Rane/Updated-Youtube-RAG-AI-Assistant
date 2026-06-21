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