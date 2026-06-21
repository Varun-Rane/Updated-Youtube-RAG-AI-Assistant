from llm import invoke_text
from prompts import CLASSIFIER_PROMPT

VALID_ROUTES = {
    "MEMORY",
    "GENERAL",
    "VIDEO_QA",
    "VIDEO_SUMMARY",
    "VIDEO_TASK",
}


def classify_query(
    question,
    chat_model,
    has_loaded_videos=False,
):
    prompt = CLASSIFIER_PROMPT.format(
        question=question
    )

    try:
        result = invoke_text(
            chat_model,
            prompt,
        ).strip().upper()

        result = result.split()[0]

        if result in VALID_ROUTES:

            if (
                result.startswith("VIDEO")
                and not has_loaded_videos
            ):
                return "GENERAL"

            return result

    except Exception:
        pass

    if has_loaded_videos:
        return "VIDEO_QA"

    return "GENERAL"