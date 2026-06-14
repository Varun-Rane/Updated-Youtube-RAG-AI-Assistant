from llm import invoke_text
from prompts import GENERAL_PROMPT


def run_general(
    question,
    history,
    chat_model,
):

    prompt = GENERAL_PROMPT.format(

        history=history,

        question=question,

    )

    answer = invoke_text(

        chat_model,

        prompt,

    )

    return {

        "mode": "GENERAL",

        "answer": answer,

        "timestamps": [],

        "source_videos": [],

        "retrieved_chunks": [],

    }

