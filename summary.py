from llm import invoke_text
from prompts import SUMMARY_PROMPT
from utils import trim_text


def build_summary_context(videos, max_chars):

    if not videos:
        return ""

    sections = []

    per_video_chars = max_chars // max(1, len(videos))

    for video in videos:

        transcript = trim_text(
            video["transcript"],
            per_video_chars,
        )

        sections.append(

            f"""

Video Title:

{video["video_title"]}

Transcript:

{transcript}

"""

        )

    return "\n\n".join(sections)


def run_summary(

    question,

    videos,

    chat_model,

    settings,

):

    context = build_summary_context(

        videos,

        settings.max_context_chars,

    )

    if not context:

        return {

            "mode":"VIDEO_SUMMARY",

            "answer":"Load at least one transcript first.",

            "timestamps":[],

            "source_videos":[],

            "retrieved_chunks":[],

        }

    prompt = SUMMARY_PROMPT.format(

        context=context,

        question=question,

    )

    answer = invoke_text(

        chat_model,

        prompt,

    )

    return {

        "mode":"VIDEO_SUMMARY",

        "answer":answer,

        "timestamps":[],

        "source_videos":[

            {

                "title":video["video_title"],

                "url":video["video_url"],

            }

            for video in videos

        ],

        "retrieved_chunks":[],

    }