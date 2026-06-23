from llm import invoke_text
from prompts import FINAL_VIDEO_TASK_PROMPT
from summary_cache import (
    summary_exists,
    get_master_summary,
)


def _unavailable_bundle():
    return {
        "mode": "VIDEO_TASK",
        "answer": "Load at least one transcript first.",
        "timestamps": [],
        "source_videos": [],
        "retrieved_chunks": [],
    }


def run_video_task(
    question,
    videos,
    chat_model,
    settings,
):
    if not videos:
        return _unavailable_bundle()

    # Single video cache
    if len(videos) == 1:

        video_id = videos[0]["video_id"]

        if summary_exists(video_id):

            material = get_master_summary(
                video_id
            )

        else:

            return {
                "mode": "VIDEO_TASK",
                "answer": (
                    "Summary not found. "
                    "Run 'Summarize this video' first."
                ),
                "timestamps": [],
                "source_videos": [],
                "retrieved_chunks": [],
            }

    else:

        summaries = []

        for video in videos:

            video_id = video["video_id"]

            if summary_exists(video_id):

                summary = get_master_summary(
                    video_id
                )

                if summary:
                    summaries.append(
                        f"Video: {video['video_title']}\n\n"
                        f"{summary}"
                    )

        if not summaries:

            return {
                "mode": "VIDEO_TASK",
                "answer": (
                    "No cached summaries found. "
                    "Run 'Summarize this video' first."
                ),
                "timestamps": [],
                "source_videos": [],
                "retrieved_chunks": [],
            }

        material = "\n\n".join(
            summaries
        )
    
    print("=" * 80)
    print("VIDEO TASK USING CACHED SUMMARY")
    print(f"SUMMARY LENGTH: {len(material)}")
    print("=" * 80)
    
    prompt = FINAL_VIDEO_TASK_PROMPT.format(
        question=question,
        material=material,
    )

    answer = invoke_text(
        chat_model,
        prompt,
    )

    return {
        "mode": "VIDEO_TASK",
        "answer": answer,
        "timestamps": [],
        "source_videos": [
            {
                "label": v.get(
                    "label",
                    "Video",
                ),
                "title": v.get(
                    "video_title",
                    "YouTube Video",
                ),
                "url": v.get(
                    "video_url",
                    "",
                ),
            }
            for v in videos
        ],
        "retrieved_chunks": [],
    }