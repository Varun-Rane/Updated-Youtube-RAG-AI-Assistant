import json
from pathlib import Path


CACHE_DIR = Path(".summary_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(video_id):
    return CACHE_DIR / f"{video_id}.json"


def summary_exists(video_id):
    return _cache_path(video_id).exists()


def save_summary(
    video_id,
    master_summary,
    video_title=None,
):
    path = _cache_path(video_id)

    data = {
        "video_id": video_id,
        "video_title": video_title,
        "master_summary": master_summary,
    }

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_summary(video_id):
    path = _cache_path(video_id)

    if not path.exists():
        return None

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return data

    except Exception:
        return None


def get_master_summary(video_id):
    data = load_summary(video_id)

    if not data:
        return None

    return data.get(
        "master_summary"
    )


def clear_summary(video_id):
    path = _cache_path(video_id)

    if path.exists():
        path.unlink()


def clear_all_summaries():
    for file in CACHE_DIR.glob("*.json"):
        try:
            file.unlink()
        except Exception:
            pass