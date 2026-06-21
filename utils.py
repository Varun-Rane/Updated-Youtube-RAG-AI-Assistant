from urllib.parse import parse_qs, urlparse


def parse_video_urls(raw_urls):
    return [line.strip() for line in raw_urls.splitlines() if line.strip()]


def get_video_id(url):
    parsed_url = urlparse(url)
    hostname = (parsed_url.hostname or "").lower()
    hostname = hostname[4:] if hostname.startswith("www.") else hostname

    if hostname in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed_url.path == "/watch":
            video_id = parse_qs(parsed_url.query).get("v", [None])[0]
            if video_id:
                return video_id

        for prefix in ("/shorts/", "/embed/", "/live/"):
            if parsed_url.path.startswith(prefix):
                return parsed_url.path.removeprefix(prefix).split("/")[0]

    if hostname == "youtu.be":
        video_id = parsed_url.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    raise ValueError(f"Invalid YouTube URL: {url}")


def format_timestamp(seconds):
    if seconds is None:
        return "00:00"

    total_seconds = max(0, int(float(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def trim_text(text, max_chars):
    if not text:
        return ""

    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned

    return cleaned[:max_chars].rsplit(" ", 1)[0].strip()


def extract_llm_text(response):
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    return str(content)