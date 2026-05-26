# Sisyphean skill — fetch YouTube video transcript (auto-subtitles or transcript API)
import sys
import re
import os
import tempfile


def extract_video_id(arg: str) -> str:
    """Accept full URL or bare ID."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", arg)
    if m:
        return m.group(1)
    # Bare 11-char ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", arg.strip()):
        return arg.strip()
    return arg.strip()


def try_yt_dlp(video_id: str) -> str | None:
    """Try yt_dlp to download auto subtitles."""
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "writeautomaticsub": True,
            "writesubtitles": True,
            "subtitleslangs": ["en"],
            "subtitlesformat": "vtt",
            "outtmpl": os.path.join(tmpdir, "sub"),
            "quiet": True,
            "no_warnings": True,
        }
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except Exception as e:
            return None

        # Find the vtt file
        for fname in os.listdir(tmpdir):
            if fname.endswith(".vtt"):
                with open(os.path.join(tmpdir, fname), encoding="utf-8") as f:
                    raw = f.read()
                # Parse VTT: strip headers, timestamps, deduplicate lines
                lines = []
                seen = set()
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("WEBVTT") or "-->" in line:
                        continue
                    # Strip VTT tags
                    clean = re.sub(r"<[^>]+>", "", line).strip()
                    if clean and clean not in seen:
                        lines.append(clean)
                        seen.add(clean)
                return " ".join(lines)
    return None


def try_transcript_api(video_id: str) -> str | None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return None
    try:
        entries = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(e["text"] for e in entries)
    except Exception as e:
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/youtube.py VIDEO_URL_OR_ID")
        return

    raw = " ".join(sys.argv[1:])
    video_id = extract_video_id(raw)

    text = try_yt_dlp(video_id)
    if text is None:
        text = try_transcript_api(video_id)

    if text is None:
        print(
            "Could not fetch transcript. Neither yt_dlp nor youtube_transcript_api is available.\n"
            "Install one of:\n"
            "    pip install yt-dlp\n"
            "    pip install youtube-transcript-api"
        )
        return

    if not text.strip():
        print(f"No transcript found for video: {video_id}")
        return

    limit = 2000
    output = text[:limit]
    if len(text) > limit:
        output += f"\n\n[... truncated at {limit} chars — {len(text)} total]"
    print(f"Transcript for {video_id}:\n")
    print(output)


if __name__ == "__main__":
    main()
