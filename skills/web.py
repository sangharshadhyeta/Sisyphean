# Sisyphean skill — fetch a URL and return condensed plain-text content (~1000 chars)
import sys
import re
import urllib.request
import urllib.parse
import html.parser


class TextExtractor(html.parser.HTMLParser):
    SKIP_TAGS = {
        "script", "style", "noscript", "nav", "footer", "header",
        "aside", "form", "button", "svg", "meta", "link",
    }

    def __init__(self):
        super().__init__()
        self._depth = {}   # tag -> nesting depth for skip tags
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._depth[tag] = self._depth.get(tag, 0) + 1
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._depth.get(tag, 0) > 0:
            self._depth[tag] -= 1
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text)


def strip_html(raw_html: str) -> str:
    parser = TextExtractor()
    try:
        parser.feed(raw_html)
    except Exception:
        pass
    text = " ".join(parser.parts)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_encoding(headers, body: bytes) -> str:
    ct = headers.get("Content-Type", "")
    m = re.search(r"charset=([^\s;]+)", ct, re.I)
    if m:
        return m.group(1)
    # Try meta charset in body
    head_bytes = body[:4096].decode("ascii", errors="ignore")
    m = re.search(r'charset=["\']?([A-Za-z0-9_-]+)', head_bytes, re.I)
    if m:
        return m.group(1)
    return "utf-8"


def fetch_url(url: str, timeout: int = 10) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        enc = detect_encoding(r.headers, body)
        try:
            html_text = body.decode(enc, errors="replace")
        except (LookupError, UnicodeDecodeError):
            html_text = body.decode("utf-8", errors="replace")
    return html_text


def condense(text: str, limit: int = 1000) -> str:
    # Remove very short fragments (likely menu items)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 30]
    joined = ". ".join(sentences)
    if not joined:
        joined = text
    return joined[:limit] + ("..." if len(joined) > limit else "")


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/web.py URL")
        return

    url = sys.argv[1]

    try:
        html_text = fetch_url(url)
    except urllib.request.HTTPError as e:
        print(f"HTTP error {e.code}: {e.reason} — {url}")
        return
    except urllib.request.URLError as e:
        print(f"URL error: {e.reason}")
        return
    except TimeoutError:
        print(f"Timeout fetching: {url}")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    plain = strip_html(html_text)
    if not plain.strip():
        print("(No readable text found on page.)")
        return

    print(condense(plain))


if __name__ == "__main__":
    main()
