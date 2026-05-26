# Sisyphean skill — search arXiv papers by keyword and return top results
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"


def search_arxiv(query: str, max_results: int = 5) -> None:
    encoded = urllib.parse.quote(query)
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&max_results={max_results}&sortBy=relevance"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Sisyphean/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"Error fetching arXiv: {e}")
        return

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"Error parsing response: {e}")
        return

    entries = root.findall(f"{{{ATOM_NS}}}entry")
    if not entries:
        print("No results found.")
        return

    for i, entry in enumerate(entries, 1):
        def text(tag):
            el = entry.find(f"{{{ATOM_NS}}}{tag}")
            return el.text.strip() if el is not None and el.text else ""

        title = text("title").replace("\n", " ")
        summary = text("summary").replace("\n", " ")
        published = text("published")[:10]
        link_el = entry.find(f"{{{ATOM_NS}}}id")
        url_val = link_el.text.strip() if link_el is not None else ""

        authors = []
        for auth in entry.findall(f"{{{ATOM_NS}}}author"):
            name_el = auth.find(f"{{{ATOM_NS}}}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += f" +{len(authors) - 3} more"

        abstract_short = summary[:150] + ("..." if len(summary) > 150 else "")

        print(f"[{i}] {title}")
        print(f"    Authors : {author_str}")
        print(f"    Date    : {published}")
        print(f"    Abstract: {abstract_short}")
        print(f"    URL     : {url_val}")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/arxiv.py QUERY")
        return
    query = " ".join(sys.argv[1:])
    search_arxiv(query)


if __name__ == "__main__":
    main()
