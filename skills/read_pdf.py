# Sisyphean skill — extract text from a PDF file, with optional page range
import sys
import os


def extract_with_pymupdf(path: str, page_start: int, page_end: int) -> str:
    try:
        import pymupdf as fitz  # newer name
    except ImportError:
        import fitz  # older name / alias
    doc = fitz.open(path)
    total = doc.page_count
    p0 = max(0, page_start - 1)
    p1 = min(total, page_end) if page_end else total
    parts = []
    for i in range(p0, p1):
        page = doc.load_page(i)
        parts.append(page.get_text())
    return "\n".join(parts)


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/read_pdf.py FILE.pdf [START_PAGE [END_PAGE]]")
        return

    path = sys.argv[1]
    page_start = int(sys.argv[2]) if len(sys.argv) >= 3 else 1
    page_end = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}")
        return

    # Try pymupdf / fitz
    try:
        text = extract_with_pymupdf(path, page_start, page_end or 9999)
    except ImportError:
        print(
            "pymupdf is not installed. Install it with:\n"
            "    pip install pymupdf\n"
            "Then re-run this script."
        )
        return
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return

    if not text.strip():
        print("(No text extracted — the PDF may be image-only or encrypted.)")
        return

    limit = 3000
    output = text[:limit]
    if len(text) > limit:
        output += f"\n\n[... truncated at {limit} chars — {len(text)} total chars in extracted range]"
    print(output)


if __name__ == "__main__":
    main()
