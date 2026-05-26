# Sisyphean Skills

Self-contained Python scripts the agent can invoke via its bash tool:
`Run python skills/SCRIPT.py ARGS`

All scripts: stdlib-first, graceful degradation when optional deps missing, exit 0 always.

---

| Script | Purpose | Usage example | Dependencies |
|---|---|---|---|
| `calc.py` | Evaluate math expressions safely | `python skills/calc.py "sqrt(144) + 2**8"` | stdlib only (`math`) |
| `arxiv.py` | Search arXiv papers (title, authors, abstract, URL) | `python skills/arxiv.py "transformer attention"` | stdlib only (`urllib`, `xml`) |
| `read_pdf.py` | Extract text from a PDF, optional page range | `python skills/read_pdf.py file.pdf 2 5` | `pymupdf` (pip install pymupdf) |
| `github_ops.py` | List issues/PRs, clone repos, search GitHub | `python skills/github_ops.py issues owner/repo` | `gh` CLI (`https://cli.github.com`) |
| `youtube.py` | Fetch YouTube video transcript | `python skills/youtube.py https://youtu.be/VIDEO_ID` | `yt-dlp` or `youtube-transcript-api` |
| `obsidian.py` | Search, read, and create Obsidian vault notes | `python skills/obsidian.py search "machine learning"` | stdlib only (`pathlib`, `re`) |
| `maps.py` | Geocode addresses, estimate routes, find nearby POIs | `python skills/maps.py geocode "Eiffel Tower"` | stdlib only (`urllib`) |
| `hf_hub.py` | Search HuggingFace models/datasets, get model info | `python skills/hf_hub.py models "llama quantized"` | stdlib only (`urllib`) |
| `ocr.py` | Extract text from images via OCR | `python skills/ocr.py screenshot.png` | `pytesseract` or `easyocr` (+ `Pillow`) |
| `web.py` | Fetch a URL and return condensed plain text | `python skills/web.py https://example.com` | stdlib only (`urllib`, `html.parser`) |

---

## Setup notes

### calc.py
No setup required. Supports all `math.*` functions: `sin`, `cos`, `sqrt`, `log`, `pi`, `e`, etc.
```
python skills/calc.py "pi * 5**2"
python skills/calc.py sqrt 2
```

### arxiv.py
No setup required. Hits the public arXiv Atom feed.
```
python skills/arxiv.py "diffusion models image generation"
```

### read_pdf.py
```
pip install pymupdf
python skills/read_pdf.py document.pdf
python skills/read_pdf.py document.pdf 3 7    # pages 3-7 only
```

### github_ops.py
Requires `gh` CLI installed and authenticated.
```
gh auth login
python skills/github_ops.py issues microsoft/vscode
python skills/github_ops.py pr pytorch/pytorch
python skills/github_ops.py clone huggingface/transformers
python skills/github_ops.py search "local llm inference"
```

### youtube.py
```
pip install yt-dlp          # preferred
# or
pip install youtube-transcript-api
python skills/youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
python skills/youtube.py dQw4w9WgXcQ    # bare video ID also works
```

### obsidian.py
Set `OBSIDIAN_VAULT` env var to your vault path, or it defaults to `~/Documents/Obsidian`.
```
set OBSIDIAN_VAULT=C:\Users\hp\Documents\MyVault
python skills/obsidian.py search "project ideas"
python skills/obsidian.py read "Meeting Notes"
python skills/obsidian.py create "New Idea" "Content goes here"
```

### maps.py
No setup required. Uses Nominatim (geocoding), OSRM (routing), and Overpass API (nearby POIs).
```
python skills/maps.py geocode "Taj Mahal, Agra"
python skills/maps.py route "New Delhi" "Mumbai"
python skills/maps.py nearby "restaurant" "28.6139,77.2090"
```

### hf_hub.py
No setup required. Uses the public HuggingFace Hub API.
```
python skills/hf_hub.py models "code generation"
python skills/hf_hub.py datasets "instruction tuning"
python skills/hf_hub.py info "mistralai/Mistral-7B-v0.1"
```

### ocr.py
```
pip install pytesseract     # also needs Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki
# or
pip install easyocr
python skills/ocr.py image.png
```

### web.py
No setup required. Strips HTML, skips nav/footer/scripts, returns ~1000 chars of body text.
```
python skills/web.py https://en.wikipedia.org/wiki/Python_(programming_language)
```
