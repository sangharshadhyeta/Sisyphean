# Sisyphean skill — search HuggingFace Hub for models, datasets, or model card info
import sys
import json
import urllib.request
import urllib.parse


HF_API = "https://huggingface.co/api"
HEADERS = {"User-Agent": "Sisyphean/1.0"}


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def fmt_number(n) -> str:
    if n is None:
        return "?"
    if isinstance(n, str):
        return n
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def cmd_models(query: str) -> None:
    url = (
        f"{HF_API}/models"
        f"?search={urllib.parse.quote(query)}"
        f"&limit=5&sort=downloads&direction=-1"
    )
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"Error: {e}")
        return
    if not data:
        print(f"No models found for: {query}")
        return
    print(f"Top models for '{query}':\n")
    for m in data:
        mid = m.get("modelId") or m.get("id", "?")
        downloads = fmt_number(m.get("downloads"))
        likes = fmt_number(m.get("likes"))
        tags = ", ".join(m.get("tags", [])[:4])
        print(f"  {mid}")
        print(f"    Downloads: {downloads}  |  Likes: {likes}")
        if tags:
            print(f"    Tags: {tags}")
        print(f"    URL: https://huggingface.co/{mid}")
        print()


def cmd_datasets(query: str) -> None:
    url = (
        f"{HF_API}/datasets"
        f"?search={urllib.parse.quote(query)}"
        f"&limit=5&sort=downloads&direction=-1"
    )
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"Error: {e}")
        return
    if not data:
        print(f"No datasets found for: {query}")
        return
    print(f"Top datasets for '{query}':\n")
    for d in data:
        did = d.get("id", "?")
        downloads = fmt_number(d.get("downloads"))
        likes = fmt_number(d.get("likes"))
        tags = ", ".join(d.get("tags", [])[:4])
        print(f"  {did}")
        print(f"    Downloads: {downloads}  |  Likes: {likes}")
        if tags:
            print(f"    Tags: {tags}")
        print(f"    URL: https://huggingface.co/datasets/{did}")
        print()


def cmd_info(model_id: str) -> None:
    url = f"{HF_API}/models/{model_id}"
    try:
        data = fetch_json(url)
    except Exception as e:
        print(f"Error fetching model info: {e}")
        return

    mid = data.get("modelId") or data.get("id", model_id)
    author = data.get("author", "?")
    downloads = fmt_number(data.get("downloads"))
    likes = fmt_number(data.get("likes"))
    pipeline = data.get("pipeline_tag", "?")
    library = data.get("library_name", "?")
    tags = ", ".join(data.get("tags", [])[:8])
    card = (data.get("cardData") or {}).get("summary", "")
    siblings = [s.get("rfilename", "") for s in data.get("siblings", [])]
    model_files = [f for f in siblings if f.endswith((".bin", ".safetensors", ".gguf"))]

    print(f"Model: {mid}")
    print(f"Author   : {author}")
    print(f"Pipeline : {pipeline}  |  Library: {library}")
    print(f"Downloads: {downloads}  |  Likes: {likes}")
    if tags:
        print(f"Tags     : {tags}")
    if card:
        print(f"Summary  : {card[:300]}")
    if model_files:
        print(f"Files    : {', '.join(model_files[:5])}")
    print(f"URL      : https://huggingface.co/{mid}")


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print('  python skills/hf_hub.py models "llama quantized"')
        print('  python skills/hf_hub.py datasets "code"')
        print('  python skills/hf_hub.py info "microsoft/phi-2"')
        return

    cmd = sys.argv[1].lower()
    arg = " ".join(sys.argv[2:])

    if cmd == "models":
        cmd_models(arg)
    elif cmd == "datasets":
        cmd_datasets(arg)
    elif cmd == "info":
        cmd_info(arg)
    else:
        print(f"Unknown command: {cmd}")
        print("Available: models, datasets, info")


if __name__ == "__main__":
    main()
