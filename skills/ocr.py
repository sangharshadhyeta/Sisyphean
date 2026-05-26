# Sisyphean skill — extract text from an image using OCR (pytesseract, easyocr, or PIL)
import sys
import os


def try_pytesseract(path: str) -> str | None:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return None
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception as e:
        return f"pytesseract error: {e}"


def try_easyocr(path: str) -> str | None:
    try:
        import easyocr
    except ImportError:
        return None
    try:
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(path, detail=0)
        return "\n".join(results)
    except Exception as e:
        return f"easyocr error: {e}"


def try_pil_basic(path: str) -> str | None:
    """Last resort: open with PIL, confirm it loads, but can't do OCR."""
    try:
        from PIL import Image
        img = Image.open(path)
        return (
            f"Image loaded ({img.size[0]}x{img.size[1]} px, mode={img.mode}) "
            "but no OCR engine available.\n"
            "Install one of:\n"
            "    pip install pytesseract   (requires Tesseract binary)\n"
            "    pip install easyocr"
        )
    except ImportError:
        return None
    except Exception as e:
        return f"PIL error: {e}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python skills/ocr.py PATH_TO_IMAGE")
        return

    path = " ".join(sys.argv[1:])

    if not os.path.isfile(path):
        print(f"Error: file not found: {path}")
        return

    # Try engines in order
    result = try_pytesseract(path)
    if result is not None:
        print(result.strip() if result.strip() else "(no text detected)")
        return

    result = try_easyocr(path)
    if result is not None:
        print(result.strip() if result.strip() else "(no text detected)")
        return

    result = try_pil_basic(path)
    if result is not None:
        print(result)
        return

    print(
        "No OCR engine or image library available.\n"
        "Install one of:\n"
        "    pip install pytesseract   (requires Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki)\n"
        "    pip install easyocr\n"
        "    pip install Pillow"
    )


if __name__ == "__main__":
    main()
