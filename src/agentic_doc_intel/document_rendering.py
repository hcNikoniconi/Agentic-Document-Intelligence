"""Render documents into VLM-readable image data URLs.

The v1 pipeline should not depend on OCR. It sends original document pages or
images to a multimodal model directly.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def image_file_to_data_url(path: str | Path) -> str:
    path = Path(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def pdf_to_data_urls(path: str | Path, *, max_pages: int, zoom: float = 2.0) -> list[str]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering requires PyMuPDF. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    path = Path(path)
    data_urls: list[str] = []

    with fitz.open(path) as document:
        page_count = min(len(document), max_pages)
        matrix = fitz.Matrix(zoom, zoom)

        for page_index in range(page_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png_bytes = pixmap.tobytes("png")
            encoded = base64.b64encode(png_bytes).decode("ascii")
            data_urls.append(f"data:image/png;base64,{encoded}")

    return data_urls


def document_to_data_urls(path: str | Path, *, max_pages: int | None = None) -> list[str]:
    path = Path(path)
    suffix = path.suffix.lower()
    max_pages = max_pages or int(os.getenv("VLM_MAX_PAGES_PER_FILE", "4"))

    if suffix == ".pdf":
        return pdf_to_data_urls(path, max_pages=max_pages)

    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [image_file_to_data_url(path)]

    raise ValueError(f"Unsupported VLM input file type: {path}")

