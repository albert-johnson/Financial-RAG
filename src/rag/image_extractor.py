from __future__ import annotations

import os
from pathlib import Path
from typing import List

from docx import Document as DocxDocument

from src.config import PAGE_IMAGES_DIR


def extract_pdf_page_images(file_path: str, dpi: int = 150) -> List[str]:
    saved: List[str] = []
    try:
        import fitz  # PyMuPDF
    except Exception:
        return saved
    try:
        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            out_path = PAGE_IMAGES_DIR / f"{Path(file_path).stem}_page_{page_num + 1}.png"
            pix.save(str(out_path))
            saved.append(str(out_path))
    except Exception:
        pass
    return saved


def extract_docx_inline_images(file_path: str) -> List[str]:
    saved: List[str] = []
    try:
        doc = DocxDocument(file_path)
        media_dir = Path(file_path).with_suffix("").name + "_media"
        target_dir = PAGE_IMAGES_DIR / media_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        # Extract embedded images from docx package
        # python-docx does not directly expose raw images; fallback by unpacking zip
        import zipfile
        with zipfile.ZipFile(file_path) as z:
            for info in z.infolist():
                if info.filename.startswith("word/media/") and info.filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    out_path = target_dir / Path(info.filename).name
                    with z.open(info.filename) as src, open(out_path, "wb") as dst:
                        dst.write(src.read())
                    saved.append(str(out_path))
    except Exception:
        pass
    return saved