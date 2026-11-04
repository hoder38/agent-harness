"""Documents → markdown (Docling on CPU; plain PDF text for long files; OCR for scanned PDFs) (§5)."""
import asyncio
import logging
import os
import tempfile
import threading
from typing import Any

from . import common

log = logging.getLogger(__name__)

_EXT = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
}
_converter = None
_conv_lock = threading.Lock()


def _docling(src: str, mime: str) -> str:
    """Docling needs the extension to pick a backend; the store keeps files by hash only."""
    global _converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "doc" + _EXT[mime])
        os.symlink(src, dst)
        with _conv_lock:
            if _converter is None:
                opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)   # scanned PDFs go to RapidOCR
                _converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            res = _converter.convert(dst)
        return res.document.export_to_markdown()


def _pdf_pages(src: str) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(src)
    try:
        return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
    finally:
        pdf.close()


def _pdf_ocr(src: str, max_pages: int) -> tuple[list[str], int]:
    import pypdfium2 as pdfium

    from .image import _ocr_sync

    pdf = pdfium.PdfDocument(src)
    try:
        n = len(pdf)
        out = [_ocr_sync(pdf[i].render(scale=2).to_pil().convert("RGB")) for i in range(min(n, max_pages))]
        return out, n
    finally:
        pdf.close()


def _paged(pages: list[str]) -> str:
    return "\n\n".join(f"## 第 {i} 頁\n{t.strip()}" for i, t in enumerate(pages, 1) if t.strip())


def _read_text(src: str) -> str:
    with open(src, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "big5", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_sync(att: dict[str, Any]) -> dict[str, Any]:
    d = common.sub("doc")
    src, mime = str(common.file_path(att["sha256"])), att["mime"]
    large_pages, large_chars = int(d.get("large_pages", 20)), int(d.get("large_chars", 40000))
    meta: dict[str, Any] = {"path": "cpu", "degraded": False}
    if mime == "text/plain":
        text = _read_text(src)
        pages = None
    elif mime == "application/pdf":
        page_texts = _pdf_pages(src)
        pages = len(page_texts)
        chars = sum(len(t.strip()) for t in page_texts)
        if pages and chars / pages < int(d.get("scanned_chars_per_page", 30)):
            ocr_pages, _ = _pdf_ocr(src, int(d.get("ocr_max_pages", 30)))
            text = _paged(ocr_pages)
            meta["ocr"] = True
            if pages > len(ocr_pages):
                meta["degraded"] = True
                text += f"\n\n(掃描檔共 {pages} 頁,只辨識了前 {len(ocr_pages)} 頁。)"
        elif pages <= int(d.get("docling_max_pages", 50)):
            try:
                text = _docling(src, mime)
            except Exception as e:
                log.warning("docling failed, using plain PDF text: %s", e)
                text, meta["docling_error"] = _paged(page_texts), str(e)[:200]
        else:
            text = _paged(page_texts)   # long PDF: plain text per page is fast; it goes to the knowledge base
    elif mime in _EXT:
        text, pages = _docling(src, mime), None
    else:
        raise ValueError(f"unsupported document type {mime}")
    text = text.strip()
    if pages is not None:
        meta["pages"] = pages
    large = (pages or 0) > large_pages or len(text) > large_chars
    first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
    summary = (f"{pages} 頁 · " if pages else "") + (first[:100] or "文件")
    return {"kind": "large_doc" if large else "doc", "text": text, "summary": summary, "meta": meta}


async def extract(att: dict[str, Any]) -> dict[str, Any]:
    return await asyncio.to_thread(_extract_sync, att)
