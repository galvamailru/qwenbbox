"""Convert PDF to list of page images (PIL) for sending to vLLM."""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None  # type: ignore

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore


def pdf_to_images(pdf_path: Path, dpi: int = 150) -> List[bytes]:
    """
    Convert PDF to list of PNG images (bytes). Prefer pdf2image (poppler); fallback PyMuPDF.
    Returns list of PNG bytes, one per page.
    """
    logger.info("pdf_to_images: конвертация %s (DPI=%s)...", pdf_path.name, dpi)
    if convert_from_path is not None:
        pil_images = convert_from_path(pdf_path, dpi=dpi, fmt="png")
        out: List[bytes] = []
        import io

        for img in pil_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out.append(buf.getvalue())
        logger.info("pdf_to_images: готово (pdf2image), страниц: %s", len(out))
        return out

    if fitz is not None:
        doc = fitz.open(pdf_path)
        out = []
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            if hasattr(pix, "getPNGData"):
                out.append(pix.getPNGData())
            else:
                import io
                from PIL import Image
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out.append(buf.getvalue())
        doc.close()
        logger.info("pdf_to_images: готово (PyMuPDF), страниц: %s", len(out))
        return out

    raise RuntimeError("Install pdf2image (with poppler) or PyMuPDF to convert PDF to images.")
