"""
Qwen bbox OCR — загрузка PDF, конвертация в изображения, распознавание через vLLM (Qwen-VL),
возврат JSON структуры документа и markdown. В интерфейсе — изображения страниц с bbox разметкой.
"""
import base64
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.document_schema import document_to_markdown
from app.pdf_utils import pdf_to_images
from app.vllm_client import run_ocr_page

app = FastAPI(
    title="Qwen Bbox OCR",
    description="PDF → изображения → vLLM (Qwen-VL) OCR: текст, таблицы, изображения, печати, подписи с bbox",
    version="0.1.0",
)

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok", "service": "qwen-bbox-ocr"}


@app.get("/")
def index():
    return JSONResponse(
        status_code=307,
        headers={"Location": "/static/index.html"},
        content={},
    )


@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    """Загрузить PDF, конвертировать в изображения, отправить в vLLM по страницам; вернуть структуру + markdown + base64 страниц для отображения bbox."""
    if file.content_type not in ("application/pdf", "application/octet-stream") and not (
        file.filename or ""
    ).lower().endswith(".pdf"):
        return JSONResponse(
            status_code=400,
            content={"error": "Нужен файл PDF"},
        )

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        page_images = pdf_to_images(tmp_path)
        if not page_images:
            return JSONResponse(
                status_code=400,
                content={"error": "В PDF нет страниц или не удалось конвертировать"},
            )

        all_elements: List[Dict[str, Any]] = []
        pages_for_ui: List[Dict[str, Any]] = []

        for i, img_bytes in enumerate(page_images):
            page_num = i + 1
            elements = run_ocr_page(img_bytes, page_num)
            all_elements.extend(elements)
            pages_for_ui.append({
                "page": page_num,
                "image_base64": base64.b64encode(img_bytes).decode("ascii"),
                "elements": elements,
            })

        markdown = document_to_markdown(all_elements)

        return {
            "filename": file.filename,
            "structure": all_elements,
            "markdown": markdown,
            "pages": pages_for_ui,
            "num_pages": len(page_images),
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка обработки: {exc!s}"},
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
