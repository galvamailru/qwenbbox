"""
Qwen bbox OCR — загрузка PDF, конвертация в изображения, распознавание через vLLM (Qwen-VL),
возврат JSON структуры документа и markdown. В интерфейсе — изображения страниц с bbox разметкой.
"""
import base64
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.document_schema import document_to_markdown
from app.pdf_utils import pdf_to_images
from app.vllm_client import run_ocr_page
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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


def _draw_bboxes_on_image(img_bytes: bytes, elements: List[Dict[str, Any]]) -> bytes:
    """Нарисовать bbox поверх PNG-страницы и вернуть новые байты PNG."""
    if not img_bytes:
        return img_bytes
    try:
        im = Image.open(BytesIO(img_bytes)).convert("RGBA")
    except Exception:
        logger.exception("parse: не удалось открыть страницу как изображение для рисования bbox")
        return img_bytes

    draw = ImageDraw.Draw(im, "RGBA")

    # Цвета как во фронтенде
    COLORS: Dict[str, tuple] = {
        "text": (37, 99, 235, 200),       # #2563eb
        "table": (22, 163, 74, 200),      # #16a34a
        "image": (180, 83, 9, 200),       # #b45309
        "stamp": (185, 28, 28, 200),      # #b91c1c
        "signature": (107, 33, 168, 200), # #6b21a8
    }

    for el in elements or []:
        bbox = el.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        t = str(el.get("type") or "text").lower()
        color = COLORS.get(t, COLORS["text"])
        # Полупрозрачная заливка + контур
        fill = (color[0], color[1], color[2], 40)
        outline = color
        draw.rectangle([x1, y1, x2, y2], outline=outline, width=2, fill=fill)

    out = BytesIO()
    im.save(out, format="PNG")
    return out.getvalue()


@app.post("/parse")
async def parse_pdf(file: UploadFile = File(...)):
    """Загрузить PDF, конвертировать в изображения, отправить в vLLM по страницам; вернуть структуру + markdown + base64 страниц для отображения bbox."""
    filename = file.filename or "unknown.pdf"
    logger.info("parse: начало обработки файла %s", filename)

    if file.content_type not in ("application/pdf", "application/octet-stream") and not (
        filename
    ).lower().endswith(".pdf"):
        logger.warning("parse: отклонён файл (не PDF): %s", filename)
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
        logger.info("parse: PDF сохранён во временный файл, размер %s байт", len(content))

        settings = get_settings()
        logger.info("parse: конвертация PDF в изображения страниц (DPI=%s)...", settings.pdf_dpi)
        page_images = pdf_to_images(tmp_path, dpi=settings.pdf_dpi)
        if not page_images:
            logger.error("parse: в PDF нет страниц или конвертация не удалась")
            return JSONResponse(
                status_code=400,
                content={"error": "В PDF нет страниц или не удалось конвертировать"},
            )
        num_pages = len(page_images)
        logger.info("parse: получено страниц: %s", num_pages)

        all_elements: List[Dict[str, Any]] = []
        pages_for_ui: List[Dict[str, Any]] = []

        for i, img_bytes in enumerate(page_images):
            page_num = i + 1
            logger.info("parse: обработка страницы %s из %s...", page_num, num_pages)
            result = run_ocr_page(img_bytes, page_num)
            elements = result["elements"]
            rotation_degrees = result.get("page_rotation_degrees", 0) or 0
            all_elements.extend(elements)
            logger.info("parse: страница %s — распознано элементов: %s, поворот: %s°", page_num, len(elements), rotation_degrees)

            # Рисуем bbox поверх изображения на бэкенде
            img_with_boxes = _draw_bboxes_on_image(img_bytes, elements)

            pages_for_ui.append({
                "page": page_num,
                "image_base64": base64.b64encode(img_with_boxes).decode("ascii"),
                "elements": elements,
                "rotation_degrees": rotation_degrees,
            })

        logger.info("parse: формирование markdown...")
        markdown = document_to_markdown(all_elements)

        logger.info("parse: готово. Файл=%s, страниц=%s, элементов=%s", filename, num_pages, len(all_elements))
        return {
            "filename": filename,
            "structure": all_elements,
            "markdown": markdown,
            "pages": pages_for_ui,
            "num_pages": num_pages,
        }
    except Exception as exc:
        logger.exception("parse: ошибка обработки файла %s: %s", filename, exc)
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка обработки: {exc!s}"},
        )
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
