"""
Call vLLM (Qwen-VL) for document OCR: one image per request, structured JSON output.
Expects OpenAI-compatible API: POST /v1/chat/completions with image_url (base64).
"""
import base64
import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a deterministic document OCR and layout analysis system.

For the given SINGLE document page image, you MUST output ONE JSON object with EXACTLY the following shape:
{
  "page_rotation_degrees": <number>,
  "elements": [
    {
      "type": "<string: text|image|table|stamp|signature>",
      "bbox": [x1, y1, x2, y2],
      "text": "<string, may be empty. For type='stamp' ALWAYS include the text inside the stamp as fully as possible, not just a generic label.>"
    },
    ...
  ]
}

Requirements:
- ALWAYS include "page_rotation_degrees" as a number (can be integer or float). This is the estimated tilt / rotation of the whole page in DEGREES:
  - 0  = page looks horizontal and upright
  - >0 = rotated CLOCKWISE (even small deskew like 1–5 degrees must be reflected)
  - <0 = rotated COUNTER-CLOCKWISE
- ALWAYS include "elements" as an array (may be empty if nothing is found).
- Each element in "elements":
  - "type": one of "text", "image", "table", "stamp", "signature" (lowercase).
  - "bbox": [x1, y1, x2, y2] — NORMALIZED coordinates in the SAME coordinate system as the input image, with origin at the top-left corner, x to the right, y down. The normalization is by page width/height so that:
    - x1 = 0 and x2 = 1000 mean the very left and very right of the page
    - y1 = 0 and y2 = 1000 mean the very top and very bottom of the page
    All four numbers MUST be in the range [0, 1000].
  - "text": recognized text for text/table/signature, or a SHORT description / label for images, stamps. For elements with type="stamp", you MUST extract and return the full readable text that is inside the stamp (if any text is visible). Do not replace it with just a generic label; include all legible words from inside the stamp.
- Do NOT include any other top-level fields.
- Do NOT wrap the result in markdown or comments.
- Do NOT output any explanations, natural language, or additional text. ONLY the JSON object.

Example of a VALID response:
{
  "page_rotation_degrees": 2.5,
  "elements": [
    {"type": "text", "bbox": [100, 50, 900, 120], "text": "Document title"},
    {"type": "table", "bbox": [80, 200, 920, 500], "text": "| A | B |\\n|--|--|\\n| 1 | 2 |"}
  ]
}"""

USER_PROMPT_TEMPLATE = (
    "Analyze ONLY this single page image. "
    "Return ONE JSON object with 'page_rotation_degrees' (page tilt in degrees, 0 if visually horizontal, "
    "positive for clockwise tilt, negative for counter-clockwise, including small scan skew like 1–5 degrees) "
    "and 'elements' (array of objects with type, bbox in NORMALIZED coordinates from 0 to 1000 relative to page width/height, and text). "
    "Do not add any prose, comments or markdown — only the JSON."
)


def _call_vllm_chat(image_base64: str, page_num: int) -> str:
    settings = get_settings()
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Install openai package: pip install openai")

    client = OpenAI(
        base_url=settings.vllm_base_url.rstrip("/"),
        api_key=settings.vllm_api_key or "dummy",
    )

    content: List[Dict[str, Any]] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"},
        },
        {"type": "text", "text": USER_PROMPT_TEMPLATE},
    ]

    payload_size_kb = (len(image_base64) * 3 // 4) // 1024  # приблизительный размер PNG в КБ
    logger.info(
        "vLLM: отправка страницы %s в модель %s (размер изображения ~%s КБ, таймаут %s с)...",
        page_num, settings.vllm_model, payload_size_kb, settings.vllm_timeout_seconds,
    )
    try:
        response = client.chat.completions.create(
            model=settings.vllm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=settings.vllm_max_tokens,
            timeout=settings.vllm_timeout_seconds,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as e:
        logger.exception(
            "vLLM: страница %s — ошибка запроса: %s: %s",
            page_num, type(e).__name__, e,
        )
        raise
    choice = response.choices[0] if response.choices else None
    if not choice or not getattr(choice, "message", None):
        logger.warning("vLLM: страница %s — пустой ответ модели", page_num)
        return "[]"
    raw = getattr(choice.message, "content", None) or ""
    logger.info("vLLM: страница %s — ответ получен, длина %s символов", page_num, len(raw))
    return raw.strip()


def _extract_json_string(raw: str) -> Optional[str]:
    """Extract JSON object or array string from model output (markdown, prefix text, etc.)."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    if raw.startswith("{") or raw.startswith("["):
        return raw
    start = raw.find("{")
    if start == -1:
        start = raw.find("[")
    if start == -1:
        return None
    return raw[start:]


def _extract_json_array_string(raw: str) -> Optional[str]:
    """Extract string that should be a JSON array from model output (markdown, prefix text, etc.)."""
    raw = raw.strip()
    # 1) Markdown code block ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    # 2) Raw string is the array
    if raw.startswith("["):
        return raw
    # 3) Find first '[' and then matching ']' by bracket count (skip "]" inside strings roughly)
    start = raw.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    quote = None
    for i in range(start, len(raw)):
        c = raw[i]
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if not in_string and c in ("'", '"'):
            in_string = True
            quote = c
            continue
        if in_string and c == quote:
            in_string = False
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    # 4) Fallback: from first '[' to last ']'
    last = raw.rfind("]")
    if last != -1 and last > start:
        return raw[start : last + 1]
    return None


def _repair_truncated_json_array(extracted: str) -> Optional[List[Dict[str, Any]]]:
    """If the model response was cut by max_tokens, find last complete object and close the array."""
    s = extracted.strip()
    if not s.startswith("["):
        return None
    # Track depth: array [ and object {. Find last position where we have }, and depth is 1,0.
    array_depth = 0
    object_depth = 0
    in_string = False
    escape = False
    quote = None
    last_complete_end = -1
    i = 0
    while i < len(s):
        c = s[i]
        if escape:
            escape = False
            i += 1
            continue
        if in_string:
            if c == "\\":
                escape = True
            elif c == quote:
                in_string = False
            i += 1
            continue
        if c in ("'", '"'):
            in_string = True
            quote = c
            i += 1
            continue
        if c == "[":
            array_depth += 1
        elif c == "]":
            array_depth -= 1
        elif c == "{":
            object_depth += 1
        elif c == "}":
            object_depth -= 1
            if array_depth == 1 and object_depth == 0:
                last_complete_end = i
        i += 1
    if last_complete_end == -1:
        return None
    repaired = s[: last_complete_end + 1] + "\n]"
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    """Extract JSON array from model output (may be wrapped in markdown or text)."""
    raw = raw.strip()
    if not raw:
        return []
    extracted = _extract_json_array_string(raw)
    if extracted is None:
        if raw:
            logger.info("vLLM: фрагмент ответа (парсинг не удался): %.800s", raw)
        return []
    # Try parse; allow trailing comma in array (replace ",]") for robustness
    normalized = extracted.replace(",]", "]").replace(",}", "}")
    try:
        return json.loads(normalized)
    except json.JSONDecodeError as e:
        repaired = _repair_truncated_json_array(extracted)
        if repaired is not None:
            logger.info("vLLM: ответ обрезан по токенам, использовано %s полных элементов", len(repaired))
            return repaired
        logger.info("vLLM: фрагмент ответа (JSON error %s): %.800s", e, extracted)
        return []


def _extract_rotation_from_raw(raw: str) -> float:
    """Try to get page_rotation_degrees from raw string (e.g. truncated object)."""
    m = re.search(r'"page_rotation_degrees"\s*:\s*(-?\d+(?:\.\d+)?)', raw)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return 0.0


def _parse_page_response(raw: str) -> tuple[List[Dict[str, Any]], float]:
    """
    Parse model response: object {page_rotation_degrees, elements} or plain array.
    Returns (elements, rotation_degrees). Each page is sent alone — no previous pages in context.
    """
    raw = raw.strip()
    if not raw:
        return [], 0.0
    extracted = _extract_json_string(raw)
    if not extracted:
        return _parse_json_array(raw), 0.0
    normalized = extracted.replace(",]", "]").replace(",}", "}")
    try:
        data = json.loads(normalized)
    except json.JSONDecodeError:
        elements, rotation = _parse_page_response_fallback(raw, extracted)
        return elements, rotation
    if isinstance(data, dict):
        elements = data.get("elements")
        rotation = float(data.get("page_rotation_degrees", 0) or 0)
        if isinstance(elements, list):
            if rotation != 0:
                logger.info("vLLM: определён поворот страницы: %s градусов", rotation)
            return elements, rotation
        if elements is not None:
            return [], rotation
        return [], 0.0
    if isinstance(data, list):
        return data, 0.0
    return [], 0.0


def _parse_page_response_fallback(raw: str, extracted: str) -> tuple[List[Dict[str, Any]], float]:
    """
    When full object parse failed (e.g. truncated): try to get 'elements' array from raw.
    """
    rotation = _extract_rotation_from_raw(raw)
    idx = extracted.find('"elements"')
    if idx == -1:
        idx = extracted.find("'elements'")
    if idx == -1:
        idx = extracted.find("elements")
    if idx != -1:
        arr_start = extracted.find("[", idx)
        if arr_start != -1:
            arr_str = extracted[arr_start:]
            normalized = arr_str.replace(",]", "]").replace(",}", "}")
            try:
                return json.loads(normalized), rotation
            except json.JSONDecodeError:
                repaired = _repair_truncated_json_array(arr_str)
                if repaired is not None:
                    logger.info("vLLM: ответ обрезан (объект), использовано %s элементов из 'elements'", len(repaired))
                    return repaired, rotation
    elements = _parse_json_array(raw)
    return elements, rotation


def run_ocr_page(image_png_bytes: bytes, page_num: int) -> Dict[str, Any]:
    """
    Send one page image to vLLM, parse response (elements + optional page_rotation_degrees).
    Returns {"elements": [...], "page_rotation_degrees": float}.
    """
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    logger.info("vLLM: страница %s — размер PNG %s байт, base64 %s символов", page_num, len(image_png_bytes), len(b64))
    raw = _call_vllm_chat(b64, page_num)
    items, rotation = _parse_page_response(raw)
    if not items and raw.strip():
        logger.warning("vLLM: страница %s — не удалось распарсить JSON из ответа (%s символов)", page_num, len(raw))
    for el in items:
        el["page"] = page_num
        if "content" in el and "text" not in el:
            el["text"] = el["content"]
    return {"elements": items, "page_rotation_degrees": rotation}


def run_ocr_all_pages(page_images: List[bytes]) -> List[Dict[str, Any]]:
    """Run OCR for each page; return concatenated list of elements with page numbers."""
    all_elements: List[Dict[str, Any]] = []
    for i, img_bytes in enumerate(page_images):
        page_num = i + 1
        result = run_ocr_page(img_bytes, page_num)
        all_elements.extend(result["elements"])
    return all_elements
