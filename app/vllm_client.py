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


SYSTEM_PROMPT = """You are a document OCR and layout analysis system. For the given document page image, output a JSON array of elements. Each element must have:
- "type": one of "text", "image", "table", "stamp", "signature"
- "bbox": [x1, y1, x2, y2] in normalized coordinates 0-1000 (width 1000, height 1000)
- "text" or "content": recognized text or short description for images/stamps/signatures

Output ONLY valid JSON array, no markdown code block, no explanation. Example:
[{"type":"text","bbox":[100,50,900,120],"text":"Document title"},{"type":"table","bbox":[80,200,920,500],"text":"| A | B |\\n|--|--|\\n| 1 | 2 |"}]"""

USER_PROMPT_TEMPLATE = "Analyze this document page and return the JSON array of elements (text, images, tables, stamps, signatures) with bbox in 0-1000 scale and text/content."


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


def run_ocr_page(image_png_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
    """
    Send one page image to vLLM, parse JSON array of elements.
    Adds "page" to each element and normalizes bbox.
    """
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    logger.info("vLLM: страница %s — размер PNG %s байт, base64 %s символов", page_num, len(image_png_bytes), len(b64))
    raw = _call_vllm_chat(b64, page_num)
    items = _parse_json_array(raw)
    if not items and raw.strip():
        logger.warning("vLLM: страница %s — не удалось распарсить JSON из ответа (%s символов)", page_num, len(raw))
    for el in items:
        el["page"] = page_num
        if "content" in el and "text" not in el:
            el["text"] = el["content"]
    return items


def run_ocr_all_pages(page_images: List[bytes]) -> List[Dict[str, Any]]:
    """Run OCR for each page; return concatenated list of elements with page numbers."""
    all_elements: List[Dict[str, Any]] = []
    for i, img_bytes in enumerate(page_images):
        page_num = i + 1
        elements = run_ocr_page(img_bytes, page_num)
        all_elements.extend(elements)
    return all_elements
