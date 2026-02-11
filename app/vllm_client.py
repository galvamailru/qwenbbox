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

    logger.info("vLLM: отправка страницы %s в модель %s...", page_num, settings.vllm_model)
    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=settings.vllm_max_tokens,
        timeout=settings.vllm_timeout_seconds,
    )
    choice = response.choices[0] if response.choices else None
    if not choice or not getattr(choice, "message", None):
        logger.warning("vLLM: страница %s — пустой ответ модели", page_num)
        return "[]"
    raw = getattr(choice.message, "content", None) or ""
    logger.info("vLLM: страница %s — ответ получен, длина %s символов", page_num, len(raw))
    return raw.strip()


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    """Extract JSON array from model output (may be wrapped in markdown or text)."""
    raw = raw.strip()
    # Strip markdown code block
    for pattern in (r"```(?:json)?\s*([\s\S]*?)\s*```", r"\[[\s\S]*\]"):
        m = re.search(pattern, raw)
        if m:
            try:
                s = m.group(1) if "```" in pattern else m.group(0)
                return json.loads(s)
            except json.JSONDecodeError:
                pass
    # Try full string
    if raw.startswith("["):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return []


def run_ocr_page(image_png_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
    """
    Send one page image to vLLM, parse JSON array of elements.
    Adds "page" to each element and normalizes bbox.
    """
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
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
