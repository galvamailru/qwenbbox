"""Document structure: elements with bbox for JSON and markdown export."""
from typing import Any, Dict, List, Literal, Optional

# Element types we expect from the model
ElementType = Literal["text", "image", "table", "stamp", "signature"]

# bbox: [x1, y1, x2, y2] in normalized 0-1000 (Qwen-VL scale)
# Frontend всегда масштабирует эти координаты под фактический размер изображения


def document_to_markdown(elements: List[Dict[str, Any]], page_separator: str = "\n\n---\n\n") -> str:
    """Build markdown from structured elements (by page, then by order)."""
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for el in elements:
        p = el.get("page") if el.get("page") is not None else 1
        by_page.setdefault(p, []).append(el)

    parts: List[str] = []
    for page_num in sorted(by_page.keys()):
        page_blocks: List[str] = []
        for el in by_page[page_num]:
            el_type = (el.get("type") or "text").lower()
            text = (el.get("text") or el.get("content") or "").strip()
            if el_type == "table":
                page_blocks.append(text if text else "*(таблица)*")
            elif el_type == "image":
                page_blocks.append(text if text else "*(изображение)*")
            elif el_type == "stamp":
                page_blocks.append(f"*[Печать: {text or '—'}]*")
            elif el_type == "signature":
                page_blocks.append(f"*[Подпись: {text or '—'}]*")
            else:
                if text:
                    page_blocks.append(text)
        parts.append("\n\n".join(page_blocks))

    return page_separator.join(parts)
