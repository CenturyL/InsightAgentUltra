from __future__ import annotations

"""统一桥接短期/长期/Insight 显式记忆。"""

from typing import Iterable

from backend.core.config import settings
from backend.core.memory import long_term_memory
from backend.services.runtime_assets_service import load_insight


def search_long_term_memory_text(user_id: str, query: str, k: int = 3) -> list[str]:
    if not user_id or not settings.POSTGRES_URL:
        return []
    try:
        return long_term_memory.search(user_id, query, k=k)
    except Exception:
        return []


def _score_markdown(query: str, content: str) -> int:
    tokens = [token for token in query.replace("\n", " ").split() if len(token) >= 2]
    if not tokens:
        return 0
    return sum(1 for token in tokens if token in content)


def search_markdown_memory_text(query: str, k: int = 2, user_id: str = "") -> list[str]:
    """
    对 insight.md 中的 Memory 部分做轻量关键词检索。
    """
    content = load_insight(user_id).strip()
    if not content:
        return []
    score = _score_markdown(query, content)
    if score <= 0:
        return []
    snippet = content[:900]
    return [f"[insight.md]\n{snippet}"]


def format_memory_sections(
    long_term_items: Iterable[str],
    markdown_items: Iterable[str],
) -> str:
    parts: list[str] = []
    long_term_items = list(long_term_items)
    markdown_items = list(markdown_items)
    if long_term_items:
        parts.append("【长期记忆】\n" + "\n".join(f"- {item}" for item in long_term_items))
    if markdown_items:
        parts.append("【显式记忆（Insight）】\n" + "\n\n".join(markdown_items))
    return "\n\n".join(parts)
