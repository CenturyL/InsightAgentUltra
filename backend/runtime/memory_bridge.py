from __future__ import annotations

"""统一桥接短期/长期/Markdown 显式记忆。"""

from pathlib import Path
from typing import Iterable

from backend.core.config import settings
from backend.core.memory import long_term_memory


def search_long_term_memory_text(user_id: str, query: str, k: int = 3) -> list[str]:
    """
    检索用户级长期记忆，失败时静默返回空。

    这是“长期记忆读链路”的统一入口：
    - ReAct 主循环在 middleware 中会走到这里
    - PAE 子流程在构造 runtime context 时也会走到这里
    """
    if not user_id or not settings.POSTGRES_URL:
        return []
    try:
        return long_term_memory.search(user_id, query, k=k)
    except Exception:
        return []


def _markdown_memory_files() -> list[Path]:
    # 显式记忆是文件化存储的，便于人工查看和直接编辑。
    # 当前约定：
    # - memory/MEMORY.md         : 稳定、长期的显式记忆
    # - memory/daily/*.md        : 运行日志/日常笔记类显式记忆
    root = Path(settings.WORKSPACE_ROOT)
    candidates: list[Path] = []
    memory_file = root / "memory" / "MEMORY.md"
    if memory_file.exists():
        candidates.append(memory_file)
    daily_dir = root / "memory" / "daily"
    if daily_dir.exists():
        candidates.extend(sorted(daily_dir.glob("*.md")))
    return candidates


def _score_markdown(query: str, content: str) -> int:
    tokens = [token for token in query.replace("\n", " ").split() if len(token) >= 2]
    if not tokens:
        return 0
    return sum(1 for token in tokens if token in content)


def search_markdown_memory_text(query: str, k: int = 2) -> list[str]:
    """
    对 Markdown 显式记忆做轻量关键词检索，返回片段文本。

    这不是向量检索，而是更轻量的关键词命中。
    目的不是替代 pgvector，而是让人工可编辑的 MEMORY.md 也能参与上下文构造。
    """
    hits: list[tuple[int, str]] = []
    for file_path in _markdown_memory_files():
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        score = _score_markdown(query, content)
        if score <= 0:
            continue
        snippet = content[:900]
        hits.append((score, f"[{file_path.name}]\n{snippet}"))
    hits.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in hits[:k]]


def format_memory_sections(
    long_term_items: Iterable[str],
    markdown_items: Iterable[str],
) -> str:
    """
    把两类记忆统一格式化成 prompt 片段。

    这里相当于“记忆层 -> prompt 层”的桥：
    上游只关心查出来什么，下游只关心最终怎么拼进上下文。
    """
    parts: list[str] = []
    long_term_items = list(long_term_items)
    markdown_items = list(markdown_items)
    if long_term_items:
        parts.append("【长期记忆】\n" + "\n".join(f"- {item}" for item in long_term_items))
    if markdown_items:
        parts.append("【显式记忆（Markdown）】\n" + "\n\n".join(markdown_items))
    return "\n\n".join(parts)
