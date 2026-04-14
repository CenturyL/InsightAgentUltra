from __future__ import annotations

"""把检索结果文档转换为轻量引用记录的辅助函数。"""

from langchain_core.documents import Document


def build_citations(docs: list[Document]) -> list[dict]:
    """标准化文档元数据，方便后续统一渲染 citations。"""
    citations: list[dict] = []
    for idx, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        citations.append(
            {
                "id": f"ref_{idx}",
                "source": metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("title")
                or "unknown",
                "page": metadata.get("page"),
                "section": metadata.get("section_path"),
                "preview": doc.page_content[:160],
            }
        )
    return citations


def format_citations(citations: list[dict]) -> str:
    """把引用列表渲染成纯文本，供 prompt 和 fallback 答案使用。"""
    if not citations:
        return "无"

    lines = []
    for citation in citations:
        source = citation.get("source", "unknown")
        page = citation.get("page")
        section = citation.get("section")
        suffix = []
        if page is not None:
            suffix.append(f"page={page}")
        if section:
            suffix.append(f"section={section}")
        label = f"{source} ({', '.join(suffix)})" if suffix else str(source)
        lines.append(f"- {label}")
    return "\n".join(lines)
