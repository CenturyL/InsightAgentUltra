from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from local_agent_api.core.config import settings
from local_agent_api.core.embedding import get_embedding_model, get_reranker_model
from local_agent_api.retrieval.citation import build_citations


@dataclass
class RetrievalBundle:
    query: str
    docs: list[Document]
    parent_docs: list[Document]
    context_text: str
    citations: list[dict]
    applied_filters: dict[str, Any]


SearchStrategy = Literal[
    "dense_only",
    "dense_rerank",
    "hybrid_only",
    "hybrid_rerank",
]


def get_vector_store() -> Chroma:
    embedding_model = get_embedding_model()
    return Chroma(
        collection_name="local_knowledge",
        embedding_function=embedding_model,
        persist_directory=settings.VECTOR_STORE_PATH,
    )

# 推断metadata
def infer_metadata_filters(query: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}

    region_keywords = ("上海", "北京", "深圳", "广州", "苏州", "杭州")
    matched_region = next((region for region in region_keywords if region in query), None)
    if matched_region:
        filters["region"] = matched_region

    year_match = re.search(r"(20\d{2})", query)
    if year_match:
        filters["year"] = year_match.group(1)

    if "pdf" in query.lower() or "附件" in query:
        filters["source_type"] = "pdf"

    return filters


def _to_chroma_where(metadata_filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metadata_filters:
        return None

    clauses: list[dict[str, Any]] = []
    for key, value in metadata_filters.items():
        if key.startswith("_"):
            continue
        if value is None or value == "":
            continue
        clauses.append({key: value})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}

# lexical用的关键词打分函数
def _keyword_score(query: str, doc: Document) -> int:
    tokens = [token for token in re.split(r"[\s,，。；;：:\n]+", query) if len(token) >= 2]
    content = doc.page_content
    return sum(1 for token in tokens if token in content)


def _merge_documents(primary: list[Document], secondary: list[Document], limit: int) -> list[Document]:
    merged: list[Document] = []
    seen = set()

    for doc in [*primary, *secondary]:
        metadata = doc.metadata or {}
        key = (
            metadata.get("source"),
            metadata.get("page"),
            doc.page_content[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break

    return merged


def compute_file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _parent_store_dir() -> Path:
    path = Path(settings.PARENT_STORE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parent_store_file(file_hash: str) -> Path:
    return _parent_store_dir() / f"{file_hash}.json"


def _split_parent_id(parent_id: str) -> tuple[str, str] | None:
    if not parent_id or "-" not in parent_id:
        return None
    file_hash, block_id = parent_id.rsplit("-", 1)
    return file_hash, block_id


def _persist_parent_documents(parent_docs: list[Document], file_hash: str) -> None:
    if not parent_docs:
        return

    payload = {
        "file_hash": file_hash,
        "documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            for doc in parent_docs
        ],
    }
    _parent_store_file(file_hash).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_parent_documents_by_hash(file_hash: str) -> list[Document]:
    store_file = _parent_store_file(file_hash)
    if not store_file.exists():
        return []

    payload = json.loads(store_file.read_text(encoding="utf-8"))
    documents = payload.get("documents", [])
    return [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {}) or {},
        )
        for item in documents
        if item.get("page_content")
    ]


def get_parent_documents(parent_ids: list[str]) -> list[Document]:
    ordered_parent_ids: list[str] = []
    seen_parent_ids: set[str] = set()
    for parent_id in parent_ids:
        if not parent_id or parent_id in seen_parent_ids:
            continue
        seen_parent_ids.add(parent_id)
        ordered_parent_ids.append(parent_id)

    # parent_id -> parent 文本的映射，避免重复读文件
    parent_lookup: dict[str, Document] = {}
    loaded_hashes: set[str] = set()
    for parent_id in ordered_parent_ids:
        parsed = _split_parent_id(parent_id)
        if not parsed:
            continue
        file_hash, _ = parsed
        if file_hash in loaded_hashes:
            continue
        loaded_hashes.add(file_hash)
        for doc in _load_parent_documents_by_hash(file_hash):
            stored_parent_id = (doc.metadata or {}).get("parent_id")
            if stored_parent_id:
                parent_lookup[stored_parent_id] = doc

    return [parent_lookup[parent_id] for parent_id in ordered_parent_ids if parent_id in parent_lookup]

# 判断是不是标题
def _heading_level(line: str) -> str | None:
    patterns = (
        r"^第[一二三四五六七八九十百]+[章节条]",
        r"^[一二三四五六七八九十]+、",
        r"^（[一二三四五六七八九十]+）",
        r"^\([一二三四五六七八九十]+\)",
        r"^\d+\.",
        r"^\d+\.\d+",
    )
    return line if any(re.match(pattern, line) for pattern in patterns) else None


def _is_table_like(line: str) -> bool:
    return "|" in line or "\t" in line or re.search(r"\s{4,}", line) is not None


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.I)
    text = re.sub(r"</(p|div|li|tr|h1|h2|h3|h4|h5|h6|table|section|article)>", "\n", text, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _table_file_to_text(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    delimiter = "\t" if suffix == ".tsv" else ","
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row_idx, row in enumerate(reader, start=1):
            if not any(cell.strip() for cell in row):
                continue
            lines.append(f"row {row_idx}: " + " | ".join(cell.strip() for cell in row))
    return "\n".join(lines)


def _ocr_image_to_text(file_path: str) -> str:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore

        engine = RapidOCR()
        result, _ = engine(file_path)
        if result:
            lines = [item[1] for item in result if len(item) >= 2 and item[1]]
            text = "\n".join(lines).strip()
            if text:
                return text
    except Exception:
        pass

    tesseract = shutil.which("tesseract")
    if not tesseract:
        raise RuntimeError("tesseract binary not found, cannot OCR image file")

    for lang in ("chi_sim+eng", "eng"):
        result = subprocess.run(
            [tesseract, file_path, "stdout", "-l", lang, "--psm", "6"],
            capture_output=True,
            text=True,
        )
        text = result.stdout.strip()
        if result.returncode == 0 and text:
            return text

    raise RuntimeError("image OCR returned empty text")

# 加载文件 & 清理 & 转换langchain的doc格式
def _load_documents(file_path: str) -> list[Document]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(file_path, encoding="utf-8").load()
    if suffix in {".html", ".htm"}:
        raw_html = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        return [Document(page_content=_html_to_text(raw_html), metadata={"source_type": "html", "modality": "web"})]
    if suffix in {".csv", ".tsv"}:
        return [Document(page_content=_table_file_to_text(file_path), metadata={"source_type": "table", "modality": "table"})]
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return [Document(page_content=_ocr_image_to_text(file_path), metadata={"source_type": "image", "modality": "image"})]
    raise ValueError(f"unsupported knowledge file type: {suffix}")


def _annotate_metadata(
    metadata: dict[str, Any],
    file_path: str,
    file_hash: str,
    content: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    annotated = dict(metadata or {})
    annotated.update(metadata_overrides or {})
    annotated["source_hash"] = file_hash
    annotated.setdefault("source", file_path)
    annotated.setdefault("source_type", "pdf" if file_path.endswith(".pdf") else "text")
    annotated.setdefault("modality", "text")
    year_match = re.search(r"(20\d{2})", content)
    if year_match:
        annotated.setdefault("year", year_match.group(1))
    return annotated


# 先 block -- 识别“结构块”（标题、表格、正文）
def _extract_structured_blocks(text: str) -> list[dict[str, str]]:
    # 统一换行符 & 逐行切分成待处理的 行们lines[]
    normalized = re.sub(r"\r\n?", "\n", text)
    lines = [line.strip() for line in normalized.split("\n")] # 按 \n 切分成行，两段去空白
    # 预先定义好 blocks&缓存区
    blocks: list[dict[str, str]] = []
    current_title = "root"
    current_lines: list[str] = []
    current_type = "text"
    # 提交当前块并刷新缓存区
    def flush() -> None:
        nonlocal current_lines, current_type
        # 过滤掉空行
        content = "\n".join(line for line in current_lines if line).strip()
        # 若有内容就提交block
        if content:
            blocks.append(
                {
                    "title": current_title,
                    "content": content,
                    "block_type": current_type,
                }
            )
        # 清空还原缓冲区
        current_lines = []
        current_type = "text"
    # 逐行扫描
    for raw_line in lines:
        # 空行：当前有内容才加上空行
        if not raw_line:
            if current_lines:
                current_lines.append("")
            continue
        # 标题行：之前积累的先提交block，打上title（标题行本身不进 content）
        heading = _heading_level(raw_line)
        if heading:
            flush()
            current_title = heading
            continue

        if _is_table_like(raw_line):
        # 表格行：
            # 有内容未提交，先提交block，改type为table
            if current_type != "table" and current_lines:
                flush()
            current_type = "table"
        # 正文行：
            # 上一行是表格，先提交block
        elif current_type == "table" and current_lines:
            flush()
        # 只有标题和表格行才能走到这里（其他的continue掉了）
        current_lines.append(raw_line)

    flush()
    return blocks


# 切正文：
# 先按段落分，长段再切分，，保留overlap
def _split_text_block(doc: Document) -> list[Document]:
    content = doc.page_content.strip()
    if not content:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if not paragraphs:
        paragraphs = [content]

    chunks: list[Document] = []
    current_parts: list[str] = []
    current_length = 0
    overlap_tail = ""

    def flush() -> None:
        nonlocal current_parts, current_length, overlap_tail
        merged = "\n\n".join(current_parts).strip()
        if not merged:
            return
        metadata = dict(doc.metadata)
        metadata["chunk_strategy"] = "semantic_parent_child"
        chunks.append(Document(page_content=merged, metadata=metadata))
        overlap_tail = current_parts[-1] if current_parts else ""
        current_parts = []
        current_length = 0

    for paragraph in paragraphs:
        if len(paragraph) > 520:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=420,
                chunk_overlap=70,
                separators=["\n", "。", "；", "，", " ", ""],
            )
            for split in splitter.split_text(paragraph):
                metadata = dict(doc.metadata)
                metadata["chunk_strategy"] = "recursive_sentence"
                chunks.append(Document(page_content=split, metadata=metadata))
            continue

        projected = current_length + len(paragraph) + (2 if current_parts else 0)
        if projected > 520 and current_parts:
            flush()
            if overlap_tail:
                current_parts.append(overlap_tail)
                current_length = len(overlap_tail)

        current_parts.append(paragraph)
        current_length += len(paragraph) + (2 if len(current_parts) > 1 else 0)

    flush()
    return chunks


# 切表格
# 若干行组成一个chunk，保留overlap
def _split_table_block(doc: Document) -> list[Document]:
    rows = [line.strip() for line in doc.page_content.splitlines() if line.strip()]
    if not rows:
        return []

    chunks: list[Document] = []
    window_size = 8
    overlap = 2
    step = max(window_size - overlap, 1)
    for start in range(0, len(rows), step):
        window = rows[start:start + window_size]
        if not window:
            continue
        metadata = dict(doc.metadata)
        metadata["chunk_strategy"] = "table_row_window"
        metadata["row_window"] = f"{start + 1}-{start + len(window)}"
        chunks.append(Document(page_content="\n".join(window), metadata=metadata))
        if start + window_size >= len(rows):
            break
    return chunks


# chunk 主函数（parent-child）
# 调用_extract_structured_blocks后
# 先切成 parent block，打metadata，再切成 child chunk
def _build_chunk_documents(
    raw_docs: list[Document],
    file_path: str,
    file_hash: str,
    metadata_overrides: dict[str, Any] | None = None,
) -> tuple[list[Document], list[Document]]:
    """
    先构建 parent block，再构建 child chunk。

    检索粒度真正落到向量库里的不是整篇原文档，也不是 parent block，
    而是这里生成的 child_docs。

    后续 add_documents(child_docs) 时，Chroma 会对每个 child chunk 单独调用
    embedding_function：
        child chunk 文本 -> embedding 模型内部 tokenizer -> 单个 chunk 向量

    因此“一个 chunk 会被切成很多 token”并不意味着会存很多检索单元；
    对向量库来说，检索单元仍然是这个 chunk 本身，因为最终只会得到一个
    固定维度向量与该 chunk 绑定。
    """

    # 构建 parent 块（切上_extract构造的block）
    parent_docs: list[Document] = []
    for doc in raw_docs:
        base_metadata = doc.metadata or {}
        blocks = _extract_structured_blocks(doc.page_content)
        if not blocks:
            blocks = [{"title": "root", "content": doc.page_content, "block_type": "text"}]
        # 遍历大block块，补metadata
        for idx, block in enumerate(blocks, start=1):
            metadata = _annotate_metadata(
                base_metadata,
                file_path,
                file_hash,
                block["content"],
                metadata_overrides=metadata_overrides,
            )
            metadata["section_path"] = block["title"]
            metadata["block_type"] = block["block_type"]
            metadata["parent_id"] = f"{file_hash}-{idx}" # parent编号为：hash-块序号
            metadata["chunk_strategy"] = "header_aware_parent"
            # 打包成langchain的doc
            parent_docs.append(Document(page_content=block["content"], metadata=metadata))
    
    # 构建child chunk（切 parent block）
    child_docs: list[Document] = []
    for parent in parent_docs:
        block_type = parent.metadata.get("block_type", "text")
        if block_type == "table":
            child_docs.extend(_split_table_block(parent))
        else:
            child_docs.extend(_split_text_block(parent))

    for child in child_docs:
        child.metadata["parent_section"] = child.metadata.get("section_path", "root")
    return parent_docs, child_docs

# 上传文件--主函数
def process_and_store_document(file_path: str, metadata_overrides: dict[str, Any] | None = None) -> int:
    # 计算hash
    file_hash = compute_file_hash(file_path)
    # 去重
    vector_store = get_vector_store()
    existing = vector_store.get(limit=10000, include=["metadatas"])
    existing_metadatas = existing.get("metadatas", []) if existing else []
    duplicated = any(
        (metadata or {}).get("source_hash") == file_hash or (metadata or {}).get("source") == file_path
        for metadata in existing_metadatas
    )
    if duplicated:
        print(f"⚠️ 文件已存在于知识库（hash={file_hash[:8]}...），跳过重复入库。")
        return 0
    # 加载文档内容 & 切片 & 元数据挂载
    docs = _load_documents(file_path)
    parent_docs, splits = _build_chunk_documents(
        docs,
        file_path,
        file_hash,
        metadata_overrides=metadata_overrides,
    )
    # Embedding & Persisting
    # 这里不会先在项目代码里手动 tokenizer。
    # Chroma.add_documents(...) 会逐个读取 split/page_content，并调用
    # embedding_function=get_embedding_model() 生成“每个 chunk 一个向量”后再落库。
    vector_store.add_documents(splits)
    _persist_parent_documents(parent_docs, file_hash)
    return len(splits)


def get_documents_by_source(source: str, limit: int = 12) -> list[Document]:
    vector_store = get_vector_store()
    for field_name in ("source", "upload_name"):
        payload = vector_store.get(
            where={field_name: source},
            limit=limit,
            include=["documents", "metadatas"],
        )
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])
        docs = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(documents, metadatas)
        ]
        if docs:
            return docs
    return []


def has_document_source(file_path: str) -> bool:
    vector_store = get_vector_store()
    existing = vector_store.get(limit=10000, include=["metadatas"])
    existing_metadatas = existing.get("metadatas", []) if existing else []
    return any((metadata or {}).get("source") == file_path for metadata in existing_metadatas)

# 基础的向量检索, 去Chroma做similarity search，初筛
def dense_search_knowledge(
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    metadata_filters: dict[str, Any] | None = None,
) -> list[Document]:
    vector_store = get_vector_store()
    search_kwargs: dict[str, Any] = {"k": max(k, candidate_k)}
    where = _to_chroma_where(metadata_filters)
    if where:
        search_kwargs["filter"] = where
    # similarity_search(query) 时，query 也不是在项目层先手动切 token；
    # Chroma 会调用同一个 embedding_function，把整条 query 文本编码成一个
    # query 向量，再和库里的 chunk 向量做相似度检索。
    return vector_store.similarity_search(query, **search_kwargs)

# 向量rerank
def dense_rerank_search_knowledge(
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    metadata_filters: dict[str, Any] | None = None,
) -> list[Document]:
    base_docs = dense_search_knowledge(
        query,
        k=k,
        candidate_k=candidate_k,
        metadata_filters=metadata_filters,
    )
    reranker_model = get_reranker_model()
    compressor = CrossEncoderReranker(model=reranker_model, top_n=k)
    reranked = compressor.compress_documents(base_docs, query)
    return list(reranked)

# 检索主函数
def search_knowledge(
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    metadata_filters: dict[str, Any] | None = None,
    strategy: SearchStrategy = "hybrid_rerank",
) -> list[Document]:
    # 最近上传文件
    recent_upload_source = (metadata_filters or {}).get("_recent_upload_source")
    dense_docs = dense_search_knowledge(
        query,
        k=k,
        candidate_k=candidate_k,
        metadata_filters=metadata_filters,
    )
    priority_docs = get_documents_by_source(recent_upload_source, limit=max(candidate_k, k * 2)) if recent_upload_source else []

    if strategy == "dense_only":
        # 直接 priority_docs 和 dense_docs 去重合并截取返回
        return _merge_documents(priority_docs, dense_docs, limit=max(k, candidate_k))[:k]
    if strategy == "dense_rerank":
        # 对dense_docs进行rerank，再拼priority_docs
        reranked = dense_rerank_search_knowledge(
            query,
            k=k,
            candidate_k=candidate_k,
            metadata_filters=metadata_filters,
        )
        return _merge_documents(priority_docs, reranked, limit=max(k, candidate_k))[:k]

    lexical_docs: list[Document] = []
    if strategy in {"hybrid_only", "hybrid_rerank"}:
        lexical_docs = lexical_search_knowledge(
            query,
            k=k,
            candidate_k=max(candidate_k * 2, 20),
            metadata_filters=metadata_filters,
        )

    merged = _merge_documents(priority_docs, _merge_documents(dense_docs, lexical_docs, limit=max(candidate_k, k * 2)), limit=max(candidate_k, k * 2))
    if strategy == "hybrid_only":
        return merged[:k]

    reranker_model = get_reranker_model()
    compressor = CrossEncoderReranker(model=reranker_model, top_n=k)
    reranked = compressor.compress_documents(merged, query)
    return list(reranked)

# 关键词召回补充，按词命中数排序，防止一些精确术语被 dense 检索漏掉
def lexical_search_knowledge(
    query: str,
    k: int = 3,
    candidate_k: int = 30,
    metadata_filters: dict[str, Any] | None = None,
) -> list[Document]:
    vector_store = get_vector_store()
    payload = vector_store.get(
        where=_to_chroma_where(metadata_filters),
        limit=candidate_k,
        include=["documents", "metadatas"],
    )
    documents = payload.get("documents", [])
    metadatas = payload.get("metadatas", [])

    candidates = [
        Document(page_content=content, metadata=metadata or {})
        for content, metadata in zip(documents, metadatas)
    ]
    ranked = sorted(candidates, key=lambda doc: _keyword_score(query, doc), reverse=True)
    return [doc for doc in ranked[:k] if _keyword_score(query, doc) > 0]


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# RAG主函数
def retrieve_knowledge_bundle(
    query: str,
    k: int = 3,
    candidate_k: int = 15,
    metadata_filters: dict[str, Any] | None = None,
    strategy: SearchStrategy = "hybrid_rerank",
) -> RetrievalBundle:
    # 生成过滤条件
    applied_filters = metadata_filters or infer_metadata_filters(query)
    # 执行检索，拿到child chunk & 获取parent id
    docs = search_knowledge(
        query,
        k=k,
        candidate_k=candidate_k,
        metadata_filters=applied_filters,
        strategy=strategy,
    )
    parent_ids = [
        (doc.metadata or {}).get("parent_id")
        for doc in docs
        if (doc.metadata or {}).get("parent_id")
    ]
    # 捞回 parent block，最终 = parent-first | child-fallback。
    parent_docs = get_parent_documents(parent_ids)
    context_docs = parent_docs or docs
    return RetrievalBundle(
        query=query,
        docs=docs,
        parent_docs=parent_docs,
        context_text=format_docs(context_docs),
        citations=build_citations(context_docs),
        applied_filters=applied_filters,
    )
