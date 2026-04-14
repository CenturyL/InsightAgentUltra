"""
长期记忆管理器 —— 基于 PostgreSQL + pgvector

存储跨会话、跨重启的用户级语义记忆（姓名、偏好、重要事项等关键事实）。
工作原理：
  写入：每轮对话结束后，用 LLM 从对话历史中提取关键事实，以向量形式存入 pg。
  读取：每次 LLM 调用前，用当前问题做语义搜索，把最相关的记忆注入 system_prompt。

与短期记忆（MemorySaver/PostgresSaver）的区别：
  短期记忆 = 同一 thread_id 内的完整对话历史（有 token 上限）
  长期记忆 = 跨 thread_id 提炼出的精华事实（永久保存、按需检索）
"""

import uuid
from typing import Optional

from langchain_postgres import PGVector
from langchain_core.documents import Document

from local_agent_api.core.config import settings
from local_agent_api.core.embedding import get_embedding_model


_ALLOWED_PREFIXES = (
    "用户姓名：",
    "用户称呼：",
    "用户身份：",
    "用户偏好：",
    "用户长期目标：",
    "用户长期约束：",
)

_FACT_TYPE_BY_PREFIX = {
    "用户姓名：": "name",
    "用户称呼：": "nickname",
    "用户身份：": "identity",
    "用户偏好：": "preference",
    "用户长期目标：": "goal",
    "用户长期约束：": "constraint",
}

_REJECT_SUBSTRINGS = (
    "无有效信息",
    "记住了AI",
    "AI的名字",
    "AI 的名字",
    "用户询问",
    "用户可能",
    "用户潜在需求",
    "用户方案",
    "工具执行",
    "天气",
    "当前时间",
)

_NAME_SUFFIX_PATTERN = ("你记住", "记住我", "记住", "呀", "啊", "哦", "哈", "吧")


def _is_identity_query(query: str) -> bool:
    text = " ".join((query or "").split()).strip()
    return any(token in text for token in ("我叫什么", "我叫", "我是谁", "我的名字", "名字", "称呼"))


def _detect_query_fact_types(query: str) -> list[str]:
    text = " ".join((query or "").split()).strip()
    wanted: list[str] = []
    if any(token in text for token in ("我叫什么", "我叫", "我的名字", "名字")):
        wanted.extend(["name", "nickname"])
    if any(token in text for token in ("称呼", "怎么叫我")):
        wanted.extend(["nickname", "name"])
    if any(token in text for token in ("我是谁", "我的身份", "我是做什么", "身份")):
        wanted.append("identity")
    if any(token in text for token in ("偏好", "喜欢", "习惯", "风格")):
        wanted.append("preference")
    if any(token in text for token in ("目标", "长期目标", "想达成")):
        wanted.append("goal")
    if any(token in text for token in ("约束", "限制", "不能", "不要")):
        wanted.append("constraint")
    deduped: list[str] = []
    for item in wanted:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _split_fact(text: str) -> tuple[str, str] | None:
    normalized = normalize_memory_fact(text)
    if not normalized:
        return None
    for prefix, fact_type in _FACT_TYPE_BY_PREFIX.items():
        if normalized.startswith(prefix):
            return fact_type, normalized[len(prefix):].strip()
    return None


def _token_overlap_score(query: str, value: str) -> int:
    query_text = " ".join((query or "").split()).strip().lower()
    value_text = " ".join((value or "").split()).strip().lower()
    if not query_text or not value_text:
        return 0
    score = 0
    if value_text and value_text in query_text:
        score += 4
    for token in query_text.replace("：", " ").replace(":", " ").split():
        if len(token) >= 2 and token in value_text:
            score += 2
    return score


def rerank_memory_facts(query: str, candidates: list[str], k: int) -> list[str]:
    wanted_types = _detect_query_fact_types(query)
    scored: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        normalized = normalize_memory_fact(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        split = _split_fact(normalized)
        if split is None:
            continue
        fact_type, value = split
        score = 100 - index
        if wanted_types:
            if fact_type in wanted_types:
                score += 100
                score += max(20 - wanted_types.index(fact_type) * 5, 0)
            else:
                score -= 20
        score += _token_overlap_score(query, value)
        if fact_type in {"name", "nickname"} and _is_identity_query(query):
            score += 50
        scored.append((score, index, normalized))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:k]]


def normalize_memory_fact(content: str) -> str | None:
    text = " ".join((content or "").split()).strip()
    if not text:
        return None
    if text.startswith(("用户姓名：", "用户称呼：")):
        prefix, value = text.split("：", 1)
        name = value.strip()
        for suffix in _NAME_SUFFIX_PATTERN:
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
        if not name:
            return None
        text = f"{prefix}：{name}"
    if any(bad in text for bad in _REJECT_SUBSTRINGS):
        return None
    if "AI" in text and "用户" not in text:
        return None
    if text in {"century", "世纪"}:
        return None
    if not text.startswith(_ALLOWED_PREFIXES):
        return None
    return text


class LongTermMemoryManager:
    """
    基于 pgvector 的长期记忆管理器（懒加载单例）。

    初始化时不立即连接数据库，首次 save/search 时才建立连接，
    确保 settings.POSTGRES_URL 未配置时不会在启动阶段崩溃。
    """

    def __init__(self):
        self._store: Optional[PGVector] = None

    @property
    def store(self) -> PGVector:
        # 懒加载：只有真正读/写长期记忆时才初始化 PGVector。
        # 这样即使没配数据库，系统启动也不会因为长期记忆模块而失败。
        if self._store is None:
            if not settings.POSTGRES_URL:
                raise RuntimeError(
                    "POSTGRES_URL 未配置，无法使用长期记忆。"
                    "请在 .env 中添加 POSTGRES_URL=postgresql://... 后重启。"
                )
            # langchain-postgres 要求 psycopg3 连接串格式
            conn = settings.POSTGRES_URL
            if conn.startswith("postgresql://") and "+psycopg" not in conn:
                conn = conn.replace("postgresql://", "postgresql+psycopg://", 1)

            self._store = PGVector(
                connection=conn,
                collection_name="agent_long_term_memory",
                embeddings=get_embedding_model(),
                use_jsonb=True,
                # 首次实例化时自动建表；pgvector extension 需手动执行：
                #   CREATE EXTENSION IF NOT EXISTS vector;
            )
        return self._store

    def save(self, user_id: str, content: str) -> None:
        """
        持久化一条长期记忆。

        调用来源通常是：
        - agent_service._extract_and_save_memories()
        - 对话结束后，从短期记忆里抽取稳定事实/偏好，再写入这里
        """
        normalized = normalize_memory_fact(content)
        if not normalized:
            return
        doc = Document(
            page_content=normalized,
            metadata={"user_id": user_id},
        )
        self.store.add_documents([doc], ids=[str(uuid.uuid4())])

    def search(self, user_id: str, query: str, k: int = 3) -> list[str]:
        """
        语义搜索：返回与 query 最相关的 k 条记忆，按 user_id 隔离。
        pgvector 用余弦相似度排序，确保检索结果与当前问题最相关。

        调用来源通常是：
        - runtime.memory_bridge.search_long_term_memory_text()
        - 在模型调用前，把相关长期记忆注入 runtime prompt
        """
        results = self.store.similarity_search(
            query,
            k=max(k * 4, 12),
            filter={"user_id": user_id},
        )
        normalized: list[str] = []
        seen: set[str] = set()
        for result in results:
            fact = normalize_memory_fact(result.page_content)
            if not fact or fact in seen:
                continue
            seen.add(fact)
            normalized.append(fact)
        return rerank_memory_facts(query, normalized, k)


# 全局单例（懒加载，POSTGRES_URL 未配置时 save/search 会抛 RuntimeError 被上层 try/except 吞掉）
long_term_memory = LongTermMemoryManager()
