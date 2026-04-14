from __future__ import annotations

import uuid
from datetime import datetime

import psycopg
from psycopg.rows import dict_row

from backend.core.config import settings


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_sessions (
    thread_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_preview TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_user_updated
ON agent_sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS agent_session_messages (
    id BIGSERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_agent_session_messages_thread_created
ON agent_session_messages (thread_id, created_at ASC);
"""


def _require_postgres_url() -> str:
    if not settings.POSTGRES_URL:
        raise RuntimeError("当前环境未启用 PostgreSQL，无法使用历史会话功能。")
    return settings.POSTGRES_URL


async def initialize_session_store() -> None:
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute(_CREATE_TABLE_SQL)
    finally:
        await conn.close()


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _truncate(text: str, limit: int) -> str:
    normalized = _normalize_whitespace(text)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def build_session_title(text: str) -> str:
    title = _truncate(text, 56)
    return title or "新会话"


def build_message_preview(text: str) -> str:
    return _truncate(text, 110)


def _serialize_session_row(row: dict) -> dict:
    return {
        "thread_id": row["thread_id"],
        "user_id": row["user_id"],
        "title": row["title"],
        "created_at": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else str(row["created_at"]),
        "updated_at": row["updated_at"].isoformat() if isinstance(row["updated_at"], datetime) else str(row["updated_at"]),
        "last_message_preview": row["last_message_preview"],
    }


async def list_sessions(user_id: str) -> list[dict]:
    if not user_id.strip():
        return []
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True, row_factory=dict_row)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at, last_message_preview
                FROM agent_sessions
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id.strip(),),
            )
            rows = await cur.fetchall()
            return [_serialize_session_row(row) for row in rows]
    finally:
        await conn.close()


async def create_empty_session(user_id: str, *, thread_id: str | None = None) -> dict:
    normalized_user = user_id.strip()
    if not normalized_user:
        raise RuntimeError("user_id 不能为空。")
    thread = thread_id or str(uuid.uuid4())
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True, row_factory=dict_row)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_sessions (thread_id, user_id, title, last_message_preview)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (thread, normalized_user, "新会话", ""),
            )
            await cur.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at, last_message_preview
                FROM agent_sessions
                WHERE thread_id = %s
                """,
                (thread,),
            )
            row = await cur.fetchone()
            if row is None:
                raise RuntimeError("创建会话失败。")
            return _serialize_session_row(row)
    finally:
        await conn.close()


async def bootstrap_sessions(user_id: str) -> dict:
    sessions = await list_sessions(user_id)
    current = await create_empty_session(user_id)
    refreshed = await list_sessions(user_id)
    return {
        "sessions": refreshed or [current],
        "current_thread_id": current["thread_id"],
    }


async def get_session(thread_id: str, user_id: str) -> dict | None:
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True, row_factory=dict_row)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT thread_id, user_id, title, created_at, updated_at, last_message_preview
                FROM agent_sessions
                WHERE thread_id = %s AND user_id = %s
                """,
                (thread_id, user_id.strip()),
            )
            row = await cur.fetchone()
            return _serialize_session_row(row) if row else None
    finally:
        await conn.close()


async def ensure_session_started(thread_id: str, user_id: str, initial_query: str) -> None:
    normalized_user = user_id.strip()
    if not normalized_user:
        return
    title = build_session_title(initial_query)
    preview = build_message_preview(initial_query)
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_sessions (thread_id, user_id, title, last_message_preview)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET
                  user_id = EXCLUDED.user_id,
                  title = CASE
                    WHEN agent_sessions.title = '新会话' THEN EXCLUDED.title
                    ELSE agent_sessions.title
                  END,
                  updated_at = NOW(),
                  last_message_preview = CASE
                    WHEN agent_sessions.last_message_preview = '' THEN EXCLUDED.last_message_preview
                    ELSE agent_sessions.last_message_preview
                  END
                """,
                (thread_id, normalized_user, title, preview),
            )
    finally:
        await conn.close()


async def touch_session_after_reply(thread_id: str, user_id: str, query: str, answer: str) -> None:
    normalized_user = user_id.strip()
    if not normalized_user:
        return
    title = build_session_title(query)
    preview = build_message_preview(answer or query)
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE agent_sessions
                SET
                  updated_at = NOW(),
                  title = CASE
                    WHEN title = '新会话' THEN %s
                    ELSE title
                  END,
                  last_message_preview = %s
                WHERE thread_id = %s AND user_id = %s
                """,
                (title, preview, thread_id, normalized_user),
            )
    finally:
        await conn.close()


async def append_session_message(thread_id: str, user_id: str, role: str, content: str) -> None:
    normalized_user = user_id.strip()
    normalized_content = _normalize_whitespace(content)
    normalized_role = role.strip().lower()
    if not normalized_user or not normalized_content or normalized_role not in {"user", "assistant"}:
        return
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO agent_session_messages (thread_id, user_id, role, content)
                VALUES (%s, %s, %s, %s)
                """,
                (thread_id, normalized_user, normalized_role, normalized_content),
            )
    finally:
        await conn.close()


async def load_session_messages(thread_id: str, user_id: str) -> list[dict]:
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True, row_factory=dict_row)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT role, content, created_at
                FROM agent_session_messages
                WHERE thread_id = %s AND user_id = %s
                ORDER BY created_at ASC, id ASC
                """,
                (thread_id, user_id.strip()),
            )
            rows = await cur.fetchall()
            return [
                {
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat() if isinstance(row["created_at"], datetime) else str(row["created_at"]),
                }
                for row in rows
            ]
    finally:
        await conn.close()


async def delete_session(thread_id: str, user_id: str) -> bool:
    normalized_user = user_id.strip()
    if not normalized_user or not thread_id.strip():
        return False
    conn = await psycopg.AsyncConnection.connect(_require_postgres_url(), autocommit=True)
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                DELETE FROM agent_session_messages
                WHERE thread_id = %s AND user_id = %s
                """,
                (thread_id, normalized_user),
            )
            await cur.execute(
                """
                DELETE FROM agent_sessions
                WHERE thread_id = %s AND user_id = %s
                """,
                (thread_id, normalized_user),
            )
            return cur.rowcount > 0
    finally:
        await conn.close()
