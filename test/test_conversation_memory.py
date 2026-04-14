from __future__ import annotations

import asyncio
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from local_agent_api.runtime import conversation_memory as cm


def _build_messages(turns: int) -> list:
    messages = []
    for idx in range(turns):
        messages.append(HumanMessage(content=f"用户问题 {idx}"))
        messages.append(AIMessage(content=f"助手回答 {idx}"))
    return messages


def test_build_effective_messages_without_summary_keeps_recent_window():
    messages = _build_messages(6)
    effective = cm.build_effective_messages(
        messages=messages,
        conversation_summary="",
        summary_upto=0,
        recent_turns=2,
    )
    assert effective == messages[-4:]


def test_build_effective_messages_with_summary_inserts_system_summary():
    messages = _build_messages(5)
    effective = cm.build_effective_messages(
        messages=messages,
        conversation_summary="【用户目标】\n- 完成任务",
        summary_upto=6,
        recent_turns=2,
    )
    assert isinstance(effective[0], SystemMessage)
    assert "历史摘要" in effective[0].content
    assert effective[1:] == messages[-4:]


def test_maybe_compact_conversation_updates_summary_incrementally(monkeypatch):
    messages = _build_messages(8)

    async def fake_summary(existing_summary: str, new_messages: list) -> str:
        return f"{existing_summary}|delta={len(new_messages)}".strip("|")

    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_ENABLED", True)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MAX_MESSAGES", 6)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MAX_CHARS", 10_000)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_RECENT_TURNS", 2)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MIN_DELTA_MESSAGES", 2)
    monkeypatch.setattr(cm, "summarize_conversation_delta", fake_summary)

    decision = asyncio.run(
        cm.maybe_compact_conversation(
            messages=messages,
            conversation_summary="old",
            summary_upto=4,
        )
    )

    assert decision.should_compact is True
    assert decision.summary == "old|delta=8"
    assert decision.summary_upto == 12
    assert isinstance(decision.effective_messages[0], SystemMessage)
    assert decision.effective_messages[1:] == messages[-4:]


def test_maybe_compact_conversation_reuses_existing_summary_when_delta_small(monkeypatch):
    messages = _build_messages(8)

    async def fake_summary(existing_summary: str, new_messages: list) -> str:
        raise AssertionError("should not summarize when delta is below threshold")

    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_ENABLED", True)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MAX_MESSAGES", 6)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MAX_CHARS", 10_000)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_RECENT_TURNS", 3)
    monkeypatch.setattr(cm.settings, "CONTEXT_COMPRESSION_MIN_DELTA_MESSAGES", 6)
    monkeypatch.setattr(cm, "summarize_conversation_delta", fake_summary)

    decision = asyncio.run(
        cm.maybe_compact_conversation(
            messages=messages,
            conversation_summary="stable-summary",
            summary_upto=8,
        )
    )

    assert decision.should_compact is False
    assert decision.summary == "stable-summary"
    assert decision.summary_upto == 8
    assert isinstance(decision.effective_messages[0], SystemMessage)
    assert decision.effective_messages[1:] == messages[-6:]
