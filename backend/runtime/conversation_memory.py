from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.core.config import settings
from backend.core.llm import get_summary_model


@dataclass
class CompactionDecision:
    should_compact: bool
    summary: str
    summary_upto: int
    effective_messages: list[BaseMessage]


def estimate_message_budget(messages: list[BaseMessage]) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += len(str(content))
        else:
            total += len(str(content or ""))
    return total


def _message_role(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return "用户"
    if isinstance(message, AIMessage):
        return "助手"
    if isinstance(message, ToolMessage):
        return "工具"
    if isinstance(message, SystemMessage):
        return "系统"
    msg_type = getattr(message, "type", "消息")
    return str(msg_type)


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return str(content).strip()
    return str(content or "").strip()


def _is_user_assistant_message(message: BaseMessage) -> bool:
    return isinstance(message, (HumanMessage, AIMessage))


def _recent_window_start(messages: list[BaseMessage], recent_turns: int) -> int:
    if not messages:
        return 0

    turns_seen = 0
    started = False
    for idx in range(len(messages) - 1, -1, -1):
        message = messages[idx]
        if isinstance(message, AIMessage):
            started = True
            continue
        if isinstance(message, HumanMessage):
            turns_seen += 1
            if turns_seen >= recent_turns:
                return idx
            started = False
        elif started and _is_user_assistant_message(message):
            continue
    return 0


def split_messages_for_compaction(
    messages: list[BaseMessage],
    recent_turns: int,
    summary_upto: int,
) -> tuple[list[BaseMessage], list[BaseMessage], int]:
    if not messages:
        return [], [], summary_upto

    recent_start = _recent_window_start(messages, recent_turns)
    recent_messages = messages[recent_start:]
    candidate_end = max(recent_start, summary_upto)
    delta_messages = messages[summary_upto:candidate_end]
    return delta_messages, recent_messages, candidate_end


def render_messages_for_summary(messages: list[BaseMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _message_text(message)
        if not text:
            continue
        role = _message_role(message)
        if isinstance(message, ToolMessage):
            text = text[:500]
        lines.append(f"{role}: {text}")
    return "\n".join(lines).strip()


def build_summary_prompt(existing_summary: str, new_history_text: str) -> str:
    existing = existing_summary.strip() or "（空）"
    incoming = new_history_text.strip() or "（无新增历史）"
    return (
        "你在为一个多轮智能体会话维护“历史摘要状态”。\n\n"
        "任务：\n"
        "基于“已有历史摘要”和“新增的较早会话记录”，生成一版新的、完整的历史摘要。\n"
        "这份摘要将供后续推理使用，必须稳定、准确、结构化，不能编造。\n\n"
        "输出要求：\n"
        "只输出下面固定结构，不要输出解释，不要输出 Markdown 代码块。\n\n"
        "【用户目标】\n"
        "- ...\n\n"
        "【已确认事实】\n"
        "- ...\n\n"
        "【已完成工作/已有结论】\n"
        "- ...\n\n"
        "【失败尝试/证据不足点】\n"
        "- ...\n\n"
        "【当前未解决问题】\n"
        "- ...\n\n"
        "约束：\n"
        "1. 保留对后续推理真正有用的信息，删除寒暄、重复表达、无价值细节。\n"
        "2. 不要改写已经确认的事实含义。\n"
        "3. 如果已有摘要中的内容被新增记录修正，以新增记录为准。\n"
        "4. 不要凭空补充任何没有在对话中出现的事实。\n"
        "5. 保持简洁，但不能遗漏关键约束、关键结论和未完成事项。\n\n"
        f"【已有历史摘要】\n{existing}\n\n"
        f"【新增较早会话记录】\n{incoming}"
    )


async def summarize_conversation_delta(
    existing_summary: str,
    new_messages: list[BaseMessage],
) -> str:
    new_history_text = render_messages_for_summary(new_messages)
    prompt = build_summary_prompt(existing_summary=existing_summary, new_history_text=new_history_text)
    response = await get_summary_model().ainvoke(prompt)
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    return str(content or "").strip()


def render_conversation_summary_message(summary: str) -> str:
    return (
        "以下是当前线程较早历史的结构化摘要。它概括的是已发生的对话事实、"
        "已完成工作和未解决问题，不是新的用户请求。\n\n"
        f"【历史摘要】\n{summary.strip()}"
    )


def build_effective_messages(
    messages: list[BaseMessage],
    conversation_summary: str,
    summary_upto: int,
    recent_turns: int,
) -> list[BaseMessage]:
    if not messages:
        return []

    _, recent_messages, _ = split_messages_for_compaction(
        messages=messages,
        recent_turns=recent_turns,
        summary_upto=summary_upto,
    )
    if not conversation_summary.strip():
        return recent_messages or messages

    summary_message = SystemMessage(content=render_conversation_summary_message(conversation_summary))
    return [summary_message, *(recent_messages or messages[summary_upto:])]


async def maybe_compact_conversation(
    messages: list[BaseMessage],
    conversation_summary: str,
    summary_upto: int,
) -> CompactionDecision:
    if not settings.CONTEXT_COMPRESSION_ENABLED or not messages:
        return CompactionDecision(
            should_compact=False,
            summary=conversation_summary,
            summary_upto=summary_upto,
            effective_messages=messages,
        )

    message_count = len(messages)
    char_budget = estimate_message_budget(messages)
    over_threshold = (
        message_count > settings.CONTEXT_COMPRESSION_MAX_MESSAGES
        or char_budget > settings.CONTEXT_COMPRESSION_MAX_CHARS
    )
    if not over_threshold:
        return CompactionDecision(
            should_compact=False,
            summary=conversation_summary,
            summary_upto=summary_upto,
            effective_messages=messages,
        )

    delta_messages, _, next_summary_upto = split_messages_for_compaction(
        messages=messages,
        recent_turns=settings.CONTEXT_COMPRESSION_RECENT_TURNS,
        summary_upto=summary_upto,
    )

    updated_summary = conversation_summary
    updated_upto = summary_upto
    should_compact = False
    if len(delta_messages) >= settings.CONTEXT_COMPRESSION_MIN_DELTA_MESSAGES:
        new_summary = await summarize_conversation_delta(
            existing_summary=conversation_summary,
            new_messages=delta_messages,
        )
        if new_summary:
            updated_summary = new_summary
            updated_upto = next_summary_upto
            should_compact = True

    if updated_summary.strip():
        effective_messages = build_effective_messages(
            messages=messages,
            conversation_summary=updated_summary,
            summary_upto=updated_upto,
            recent_turns=settings.CONTEXT_COMPRESSION_RECENT_TURNS,
        )
    else:
        effective_messages = messages
    return CompactionDecision(
        should_compact=should_compact,
        summary=updated_summary,
        summary_upto=updated_upto,
        effective_messages=effective_messages,
    )
