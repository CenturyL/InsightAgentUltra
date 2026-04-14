"""
agent_service.py — 智能体服务

V2 设计：
1. 唯一主循环：ReAct
2. PAE 作为高级工具 run_plan_and_execute 被主循环调用
3. runtime 统一提供动态 prompt、硬触发 PAE、工具上下文
4. thread_id 仍驱动短期记忆，user_id 驱动长期记忆写回
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from local_agent_api.core.config import settings
from local_agent_api.core.llm import deepseek_model, get_model_label, local_model
from local_agent_api.core.middleware import (
    bind_selected_model,
    compact_conversation_context,
    handle_tool_errors,
    inject_runtime_context,
)
from local_agent_api.runtime.engine import (
    RuntimeRequest,
    judge_runtime_route,
    classify_complexity,
    react_recursion_limit,
)
from local_agent_api.runtime.context_builder import build_skill_catalog_rows
from local_agent_api.runtime.skill_loader import build_skill_catalog
from local_agent_api.runtime.workflow import run_plan_and_execute_once, stream_plan_and_execute
from local_agent_api.services.tool_context import (
    consume_last_pae_result,
    drain_tool_trace,
    get_tool_trace_queue,
    reset_tool_request_context,
    set_tool_request_context,
)
from local_agent_api.services.session_service import (
    append_session_message,
    ensure_session_started,
    initialize_session_store,
    load_session_messages,
    touch_session_after_reply,
)


def _extract_explicit_user_facts(messages: list[BaseMessage]) -> list[str]:
    facts: list[str] = []
    name_pattern = re.compile(r"^我叫\s*([A-Za-z\u4e00-\u9fa5·]{2,20})")
    identity_pattern = re.compile(r"^我是\s*([A-Za-z\u4e00-\u9fa5·]{2,20})")
    suffix_pattern = re.compile(r"(你记住|记住我|记住|呀|啊|哦|哈|吧)+$")
    for message in messages:
        if not isinstance(message, HumanMessage):
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        text = " ".join(content.split())
        name_match = name_pattern.match(text)
        identity_match = identity_pattern.match(text)
        if name_match:
            name = suffix_pattern.sub("", name_match.group(1)).strip()
            if name:
                facts.append(f"用户姓名：{name}")
        elif identity_match and len(text) <= 24:
            name = suffix_pattern.sub("", identity_match.group(1)).strip()
            if name:
                facts.append(f"用户姓名：{name}")
    return facts
from local_agent_api.services.tools import get_agent_tools, initialize_runtime_tools


class AgentState(TypedDict):
    # messages 是 LangGraph/agent runtime 会持久化的核心字段之一。
    # 同一个 thread_id 下，多轮对话的短期记忆主要就是靠这里恢复。
    messages: list[BaseMessage]
    # user_id 不参与短期记忆主键，而是供 runtime 注入长期记忆、对话后写回长期记忆使用。
    user_id: str
    # plan_mode 控制当前请求的规划倾向，例如 auto / strict_plan。
    plan_mode: str
    # model_choice 由前端传入，主循环每次模型调用前由 middleware 动态绑定。
    model_choice: str
    # conversation_summary 是当前 thread 级的早期历史压缩摘要。
    conversation_summary: str
    # summary_upto 记录摘要已覆盖到 messages 的位置，用于增量压缩。
    summary_upto: int
    runtime_route: dict[str, Any]


_agent = None
_checkpointer = None
_connection_pool = None


def _should_suppress_stream_chunk(chunk) -> bool:
    tool_call_chunks = getattr(chunk, "tool_call_chunks", None)
    if tool_call_chunks:
        return True
    chunk_content = getattr(chunk, "content", None)
    if not isinstance(chunk_content, str):
        return False
    stripped = chunk_content.strip()
    if not stripped:
        return False
    if stripped in {"[", "]", "{", "}", ":", ","}:
        return True
    if stripped in {'{"', '"selected"', 'selected', '":[]', '[]', '":'}:
        return True
    if any(marker in stripped for marker in ('"step_id"', '"required_capability"', '"expected_output"', '"status"')):
        return True
    if 'selected' in stripped:
        return True
    return False


def _extract_tool_output_text(event: dict[str, Any]) -> str:
    output = event.get("data", {}).get("output")
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")).strip())
            else:
                parts.append(str(getattr(item, "text", "") or item).strip())
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _format_filesystem_listing(tool_text: str) -> str:
    lines = [line.strip() for line in (tool_text or "").splitlines() if line.strip()]
    files: list[str] = []
    dirs: list[str] = []
    for line in lines:
        if line.startswith("[FILE] "):
            files.append(line.removeprefix("[FILE] ").strip())
        elif line.startswith("[DIR] "):
            dirs.append(line.removeprefix("[DIR] ").strip())

    if not files and not dirs:
        return tool_text.strip()

    sections: list[str] = []
    if dirs:
        sections.append("文件夹：\n" + "\n".join(f"- {name}" for name in dirs))
    if files:
        sections.append("文件：\n" + "\n".join(f"- {name}" for name in files))
    return "\n\n".join(sections).strip()


def _build_agent(checkpointer):
    """
    构建唯一的 ReAct 主循环 Agent。
    
    架构设计：
      Agent 本身是无状态的（只有一份），但当它被调用时：
        1. Service 层把 user_id/plan_mode/model_choice 等信息塞进 AgentState
        2. Agent 内部的主循环每次调用 LLM 前：
          - middleware[0] inject_runtime_context 拦截，根据当前 state 动态拼最新的 prompt
          - middleware[1] bind_selected_model 拦截，动态绑定正确的模型
          - middleware[2] handle_tool_errors 拦截，捕获工具错误，不中断主循环
        3. 然后推理 LLM
        4. 返回结果维持 service 处置
        
      这样做的好处：
        - Service 层很简洁，唯一职责是推理流程和流式输出
        - Prompt、skill、memory 的全部都组件化到 runtime 模块
        - Agent 不需要知道 runtime 的存在—它只是一段通用的推理器

    这里的 runtime 不是某个单独类实例，而是一组模块协作：
      - middleware：负责在“每次模型调用前”拦截
      - context_builder：负责收集长期记忆 / markdown memory / skills / complexity
      - prompt_manager：负责把这些原材料拼成动态 system prompt
    所以 _build_agent() 里注册 middleware，就是把 runtime 接到 ReAct 主循环上的关键动作。
    """
    # create_agent 本身就是 LangChain/LangGraph 提供的 ReAct/tool-calling agent runtime。
    # 这里没有手写“推理 -> 调工具 -> 再推理”的 while 循环，
    # 循环逻辑由 create_agent + astream_events 在框架内部完成。
    return create_agent(
        model=local_model,
        tools=get_agent_tools(),
        # 关键：middleware 是最核心的变化点。
        # 旧方式：service 层硬编码 prompt，每个请求一样。
        # 新方式：middleware 拦截 LLM 调用，每次都动态拼一次。
        middleware=[
            compact_conversation_context,  # 第 0 把：压缩早期会话，保留最近窗口
            inject_runtime_context,  # 第 1 把：动态注入 skill + memory + persona
            bind_selected_model,     # 第 2 把：动态绑定正确的模型
            handle_tool_errors,      # 第 3 把：捕获工具错误，不中断主循环
        ],
        checkpointer=checkpointer,
        state_schema=AgentState,
        system_prompt=(
            # 这是主循环的“基础 system prompt”：
            # - 定义 agent 的默认身份与总规则
            # - 不是最终完整 prompt
            # 真正发送给 LLM 时，middleware 还会在此基础上追加 runtime_context.system_prompt。
            #
            # 因此它和 persona / memory / skills 的关系是：
            # - 基础 system_prompt 提供稳定的底座规则
            # - inject_runtime_context 提供每次请求都可能变化的动态上下文
            # - 两者会顺序拼接后一起发给 LLM，不是二选一
            "你是 InsightAgentPro，一个通用智能体平台中的主执行智能体。"
            "默认在 ReAct 主循环中工作。"
            "如果任务涉及比较、提取多个字段、生成报告/方案、多步研究、需要多工具协作、"
            "或你无法确定单轮工具调用就能稳定完成，则必须先调用 run_plan_and_execute，禁止直接尝试一次性回答。"
            "但如果任务是单步可验证的简单工具任务，例如列目录、读取单个文件、获取时间、抓取单个页面或一次明确检索，"
            "则应直接调用相应工具完成，禁止进入 PAE。"
            "一旦简单工具任务的最近一次工具返回已经足以回答用户问题，必须立即停止继续思考和铺垫，直接给出最终答案。"
            "不要在工具调用前后重复解释“我将要做什么”或“接下来准备做什么”。"
        ),
    )


async def initialize() -> None:
    """初始化 Agent 及 checkpointer。"""
    global _agent, _checkpointer, _connection_pool

    # 短期记忆的存储介质在这里决定：
    # - 有 POSTGRES_URL：使用 AsyncPostgresSaver，thread 级 state 会持久化到 PostgreSQL
    # - 无 POSTGRES_URL：使用 MemorySaver，只保存在当前进程内
    if settings.POSTGRES_URL:
        try:
            from psycopg_pool import AsyncConnectionPool
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            _connection_pool = AsyncConnectionPool(
                conninfo=settings.POSTGRES_URL,
                max_size=10,
                open=False,
                kwargs={"autocommit": True, "prepare_threshold": 0},
            )
            await _connection_pool.open()

            _checkpointer = AsyncPostgresSaver(_connection_pool)
            await _checkpointer.setup()
            print("✅ [Checkpointer] 已连接 PostgreSQL，使用 AsyncPostgresSaver")
        except Exception as e:
            print(f"⚠️ [Checkpointer] PostgreSQL 连接失败（{e}），降级为 MemorySaver")
            _checkpointer = MemorySaver()
    else:
        _checkpointer = MemorySaver()
        print("ℹ️ [Checkpointer] 未配置 POSTGRES_URL，使用 MemorySaver（进程内短期记忆）")

    await initialize_runtime_tools()
    await initialize_session_store()
    _agent = _build_agent(_checkpointer)
    print("✅ [Agent] 初始化完成")


async def cleanup() -> None:
    """关闭数据库连接池。"""
    global _connection_pool
    if _connection_pool:
        await _connection_pool.close()
        print("🔒 [Agent] 数据库连接池已关闭")


def _get_agent():
    """同步获取 Agent 单例。"""
    global _agent, _checkpointer
    if _agent is None:
        _checkpointer = MemorySaver()
        _agent = _build_agent(_checkpointer)
    return _agent


async def _extract_and_save_memories(thread_id: str, user_id: str) -> None:
    """对话结束后，异步提取长期记忆。"""
    if not user_id or not settings.POSTGRES_URL:
        return
    try:
        agent = _get_agent()
        # 这里显式读取同一 thread_id 的短期记忆 state。
        # 注意：按 thread_id 恢复 state 的逻辑不是这里手写实现的，
        # 而是 LangGraph checkpointer 自带的能力；我们这里只是通过 thread_id 去取。
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        recent = messages[-6:] if len(messages) > 6 else messages
        conversation = "\n".join(
            f"{'用户' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in recent
            if hasattr(m, "content") and m.content
        )
        if not conversation.strip():
            return

        deterministic_facts = _extract_explicit_user_facts(recent)

        extraction_prompt = (
            "你在做长期记忆提取。"
            "只提取用户本人明确表达的、跨会话仍有价值的稳定事实。"
            "只允许提取：用户姓名/称呼、身份背景、长期偏好、长期约束、长期目标。"
            "禁止提取 AI 的名字、AI 的能力、AI 的计划、工具执行过程、临时任务内容。"
            "如果一句话是在描述 AI，而不是用户，就绝对不要写入。"
            "每行一条，尽量写成“用户姓名：xxx”或“用户偏好：xxx”这种明确形式。"
            "无有效信息则输出空。\n\n"
            f"{conversation}"
        )
        response = await deepseek_model.ainvoke(extraction_prompt)
        facts = deterministic_facts + [line.strip() for line in response.content.splitlines() if line.strip()]

        from local_agent_api.core.memory import long_term_memory
        for fact in facts:
            long_term_memory.save(user_id, fact)
        if facts:
            print(f"💾 [长期记忆] 为 {user_id} 保存了 {len(facts)} 条记忆")
    except Exception as e:
        print(f"⚠️ [长期记忆] 记忆提取失败（{e}），跳过")


async def get_agent_stream(
    query: str,
    thread_id: str = "default",
    user_id: str = "",
    plan_mode: str | None = None,
    model_choice: str = "local_qwen",
    metadata_filters: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, str], None]:
    """
    唯一智能体主入口：始终走 ReAct 主循环，必要时由模型自行调用 run_plan_and_execute。
    """
    agent = _get_agent()
    request = RuntimeRequest(
        query=query,
        thread_id=thread_id,
        user_id=user_id,
        plan_mode=plan_mode or "auto",
        model_choice=model_choice,
        metadata_filters=metadata_filters or {},
    )
    context_tokens = set_tool_request_context(
        thread_id=thread_id,
        user_id=user_id,
        plan_mode=request.plan_mode,
        model_choice=request.model_choice,
        metadata_filters=request.metadata_filters,
    )
    tool_trace_queue = get_tool_trace_queue(context_tokens)
    # 本轮请求进入主循环时，新的 HumanMessage 只追加当前消息。
    # 历史上下文不在这里手工拼接，而是由 checkpointer 按 thread_id 自动恢复。
    inputs: AgentState = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "plan_mode": request.plan_mode,
        "model_choice": request.model_choice,
        "conversation_summary": "",
        "summary_upto": 0,
        "runtime_route": {},
    }
    # LangGraph 会使用 configurable.thread_id 作为短期记忆的会话主键。
    # 同一个 thread_id 连续请求时，旧 state 会被自动读取并续上。
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": react_recursion_limit(),
    }

    try:
        await ensure_session_started(thread_id, user_id, query)
        await append_session_message(thread_id, user_id, "user", query)
        skill_catalog = build_skill_catalog()
        runtime_route = await judge_runtime_route(
            query=query,
            plan_mode=request.plan_mode,
            complexity=classify_complexity(query, request.plan_mode),
            available_tool_names=[tool.name for tool in get_agent_tools()],
            skill_catalog_rows=build_skill_catalog_rows(skill_catalog),
        )
        inputs["runtime_route"] = {
            "pae_action": runtime_route.pae_action,
            "pae_reason": runtime_route.pae_reason,
            "selected_skills": runtime_route.selected_skills,
        }
        model_label = get_model_label(request.model_choice)
        yield {"type": "trace", "content": f"> 模型来源：{model_label}"}
        yield {"type": "trace", "content": f"🧠 [主循环] 已进入 ReAct 主循环。plan_mode={request.plan_mode}"}
        yield {"type": "trace", "content": f"🧭 [PAE路由] {runtime_route.pae_action} | {runtime_route.pae_reason}"}

        if runtime_route.pae_action == "run_plan_and_execute":
            yield {"type": "trace", "content": "🧭 [硬触发] PAE 路由器判定当前请求应直接进入 PAE。"}
            final_result = None
            async for line, result in stream_plan_and_execute(
                query=query,
                thread_id=thread_id,
                user_id=user_id,
                plan_mode=request.plan_mode,
                model_choice=request.model_choice,
                metadata_filters=request.metadata_filters,
            ):
                if line:
                    yield {"type": "trace", "content": line}
                if result is not None:
                    final_result = result
            if final_result:
                if final_result.get("status") == "planning_failed":
                    planning_reason = str(final_result.get("planning_reason", "")).strip() or "PAE 未能生成可执行计划。"
                    yield {"type": "trace", "content": f"⚠️ [PAE规划失败] {planning_reason}"}
                    yield {"type": "trace", "content": "↩️ [返回主循环] 已停止 PAE，交由主循环决定是否澄清需求或缩小任务范围。"}
                    inputs["runtime_route"] = {
                        "pae_action": "direct_or_simple_tools",
                        "pae_reason": f"上一轮 PAE 规划失败：{planning_reason}。优先向用户澄清或缩小任务范围，不要再次直接进入 PAE。",
                        "selected_skills": runtime_route.selected_skills,
                    }
                else:
                    final_answer = final_result.get("final_answer", "")
                    if final_answer:
                        await append_session_message(thread_id, user_id, "assistant", final_answer)
                    yield {"type": "answer", "content": final_answer}
                    if user_id:
                        asyncio.ensure_future(_extract_and_save_memories(thread_id, user_id))
                    return

        stream_had_content = False
        emitted_trace_lines: set[str] = set()
        event_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        pae_tool_invoked = False
        parser_state = {"in_thought": False, "pending": ""}

        def _emit_text(kind: str, text: str) -> list[dict[str, str]]:
            if not text or not text.strip():
                return []
            return [{"type": kind, "content": text}]

        def _flush_model_chunk(text: str, *, final: bool = False) -> list[dict[str, str]]:
            parser_state["pending"] += text or ""
            pending = parser_state["pending"]
            events: list[dict[str, str]] = []
            open_tag = "<think>"
            close_tag = "</think>"

            while pending:
                if parser_state["in_thought"]:
                    end = pending.find(close_tag)
                    if end >= 0:
                        events.extend(_emit_text("thought", pending[:end]))
                        pending = pending[end + len(close_tag):]
                        parser_state["in_thought"] = False
                        continue
                    if final:
                        events.extend(_emit_text("thought", pending.replace(close_tag, "")))
                        pending = ""
                        break
                    keep = len(close_tag) - 1
                    flush_upto = max(len(pending) - keep, 0)
                    if flush_upto <= 0:
                        break
                    events.extend(_emit_text("thought", pending[:flush_upto]))
                    pending = pending[flush_upto:]
                    break

                start = pending.find(open_tag)
                if start >= 0:
                    events.extend(_emit_text("answer", pending[:start]))
                    pending = pending[start + len(open_tag):]
                    parser_state["in_thought"] = True
                    continue
                if final:
                    events.extend(_emit_text("answer", pending.replace(open_tag, "")))
                    pending = ""
                    break
                keep = len(open_tag) - 1
                flush_upto = max(len(pending) - keep, 0)
                if flush_upto <= 0:
                    break
                events.extend(_emit_text("answer", pending[:flush_upto]))
                pending = pending[flush_upto:]
                break

            parser_state["pending"] = pending
            return events

        async def _pump_events() -> None:
            # astream_events 是 ReAct 主循环真正执行的地方：
            # 框架内部会负责模型推理、工具调用、再推理，直到自然收敛或达到 recursion_limit。
            #
            # 这里单独起一个 asyncio task 去“泵”事件，是为了把框架事件流和我们自己的输出逻辑解耦：
            # - agent.astream_events(...) 负责持续产生事件
            # - 我们在外层 while 里一边消费事件，一边消费 tool trace
            # - 两边谁先到，就先把谁流式 yield 给前端
            async for event in agent.astream_events(inputs, config=config, version="v2"):
                await event_queue.put(event)
            await event_queue.put(None)

        # create_task 会把 _pump_events 这个协程交给事件循环后台执行。
        # 当前协程不会阻塞在“拉取框架事件”这件事上，而是可以继续同时等待：
        # - event_queue（模型 / tool runtime 事件）
        # - tool_trace_queue（我们自定义的工具 trace）
        pump_task = asyncio.create_task(_pump_events())
        event_stream_done = False
        assistant_answer_parts: list[str] = []

        try:
            while not event_stream_done:
                pending = []
                event_task = asyncio.create_task(event_queue.get())
                pending.append(event_task)
                trace_task = None
                if tool_trace_queue is not None:
                    trace_task = asyncio.create_task(tool_trace_queue.get())
                    pending.append(trace_task)

                done, pending_tasks = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in pending_tasks:
                    task.cancel()

                if trace_task is not None and trace_task in done:
                    line = trace_task.result()
                    if line:
                        emitted_trace_lines.add(line)
                        yield {"type": "trace", "content": line}
                    continue

                if event_task not in done:
                    continue

                event = event_task.result()
                if event is None:
                    event_stream_done = True
                    break

                kind = event["event"]

                if kind == "on_chat_model_stream":
                    if pae_tool_invoked:
                        continue
                    chunk = event["data"]["chunk"]
                    if _should_suppress_stream_chunk(chunk):
                        continue
                    chunk_content = chunk.content
                    if chunk_content:
                        stream_had_content = True
                        for item in _flush_model_chunk(str(chunk_content)):
                            if item["type"] == "answer":
                                assistant_answer_parts.append(item["content"])
                            yield item

                elif kind == "on_tool_start":
                    tool_name = event["name"]
                    if tool_name == "run_plan_and_execute":
                        pae_tool_invoked = True
                        yield {"type": "trace", "content": "🧭 [PAE调用] 主循环决定进入 Plan-and-Execute 子流程。"}
                    elif tool_name.startswith("mcp__"):
                        yield {"type": "trace", "content": f"🧩 [MCP工具调用] 决定调用 MCP 工具：【{tool_name}】..."}
                    else:
                        yield {"type": "trace", "content": f"🛠️ [工具调用] 决定调用工具：【{tool_name}】..."}

                elif kind == "on_tool_end":
                    tool_name = event["name"]
                    if tool_name == "run_plan_and_execute":
                        for line in drain_tool_trace():
                            if line not in emitted_trace_lines:
                                emitted_trace_lines.add(line)
                                yield {"type": "trace", "content": line}
                        final_result = consume_last_pae_result()
                        if final_result and final_result.get("status") == "planning_failed":
                            planning_reason = str(final_result.get("planning_reason", "")).strip() or "PAE 未能生成可执行计划。"
                            yield {"type": "trace", "content": f"⚠️ [PAE规划失败] {planning_reason}"}
                            yield {"type": "trace", "content": "↩️ [继续主循环] 已把规划失败结果返回给主循环，由模型自行决定是否澄清或缩小任务范围。"}
                            pae_tool_invoked = False
                            continue
                        yield {"type": "trace", "content": "✅ [PAE完成] Plan-and-Execute 子流程执行完成，已直接输出最终结果。"}
                        if final_result and final_result.get("final_answer"):
                            stream_had_content = True
                            assistant_answer_parts.append(final_result["final_answer"])
                            yield {"type": "answer", "content": final_result["final_answer"]}
                        event_stream_done = True
                        break
                    elif tool_name.startswith("mcp__"):
                        yield {"type": "trace", "content": f"✅ [MCP工具完成] MCP 工具【{tool_name}】执行完成，正在继续推理..."}
                        if (
                            runtime_route.pae_action == "direct_or_simple_tools"
                            and tool_name in {
                                "mcp__filesystem__list_directory",
                                "mcp__filesystem__list_directory_with_sizes",
                                "mcp__filesystem__directory_tree",
                            }
                        ):
                            tool_text = _extract_tool_output_text(event)
                            if tool_text:
                                formatted = _format_filesystem_listing(tool_text)
                                stream_had_content = True
                                yield {"type": "trace", "content": "✅ [直接收尾] 已基于 MCP 工具结果直接返回最终答案。"}
                                assistant_answer_parts.append(formatted)
                                yield {"type": "answer", "content": formatted}
                                event_stream_done = True
                                break
                    else:
                        yield {"type": "trace", "content": f"✅ [工具完成] 工具【{tool_name}】执行完成，正在继续推理..."}
        finally:
            if not pump_task.done():
                pump_task.cancel()

        for item in _flush_model_chunk("", final=True):
            if item["type"] == "answer":
                assistant_answer_parts.append(item["content"])
            yield item

        if stream_had_content and user_id:
            final_answer_preview = "".join(assistant_answer_parts).strip()
            if final_answer_preview:
                await append_session_message(thread_id, user_id, "assistant", final_answer_preview)
            await touch_session_after_reply(thread_id, user_id, query, final_answer_preview)
            asyncio.ensure_future(_extract_and_save_memories(thread_id, user_id))
    finally:
        reset_tool_request_context(context_tokens)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


async def get_thread_messages(thread_id: str, user_id: str) -> list[dict[str, str]]:
    rows = await load_session_messages(thread_id, user_id)
    return [{"role": row["role"], "content": row["content"]} for row in rows]
