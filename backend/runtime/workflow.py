from __future__ import annotations

"""把现有 PAE 流程封装成可被主循环调用的高级子执行器。"""

from typing import Any, AsyncGenerator, Callable

from langchain_core.messages import HumanMessage

from backend.agents.executor import execute_plan_events, executor_node
from backend.agents.orchestrator import build_synthesizer_prompt
from backend.agents.planner import planner_node
from backend.agents.reflection import reflection_node
from backend.agents.state import OrchestratorState
from backend.core.llm import get_model_by_choice
from backend.runtime.context_builder import build_runtime_context
from backend.runtime.tool_registry import get_tool_names


def _resolve_pae_model_choice(model_choice: str) -> str:
    """
    PAE 内部强制使用更稳定的模型。

    主循环允许用户手动选本地模型，但 PAE 的 planner/synthesizer 对结构化规划和长文本综合更敏感，
    因此这里会把 local_qwen 提升到 deepseek，避免在复杂任务上卡死。
    """
    choice = (model_choice or "deepseek").lower()
    if choice == "local_qwen":
        return "deepseek"
    return choice


async def run_plan_and_execute_once(
    query: str,
    thread_id: str,
    user_id: str,
    plan_mode: str | None = None,
    model_choice: str = "deepseek",
    metadata_filters: dict[str, Any] | None = None,
    trace_sink: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    执行一次完整 PAE，并把 trace 和结果一起返回。

    这条函数是“PAE tool 的同步版”：
    - 主循环里的 run_plan_and_execute 工具最终会落到这里或其流式版本
    - 进入 PAE 后不再重新判断是否要进入 PAE，而是直接执行 planner -> executor -> reflection -> synthesizer
    """
    pae_model_choice = _resolve_pae_model_choice(model_choice)
    # PAE 也复用与主循环相同的 runtime context 组装逻辑，
    # 所以 persona / skills / 长期记忆 / Markdown 记忆在这里同样生效。
    runtime_context = await build_runtime_context(
        query=query,
        user_id=user_id,
        plan_mode=plan_mode,
        available_tool_names=get_tool_names(),
        route_decision={
            "pae_action": "run_plan_and_execute",
            "pae_reason": "当前已进入 Plan-and-Execute 子流程。",
            "selected_skills": [],
        },
    )
    # 注意这里的 state 是 PAE 子流程自己的运行状态，不是主循环的 AgentState。
    # 两者分开，避免主循环和 PAE 内部状态相互污染。
    state: OrchestratorState = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "thread_id": thread_id,
        "plan_mode": plan_mode or "auto",
        "model_choice": pae_model_choice,
        "metadata_filters": metadata_filters or {},
        "is_complex": True,
        "runtime_system_prompt": runtime_context.system_prompt,
        "activated_skill_names": [item.package.metadata.name for item in runtime_context.activated_skills],
        "planner_hints": runtime_context.skill_effects.planner_hints,
        "executor_hints": runtime_context.skill_effects.executor_hints,
        "output_format_hints": runtime_context.skill_effects.output_format_hints,
    }

    traces: list[str] = []

    def emit(line: str) -> None:
        traces.append(line)
        if trace_sink:
            trace_sink(line)

    emit("🧭 [计划模式] 已进入 Plan-and-Execute 子流程。")
    if pae_model_choice != model_choice:
        emit(f"☁️ [PAE模型] 已将 PAE 内部模型切换为 {pae_model_choice}。")
    state = await planner_node(state)
    if state.get("planning_failed"):
        planning_reason = state.get("planning_reason", "Planner 未返回可执行计划。")
        final_answer = (
            "当前任务暂时无法稳定规划。"
            f"\n规划失败原因：{planning_reason}"
            "\n请先补充关键信息、缩小任务范围，或改成更明确的单一目标后再继续。"
        )
        emit(f"⚠️ [规划失败] {planning_reason}")
        emit("↩️ [返回主循环] 当前不执行假计划，交由主循环决定是否澄清或改用直接回答。")
        state["final_answer"] = final_answer
        return {
            "mode": "plan_and_execute",
            "status": "planning_failed",
            "planning_reason": planning_reason,
            "plan": [],
            "step_results": [],
            "citations": [],
            "trace": traces,
            "final_answer": final_answer,
        }
    plan = state.get("plan", [])
    if plan:
        emit("📝 [任务规划] 已生成执行计划：")
        for step in plan:
            emit(f"- {step['step_id']} | {step['required_capability']} | {step['goal']}")

    state = await executor_node(state, trace_sink=emit)

    before = state.get("step_results", [])
    state = await reflection_node(state)
    after = state.get("step_results", [])
    if after != before:
        emit("🔁 [反思修正] 已对证据不足或失败步骤进行补救。")
        for result in after:
            emit(f"📎 [反思结果] {result['step_id']} | {result['status']}")
    else:
        emit("🔁 [反思修正] 未发现需要补救的步骤。")

    emit("✍️ [答案生成] 正在生成最终结果。")
    prompt = build_synthesizer_prompt(state)
    model = get_model_by_choice(pae_model_choice)
    response = await model.ainvoke(prompt)
    final_answer = response.content.strip()
    state["final_answer"] = final_answer
    emit("✅ [PAE完成] Plan-and-Execute 子流程执行完成，已返回主循环。")
    return {
        "mode": "plan_and_execute",
        "plan": plan,
        "step_results": state.get("step_results", []),
        "citations": state.get("citations", []),
        "trace": traces,
        "final_answer": final_answer,
    }


async def stream_plan_and_execute(
    query: str,
    thread_id: str,
    user_id: str,
    plan_mode: str | None = None,
    model_choice: str = "deepseek",
    metadata_filters: dict[str, Any] | None = None,
    trace_sink: Callable[[str], None] | None = None,
) -> AsyncGenerator[tuple[str, dict[str, Any] | None], None]:
    """
    逐阶段流式执行 PAE。

    这是前端可观测性的主要来源：
    planner / executor / reflection / synthesizer 的关键阶段都会转成 trace 行，
    供聊天区和右侧运行面板实时展示。
    """
    pae_model_choice = _resolve_pae_model_choice(model_choice)
    runtime_context = await build_runtime_context(
        query=query,
        user_id=user_id,
        plan_mode=plan_mode,
        available_tool_names=get_tool_names(),
        route_decision={
            "pae_action": "run_plan_and_execute",
            "pae_reason": "当前已进入 Plan-and-Execute 子流程。",
            "selected_skills": [],
        },
    )
    state: OrchestratorState = {
        "messages": [HumanMessage(content=query)],
        "user_id": user_id,
        "thread_id": thread_id,
        "plan_mode": plan_mode or "auto",
        "model_choice": pae_model_choice,
        "metadata_filters": metadata_filters or {},
        "is_complex": True,
        "runtime_system_prompt": runtime_context.system_prompt,
        "activated_skill_names": [item.package.metadata.name for item in runtime_context.activated_skills],
        "planner_hints": runtime_context.skill_effects.planner_hints,
        "executor_hints": runtime_context.skill_effects.executor_hints,
        "output_format_hints": runtime_context.skill_effects.output_format_hints,
    }

    def emit(line: str) -> str:
        if trace_sink:
            trace_sink(line)
        return line

    yield emit("🧭 [计划模式] 已进入 Plan-and-Execute 子流程。"), None
    if pae_model_choice != model_choice:
        yield emit(f"☁️ [PAE模型] 已将 PAE 内部模型切换为 {pae_model_choice}。"), None

    state = await planner_node(state)
    if state.get("planning_failed"):
        planning_reason = state.get("planning_reason", "Planner 未返回可执行计划。")
        final_answer = (
            "当前任务暂时无法稳定规划。"
            f"\n规划失败原因：{planning_reason}"
            "\n请先补充关键信息、缩小任务范围，或改成更明确的单一目标后再继续。"
        )
        yield emit(f"⚠️ [规划失败] {planning_reason}"), None
        yield emit("↩️ [返回主循环] 当前不执行假计划，交由主循环决定是否澄清或改用直接回答。"), None
        state["final_answer"] = final_answer
        yield "", {
            "mode": "plan_and_execute",
            "status": "planning_failed",
            "planning_reason": planning_reason,
            "plan": [],
            "step_results": [],
            "citations": [],
            "final_answer": final_answer,
        }
        return
    plan = state.get("plan", [])
    if plan:
        yield emit("📝 [任务规划] 已生成执行计划："), None
        for step in plan:
            yield emit(f"- {step['step_id']} | {step['required_capability']} | {step['goal']}"), None

    async for event in execute_plan_events(state):
        if event["type"] == "step_start":
            yield emit(
                f"🛠️ [步骤开始] {event['step_id']} | {event['capability']} | {event['goal']}"
            ), None
        elif event["type"] == "step_end":
            result = event["result"]
            yield emit(
                f"🛠️ [步骤执行] {result['step_id']} | {result['status']} | {result['goal']}"
            ), None
        elif event["type"] == "done":
            state = event["state"]

    before = list(state.get("step_results", []))
    state = await reflection_node(state)
    after = state.get("step_results", [])
    if after != before:
        yield emit("🔁 [反思修正] 已对证据不足或失败步骤进行补救。"), None
        for result in after:
            yield emit(f"📎 [反思结果] {result['step_id']} | {result['status']}"), None
    else:
        yield emit("🔁 [反思修正] 未发现需要补救的步骤。"), None

    yield emit("✍️ [答案生成] 正在生成最终结果。"), None
    prompt = build_synthesizer_prompt(state)
    model = get_model_by_choice(pae_model_choice)
    response = await model.ainvoke(prompt)
    final_answer = response.content.strip()
    state["final_answer"] = final_answer
    yield emit("✅ [PAE完成] Plan-and-Execute 子流程执行完成，已返回主循环。"), None
    yield "", {
        "mode": "plan_and_execute",
        "plan": plan,
        "step_results": state.get("step_results", []),
        "citations": state.get("citations", []),
        "final_answer": final_answer,
    }
