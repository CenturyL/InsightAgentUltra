from __future__ import annotations

"""执行 planner 生成的步骤：按 capability 分发工具，而不是只做 RAG。"""

from typing import Any, AsyncGenerator, Callable

from local_agent_api.agents.state import OrchestratorState, StepResult
from local_agent_api.core.llm import get_model_by_choice
from local_agent_api.services.tool_context import set_tool_request_context, reset_tool_request_context


def _build_step_query(user_query: str, step_goal: str) -> str:
    return f"{user_query}\n\n请优先完成该子任务：{step_goal}"


async def _run_analysis_step(state: OrchestratorState, step_query: str, step_goal: str) -> str:
    model = get_model_by_choice(state.get("model_choice", "deepseek"))
    prompt = (
        "你正在执行计划中的一个分析步骤。\n"
        f"原始问题：{state['messages'][-1].content if state.get('messages') else ''}\n"
        f"当前子目标：{step_goal}\n"
        f"子任务查询：{step_query}\n"
        f"执行提示：{'; '.join(state.get('executor_hints', [])) or '无'}\n"
        "请给出简洁、结构化的中间结论；若证据不足请明确说明。"
    )
    response = await model.ainvoke(prompt)
    return response.content.strip()


async def execute_plan_events(
    state: OrchestratorState,
    trace_sink: Callable[[str], None] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """逐步执行计划，并在开始/完成时产出事件。"""
    plan = state.get("plan", [])
    user_query = str(state["messages"][-1].content) if state.get("messages") else ""
    results: list[StepResult] = []
    collected_docs = []
    citations: list[dict] = list(state.get("citations", []))

    context_tokens = set_tool_request_context(
        thread_id=state.get("thread_id", "default"),
        user_id=state.get("user_id", ""),
        plan_mode=state.get("plan_mode"),
        model_choice=state.get("model_choice", "local_qwen"),
        metadata_filters=state.get("metadata_filters"),
        in_pae=True,
    )
    try:
        from local_agent_api.runtime.tool_registry import INTERNAL_TOOL_HANDLERS

        for step in plan:
            capability = step.get("required_capability", "rag_search")
            step_query = _build_step_query(user_query, step["goal"])
            start_line = f"🛠️ [步骤开始] {step['step_id']} | {capability} | {step['goal']}"
            if trace_sink:
                trace_sink(start_line)
            yield {
                "type": "step_start",
                "step_id": step["step_id"],
                "capability": capability,
                "goal": step["goal"],
            }
            try:
                if capability in INTERNAL_TOOL_HANDLERS:
                    evidence, artifact = INTERNAL_TOOL_HANDLERS[capability](step_query)
                    status = "completed" if evidence else "partial"
                    if isinstance(artifact, dict):
                        collected_docs.extend([item for item in artifact.get("docs", []) if hasattr(item, "metadata")])
                        citations.extend(artifact.get("citations", []))
                    elif isinstance(artifact, list):
                        collected_docs.extend([item for item in artifact if hasattr(item, "metadata")])
                elif capability == "analysis":
                    evidence = await _run_analysis_step(state, step_query, step["goal"])
                    status = "completed" if evidence else "partial"
                elif capability == "synthesis":
                    evidence = "最终综合结论将在 synthesizer 阶段生成。"
                    status = "completed"
                else:
                    evidence, artifact = INTERNAL_TOOL_HANDLERS["rag_search"](step_query)
                    status = "completed" if evidence else "partial"
                    if isinstance(artifact, list):
                        collected_docs.extend([item for item in artifact if hasattr(item, "metadata")])

                step["status"] = status
                result: StepResult = {
                    "step_id": step["step_id"],
                    "goal": step["goal"],
                    "query": step_query,
                    "evidence": evidence or "未获得有效结果。",
                    "capability": capability,
                    "status": status,
                }
                results.append(result)
            except Exception as exc:
                step["status"] = "failed"
                result = {
                    "step_id": step["step_id"],
                    "goal": step["goal"],
                    "query": step_query,
                    "evidence": f"步骤执行失败：{exc}",
                    "capability": capability,
                    "status": "failed",
                }
                results.append(result)

            end_line = f"🛠️ [步骤执行] {result['step_id']} | {result['status']} | {result['goal']}"
            if trace_sink:
                trace_sink(end_line)
            yield {"type": "step_end", "result": result}
    finally:
        reset_tool_request_context(context_tokens)

    yield {
        "type": "done",
        "state": {
            **state,
            "plan": plan,
            "step_results": results,
            "retrieved_docs": collected_docs,
            "citations": citations,
            "current_step": len(plan),
        },
    }


async def executor_node(
    state: OrchestratorState,
    trace_sink: Callable[[str], None] | None = None,
) -> OrchestratorState:
    """逐步执行计划，并把工具结果或分析结果写回状态。"""
    final_state = state
    async for event in execute_plan_events(state, trace_sink=trace_sink):
        if event["type"] == "done":
            final_state = event["state"]
    return final_state
