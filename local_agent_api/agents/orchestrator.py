from __future__ import annotations

"""把 planner / executor / reflection / synthesizer 串成复杂任务工作流。"""

from local_agent_api.core.llm import get_model_by_choice
from local_agent_api.agents.state import OrchestratorState
from local_agent_api.retrieval.citation import format_citations


SYNTHESIZER_PROMPT = """你是一个通用智能体中的综合分析器。请基于给定执行计划和每一步的证据，生成最终答案。

要求：
1. 明确回答用户问题
2. 优先使用检索证据，不要编造
3. 若证据不足，要明确指出
4. 输出格式必须遵循任务模式要求
5. 在答案末尾追加“证据来源”小节，列出来源路径或标识

用户问题：
{query}

计划模式：
{plan_mode}

输出格式要求：
{format_instruction}

Skill 输出提示：
{skill_output_hints}

执行计划：
{plan_text}

步骤结果：
{results_text}

证据来源：
{sources_text}

补充上下文（人格 / skill / 记忆 / 提示词）：
{runtime_context}
"""

# 根据query类型 提供不同答案格式
def _format_instruction_for_mode(plan_mode: str | None) -> str:
    """为不同任务模式提供最终答案格式要求。"""
    mode = plan_mode or "auto"
    if mode == "compare":
        return "使用“结论 / 对比维度 / 差异摘要 / 建议”结构，尽量用表格表达核心差异。"
    if mode == "extract":
        return "使用“字段名: 字段值”或 JSON 风格的结构化结果，并在字段缺失时明确标注未找到。"
    if mode == "report":
        return "使用“摘要 / 关键发现 / 详细分析 / 风险与不确定性 / 下一步建议”结构。"
    if mode == "research":
        return "使用“问题定义 / 主要发现 / 来源归纳 / 不确定性 / 下一步建议”结构。"
    return "优先使用清晰分点回答；若信息较多，可以补充短表格。"


# 拼 synthesizer 看的最终提示词
def build_synthesizer_prompt(state: OrchestratorState) -> str:
    """把计划、步骤证据和引用来源拼成 synthesizer 最终提示词。"""
    query = str(state["messages"][-1].content) if state.get("messages") else ""
    plan_mode = state.get("plan_mode", "auto")
    plan_lines = [
        f"{item['step_id']}: {item['goal']} ({item['status']})"
        for item in state.get("plan", [])
    ]
    result_lines = [
        f"{item['step_id']} | {item['status']} | {item['evidence']}"
        for item in state.get("step_results", [])
    ]
    return SYNTHESIZER_PROMPT.format(
        query=query,
        plan_mode=plan_mode,
        format_instruction=_format_instruction_for_mode(plan_mode),
        skill_output_hints="\n".join(state.get("output_format_hints", [])) or "无",
        plan_text="\n".join(plan_lines) or "无",
        results_text="\n\n".join(result_lines) or "无",
        sources_text=format_citations(state.get("citations", [])) if state.get("citations") else "无",
        runtime_context=state.get("runtime_system_prompt", "无"),
    )

# 生成最终答案（synthesizer）
async def synthesizer_node(state: OrchestratorState) -> OrchestratorState:
    """根据中间执行状态生成最终给用户看的答案。"""
    plan_mode = state.get("plan_mode", "auto")
    prompt = build_synthesizer_prompt(state)
    plan_lines = [
        f"{item['step_id']}: {item['goal']} ({item['status']})"
        for item in state.get("plan", [])
    ]
    result_lines = [
        f"{item['step_id']} | {item['status']} | {item['evidence']}"
        for item in state.get("step_results", [])
    ]

    try:
        model = get_model_by_choice(state.get("model_choice", "deepseek"))
        response = await model.ainvoke(prompt)
        final_answer = response.content.strip()
    except Exception:
        fallback_sections = [
            "已进入 Plan-and-Execute 模式。",
            f"计划模式：{plan_mode}",
            "",
            "计划步骤：",
            "\n".join(plan_lines) or "无",
            "",
            "步骤结果：",
            "\n\n".join(result_lines) or "无",
            "",
            "证据来源：",
            format_citations(state.get("citations", [])) if state.get("citations") else "无",
        ]
        final_answer = "\n".join(fallback_sections)

    return {
        **state,
        "final_answer": final_answer,
    }
