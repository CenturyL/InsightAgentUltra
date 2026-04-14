from __future__ import annotations

"""统一拼装 system prompt、insight、工具策略。"""

from backend.runtime.skill_loader import ActivatedSkill, SkillPackage
from backend.runtime.skill_runtime import SkillRuntimeEffects, format_skill_catalog
from backend.services.runtime_assets_service import load_insight


def format_activated_skill_sections(skills: list[ActivatedSkill]) -> str:
    if not skills:
        return ""
    sections = []
    for skill in skills:
        body = (skill.package.body or "").strip()
        if not body:
            continue
        sections.append(
            f"### Activated Skill: {skill.package.metadata.name}\n"
            f"触发原因：{skill.reason}\n\n"
            f"{body}"
        )
    if not sections:
        return ""
    return "【当前激活的 Skills】\n" + "\n\n".join(sections)


def format_output_schema_section(skill_effects: SkillRuntimeEffects) -> str:
    if not skill_effects.output_format_hints:
        return ""
    return "【Skill 输出约束】\n" + "\n".join(f"- {item}" for item in skill_effects.output_format_hints)


def build_runtime_system_prompt(
    query: str,
    plan_mode: str | None,
    skill_catalog: list[SkillPackage],
    activated_skills: list[ActivatedSkill],
    skill_effects: SkillRuntimeEffects,
    memory_text: str,
    available_tool_names: list[str],
    complexity: str,
    pae_route_action: str,
    pae_route_reason: str,
    user_id: str = "",
) -> str:
    insight_text = load_insight(user_id).strip()
    skill_catalog_text = format_skill_catalog(skill_catalog)
    activated_skills_text = format_activated_skill_sections(activated_skills)
    output_schema_text = format_output_schema_section(skill_effects)
    mode = plan_mode or "auto"
    tool_text = "、".join(available_tool_names)

    sections = [
        "你是 InsightAgentUltra，一个通用智能体平台中的主执行智能体。",
        "你的能力来源于 tools、skills、记忆、workspace 提示词和 MCP 接入的外部能力。",
        f"当前 plan_mode={mode}。plan_mode 只表示规划/输出倾向，不表示强制路径。",
        f"当前复杂度评估：complexity={complexity}。",
        f"当前 PAE 路由建议：{pae_route_action}。",
        f"当前 PAE 路由原因：{pae_route_reason or '无'}。",
        f"当前可用工具：{tool_text or '无'}。",
        "PAE 是否启用已经由独立路由器先做了一次判断。你应优先参考该判断，而不是依据本地启发式词规则自行脑补。",
        "如果当前 PAE 路由建议是 run_plan_and_execute，则优先进入 PAE；如果是 direct_or_simple_tools，则优先先用普通工具或直接回答完成任务。",
        "如果当前 PAE 路由建议是 direct_or_simple_tools，那么一旦最近一次工具结果已经足以回答问题，就立刻直接输出最终答案。",
        "在 direct_or_simple_tools 下，禁止长篇铺垫、禁止重复解释计划、禁止在工具前后反复描述「准备怎么做」。",
        "停止条件：如果已有足够证据回答问题、最近一次工具返回已提供完整答案候选、继续调用工具收益很低、或证据不足但已完成必要补救，则应直接收尾输出。",
        "禁止在证据不足时编造；能用工具和记忆解决的问题，优先用工具和记忆。",
    ]

    if skill_catalog_text:
        sections.append(skill_catalog_text)
    if activated_skills_text:
        sections.append(activated_skills_text)
    if output_schema_text:
        sections.append(output_schema_text)
    if insight_text:
        sections.append(f"【Insight（Persona / Style / Memory）】\n{insight_text[:3600]}")
    if memory_text:
        sections.append(memory_text)
    sections.append(f"【当前用户请求】\n{query}")
    return "\n\n".join(section for section in sections if section)
