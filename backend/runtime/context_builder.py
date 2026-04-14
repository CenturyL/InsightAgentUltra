from __future__ import annotations

"""统一构造最终送给 LLM 的动态上下文。"""

from dataclasses import dataclass

from backend.runtime.memory_bridge import (
    format_memory_sections,
    search_long_term_memory_text,
    search_markdown_memory_text,
)
from backend.runtime.engine import RuntimeRouteDecision, classify_complexity, judge_runtime_route
from backend.runtime.prompt_manager import build_runtime_system_prompt
from backend.runtime.skill_loader import ActivatedSkill, SkillPackage, build_skill_catalog, skill_prompt_path
from backend.runtime.skill_runtime import (
    SkillRuntimeEffects,
    compile_skill_effects,
    select_skills_from_route,
)


@dataclass
class RuntimeContext:
    system_prompt: str
    skill_catalog: list[SkillPackage]
    activated_skills: list[ActivatedSkill]
    skill_effects: SkillRuntimeEffects
    long_term_memories: list[str]
    markdown_memories: list[str]
    complexity: str
    pae_route_action: str
    pae_route_reason: str


def build_skill_catalog_rows(catalog: list[SkillPackage]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for package in catalog:
        meta = package.metadata
        if not meta.description:
            continue
        rows.append(
            {
                "name": meta.name,
                "description": meta.description,
                "path": skill_prompt_path(package),
            }
        )
    return rows


async def build_runtime_context(
    query: str,
    user_id: str,
    plan_mode: str | None,
    available_tool_names: list[str],
    route_decision: RuntimeRouteDecision | dict | None = None,
) -> RuntimeContext:
    complexity = classify_complexity(query, plan_mode)
    skill_catalog = build_skill_catalog()
    if isinstance(route_decision, dict) and route_decision:
        route_decision = RuntimeRouteDecision(
            pae_action=str(route_decision.get("pae_action", "")).strip() or "direct_or_simple_tools",
            pae_reason=str(route_decision.get("pae_reason", "")).strip() or "统一路由器未返回明确原因。",
            selected_skills=list(route_decision.get("selected_skills", []) or []),
        )
    if route_decision is None:
        route_decision = await judge_runtime_route(
            query=query,
            plan_mode=plan_mode,
            complexity=complexity,
            available_tool_names=available_tool_names,
            skill_catalog_rows=build_skill_catalog_rows(skill_catalog),
        )
    activated_skills = select_skills_from_route(
        catalog=skill_catalog,
        selected_rows=route_decision.selected_skills,
    )
    skill_effects = compile_skill_effects(activated_skills)
    long_term_memories = search_long_term_memory_text(user_id, query, k=3)
    markdown_memories = search_markdown_memory_text(query, k=2, user_id=user_id)
    memory_text = format_memory_sections(long_term_memories, markdown_memories)

    system_prompt = build_runtime_system_prompt(
        query=query,
        plan_mode=plan_mode,
        skill_catalog=skill_catalog,
        activated_skills=activated_skills,
        skill_effects=skill_effects,
        memory_text=memory_text,
        available_tool_names=available_tool_names,
        complexity=complexity,
        pae_route_action=route_decision.pae_action,
        pae_route_reason=route_decision.pae_reason,
        user_id=user_id,
    )
    return RuntimeContext(
        system_prompt=system_prompt,
        skill_catalog=skill_catalog,
        activated_skills=activated_skills,
        skill_effects=skill_effects,
        long_term_memories=long_term_memories,
        markdown_memories=markdown_memories,
        complexity=complexity,
        pae_route_action=route_decision.pae_action,
        pae_route_reason=route_decision.pae_reason,
    )
