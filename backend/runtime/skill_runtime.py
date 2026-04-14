from __future__ import annotations

"""skill 的 progressive disclosure 运行时选择与语义编译。"""

from dataclasses import dataclass, field

from backend.runtime.skill_loader import (
    ActivatedSkill,
    SkillPackage,
    load_skill_body,
    skill_prompt_path,
)


@dataclass
class SkillRuntimeEffects:
    prompt_sections: list[str] = field(default_factory=list)
    planner_hints: list[str] = field(default_factory=list)
    executor_hints: list[str] = field(default_factory=list)
    output_format_hints: list[str] = field(default_factory=list)


def format_skill_catalog(catalog: list[SkillPackage]) -> str:
    if not catalog:
        return ""
    lines = []
    for package in catalog:
        meta = package.metadata
        if not meta.description:
            continue
        lines.append(
            f"- name: {meta.name} | description: {meta.description} | path: {skill_prompt_path(package)}"
        )
    if not lines:
        return ""
    return "【可用 Skills（仅元数据）】\n" + "\n".join(lines)


def select_skills_from_route(
    *,
    catalog: list[SkillPackage],
    selected_rows: list[dict[str, str]] | None,
    limit: int = 3,
) -> list[ActivatedSkill]:
    if not catalog or not selected_rows:
        return []
    by_name = {package.metadata.name: package for package in catalog}
    activated: list[ActivatedSkill] = []
    seen: set[str] = set()
    for row in selected_rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name or name in seen or name not in by_name:
            continue
        seen.add(name)
        activated.append(
            ActivatedSkill(
                package=load_skill_body(by_name[name]),
                reason=str(row.get("reason", "")).strip() or "LLM 认为该 skill 与当前请求相关",
                score=max(limit - len(activated), 0),
            )
        )
        if len(activated) >= limit:
            break
    return activated


def render_skill_prompt_sections(skills: list[ActivatedSkill]) -> list[str]:
    sections: list[str] = []
    for skill in skills:
        body = (skill.package.body or "").strip()
        if not body:
            continue
        sections.append(
            f"### Activated Skill: {skill.package.metadata.name}\n"
            f"path: {skill_prompt_path(skill.package)}\n"
            f"触发原因：{skill.reason}\n\n"
            f"{body}"
        )
    return sections


def compile_skill_effects(skills: list[ActivatedSkill]) -> SkillRuntimeEffects:
    effects = SkillRuntimeEffects()
    if not skills:
        return effects

    effects.prompt_sections = render_skill_prompt_sections(skills)
    for skill in skills:
        meta = skill.package.metadata
        effects.planner_hints.append(f"{meta.name}: {meta.description}")
        effects.executor_hints.append(f"{meta.name}: 遵循该 skill 的完整指令。")
        if meta.argument_hint:
            effects.executor_hints.append(f"{meta.name} argument-hint: {meta.argument_hint}")
        effects.output_format_hints.append(f"{meta.name}: 输出应遵循该 skill 正文中的结构要求。")
    return effects
