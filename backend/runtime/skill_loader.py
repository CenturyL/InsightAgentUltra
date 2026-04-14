from __future__ import annotations

"""兼容 Claude Code 风格 skill package 的技能发现与解析。"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml

from backend.core.config import settings

GENERIC_SKILL_TOKENS = {
    "skill",
    "skills",
    "guide",
    "using",
    "use",
    "when",
    "tool",
    "tools",
    "create",
    "creating",
    "build",
    "building",
    "builder",
    "server",
    "servers",
    "mcp",
}


def _backend_root() -> Path:
    return Path(settings.BACKEND_ROOT)


@dataclass
class SkillMetadata:
    name: str
    description: str
    argument_hint: str | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    effort: str | None = None
    context: str | None = None
    agent: str | None = None


@dataclass
class SkillPackage:
    root_dir: Path
    skill_md_path: Path
    metadata: SkillMetadata
    body: str | None = None
    support_files: dict[str, Path] = field(default_factory=dict)


@dataclass
class ActivatedSkill:
    package: SkillPackage
    reason: str
    arguments: str | None = None
    score: int = 0


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, flags=re.S)
    if not match:
        return {}, raw.strip()
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return frontmatter, match.group(2).strip()


def _normalize_metadata(frontmatter: dict[str, Any], fallback_name: str) -> SkillMetadata:
    return SkillMetadata(
        name=str(frontmatter.get("name") or fallback_name),
        description=str(frontmatter.get("description") or "").strip(),
        argument_hint=_optional_text(frontmatter.get("argument-hint")),
        disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
        user_invocable=bool(frontmatter.get("user-invocable", True)),
        allowed_tools=_ensure_str_list(frontmatter.get("allowed-tools")),
        model=_optional_text(frontmatter.get("model")),
        effort=_optional_text(frontmatter.get("effort")),
        context=_optional_text(frontmatter.get("context")),
        agent=_optional_text(frontmatter.get("agent")),
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _skill_dirs() -> list[Path]:
    skill_dir = _backend_root() / ".claude" / "skills"
    return [skill_dir] if skill_dir.exists() else []


def _iter_skill_files() -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for base_dir in _skill_dirs():
        for skill_file in sorted(base_dir.glob("*/SKILL.md")):
            resolved = skill_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(skill_file)
        for legacy_file in sorted(base_dir.glob("*.md")):
            resolved = legacy_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(legacy_file)
    return discovered


def parse_claude_skill(path: Path) -> SkillPackage:
    raw = path.read_text(encoding="utf-8").strip()
    frontmatter, body = _parse_frontmatter(raw)
    fallback_name = path.parent.name if path.name == "SKILL.md" else path.stem
    metadata = _normalize_metadata(frontmatter, fallback_name=fallback_name)
    support_files: dict[str, Path] = {}
    root_dir = path.parent if path.name == "SKILL.md" else path.parent / path.stem
    if path.name == "SKILL.md":
        for support_name in ("reference.md", "examples.md"):
            support_path = path.parent / support_name
            if support_path.exists():
                support_files[support_name] = support_path
        templates_dir = path.parent / "templates"
        if templates_dir.exists():
            support_files["templates"] = templates_dir
        scripts_dir = path.parent / "scripts"
        if scripts_dir.exists():
            support_files["scripts"] = scripts_dir
    return SkillPackage(
        root_dir=root_dir,
        skill_md_path=path,
        metadata=metadata,
        body=None if path.name == "SKILL.md" else raw,
        support_files=support_files,
    )


def discover_skill_packages() -> list[SkillPackage]:
    packages: list[SkillPackage] = []
    for skill_file in _iter_skill_files():
        try:
            packages.append(parse_claude_skill(skill_file))
        except Exception:
            continue
    return packages


def build_skill_catalog() -> list[SkillPackage]:
    catalog = discover_skill_packages()
    return sorted(catalog, key=lambda item: item.metadata.name.lower())


def load_skill_body(package: SkillPackage) -> SkillPackage:
    if package.body is not None:
        return package
    raw = package.skill_md_path.read_text(encoding="utf-8").strip()
    _, body = _parse_frontmatter(raw)
    return SkillPackage(
        root_dir=package.root_dir,
        skill_md_path=package.skill_md_path,
        metadata=package.metadata,
        body=body if package.skill_md_path.name == "SKILL.md" else raw,
        support_files=package.support_files,
    )


def skill_prompt_path(package: SkillPackage) -> str:
    root = _backend_root()
    try:
        return str(package.skill_md_path.relative_to(root))
    except Exception:
        return str(package.skill_md_path)


def _skill_search_text(package: SkillPackage) -> str:
    metadata = package.metadata
    parts = [metadata.name, metadata.description, metadata.argument_hint or ""]
    return " ".join(parts).lower()


def _should_use_match_token(token: str) -> bool:
    token = token.strip().lower()
    if len(token) < 2:
        return False
    if token in GENERIC_SKILL_TOKENS:
        return False
    if re.fullmatch(r"[a-z0-9]+", token):
        return len(token) >= 4
    return True


def _matches_name_intent(metadata: SkillMetadata, query_lower: str) -> bool:
    name = metadata.name.lower()
    if name in query_lower:
        return True

    create_words = ("创建", "新建", "构建", "搭建", "开发", "实现", "编写", "build", "create", "implement")
    parts = [item for item in re.split(r"[-_/]+", name) if item]
    if len(parts) >= 2:
        base_parts = parts[:-1]
        suffix = parts[-1]
        if suffix in {"builder", "creator"}:
            if any(part in query_lower for part in base_parts) and any(word in query_lower for word in create_words):
                return True
    return False


def _score_package(
    package: SkillPackage,
    query: str,
    plan_mode: str | None,
    complexity: str,
    available_tool_names: list[str],
) -> tuple[int, str] | None:
    metadata = package.metadata
    if not metadata.description:
        return None

    if metadata.disable_model_invocation:
        explicit_markers = {
            f"${metadata.name.lower()}",
            f"skill:{metadata.name.lower()}",
            f"/{metadata.name.lower()}",
        }
        query_lower = query.lower()
        if not any(marker in query_lower for marker in explicit_markers):
            return None

    if metadata.allowed_tools and not any(tool_name in available_tool_names for tool_name in metadata.allowed_tools):
        return None

    score = 0
    reasons: list[str] = []
    query_lower = query.lower()
    search_text = _skill_search_text(package)
    if _matches_name_intent(metadata, query_lower):
        score += 8
        reasons.append("命中 skill 名称")
    if metadata.argument_hint and metadata.argument_hint.lower() in query_lower:
        score += 5
        reasons.append("命中 argument_hint")
    for token in re.split(r"[\s,，。；;：:/_-]+", search_text):
        token = token.strip()
        if not _should_use_match_token(token):
            continue
        if token in query_lower:
            score += 2
            reasons.append(f"命中描述词:{token}")
    mode = (plan_mode or "auto").lower()
    if mode != "auto" and mode in search_text:
        score += 4
        reasons.append("命中 plan_mode")
    if complexity in {"medium", "high"} and any(word in search_text for word in ("research", "comparison", "extraction", "general")):
        score += 1

    if score <= 0:
        return None
    return score, "；".join(dict.fromkeys(reasons)) or "描述匹配"


def match_skill_candidates(
    query: str,
    plan_mode: str | None,
    complexity: str,
    available_tool_names: list[str],
    limit: int = 3,
) -> tuple[list[SkillPackage], list[ActivatedSkill]]:
    catalog = build_skill_catalog()
    activated: list[ActivatedSkill] = []
    for package in catalog:
        matched = _score_package(package, query, plan_mode, complexity, available_tool_names)
        if matched is None:
            continue
        score, reason = matched
        activated.append(
            ActivatedSkill(
                package=load_skill_body(package),
                reason=reason,
                score=score,
            )
        )
    activated.sort(key=lambda item: (-item.score, item.package.metadata.name.lower()))
    return catalog, activated[:limit]
