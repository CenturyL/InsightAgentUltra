from __future__ import annotations

"""运行时资产读写：persona / markdown memory / skills。"""

from pathlib import Path

from backend.core.config import settings


def _workspace_root() -> Path:
    return Path(settings.WORKSPACE_ROOT)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _skill_roots() -> dict[str, Path]:
    root = _workspace_root()
    backend = Path(settings.BACKEND_ROOT)
    return {
        "project": root / "skills",
        "claude": backend / ".claude" / "skills",
    }


def _safe_skill_name(filename: str) -> str:
    normalized = filename.strip().replace("\\", "/").lstrip("/")
    if not normalized:
        normalized = "untitled/SKILL.md"
    if normalized.endswith("/"):
        normalized = f"{normalized}SKILL.md"
    path = Path(normalized)
    if path.suffix.lower() != ".md":
        if path.name == path.stem:
            path = path / "SKILL.md"
        else:
            path = path.with_suffix(".md")
    return str(path)


def _iter_skill_assets(skills_dir: Path) -> list[Path]:
    files: list[Path] = []
    if not skills_dir.exists():
        return files
    files.extend(sorted(skills_dir.glob("*/SKILL.md")))
    files.extend(sorted(skills_dir.glob("*.md")))
    return files


def _prune_empty_dirs(path: Path, stop_at: Path) -> None:
    current = path.parent
    while current != stop_at and current.exists():
        if any(current.iterdir()):
            break
        current.rmdir()
        current = current.parent


def load_runtime_assets() -> dict:
    root = _workspace_root()
    persona_dir = root / "persona"
    memory_dir = root / "memory"

    skills: list[dict[str, str]] = []
    for source, skills_dir in _skill_roots().items():
        if not skills_dir.exists():
            continue
        for file_path in _iter_skill_assets(skills_dir):
            skills.append(
                {
                    "filename": str(file_path.relative_to(skills_dir)),
                    "content": _read_text(file_path),
                    "source": source,
                }
            )
    skills.sort(key=lambda item: (item.get("source", ""), item.get("filename", "")))

    return {
        "agents_md": _read_text(persona_dir / "AGENTS.md"),
        "soul_md": _read_text(persona_dir / "SOUL.md"),
        "memory_md": _read_text(memory_dir / "MEMORY.md"),
        "skills": skills,
    }


def save_runtime_assets(
    *,
    agents_md: str,
    soul_md: str,
    memory_md: str,
    skills: list[dict[str, str]],
) -> dict:
    root = _workspace_root()
    persona_dir = root / "persona"
    memory_dir = root / "memory"

    agents_path = persona_dir / "AGENTS.md"
    soul_path = persona_dir / "SOUL.md"
    memory_path = memory_dir / "MEMORY.md"

    _ensure_parent(agents_path)
    _ensure_parent(soul_path)
    _ensure_parent(memory_path)
    for skills_dir in _skill_roots().values():
        skills_dir.mkdir(parents=True, exist_ok=True)

    agents_path.write_text(agents_md or "", encoding="utf-8")
    soul_path.write_text(soul_md or "", encoding="utf-8")
    memory_path.write_text(memory_md or "", encoding="utf-8")

    desired_paths: dict[str, set[Path]] = {source: set() for source in _skill_roots()}
    for skill in skills:
        source = str(skill.get("source") or "project").strip().lower()
        if source not in _skill_roots():
            source = "project"
        filename = _safe_skill_name(skill.get("filename", "untitled.md"))
        content = skill.get("content", "")
        target = _skill_roots()[source] / filename
        _ensure_parent(target)
        target.write_text(content, encoding="utf-8")
        desired_paths[source].add(target.resolve())

    for source, skills_dir in _skill_roots().items():
        for existing in _iter_skill_assets(skills_dir):
            if existing.resolve() not in desired_paths[source]:
                existing.unlink(missing_ok=True)
                _prune_empty_dirs(existing, skills_dir)

    return load_runtime_assets()
