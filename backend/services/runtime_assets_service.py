from __future__ import annotations

"""运行时资产读写：insight.md (per user) / skills。"""

from pathlib import Path

from backend.core.config import settings


def _backend_root() -> Path:
    return Path(settings.BACKEND_ROOT)


def _workspace_root() -> Path:
    return Path(settings.WORKSPACE_ROOT)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


# ── insight.md: per-user, stored in backend/insights/{user_id}.md ──────────

def _insight_path(user_id: str) -> Path:
    safe_id = user_id.strip() or "_default"
    return _backend_root() / "insights" / f"{safe_id}.md"


def _default_insight() -> str:
    default_path = _backend_root() / "insights" / "_default.md"
    return _read_text(default_path)


def load_insight(user_id: str) -> str:
    path = _insight_path(user_id)
    content = _read_text(path)
    if not content:
        content = _default_insight()
    return content


def save_insight(user_id: str, content: str) -> str:
    path = _insight_path(user_id)
    _ensure_parent(path)
    path.write_text(content or "", encoding="utf-8")
    return _read_text(path)


# ── skills ─────────────────────────────────────────────────────────────────

def _skill_roots() -> dict[str, Path]:
    root = _workspace_root()
    backend = _backend_root()
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


# ── unified load / save ────────────────────────────────────────────────────

def load_runtime_assets(user_id: str = "") -> dict:
    insight_md = load_insight(user_id)

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
        "insight_md": insight_md,
        "skills": skills,
    }


def save_runtime_assets(
    *,
    user_id: str = "",
    insight_md: str,
    skills: list[dict[str, str]],
) -> dict:
    save_insight(user_id, insight_md)

    desired_paths: dict[str, set[Path]] = {source: set() for source in _skill_roots()}
    for skills_dir in _skill_roots().values():
        skills_dir.mkdir(parents=True, exist_ok=True)

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

    return load_runtime_assets(user_id)
