from __future__ import annotations

"""运行时资产读写：insight.md (per user) / skills。"""

from pathlib import Path

from backend.core.config import settings


def _backend_root() -> Path:
    return Path(settings.BACKEND_ROOT)


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


# ── skills (single source: backend/.claude/skills/) ───────────────────────

def _skills_dir() -> Path:
    return _backend_root() / ".claude" / "skills"


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


def _iter_skill_assets(base: Path) -> list[Path]:
    files: list[Path] = []
    if not base.exists():
        return files
    files.extend(sorted(base.glob("*/SKILL.md")))
    files.extend(sorted(base.glob("*.md")))
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
    base = _skills_dir()

    skills: list[dict[str, str]] = []
    if base.exists():
        for file_path in _iter_skill_assets(base):
            skills.append(
                {
                    "filename": str(file_path.relative_to(base)),
                    "content": _read_text(file_path),
                    "source": "claude",
                }
            )
    skills.sort(key=lambda item: item.get("filename", ""))

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

    base = _skills_dir()
    base.mkdir(parents=True, exist_ok=True)

    desired_paths: set[Path] = set()
    for skill in skills:
        filename = _safe_skill_name(skill.get("filename", "untitled.md"))
        content = skill.get("content", "")
        target = base / filename
        _ensure_parent(target)
        target.write_text(content, encoding="utf-8")
        desired_paths.add(target.resolve())

    for existing in _iter_skill_assets(base):
        if existing.resolve() not in desired_paths:
            existing.unlink(missing_ok=True)
            _prune_empty_dirs(existing, base)

    return load_runtime_assets(user_id)
