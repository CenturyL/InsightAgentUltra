from __future__ import annotations

from pathlib import Path

from backend.runtime.skill_loader import build_skill_catalog, match_skill_candidates
from backend.core.config import settings


def test_build_skill_catalog_reads_claude_skill_packages(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills" / "review-pr"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
name: review-pr
description: Review pull requests and summarize issues
allowed-tools:
  - web_search
---

Review carefully.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path))
    catalog = build_skill_catalog()
    assert len(catalog) == 1
    assert catalog[0].metadata.name == "review-pr"
    assert catalog[0].metadata.allowed_tools == ["web_search"]


def test_match_skill_candidates_activates_by_name_and_description(tmp_path, monkeypatch):
    skills_dir = tmp_path / "skills" / "comparison"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text(
        """---
name: comparison
description: Compare documents and summarize differences
---

Compare carefully.
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path))
    catalog, activated = match_skill_candidates(
        query="请做一个 comparison，对比两个方案差异",
        plan_mode="compare",
        complexity="high",
        available_tool_names=["rag_search", "run_plan_and_execute"],
    )
    assert len(catalog) == 1
    assert len(activated) == 1
    assert activated[0].package.metadata.name == "comparison"
