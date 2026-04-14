from __future__ import annotations

"""读取 workspace 中的 MCP 补充说明，仅作人类/模型高层提示。"""

from pathlib import Path

from local_agent_api.core.config import settings


def load_mcp_notes() -> str:
    root = Path(settings.WORKSPACE_ROOT)
    candidates = [
        root / "mcp" / "TOOLS.md",
        root / "mcp" / "tools.md",
    ]
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if content:
            return f"【MCP 补充说明】\n{content[:1800]}"
    return ""
