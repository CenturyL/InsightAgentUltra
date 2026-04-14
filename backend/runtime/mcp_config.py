from __future__ import annotations

"""标准 .mcp.json 配置读取。"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.core.config import settings


@dataclass
class MCPServerConfig:
    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class MCPConfig:
    servers: dict[str, MCPServerConfig] = field(default_factory=dict)


def resolve_project_mcp_json() -> Path | None:
    if settings.MCP_CONFIG_PATH:
        candidate = Path(settings.MCP_CONFIG_PATH)
        return candidate if candidate.exists() else None
    candidate = Path(settings.BACKEND_ROOT) / ".mcp.json"
    return candidate if candidate.exists() else None


def expand_env_vars(value: str) -> str:
    pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        default = match.group(2) or ""
        return os.getenv(key, default)

    return pattern.sub(repl, value)


def _expand_value(value: Any) -> Any:
    if isinstance(value, str):
        return expand_env_vars(value)
    if isinstance(value, dict):
        return {str(k): _expand_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_value(item) for item in value]
    return value


def load_mcp_config() -> MCPConfig:
    path = resolve_project_mcp_json()
    if path is None:
        return MCPConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    servers_payload = payload.get("mcpServers", {}) or {}
    servers: dict[str, MCPServerConfig] = {}
    for name, raw in servers_payload.items():
        raw = _expand_value(raw or {})
        transport = "http" if raw.get("url") else "stdio"
        servers[name] = MCPServerConfig(
            name=name,
            transport=transport,
            command=raw.get("command"),
            args=list(raw.get("args", []) or []),
            cwd=raw.get("cwd"),
            env=dict(raw.get("env", {}) or {}),
            url=raw.get("url"),
            headers=dict(raw.get("headers", {}) or {}),
        )
    return MCPConfig(servers=servers)
