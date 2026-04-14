from __future__ import annotations

"""MCP 配置读写与状态查看。"""

import json
from pathlib import Path

from backend.core.config import settings
from backend.runtime.mcp_config import load_mcp_config, resolve_project_mcp_json
from backend.runtime.tool_registry import get_mcp_status, initialize_runtime_tools


def _mcp_path() -> Path:
    path = resolve_project_mcp_json()
    if path is not None:
        return path
    if settings.MCP_CONFIG_PATH:
        return Path(settings.MCP_CONFIG_PATH)
    return Path(settings.BACKEND_ROOT) / ".mcp.json"


def load_runtime_mcp_config() -> dict:
    path = _mcp_path()
    raw_text = path.read_text(encoding="utf-8") if path.exists() else "{\n  \"mcpServers\": {}\n}"
    parsed = load_mcp_config()
    return {
        "config_text": raw_text,
        "servers": [
            {
                "server_name": name,
                "transport": server.transport,
                "command": server.command,
                "args": server.args,
                "cwd": server.cwd,
                "url": server.url,
                "headers": server.headers,
            }
            for name, server in sorted(parsed.servers.items())
        ],
        "status": get_mcp_status(),
    }


async def save_runtime_mcp_config(config_text: str) -> dict:
    payload = json.loads(config_text or "{\n  \"mcpServers\": {}\n}")
    path = _mcp_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    await initialize_runtime_tools()
    return load_runtime_mcp_config()


async def reload_runtime_mcp_config() -> dict:
    await initialize_runtime_tools()
    return load_runtime_mcp_config()
