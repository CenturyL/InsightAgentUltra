from __future__ import annotations

import json

from local_agent_api.core.config import settings
from local_agent_api.runtime.mcp_config import expand_env_vars, load_mcp_config


def test_expand_env_vars_with_default(monkeypatch):
    monkeypatch.delenv("MCP_API_KEY", raising=False)
    assert expand_env_vars("${MCP_API_KEY:-demo}") == "demo"


def test_load_mcp_config_reads_project_file(tmp_path, monkeypatch):
    payload = {
        "mcpServers": {
            "demo-http": {
                "url": "https://example.com/mcp",
                "headers": {"Authorization": "Bearer ${TOKEN:-abc}"},
            },
            "demo-stdio": {
                "command": "python",
                "args": ["server.py"],
                "cwd": "./mcp",
            },
        }
    }
    (tmp_path / ".mcp.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "MCP_CONFIG_PATH", None)
    config = load_mcp_config()
    assert set(config.servers) == {"demo-http", "demo-stdio"}
    assert config.servers["demo-http"].transport == "http"
    assert config.servers["demo-http"].headers["Authorization"] == "Bearer abc"
    assert config.servers["demo-stdio"].transport == "stdio"
