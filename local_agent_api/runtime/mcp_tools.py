from __future__ import annotations

"""把 MCP tool spec 适配为 LangChain tool。"""

from typing import Any

from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from local_agent_api.core.config import settings
from local_agent_api.runtime.mcp_client import MCPClientSession, MCPToolSpec, call_mcp_tool


def _field_from_schema(name: str, schema: dict[str, Any]) -> tuple[Any, Any]:
    schema_type = schema.get("type")
    description = schema.get("description", "")
    default = ... if name in schema.get("__required__", set()) else None
    if schema_type == "integer":
        return int, Field(default=default, description=description)
    if schema_type == "number":
        return float, Field(default=default, description=description)
    if schema_type == "boolean":
        return bool, Field(default=default, description=description)
    if schema_type == "array":
        return list[Any], Field(default=default if default is ... else [], description=description)
    if schema_type == "object":
        return dict[str, Any], Field(default=default if default is ... else {}, description=description)
    return str, Field(default=default, description=description)


def _build_args_schema(tool_spec: MCPToolSpec) -> type[BaseModel]:
    input_schema = dict(tool_spec.input_schema or {})
    properties = dict(input_schema.get("properties", {}) or {})
    required = set(input_schema.get("required", []) or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop_schema in properties.items():
        schema = dict(prop_schema or {})
        schema["__required__"] = required
        fields[name] = _field_from_schema(name, schema)
    if not fields:
        fields["payload"] = (dict[str, Any], Field(default_factory=dict, description="MCP tool arguments"))
    model_name = f"MCPArgs_{tool_spec.server_name}_{tool_spec.tool_name}".replace("-", "_")
    return create_model(model_name, **fields)  # type: ignore[arg-type]


def _normalize_mcp_arguments(tool_spec: MCPToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments or {})
    if tool_spec.server_name != "filesystem":
        return normalized
    if tool_spec.tool_name not in {"list_directory", "list_directory_with_sizes", "directory_tree"}:
        return normalized
    path = normalized.get("path")
    if path is None:
        normalized["path"] = settings.WORKSPACE_ROOT
        return normalized
    if isinstance(path, str) and path.strip() in {"", ".", "./", "/"}:
        normalized["path"] = settings.WORKSPACE_ROOT
    return normalized


def build_mcp_tool_proxy(tool_spec: MCPToolSpec, session: MCPClientSession):
    args_schema = _build_args_schema(tool_spec)

    async def _invoke(**kwargs):
        if "payload" in kwargs and len(kwargs) == 1 and isinstance(kwargs["payload"], dict):
            arguments = kwargs["payload"]
        else:
            arguments = {key: value for key, value in kwargs.items() if value is not None}
        arguments = _normalize_mcp_arguments(tool_spec, arguments)
        result = await call_mcp_tool(session, tool_spec.tool_name, arguments)
        content = getattr(result, "content", None)
        if isinstance(content, list):
            text = "\n".join(
                (
                    item.get("text", "")
                    if isinstance(item, dict)
                    else getattr(item, "text", str(item))
                )
                for item in content
            ).strip()
        else:
            text = str(content or result).strip()
        return text or f"MCP 工具 {tool_spec.tool_name} 已执行。", result

    name = f"mcp__{tool_spec.server_name}__{tool_spec.tool_name}"
    description = tool_spec.description or f"MCP tool proxy for {tool_spec.server_name}/{tool_spec.tool_name}"
    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=args_schema,
        response_format="content_and_artifact",
    )


def build_all_mcp_tool_proxies(sessions: dict[str, MCPClientSession]) -> list:
    tools = []
    for session in sessions.values():
        for tool_spec in session.tools:
            tools.append(build_mcp_tool_proxy(tool_spec, session))
    return tools
