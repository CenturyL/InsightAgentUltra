from __future__ import annotations

"""标准 MCP client：优先使用官方 Python SDK。"""

from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from local_agent_api.runtime.mcp_config import MCPConfig, MCPServerConfig


@dataclass
class MCPToolSpec:
    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPClientSession:
    server_name: str
    config: MCPServerConfig
    capabilities: dict[str, Any] = field(default_factory=dict)
    tools: list[MCPToolSpec] = field(default_factory=list)
    client: ClientSession | None = None
    stack: AsyncExitStack | None = None


async def connect_mcp_server(config: MCPServerConfig) -> MCPClientSession:
    stack = AsyncExitStack()
    await stack.__aenter__()

    if config.transport == "http":
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(url=config.url, headers=config.headers)
        )
    else:
        read, write = await stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env or None,
                )
            )
        )

    session = await stack.enter_async_context(ClientSession(read, write))
    init_result = await session.initialize()
    capabilities = getattr(init_result, "capabilities", {}) or {}
    return MCPClientSession(
        server_name=config.name,
        config=config,
        capabilities=capabilities,
        client=session,
        stack=stack,
    )


async def list_mcp_tools(session: MCPClientSession) -> list[MCPToolSpec]:
    if session.client is None:
        raise RuntimeError(f"MCP server {session.server_name} 未建立连接")
    result = await session.client.list_tools()
    tools = [
        MCPToolSpec(
            server_name=session.server_name,
            tool_name=tool.name,
            description=tool.description or "",
            input_schema=tool.inputSchema or {},
        )
        for tool in result.tools
    ]
    session.tools = tools
    return tools


async def call_mcp_tool(session: MCPClientSession, tool_name: str, arguments: dict) -> Any:
    if session.client is None:
        raise RuntimeError(f"MCP server {session.server_name} 未建立连接")
    return await session.client.call_tool(tool_name, arguments=arguments or {})


async def initialize_mcp_sessions(config: MCPConfig) -> dict[str, MCPClientSession]:
    sessions: dict[str, MCPClientSession] = {}
    for server_name, server_config in config.servers.items():
        try:
            session = await connect_mcp_server(server_config)
            await list_mcp_tools(session)
            sessions[server_name] = session
        except Exception as exc:
            print(f"⚠️ [MCP] server {server_name} 初始化失败：{exc}")
    return sessions


async def close_mcp_sessions(sessions: dict[str, MCPClientSession]) -> None:
    for session in sessions.values():
        if session.stack is not None:
            try:
                await session.stack.aclose()
            except Exception:
                pass
