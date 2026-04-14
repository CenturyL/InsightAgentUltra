"""兼容层：统一从 runtime.tool_registry 暴露工具。"""

from backend.runtime.tool_registry import (
    get_current_time,
    get_mcp_tools,
    get_runtime_tools,
    get_tool_descriptions,
    get_tool_names,
    initialize_runtime_tools,
    rag_search,
    rag_search_uploaded,
    run_plan_and_execute,
    search_long_term_memory,
    web_search,
)


def get_agent_tools():
    return get_runtime_tools()


__all__ = [
    "get_agent_tools",
    "get_current_time",
    "get_mcp_tools",
    "get_tool_descriptions",
    "get_tool_names",
    "initialize_runtime_tools",
    "rag_search",
    "rag_search_uploaded",
    "run_plan_and_execute",
    "search_long_term_memory",
    "web_search",
]
