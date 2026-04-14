from __future__ import annotations

"""统一注册 ReAct 和 PAE 共用的工具。"""

import datetime
import re
import urllib.parse
import urllib.request
from typing import Any
from langchain_core.tools import tool

from local_agent_api.core.config import settings
from local_agent_api.retrieval.pipeline import retrieve_knowledge_bundle
from local_agent_api.runtime.mcp_client import MCPClientSession, close_mcp_sessions, initialize_mcp_sessions
from local_agent_api.runtime.mcp_config import load_mcp_config
from local_agent_api.runtime.mcp_tools import build_all_mcp_tool_proxies
from local_agent_api.runtime.memory_bridge import search_long_term_memory_text
from local_agent_api.services.tool_context import (
    append_tool_trace,
    set_last_pae_result,
    get_tool_model_choice,
    get_tool_metadata_filters,
    get_tool_plan_mode,
    get_tool_thread_id,
    get_tool_user_id,
    is_in_pae,
)


_MCP_SESSIONS: dict[str, MCPClientSession] = {}
_MCP_TOOLS: list = []


def _merged_filters(scope: str, metadata_filters: dict[str, Any] | None) -> dict[str, Any] | None:
    filters = dict(metadata_filters or {})
    if scope == "policy":
        filters.setdefault("doc_category", "policy")
    elif scope == "tender":
        filters.setdefault("doc_category", "tender")
    elif scope == "company_rules":
        filters.setdefault("scope", "company_rules")
    if scope == "uploaded" and not filters.get("_recent_upload_source"):
        return metadata_filters
    return filters or None


def _rag_search_impl(
    query: str,
    scope: str = "general",
    strategy: str = "hybrid_rerank",
) -> tuple[str, dict[str, Any]]:
    filters = _merged_filters(scope, get_tool_metadata_filters())
    bundle = retrieve_knowledge_bundle(
        query,
        k=3,
        metadata_filters=filters,
        strategy=strategy,
    )
    if not bundle.docs:
        return "未能找到相关信息。", {"docs": [], "citations": [], "parent_docs": []}
    return bundle.context_text, {
        "docs": bundle.docs,
        "citations": bundle.citations,
        "parent_docs": bundle.parent_docs,
    }


# ============================================================================
# 工具化设计说明：
#   旧方式：service 层硬调 RAG，LLM 无法拒绝或优化
#   新方式：RAG 注册为工具，LLM 自己决定何时调用、如何调用（scope、strategy）
#   好处：简单问题可以不调 RAG，复杂问题由 LLM 自动组织多工具调用
# ============================================================================

@tool(response_format="content_and_artifact")
def rag_search(
    query: str,
    scope: str = "general",
    strategy: str = "hybrid_rerank",
) -> tuple[str, dict[str, Any]]:
    """通用检索工具。scope 可取 general、policy、tender、company_rules、uploaded。需要文档证据时优先调用。"""
    return _rag_search_impl(query=query, scope=scope, strategy=strategy)


@tool(response_format="content_and_artifact")
def rag_search_uploaded(
    query: str,
    strategy: str = "hybrid_rerank",
) -> tuple[str, dict[str, Any]]:
    """针对当前上传文件或最近上传来源优先检索；适合刚上传文档后立刻问答。"""
    return _rag_search_impl(query=query, scope="uploaded", strategy=strategy)


def _extract_search_results(html: str) -> list[dict[str, str]]:
    pattern = re.compile(
        r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
        flags=re.S,
    )
    matches = pattern.findall(html)
    results = []
    for url, title in matches[:5]:
        clean_title = re.sub(r"<.*?>", "", title).strip()
        clean_url = url.replace("&amp;", "&")
        if clean_title and clean_url:
            results.append({"title": clean_title, "url": clean_url})
    return results


def _web_search_impl(query: str) -> tuple[str, list[dict[str, str]]]:
    headers = {"User-Agent": "InsightAgentPro/1.0"}
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    request = urllib.request.Request(
        "https://html.duckduckgo.com/html/",
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8", errors="ignore")
    results = _extract_search_results(html)
    if not results:
        return "联网搜索未返回有效结果。", []
    content = "\n".join(
        f"- {item['title']}\n  {item['url']}" for item in results
    )
    return f"以下是联网搜索结果：\n{content}", results


@tool(response_format="content_and_artifact")
def web_search(query: str) -> tuple[str, list[dict[str, str]]]:
    """联网公开搜索工具。用于需要最新公开信息、外部网页资料或在线验证时。"""
    return _web_search_impl(query)


def _search_long_term_memory_impl(query: str, k: int = 3) -> tuple[str, list[str]]:
    user_id = get_tool_user_id()
    if not user_id:
        return "当前请求没有 user_id，无法检索长期记忆。", []
    memories = search_long_term_memory_text(user_id, query, k=k)
    if not memories:
        return "未检索到相关长期记忆。", []
    return "检索到以下长期记忆：\n" + "\n".join(f"- {item}" for item in memories), memories


@tool(response_format="content_and_artifact")
def search_long_term_memory(query: str, k: int = 3) -> tuple[str, list[str]]:
    """检索当前用户的长期记忆，用于偏好、历史事实和跨会话信息。"""
    return _search_long_term_memory_impl(query=query, k=k)


def _get_current_time_text(timezone: str = "Asia/Shanghai") -> str:
    now = datetime.datetime.now()
    return f"当前时间是: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})"


@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取当前系统时间。用于现实时间相关问题。"""
    return _get_current_time_text(timezone)


def _get_current_time_impl(timezone: str = "Asia/Shanghai") -> tuple[str, list]:
    return _get_current_time_text(timezone), []


async def _run_plan_and_execute_impl(query: str) -> tuple[str, dict[str, Any]]:
    # 防递归：已经在 PAE 流程里了，禁止再次调用 run_plan_and_execute
    # is_in_pae() 是通过 ContextVar 检查的，每个请求独立
    if is_in_pae():
        return "当前已在计划执行流程中，禁止递归再次调用 run_plan_and_execute。", {
            "mode": "plan_and_execute",
            "plan": [],
            "step_results": [],
            "citations": [],
            "trace": [],
            "final_answer": "当前已在计划执行流程中，禁止递归再次调用 run_plan_and_execute。",
        }
    # 通过 ContextVar 隐式获取当前请求的上下文信息，无需显式传参
    # 这是新架构的核心：工具函数可以透明地访问请求级隔离的上下文
    thread_id = get_tool_thread_id() or "default"
    user_id = get_tool_user_id() or ""
    plan_mode = get_tool_plan_mode()
    model_choice = get_tool_model_choice()
    metadata_filters = get_tool_metadata_filters()
    from local_agent_api.runtime.workflow import run_plan_and_execute_once
    result = await run_plan_and_execute_once(
        query=query,
        thread_id=thread_id,
        user_id=user_id,
        plan_mode=plan_mode,
        model_choice=model_choice,
        metadata_filters=metadata_filters,
        trace_sink=append_tool_trace,
    )
    set_last_pae_result(result)
    return result["final_answer"], result


# ============================================================================
# 最重要的工具：run_plan_and_execute（PAE）
# 
# 工具化的突破点：
#   旧方式：if complexity_is_high: 执行 PAE; else: 直接调 LLM
#           → 决策规则硬编码，不灵活
#   
#   新方式：把 PAE 注册为工具，LLM 看到后自己决定何时调用
#           → LLM 可以根据对话上下文更智能地判断
#           → 同一个请求可以先试连续简单工具，不行了再调 PAE
#           → 多工具组合更灵活
#
# 工具内部流程（完整的 Plan-and-Execute）：
#   1. planner：把自然语言任务转换为结构化步骤列表
#   2. executor：逐步执行计划，每步都可能调用 rag_search / web_search / analysis
#   3. reflection：如果有步骤失败或证据不足，自动扩大召回范围重试
#   4. synthesizer：把所有步骤结果综合成最终答案，按 plan_mode（比较/提取/报告）格式输出
# ============================================================================

@tool(response_format="content_and_artifact")
async def run_plan_and_execute(query: str) -> tuple[str, dict[str, Any]]:
    """
    复杂任务首选工具。 Plan-and-Execute 子流程，用于比较、提取多字段、报告生成、多步研究等。
    
    什么时候 LLM 应该调我？
      - 需要比较两个或多个对象的差异
      - 需要从多个地方提取结构化信息
      - 需要生成报告或方案（多段落、多维度）
      - 需要多步研究（先查资料、再分析、再汇总）
      - 单轮简单工具调用预期成功率低时
    """
    return await _run_plan_and_execute_impl(query=query)


INTERNAL_TOOL_HANDLERS = {
    "rag_search": _rag_search_impl,
    "rag_search_uploaded": lambda query: _rag_search_impl(query=query, scope="uploaded"),
    "web_search": _web_search_impl,
    "search_long_term_memory": _search_long_term_memory_impl,
    "get_current_time": lambda query="": _get_current_time_impl(),
}


def get_builtin_tools():
    return [
        rag_search,
        rag_search_uploaded,
        web_search,
        search_long_term_memory,
        get_current_time,
        run_plan_and_execute,
    ]


async def initialize_runtime_tools() -> None:
    global _MCP_SESSIONS, _MCP_TOOLS
    if _MCP_SESSIONS:
        await close_mcp_sessions(_MCP_SESSIONS)
        _MCP_SESSIONS = {}
        _MCP_TOOLS = []
    if not settings.MCP_ENABLED:
        _MCP_SESSIONS = {}
        _MCP_TOOLS = []
        return
    config = load_mcp_config()
    _MCP_SESSIONS = await initialize_mcp_sessions(config)
    _MCP_TOOLS = build_all_mcp_tool_proxies(_MCP_SESSIONS)


def get_mcp_tools() -> list:
    return list(_MCP_TOOLS)


def get_mcp_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, session in _MCP_SESSIONS.items():
        rows.append(
            {
                "server_name": name,
                "transport": session.config.transport,
                "connected": True,
                "tool_names": [tool.tool_name for tool in session.tools],
            }
        )
    return sorted(rows, key=lambda item: item["server_name"])


def get_runtime_tools(allowed_tool_names: list[str] | None = None) -> list:
    tools = [*get_builtin_tools(), *get_mcp_tools()]
    if not allowed_tool_names:
        return tools
    allowed = set(allowed_tool_names)
    return [item for item in tools if item.name in allowed]


def get_tool_names() -> list[str]:
    return [tool.name for tool in get_runtime_tools()]


def get_tool_descriptions() -> list[str]:
    descriptions = []
    for item in get_runtime_tools():
        descriptions.append(f"{item.name}: {item.description}")
    return descriptions
