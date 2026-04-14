from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import re
from typing import Any

from local_agent_api.core.config import settings
from local_agent_api.core.llm import get_router_model


@dataclass
class RuntimeRequest:
    query: str
    thread_id: str
    user_id: str
    plan_mode: str | None
    model_choice: str
    metadata_filters: dict | None


@dataclass
class RuntimeRouteDecision:
    pae_action: str
    pae_reason: str
    selected_skills: list[dict[str, str]]


_ROUTE_CACHE: "OrderedDict[str, RuntimeRouteDecision]" = OrderedDict()
_ROUTE_CACHE_SIZE = 128


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _normalize_route_payload(payload: dict[str, Any]) -> tuple[str, str, list[Any]]:
    pae_action = str(payload.get("pae_action", "") or payload.get("p", "")).strip()
    pae_reason = str(payload.get("pae_reason", "") or payload.get("r", "")).strip()
    selected_skills = payload.get("selected_skills")
    if selected_skills is None:
        selected_skills = payload.get("s", [])
    return pae_action, pae_reason, selected_skills


def classify_complexity(query: str, plan_mode: str | None) -> str:
    qlen = len(query or "")
    if qlen >= 160:
        return "high"
    if qlen >= 80:
        return "medium"
    return "low"


def _tool_summary(available_tool_names: list[str]) -> str:
    names = available_tool_names or []
    parts: list[str] = []
    if "run_plan_and_execute" in names:
        parts.append("pae")
    if any(name.startswith("mcp__filesystem__") for name in names):
        parts.append("mcp:filesystem")
    if any(name.startswith("mcp__fetch__") for name in names):
        parts.append("mcp:fetch")
    if "rag_search" in names or "rag_search_uploaded" in names:
        parts.append("rag")
    if "web_search" in names:
        parts.append("web")
    if "search_long_term_memory" in names:
        parts.append("memory")
    if "get_current_time" in names:
        parts.append("time")
    return ", ".join(parts) or "none"


def _catalog_summary(skill_catalog_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], str]:
    compact_rows: list[dict[str, str]] = []
    digest_parts: list[str] = []
    for row in skill_catalog_rows:
        name = str(row.get("name", "")).strip()
        description = str(row.get("description", "")).strip().replace("\n", " ")
        path = str(row.get("path", "")).strip()
        if not name or not description:
            continue
        compact_desc = description[:96]
        compact_rows.append({"name": name, "description": compact_desc, "path": path})
        digest_parts.append(f"{name}:{compact_desc}:{path}")
    return compact_rows, "|".join(digest_parts)


def _route_cache_key(
    *,
    query: str,
    plan_mode: str | None,
    complexity: str,
    tool_summary: str,
    catalog_digest: str,
) -> str:
    return json.dumps(
        {
            "q": query.strip(),
            "m": (plan_mode or "auto").strip().lower(),
            "c": complexity,
            "t": tool_summary,
            "s": catalog_digest,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _route_cache_get(key: str) -> RuntimeRouteDecision | None:
    cached = _ROUTE_CACHE.get(key)
    if cached is None:
        return None
    _ROUTE_CACHE.move_to_end(key)
    return cached


def _route_cache_set(key: str, decision: RuntimeRouteDecision) -> None:
    _ROUTE_CACHE[key] = decision
    _ROUTE_CACHE.move_to_end(key)
    while len(_ROUTE_CACHE) > _ROUTE_CACHE_SIZE:
        _ROUTE_CACHE.popitem(last=False)


async def judge_runtime_route(
    *,
    query: str,
    plan_mode: str | None,
    complexity: str,
    available_tool_names: list[str],
    skill_catalog_rows: list[dict[str, str]],
) -> RuntimeRouteDecision:
    compact_catalog_rows, catalog_digest = _catalog_summary(skill_catalog_rows)
    tool_summary = _tool_summary(available_tool_names)
    cache_key = _route_cache_key(
        query=query,
        plan_mode=plan_mode,
        complexity=complexity,
        tool_summary=tool_summary,
        catalog_digest=catalog_digest,
    )
    cached = _route_cache_get(cache_key)
    if cached is not None:
        return cached

    prompt = (
        "你是路由器，只做两件事：1) 判断是否进 PAE；2) 选要加载正文的 skill。\n"
        "单步可验证任务=>direct_or_simple_tools；明显多步/比较/提取/报告/研究=>run_plan_and_execute。\n"
        "skill 只看元数据，明显相关才选，最多 3 个；普通请求不要选 creator/builder/simple-direct 任务通常不要选 general。\n"
        "只输出 JSON，不要解释。短键：p=PAE动作，r=原因，s=skills。\n"
        '{"p":"run_plan_and_execute|direct_or_simple_tools","r":"一句话","s":[{"name":"skill-name","reason":"一句话"}]}\n'
        f"q:{query}\n"
        f"m:{plan_mode or 'auto'}\n"
        f"c:{complexity}\n"
        f"t:{tool_summary}\n"
        f"s:{json.dumps(compact_catalog_rows, ensure_ascii=False, separators=(',', ':'))}"
    )
    response = await get_router_model().ainvoke(prompt)
    payload = _extract_json_object(str(response.content))
    pae_action, pae_reason, selected_skills = _normalize_route_payload(payload)

    if pae_action not in {"run_plan_and_execute", "direct_or_simple_tools"}:
        pae_action = "direct_or_simple_tools"
    if not pae_reason:
        pae_reason = "统一路由器未返回明确原因。"
    if not isinstance(selected_skills, list):
        selected_skills = []

    normalized: list[dict[str, str]] = []
    for row in selected_skills:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        reason = str(row.get("reason", "")).strip() or "LLM 认为该 skill 与当前请求相关"
        if not name:
            continue
        normalized.append({"name": name, "reason": reason})

    decision = RuntimeRouteDecision(
        pae_action=pae_action,
        pae_reason=pae_reason,
        selected_skills=normalized,
    )
    _route_cache_set(cache_key, decision)
    return decision


def react_recursion_limit() -> int:
    return max(4, settings.REACT_MAX_TOOL_CALLS * 2)
