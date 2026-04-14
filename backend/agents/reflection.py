from __future__ import annotations

"""对第一轮执行中证据不足或失败的步骤做补救重试。"""

from backend.agents.state import OrchestratorState
from backend.services.tool_context import set_tool_request_context, reset_tool_request_context


def _needs_reflection(state: OrchestratorState) -> bool:
    """
    当前 reflection 是“规则驱动补救”，不是 LLM judge。

    触发条件非常直接：
    - 只要 step_results 里出现 partial / failed
    - 就进入 reflection 尝试补救
    它不会把“当前目标 + 当前结果”再发给大模型做一次总体判断。
    """
    results = state.get("step_results", [])
    if not results:
        return False
    return any(result["status"] in {"failed", "partial"} for result in results)


async def reflection_node(state: OrchestratorState) -> OrchestratorState:
    """
    对第一轮执行中的弱步骤做补救。

    当前实现的策略是：
    1. 只处理 partial / failed 的步骤
    2. 如果步骤是检索/搜索类能力（rag_search / web_search / long_term_memory）
       就构造一个“扩大召回范围”的 retry query 再试一次
    3. 如果是其他能力（例如 analysis），暂不做通用自动补救

    也就是说，这里更像“工程化 retry policy”，而不是完整的 LLM reflection。
    """
    if not _needs_reflection(state):
        return state

    updated_results = []
    updated_plan = list(state.get("plan", []))
    citations = list(state.get("citations", []))
    retrieved_docs = list(state.get("retrieved_docs", []))

    context_tokens = set_tool_request_context(
        thread_id=state.get("thread_id", "default"),
        user_id=state.get("user_id", ""),
        plan_mode=state.get("plan_mode"),
        model_choice=state.get("model_choice", "local_qwen"),
        metadata_filters=state.get("metadata_filters"),
        in_pae=True,
    )
    try:
        from backend.runtime.tool_registry import INTERNAL_TOOL_HANDLERS
        for result in state.get("step_results", []):
            if result["status"] == "completed":
                updated_results.append(result)
                continue

            capability = result.get("capability", "rag_search")
            # 这里的补救策略不是让模型自由反思，而是明确告诉检索类工具：
            # “请扩大召回范围，优先补充可直接回答该步骤的证据”
            retry_query = f"{result['goal']}\n请扩大召回范围，优先补充可直接回答该步骤的证据。"
            try:
                if capability in {"rag_search", "rag_search_uploaded", "web_search", "search_long_term_memory"}:
                    evidence, artifact = INTERNAL_TOOL_HANDLERS[capability](retry_query)
                    if evidence:
                        result = {
                            **result,
                            "query": retry_query,
                            "evidence": evidence,
                            "status": "completed",
                        }
                        if isinstance(artifact, dict):
                            retrieved_docs.extend([item for item in artifact.get("docs", []) if hasattr(item, "metadata")])
                            citations.extend(artifact.get("citations", []))
                        elif isinstance(artifact, list):
                            retrieved_docs.extend([item for item in artifact if hasattr(item, "metadata")])
                    else:
                        result = {
                            **result,
                            "query": retry_query,
                            "status": "partial",
                            "evidence": result["evidence"] + "\n\n[reflection] 扩大召回后仍未找到足够证据。",
                        }
                else:
                    result = {
                        **result,
                        "status": "partial",
                        "evidence": result["evidence"] + "\n\n[reflection] 当前步骤类型不做自动补救。",
                    }
            except Exception as exc:
                result = {
                    **result,
                    "status": "failed",
                    "evidence": result["evidence"] + f"\n\n[reflection] 重试失败：{exc}",
                }

            updated_results.append(result)

            for plan_step in updated_plan:
                if plan_step["step_id"] == result["step_id"]:
                    plan_step["status"] = result["status"]
                    break
    finally:
        reset_tool_request_context(context_tokens)

    return {
        **state,
        "plan": updated_plan,
        "step_results": updated_results,
        "citations": citations,
        "retrieved_docs": retrieved_docs,
    }
