from __future__ import annotations

"""
基于 ContextVar 的请求级工具上下文隔离。

为什么不用传参数？（对比两种做法）：
  
  方案 A（参数传递）：
    def rag_search(query, user_id, thread_id, model_choice, metadata_filters):
        ...
    def web_search(query, user_id, thread_id, model_choice, metadata_filters):
        ...
    问题：每个工具签名都冗长，多用户隔离逻辑重复
  
  方案 B（ContextVar，当前采用）：
    def rag_search(query):  # 签名很干净
        user_id = get_tool_user_id()  # 从上下文隐式拿到
        thread_id = get_tool_thread_id()
        ...
    def web_search(query):  # 签名一样干净
        user_id = get_tool_user_id()  # 从上下文隐式拿到
        ...
    好处：工具签名不膨胀、多用户隔离天然实现、async 友好

工作机制：
  1. Service 层在流式输出开始前，调用 set_tool_request_context(...)
  2. ContextVar 把所有请求信息（user_id/thread_id/model_choice/filters）存到线程本地储存
  3. 工具函数可以随时调用 get_tool_user_id() 等，无需知道当前请求是哪个
  4. 流式输出结束后，调用 reset_tool_request_context() 恢复上一个请求的上下文
  5. 多个并发请求互不污染（async context vars 自动隔离）

ContextVar 的好处（对比全局变量）：
  ✓ 并发安全：每个 async task 有自己独立的值，不需要 Lock
  ✓ 中间件友好：可以在调用链的任何地方访问，不需要显式传参
  ✓ 生命周期管理：token 模式确保即使异常也能恢复
"""

import asyncio
from contextvars import ContextVar
from typing import Any


_metadata_filters_var: ContextVar[dict[str, Any] | None] = ContextVar(
    "tool_metadata_filters",
    default=None,
)
_thread_id_var: ContextVar[str] = ContextVar("tool_thread_id", default="")
_user_id_var: ContextVar[str] = ContextVar("tool_user_id", default="")
_plan_mode_var: ContextVar[str | None] = ContextVar("tool_plan_mode", default=None)
_model_choice_var: ContextVar[str] = ContextVar("tool_model_choice", default="local_qwen")
_in_pae_var: ContextVar[bool] = ContextVar("tool_in_pae", default=False)
_tool_trace_var: ContextVar[list[str]] = ContextVar("tool_trace", default=[])
_tool_trace_queue_var: ContextVar[asyncio.Queue[str] | None] = ContextVar("tool_trace_queue", default=None)
_last_pae_result_var: ContextVar[dict[str, Any] | None] = ContextVar("last_pae_result", default=None)


def set_tool_metadata_filters(metadata_filters: dict[str, Any] | None):
    """保存当前请求的 metadata_filters，供工具函数隐式复用。"""
    return _metadata_filters_var.set(metadata_filters or None)


def reset_tool_metadata_filters(token) -> None:
    """在流式输出结束后恢复上一个请求上下文。"""
    _metadata_filters_var.reset(token)


def get_tool_metadata_filters() -> dict[str, Any] | None:
    """在工具函数内部读取当前请求的 metadata_filters。"""
    return _metadata_filters_var.get()


def set_tool_request_context(
    *,
    thread_id: str,
    user_id: str,
    plan_mode: str | None,
    model_choice: str,
    metadata_filters: dict[str, Any] | None,
    in_pae: bool = False,
) -> dict[str, Any]:
    """
    一次性设置当前请求的工具上下文。

    这里顺手为每个请求创建一个独立的 asyncio.Queue 作为 tool trace 通道：
    - 工具函数可随时 append trace
    - service 层可并发监听这个 queue
    - 多个请求之间因为 ContextVar 隔离，不会串线
    """
    trace_queue: asyncio.Queue[str] = asyncio.Queue()
    return {
        "metadata": _metadata_filters_var.set(metadata_filters or None),
        "thread_id": _thread_id_var.set(thread_id),
        "user_id": _user_id_var.set(user_id),
        "plan_mode": _plan_mode_var.set(plan_mode),
        "model_choice": _model_choice_var.set(model_choice),
        "in_pae": _in_pae_var.set(in_pae),
        "trace": _tool_trace_var.set([]),
        "trace_queue": _tool_trace_queue_var.set(trace_queue),
        "trace_queue_obj": trace_queue,
        "last_pae_result": _last_pae_result_var.set(None),
    }


def reset_tool_request_context(tokens: dict[str, Any]) -> None:
    """恢复之前的工具上下文。"""
    _metadata_filters_var.reset(tokens["metadata"])
    _thread_id_var.reset(tokens["thread_id"])
    _user_id_var.reset(tokens["user_id"])
    _plan_mode_var.reset(tokens["plan_mode"])
    _model_choice_var.reset(tokens["model_choice"])
    _in_pae_var.reset(tokens["in_pae"])
    _tool_trace_var.reset(tokens["trace"])
    _tool_trace_queue_var.reset(tokens["trace_queue"])
    _last_pae_result_var.reset(tokens["last_pae_result"])


def get_tool_thread_id() -> str:
    return _thread_id_var.get()


def get_tool_user_id() -> str:
    return _user_id_var.get()


def get_tool_plan_mode() -> str | None:
    return _plan_mode_var.get()


def get_tool_model_choice() -> str:
    return _model_choice_var.get()


def is_in_pae() -> bool:
    return _in_pae_var.get()


def append_tool_trace(line: str) -> None:
    # trace 一份存到内存列表里，便于结束后 drain；
    # 一份立即 put 到 queue 里，便于 service 层做实时流式输出。
    traces = list(_tool_trace_var.get())
    traces.append(line)
    _tool_trace_var.set(traces)
    queue = _tool_trace_queue_var.get()
    if queue is not None:
        queue.put_nowait(line)


def drain_tool_trace() -> list[str]:
    traces = list(_tool_trace_var.get())
    _tool_trace_var.set([])
    return traces


def get_tool_trace_queue(tokens: dict[str, Any]) -> asyncio.Queue[str] | None:
    return tokens.get("trace_queue_obj")


def set_last_pae_result(result: dict[str, Any]) -> None:
    _last_pae_result_var.set(result)


def consume_last_pae_result() -> dict[str, Any] | None:
    result = _last_pae_result_var.get()
    _last_pae_result_var.set(None)
    return result
