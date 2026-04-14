from __future__ import annotations

"""复杂任务编排图共享的 TypedDict 状态定义。"""

from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import NotRequired, TypedDict


class PlanStep(TypedDict):
    """planner 生成的一条步骤定义，供 executor 按步执行。"""
    step_id: str
    goal: str
    reason: str
    required_capability: str
    expected_output: str
    status: Literal["pending", "completed", "failed", "partial"]


class StepResult(TypedDict):
    """单个步骤的执行结果，供 reflection 和 synthesis 使用。"""
    step_id: str
    goal: str
    query: str
    evidence: str
    capability: str
    status: Literal["completed", "failed", "partial"]


class OrchestratorState(TypedDict):
    """LangGraph 复杂任务节点之间传递的共享状态。"""
    messages: list[BaseMessage]
    user_id: str
    thread_id: str
    plan_mode: NotRequired[str]
    model_choice: NotRequired[str]
    metadata_filters: NotRequired[dict[str, Any]]
    is_complex: bool
    planning_reason: NotRequired[str]
    planning_failed: NotRequired[bool]
    plan: NotRequired[list[PlanStep]]
    current_step: NotRequired[int]
    step_results: NotRequired[list[StepResult]]
    retrieved_docs: NotRequired[list[Document]]
    citations: NotRequired[list[dict[str, Any]]]
    final_answer: NotRequired[str]
    runtime_system_prompt: NotRequired[str]
    activated_skill_names: NotRequired[list[str]]
    planner_hints: NotRequired[list[str]]
    executor_hints: NotRequired[list[str]]
    output_format_hints: NotRequired[list[str]]
