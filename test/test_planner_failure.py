import asyncio
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from local_agent_api.agents.planner import planner_node
from local_agent_api.runtime.workflow import stream_plan_and_execute


class _FakeModel:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)

    async def ainvoke(self, _prompt: str):
        if not self._outputs:
            raise RuntimeError("no more fake outputs")
        return SimpleNamespace(content=self._outputs.pop(0))


def test_planner_node_marks_planning_failed_after_two_invalid_attempts(monkeypatch):
    async def _run():
        monkeypatch.setattr(
            "local_agent_api.agents.planner.get_model_by_choice",
            lambda _choice: _FakeModel(["not json", '{"bad":"shape"}']),
        )
        state = {
            "messages": [HumanMessage(content="请比较两个方案")],
            "user_id": "u1",
            "thread_id": "t1",
            "plan_mode": "auto",
            "model_choice": "deepseek_chat",
            "is_complex": True,
            "runtime_system_prompt": "无",
            "planner_hints": [],
        }
        result = await planner_node(state)
        assert result["planning_failed"] is True
        assert result["plan"] == []
        assert result["step_results"] == []
        assert "Planner 未能生成合法执行计划" in result["planning_reason"]

    asyncio.run(_run())


def test_stream_plan_and_execute_returns_planning_failed_without_executor(monkeypatch):
    async def _run():
        async def _fake_build_runtime_context(**_kwargs):
            return SimpleNamespace(
                system_prompt="runtime",
                activated_skills=[],
                skill_effects=SimpleNamespace(
                    planner_hints=[],
                    executor_hints=[],
                    output_format_hints=[],
                ),
            )

        async def _fake_planner_node(state):
            return {
                **state,
                "planning_failed": True,
                "planning_reason": "缺少明确比较对象",
                "plan": [],
                "current_step": 0,
                "step_results": [],
            }

        monkeypatch.setattr("local_agent_api.runtime.workflow.build_runtime_context", _fake_build_runtime_context)
        monkeypatch.setattr("local_agent_api.runtime.workflow.get_tool_names", lambda: [])
        monkeypatch.setattr("local_agent_api.runtime.workflow.planner_node", _fake_planner_node)

        lines: list[str] = []
        final = None
        async for line, result in stream_plan_and_execute(
            query="请比较一下",
            thread_id="t1",
            user_id="u1",
            plan_mode="auto",
            model_choice="deepseek_chat",
        ):
            if line:
                lines.append(line)
            if result is not None:
                final = result

        assert any("规划失败" in line for line in lines)
        assert final is not None
        assert final["status"] == "planning_failed"
        assert final["planning_reason"] == "缺少明确比较对象"

    asyncio.run(_run())
