from typing import Callable

from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage

from local_agent_api.core.llm import get_model_by_choice
from local_agent_api.runtime.conversation_memory import maybe_compact_conversation
from local_agent_api.runtime.context_builder import build_runtime_context
from local_agent_api.runtime.tool_registry import get_runtime_tools, get_tool_names
from local_agent_api.services.tool_context import append_tool_trace


"""
Middleware 设计说明：
  runtime 不是一个“单独的大对象”，而是一层能力编排：
  - context_builder：决定当前请求要收集哪些上下文原材料
  - prompt_manager：决定这些原材料最终怎么拼成 prompt
  - middleware：把 runtime 生成的动态 prompt 真正接到每一次 LLM 调用前

  因此关系是：
    agent_service.create_agent(...)
      -> 注册 wrap_model_call middleware
      -> middleware 在每次模型调用前调用 build_runtime_context(...)
      -> build_runtime_context(...) 查询长期记忆 / markdown memory / skills
      -> prompt_manager 拼出动态 system prompt
      -> middleware 把它追加到基础 system prompt 后面一起发给 LLM

  好处：
    ✓ ReAct 主循环每一轮都会拿到最新上下文
    ✓ PAE 子流程复用同一套 runtime 逻辑
    ✓ service 层不用手写 prompt 拼接
"""


@wrap_tool_call
async def handle_tool_errors(request, handler):
    try:
        return await handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"工具调用失败，请换个方式提问或跳过此步骤。错误详情：{str(e)}",
            tool_call_id=request.tool_call["id"],
        )


@wrap_model_call
async def compact_conversation_context(
    request: ModelRequest,
    handler: Callable,
) -> ModelResponse:
    messages = request.state.get("messages", [])
    decision = await maybe_compact_conversation(
        messages=messages,
        conversation_summary=request.state.get("conversation_summary", ""),
        summary_upto=request.state.get("summary_upto", 0),
    )
    if decision.should_compact or decision.effective_messages is not messages:
        updated_state = {
            **request.state,
            "conversation_summary": decision.summary,
            "summary_upto": decision.summary_upto,
        }
        request = request.override(messages=decision.effective_messages, state=updated_state)
    return await handler(request)


@wrap_model_call
async def inject_runtime_context(
    request: ModelRequest,
    handler: Callable,
) -> ModelResponse:
    """
    在 LLM 被求情体的每一次调用前，拦截并注入动态上下文。
    
    拦截时机：
      不是路由层或 service 层初化时，而是 LLM.ainvoke() 被真正调用前。
      这样可以确保：
        ✓ Agent 主循环中的每次中間推理卿都能拦截
        ✓ PAE 子流程中的每次求且也能拦截
        ✓ 不需要告诉 service/agent 。middleware 自动化
    
    注入的内容：
      动态 prompt = Skill + Memory + 人格 + 上下文
      都是根据当前请求（user_id, query, plan_mode）实时计算
    """
    # 这里运行在“每次模型调用之前”，而不是只在请求入口执行一次。
    # 因此：
    # - ReAct 主循环每一轮推理都会重新注入最新的 persona / skills / memory / prompt
    # - 主循环里 tool 调完后的下一轮推理，也会拿到更新后的上下文
    user_id = request.state.get("user_id", "")
    messages = request.messages
    human_msgs = [m for m in messages if hasattr(m, "type") and m.type == "human"]
    if human_msgs:
        query = str(human_msgs[-1].content)
        # ⚡关键：在即将獻辐的一刻之前，根据当前 user_id + query 实时计算最新的能力
        # 这样高效率且永远是最新的
        # build_runtime_context 会统一把：
        # - 长期记忆
        # - Markdown 显式记忆
        # - persona
        # - skills
        # - 当前任务复杂度与推荐动作
        # 拼成新的 runtime system prompt
        runtime_route = request.state.get("runtime_route") or {}
        runtime_context = await build_runtime_context(
            query=query,
            user_id=user_id,
            plan_mode=request.state.get("plan_mode"),
            available_tool_names=get_tool_names(),
            route_decision=runtime_route if isinstance(runtime_route, dict) and runtime_route else None,
        )
        # 这里不是覆盖掉 _build_agent() 里写死的基础 system_prompt，
        # 而是在其后面追加 runtime 生成的动态 prompt。
        # 所以最终送给 LLM 的 system prompt 结构大致是：
        #   基础 system_prompt + persona/skills/memory/plan_mode 等动态上下文
        #
        # 这会带来一定“语义重复”的可能，例如基础 prompt 说“你是主执行智能体”，
        # persona 文件里也可能再次描述身份。但这种重复目前是可控的：
        # - 基础 prompt 负责稳定规则
        # - persona / memory 负责补充更具体、可编辑的上下文
        # 如果后面发现冲突，应优先收敛 persona 文本，而不是移除基础 prompt。
        current_prompt = request.system_prompt or ""
        updated_state = {
            **request.state,
            "activated_skill_names": [item.package.metadata.name for item in runtime_context.activated_skills],
            "planner_hints": runtime_context.skill_effects.planner_hints,
            "executor_hints": runtime_context.skill_effects.executor_hints,
            "output_format_hints": runtime_context.skill_effects.output_format_hints,
        }
        if runtime_context.activated_skills:
            skill_names = ", ".join(item.package.metadata.name for item in runtime_context.activated_skills)
            append_tool_trace(f"🧩 [Skill激活] {skill_names}")
        request = request.override(
            system_prompt=current_prompt + "\n\n" + runtime_context.system_prompt,
            state=updated_state,
        )
    return await handler(request)


@wrap_model_call
async def bind_selected_model(
    request: ModelRequest,
    handler: Callable,
) -> ModelResponse:
    """
    下一个 middleware：根据君主选择的 model_choice，动态绑定正确的模型实例。
    
    设计：
      不是 hardcode 一个东幸的模型，而是每次调用前按 model_choice 刚刚好。
      比如：AgentPass request.state["model_choice"] = "deepseek"，
          下次 LLM 调用前，middleware 会自动切为 deepseek_model。
    """
    model_choice = request.state.get("model_choice", "local_qwen")
    # 主循环的模型不是在 service 里硬编码的，而是在每次模型调用前动态绑定。
    # 这样同一个 ReAct runtime 可以按请求切换 local_qwen / deepseek / minimax。
    model = get_model_by_choice(model_choice).bind_tools(get_runtime_tools())
    return await handler(request.override(model=model))
