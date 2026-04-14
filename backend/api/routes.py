"""FastAPI 路由层：把 HTTP 接口映射到聊天、入库和评估服务。"""

import os
import uuid
import shutil
import json
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from backend.api.schemas import (
    ChatRequest,
    RuntimeAssetsResponse,
    RuntimeAssetsUpdateRequest,
    RuntimeMCPConfigResponse,
    RuntimeMCPConfigUpdateRequest,
    SessionBootstrapResponse,
    SessionListResponse,
    SessionMessagesResponse,
    SessionRequest,
    SessionSummary,
)
from langchain_core.messages import HumanMessage
from backend.retrieval.pipeline import process_and_store_document
from backend.services.agent_service import get_agent_stream, get_thread_messages
from backend.services.runtime_assets_service import (
    load_runtime_assets,
    save_runtime_assets,
)
from backend.services.runtime_mcp_service import (
    load_runtime_mcp_config,
    reload_runtime_mcp_config,
    save_runtime_mcp_config,
)
from backend.services.session_service import (
    bootstrap_sessions,
    create_empty_session,
    delete_session,
    get_session,
    list_sessions,
)
from backend.core.llm import create_basic_model

router = APIRouter()


# 不走RAG，不走TOOL，不走AGENT，纯本地直接问答
@router.post("/chat/stream", summary="基础流式对话接口(不借用知识库)")
async def chat_stream(request: ChatRequest):
    """
    与本地 Qwen 模型进行直接流式对话。
    """
    # 按请求中的 temperature 动态创建模型实例（轻量，不缓存）
    llm = create_basic_model(temperature=request.temperature)

    # 2. 构建消息体
    messages = [HumanMessage(content=request.query)]

    # 3. 定义异步生成器实现流式 (Streaming) 输出
    async def generate_chat():
        # 它会在收到模型的每个 token 时理解 yield 出来，而不是等全句生成完
        async for chunk in llm.astream(messages): # astream() 是 LangChain 核心的异步流式 API
            yield chunk.content

    # StreamingResponse 是 FastAPI/Starlette 提供的“流式 HTTP 响应包装器”：
    # - 它接收一个（异步）生成器
    # - 生成器每 yield 一段，HTTP 响应体就继续往前端写一段
    # - 不需要等完整答案准备好才返回
    return StreamingResponse(generate_chat(), media_type="text/plain")

# 动态智能体 Agent 接口
@router.post("/chat/agent", summary="通用智能体 Agent 接口（ReAct 主循环 + Tools + PAE）")
async def chat_agent_endpoint(request: ChatRequest):
    """
    统一智能体入口：默认走 ReAct 主循环，必要时由模型主动调用 PAE 工具。
    - 传入 thread_id 可保持多轮对话记忆（相同 ID 自动拼接历史）
    - 不传 thread_id 则每次独立会话
    - RAG 检索已收敛至 search_company_rules 工具，无需单独调用 /chat/rag
    """
    # 未指定 thread_id 时自动分配 UUID，保证无状态调用的隔离性
    thread_id = request.thread_id or str(uuid.uuid4())
    # user_id 不传则为空串，inject_long_term_memory 中间件会跳过记忆读写
    user_id = request.user_id or ""

    async def generate_agent_output():
        # 这里返回的是“异步生成器”，不是一次性算完整答案后再 return。
        # FastAPI 的 StreamingResponse 会边迭代、边把 chunk 刷给前端，
        # 因此主循环 token、工具 trace、PAE 阶段信息都能实时显示。
        plan_mode = request.plan_mode or request.task_mode
        async for chunk in get_agent_stream(
                request.query,
                thread_id=thread_id,
                user_id=user_id,
                plan_mode=plan_mode,
                model_choice=request.model_choice or "local_qwen",
                metadata_filters=request.metadata_filters,
        ):
            yield json.dumps(chunk, ensure_ascii=False) + "\n"

    # 这里返回的不是普通 JSONResponse，而是 StreamingResponse。
    # 因此 Agent 主循环里的 token、工具 trace、PAE 阶段信息都可以边产生边发送给前端。
    return StreamingResponse(generate_agent_output(), media_type="application/x-ndjson")

# 上传文件接口
@router.post("/knowledge/upload", summary="上传文件并录入本地知识库")
async def upload_knowledge(file: UploadFile = File(...)):
    """
    接收用户上传的 TXT / MD / PDF / HTML / CSV / 图片文件，
    进行结构化切块、Embedding 与入库，最终保存到本地 Chroma 向量库。
    """
    os.makedirs("backend/data/temp", exist_ok=True)
    temp_file_path = f"backend/data/temp/{file.filename}"

    # 1. 临时保存文件
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. 调用存库逻辑，并保留原始上传文件名作为知识库 source
        chunks_count = process_and_store_document(
            temp_file_path,
            metadata_overrides={
                "source": file.filename,
                "upload_name": file.filename,
            },
        )
        return {
            "code": 200,
            "message": "录入成功！",
            "filename": file.filename,
            "source": file.filename,
            "chunks_inserted": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理失败: {str(e)}")
    finally:
        # 3. 擦屁股：存进向量库后删掉服务器上的临时原文件
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@router.get("/runtime/assets", response_model=RuntimeAssetsResponse, summary="读取运行时资产")
async def get_runtime_assets():
    """读取 persona / markdown memory / skills，供前端编辑。"""
    return load_runtime_assets()


@router.put("/runtime/assets", response_model=RuntimeAssetsResponse, summary="更新运行时资产")
async def update_runtime_assets(request: RuntimeAssetsUpdateRequest):
    """更新 persona / markdown memory / skills。"""
    return save_runtime_assets(
        agents_md=request.agents_md,
        soul_md=request.soul_md,
        memory_md=request.memory_md,
        skills=[skill.model_dump() for skill in request.skills],
    )


@router.get("/runtime/mcp/config", response_model=RuntimeMCPConfigResponse, summary="读取 MCP 配置与状态")
async def get_runtime_mcp_config():
    return load_runtime_mcp_config()


@router.put("/runtime/mcp/config", response_model=RuntimeMCPConfigResponse, summary="更新 MCP 配置并重载")
async def update_runtime_mcp_config(request: RuntimeMCPConfigUpdateRequest):
    return await save_runtime_mcp_config(request.config_text)


@router.post("/runtime/mcp/reload", response_model=RuntimeMCPConfigResponse, summary="重载 MCP 配置")
async def reload_runtime_mcp():
    return await reload_runtime_mcp_config()


@router.post("/sessions/bootstrap", response_model=SessionBootstrapResponse, summary="加载用户历史 session 并创建新会话")
async def bootstrap_user_sessions(request: SessionRequest):
    return await bootstrap_sessions(request.user_id)


@router.post("/sessions", response_model=SessionSummary, summary="为用户创建新会话")
async def create_user_session(request: SessionRequest):
    return await create_empty_session(request.user_id)


@router.get("/sessions", response_model=SessionListResponse, summary="按 USERID 列出历史 session")
async def list_user_sessions(user_id: str = Query(..., description="用户ID")):
    return {"sessions": await list_sessions(user_id)}


@router.get("/sessions/{thread_id}/messages", response_model=SessionMessagesResponse, summary="加载指定历史 session 消息")
async def get_session_messages(thread_id: str, user_id: str = Query(..., description="用户ID")):
    session = await get_session(thread_id, user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="未找到该用户下的会话。")
    return {
        "thread_id": thread_id,
        "messages": await get_thread_messages(thread_id, user_id),
    }


@router.delete("/sessions/{thread_id}", summary="删除指定历史 session")
async def delete_user_session(thread_id: str, user_id: str = Query(..., description="用户ID")):
    session = await get_session(thread_id, user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="未找到该用户下的会话。")
    deleted = await delete_session(thread_id, user_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="删除会话失败。")
    return {"deleted": True, "thread_id": thread_id}
