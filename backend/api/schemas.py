"""FastAPI 路由共享的 Pydantic 请求模型。"""

from typing import Any, Optional
from pydantic import BaseModel, Field

# 用Pydantic定义传入参数规范，FastAPI靠它来拦截所有非法请求
class ChatRequest(BaseModel):
    """统一聊天请求体，供直连 chat 和 agent 接口复用。"""
    query: str = Field(..., description="用户的提问内容")
    temperature: float = Field(default=0.7, description="模型生成的温度值，越高越有创造性")
    # 会话 ID，传入相同 thread_id 可保持多轮对话记忆
    # 不传则每次独立（自动分配 UUID），传固定值则持续累积历史
    thread_id: Optional[str] = Field(default=None, description="会话ID，相同ID保持多轮记忆，不传则每次独立")
    # 长期记忆：用户唯一标识（如用户名/邮箱），用于 pgvector 按用户隔离存取历史记忆
    # 不传则不启用长期记忆功能
    user_id: Optional[str] = Field(default=None, description="用户ID，用于长期记忆隔离；不传则跳过长期记忆")
    plan_mode: Optional[str] = Field(default=None, description="可选计划模式：auto、compare、extract、report、research、strict_plan")
    task_mode: Optional[str] = Field(default=None, description="旧字段兼容：将被映射为 plan_mode")
    model_choice: Optional[str] = Field(default="local_qwen", description="手动选择模型：local_qwen、deepseek、minimax")
    metadata_filters: Optional[dict[str, Any]] = Field(default=None, description="可选元数据过滤条件，如 region、year、source_type")


class RuntimeSkillAsset(BaseModel):
    """前端可编辑的 Skill 文件。"""
    filename: str = Field(..., description="skill 文件名，如 research.md")
    content: str = Field(..., description="skill 文件内容")
    source: str = Field(default="project", description="skill 来源：project 或 claude")


class RuntimeAssetsResponse(BaseModel):
    """运行时资产读取结果。"""
    agents_md: str = Field(default="", description="persona/AGENTS.md 内容")
    soul_md: str = Field(default="", description="persona/SOUL.md 内容")
    memory_md: str = Field(default="", description="memory/MEMORY.md 内容")
    skills: list[RuntimeSkillAsset] = Field(default_factory=list, description="skills 目录下的 markdown skills")


class RuntimeAssetsUpdateRequest(BaseModel):
    """运行时资产更新请求。"""
    agents_md: str = Field(default="", description="persona/AGENTS.md 新内容")
    soul_md: str = Field(default="", description="persona/SOUL.md 新内容")
    memory_md: str = Field(default="", description="memory/MEMORY.md 新内容")
    skills: list[RuntimeSkillAsset] = Field(default_factory=list, description="需要保存的技能文件内容")


class RuntimeMCPServerStatus(BaseModel):
    server_name: str
    transport: str
    connected: bool = False
    tool_names: list[str] = Field(default_factory=list)


class RuntimeMCPServerConfig(BaseModel):
    server_name: str
    transport: str
    command: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: dict[str, str] = Field(default_factory=dict)


class RuntimeMCPConfigResponse(BaseModel):
    config_text: str = Field(default="", description=".mcp.json 原始文本")
    servers: list[RuntimeMCPServerConfig] = Field(default_factory=list, description="解析后的 MCP servers")
    status: list[RuntimeMCPServerStatus] = Field(default_factory=list, description="当前 MCP 连接状态")


class RuntimeMCPConfigUpdateRequest(BaseModel):
    config_text: str = Field(..., description=".mcp.json 新内容")


class SessionRequest(BaseModel):
    user_id: str = Field(..., description="用户ID，直接作为历史会话归属键")


class SessionSummary(BaseModel):
    thread_id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    last_message_preview: str


class SessionBootstrapResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)
    current_thread_id: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary] = Field(default_factory=list)


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionMessagesResponse(BaseModel):
    thread_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
