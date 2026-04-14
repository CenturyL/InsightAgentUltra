from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

# .env 文件与本文件同目录（backend/），无论从哪个目录启动都能找到
_ENV_FILE = Path(__file__).parent.parent / ".env"
_PROJECT_ROOT = Path(__file__).parent.parent

# 自动读取环境变量，类型转换与校验
class Settings(BaseSettings):
    # 默认配置：如果环境变量里没有设置，就用这些默认值
    PROJECT_NAME: str = "Local Knowledge Agent API"
    WORKSPACE_ROOT: str = str(_PROJECT_ROOT.parent)
    BACKEND_ROOT: str = str(_PROJECT_ROOT)
    # 基础模型提供方：支持 ollama 或 openai_compatible（vLLM / LM Studio / OneAPI 等）
    BASIC_MODEL_PROVIDER: str = "ollama"
    BASIC_MODEL_BASE_URL: str = "http://10.144.144.7:11434"
    BASIC_MODEL_NAME: str = "qwen3.5:9b"
    BASIC_MODEL_API_KEY: str = "EMPTY"

    # 兼容旧配置名，避免已有 .env / 文档 立即失效
    OLLAMA_BASE_URL: Optional[str] = None
    LLM_MODEL: Optional[str] = None
    
    # 【新增】DeepSeek 官方 API 配置 (完全兼容 OpenAI SDK)
    # 必须在 .env 文件中设置：DEEPSEEK_API_KEY=your_real_key_here
    # 禁止在代码中硬编码 API Key，否则提交 git 会造成密钥泄露
    DEEPSEEK_API_KEY: str  # 无默认值，强制从环境变量/.env 文件读取
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-reasoner"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"
    MINIMAX_API_KEY: Optional[str] = None
    MINIMAX_BASE_URL: str = "https://api.minimax.io/v1"
    MINIMAX_MODEL: str = "MiniMax-M2.5"
    MIMO_API_KEY: Optional[str] = None
    MIMO_BASE_URL: str = "https://api.xiaomimimo.com/v1"
    MIMO_MODEL: str = "mimo-v2-flash"
    MIMO_PRO_MODEL: str = "mimo-v2-pro"
    
    # RAG 向量数据库与模型配置
    VECTOR_STORE_PATH: str = str(_PROJECT_ROOT / "data" / "chroma_db")
    PARENT_STORE_PATH: str = str(_PROJECT_ROOT / "data" / "parent_store")
    # 这里选择一个在 Mac 本地跑得快、且对中文友好的轻量级开源 Embedding 模型
    EMBEDDING_MODEL: str = "shibing624/text2vec-base-chinese" 
    EMBEDDING_DEVICE: str = "auto"
    
    # 【工业级新增】本地重排 (Reranker) 模型
    # BAAI (智源研究院) 的 bge-reranker 目前是开源中最顶尖的中文重排模型之一
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_DEVICE: str = "auto"
    REACT_MAX_TOOL_CALLS: int = 10
    REACT_MAX_NO_PROGRESS_CALLS: int = 3
    PAE_MAX_CALLS_PER_REQUEST: int = 1
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_COMPRESSION_MAX_MESSAGES: int = 14
    CONTEXT_COMPRESSION_MAX_CHARS: int = 12000
    CONTEXT_COMPRESSION_RECENT_TURNS: int = 4
    CONTEXT_COMPRESSION_MIN_DELTA_MESSAGES: int = 4
    MCP_ENABLED: bool = True
    MCP_CONFIG_PATH: Optional[str] = None
    SKILL_COMPAT_MODE: str = "claude"

    # ── 长期记忆 & 持久化 Checkpointer（可选）────────────────────────────────
    # 格式：postgresql://用户名:密码@主机:端口/数据库名
    # 配置后：① 对话历史跨重启保留（PostgresSaver）② 用户事实跨会话记忆（pgvector）
    # 不配置则自动降级：① MemorySaver（进程内）② 长期记忆功能关闭
    POSTGRES_URL: Optional[str] = None
    
    # 内部配置类：使用绝对路径定位 .env，避免因工作目录不同而读取失败
    class Config:
        env_file = str(_ENV_FILE)
        env_file_encoding = "utf-8"

# 实例化一个全局配置对象
settings = Settings()
