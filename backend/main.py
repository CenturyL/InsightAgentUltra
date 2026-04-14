from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
from backend.api.routes import router as chat_router
from backend.core.config import settings
from backend.services import agent_service


# ── Lifespan：负责 Agent / 数据库连接池的初始化与清理 ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子：启动时初始化，关闭时清理资源。"""
    await agent_service.initialize()  # 连接 PostgreSQL / 构建 Agent
    yield
    await agent_service.cleanup()     # 关闭数据库连接池


# 1. 创建 FastAPI 实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="基于 FastAPI 的通用智能体平台 API",
    version="3.0.0",
    lifespan=lifespan,
)

# 2. 挂载我们在 routes.py 中编写的路由
app.include_router(chat_router, prefix="/api/v3", tags=["LLM Chat"])

# 3. 根路由（用来健康检查）
@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}. The API is running correctly!"}

if __name__ == "__main__":
    # 使用 uvicorn 启动服务器
    # reload=True 支持代码修改后热更新，方便开发
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
