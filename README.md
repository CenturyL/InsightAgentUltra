# InsightAgentUltra

一个面向真实任务执行的通用 Agent Runtime 平台。  
项目不是"单轮 RAG 问答 Demo"，而是围绕 **会话管理、上下文压缩、统一路由、Skill、MCP、RAG、长期记忆、PAE（Plan-and-Execute）** 组织起来的一套完整运行时。

## 1. 项目定位

当前系统的核心目标：

- 用 **ReAct** 作为默认主循环处理普通问题与工具调用
- 用 **Plan-and-Execute** 处理复杂、多步、跨工具任务
- 用 **统一 runtime** 在每次请求时动态决定：
  - 是否进入 PAE
  - 是否加载 Skill
  - 使用哪个模型
  - 注入哪些长期记忆 / markdown memory
  - 暴露哪些工具能力
- 支持 **历史 Session**、**thread 级短期记忆**、**跨 Session 长期记忆**
- 支持 **MCP 工具接入**

## 2. 技术栈

后端：

- FastAPI
- LangChain
- LangGraph
- PostgreSQL
- pgvector
- Chroma
- sentence-transformers
- Ollama / OpenAI-Compatible API / DeepSeek / MiniMax / Xiaomi MiMo

前端：

- React
- TypeScript
- Vite

## 3. 核心能力

### 3.1 通用 Agent Runtime

- ReAct 主循环
- PAE 作为高级工具暴露给主循环
- 统一路由层：
  - 判断是否进入 PAE
  - 选择需要加载的 Skill
  - 结合 `plan_mode` 调整执行倾向
- 运行时动态拼装上下文，而不是写死一段 prompt

### 3.2 会话与记忆

- `user_id` 级历史 Session 管理
- `thread_id` 级短期运行状态恢复
- 长会话自动压缩为结构化摘要
- PostgreSQL + pgvector 长期记忆
- Markdown Memory / Persona 可编辑

### 3.3 检索与知识库

- 文本、Markdown、HTML、CSV、PDF、图片入库
- Chroma 本地向量库
- Hybrid Retrieval + Rerank
- 上传文件优先检索

### 3.4 Skill 与 MCP

- 支持 Claude-compatible `SKILL.md`
- Skill 采用 progressive disclosure：
  - 先只暴露 skill 元数据
  - 路由命中后再加载全文
- 支持 `.mcp.json`
- 已实现 MCP tools 接入：
  - `initialize`
  - `list_tools`
  - `call_tool`

## 4. 项目结构

```text
.
├── backend/                              # 后端
│   ├── agents/                           # planner / executor / reflection / synthesizer
│   ├── api/                              # FastAPI 路由与请求/响应 schema
│   ├── core/                             # 配置、模型、memory、middleware 等核心能力
│   ├── retrieval/                        # 文档入库、切块、检索、rerank
│   ├── runtime/                          # runtime 路由、上下文、skill、MCP、workflow
│   ├── services/                         # agent service、session service、runtime assets service
│   ├── data/                             # Chroma、parent store、临时文件、测试文档
│   ├── .claude/skills/                   # 导入的 Claude 风格 Skill 包
│   ├── .mcp.json                         # MCP server 配置
│   ├── requirements.txt                  # Python 依赖
│   └── main.py                           # FastAPI 入口（挂载 /api/v3）
├── frontend/                             # React 前端
│   ├── src/App.tsx                       # 主界面与核心前端状态
│   ├── package.json
│   └── index.html
├── skills/                               # 项目内 Skill
├── persona/                              # AGENTS.md / SOUL.md
├── memory/                               # MEMORY.md
├── prompts/                              # 手工 Prompt 资产
├── mcp/                                  # MCP 相关说明文件
├── test/                                 # pytest 测试
└── README.md
```

## 5. 数据存储与数据库要求

当前版本要想完整跑起来，**PostgreSQL 是必需的**。

原因：

- 历史 Session 功能依赖 PostgreSQL
- 长期记忆依赖 PostgreSQL + pgvector
- LangGraph checkpointer 持久化依赖 PostgreSQL

### 5.1 必需数据库能力

你需要一个 PostgreSQL 实例，并启用 `pgvector` 扩展。

### 5.2 推荐版本

- PostgreSQL 14+
- pgvector 已安装

### 5.3 初始化数据库

进入 PostgreSQL 后执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 5.4 项目会自动创建的表

应用启动后会自动初始化：

- 历史 Session 相关表
  - `agent_sessions`
  - `agent_session_messages`
- LangGraph checkpointer 相关表
  - `checkpoint_migrations`
  - `checkpoints`
  - `checkpoint_blobs`
  - `checkpoint_writes`

长期记忆的 pgvector 相关表由运行时在首次使用时创建。

## 6. 环境变量配置

后端从：

- `backend/.env`

读取配置。  
仓库中不包含真实密钥，请先复制模板：

```bash
cp backend/.env.example backend/.env
```

最小可启动配置建议至少填写：

- `DEEPSEEK_API_KEY`
- `POSTGRES_URL`

如果你要启用额外模型，再填写：

- `MINIMAX_API_KEY`
- `MIMO_API_KEY`

如果你要走本地模型，还要保证 Ollama 或 OpenAI-compatible 本地服务可用。

### 6.1 关键环境变量说明

| 变量名 | 说明 | 是否必需 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key。当前代码会在启动时初始化 DeepSeek 模型对象，因此必需 | 是 |
| `POSTGRES_URL` | PostgreSQL 连接串，历史 Session、长期记忆、checkpointer 都依赖它 | 是 |
| `BASIC_MODEL_PROVIDER` | 本地基础模型提供方：`ollama` 或 `openai_compatible` | 否 |
| `BASIC_MODEL_BASE_URL` | 本地基础模型服务地址 | 否 |
| `BASIC_MODEL_NAME` | 本地基础模型名称 | 否 |
| `MINIMAX_API_KEY` | MiniMax API Key | 否 |
| `MIMO_API_KEY` | Xiaomi MiMo API Key | 否 |
| `VECTOR_STORE_PATH` | Chroma 存储目录 | 否 |
| `WORKSPACE_ROOT` | 工作区根目录，MCP filesystem 与文件检索依赖它 | 否 |

## 7. 安装与启动

### 7.1 克隆仓库

```bash
git clone git@github.com:CenturyL/InsightAgentUltra.git
cd InsightAgentUltra
```

### 7.2 后端环境

推荐 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

或者使用 conda：

```bash
conda create -n insightagent python=3.11 -y
conda activate insightagent
pip install -r backend/requirements.txt
```

### 7.3 前端环境

```bash
cd frontend
npm install
cd ..
```

### 7.4 启动后端

请在**项目根目录**执行：

```bash
uvicorn backend.main:app --reload
```

默认地址：

- 后端 API: `http://127.0.0.1:8000`
- 健康检查: `http://127.0.0.1:8000/`

### 7.5 启动前端

```bash
cd frontend
npm run dev
```

默认地址：

- 前端: `http://127.0.0.1:5173`

## 8. 本地启动后的功能验证

### 8.1 基础页面

打开前端后，确认：

- 左侧能看到 Session / Runtime 面板
- 中间聊天区正常显示
- 右侧 Agent Run 面板正常显示

### 8.2 MCP

使用体验示例中的：

```text
请列出当前项目根目录下有哪些一级文件和文件夹，只使用 MCP 工具完成，不要猜测。
```

应看到：

- 右侧 trace 出现 MCP tool 调用
- 正文返回文件夹/文件列表

### 8.3 Session

输入一个 `UserID` 后点击"确定"，应看到：

- 历史 Session 列表
- 自动创建一个新会话
- 可以切换、删除历史 Session

### 8.4 长期记忆

在同一个 `UserID` 下：

1. 一个 session 里说：`我叫张三`
2. 新建 session 再问：`我叫什么`

应能通过长期记忆召回姓名。

## 9. MCP 配置

当前后端目录下使用：

- `backend/.mcp.json`

来声明 MCP server。

当前系统已支持：

- 读取 `.mcp.json`
- 初始化 MCP session
- 动态发现工具
- 注册为统一 runtime tool

当前实现的是 **MCP tools 子集**，还没有完整接入：

- resources
- prompts

## 10. Skill 机制

当前支持两类 Skill 来源：

- `skills/`
- `backend/.claude/skills/`

Skill 采用 Claude-compatible 格式：

- `SKILL.md`
- YAML frontmatter
- Markdown 正文

当前运行机制：

1. 先建立 skill catalog
2. 路由器只看 skill 元数据
3. 命中后再加载 skill 正文
4. skill 正文进入 runtime prompt

也就是说，当前 Skill 是"结构化策略文件"，不是"自动执行插件"。

## 11. 常用接口

后端挂载前缀是：

- `/api/v3`

核心接口包括：

- `POST /api/v3/chat/agent`
- `POST /api/v3/chat/stream`
- `POST /api/v3/knowledge/upload`
- `GET /api/v3/runtime/assets`
- `PUT /api/v3/runtime/assets`
- `GET /api/v3/runtime/mcp/config`
- `PUT /api/v3/runtime/mcp/config`
- `POST /api/v3/runtime/mcp/reload`
- `POST /api/v3/sessions/bootstrap`
- `POST /api/v3/sessions`
- `GET /api/v3/sessions`
- `GET /api/v3/sessions/{thread_id}/messages`
- `DELETE /api/v3/sessions/{thread_id}`

## 12. 测试

在项目根目录执行：

```bash
pytest -q
```

如果只想先检查后端模块是否能 import：

```bash
python -m py_compile backend/main.py
```

前端构建检查：

```bash
cd frontend
npm run build
```

## 13. 当前版本的已知边界

- MCP 当前只完整接入了 tools 子集
- Skill 当前主要还是 prompt/runtime 策略层，未实现 skill scripts 自动执行
- 扫描版 PDF 目前没有完整做"逐页图片 OCR"
- 当前默认前端模型选择是 `deepseek_chat`

## 14. 推荐阅读顺序

如果你想从代码角度快速理解系统，建议按这个顺序看：

1. `backend/services/agent_service.py`
2. `backend/core/middleware.py`
3. `backend/runtime/engine.py`
4. `backend/runtime/context_builder.py`
5. `backend/runtime/tool_registry.py`
6. `backend/retrieval/pipeline.py`
7. `backend/core/memory.py`
