# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**InsightAgentMax** — a general-purpose Agent Runtime platform. Not a single-turn RAG demo; it is a full runtime for real-world task execution combining: ReAct main loop + Plan-and-Execute (PAE) + Session management + Long-term memory + MCP + Skills + RAG.

## Deployment

三版本并行部署，通过 EasyTier 组虚拟局域网打通阿里云 ECS 和本地 Ubuntu 机器。

### 网络拓扑

```
用户浏览器 → 阿里云 ECS (139.196.39.24, ssh别名: ecs)
                ├── nginx 托管前端静态文件 (/var/www/)
                └── nginx 反向代理 API 请求 → EasyTier 虚拟网络 → Ubuntu (10.144.144.11, ssh别名: su5090dd)
                                                                      ├── port 18000: V1 (~/project/InsightAgent/)
                                                                      ├── port 18001: V2 (~/project/InsightAgentPro/)
                                                                      └── port 18002: V3 (~/apps/InsightAgentMax/)
```

### 路由规则 (nginx on ECS)

| URL 路径 | 前端静态文件 | API 代理目标 |
|----------|-------------|-------------|
| `/` | `/var/www/insightagent/` (V3) | `/api/v3/` → `10.144.144.11:18002` |
| `/pro/` | `/var/www/pro/` (V2) | `/api/v2/` → `10.144.144.11:18001` |
| `/legacy/` | `/var/www/insightagent-legacy/` (V1) | `/api/v1/` → `10.144.144.11:18000` |

### 部署工作流

1. 本机 Mac 上开发调试
2. push 到 GitHub
3. SSH 到 Ubuntu (`ssh su5090dd`) clone/pull 后端代码，用 conda 环境 `insightagent` 启动 uvicorn
4. SSH 到 ECS (`ssh ecs`) 更新前端 build 产物到 `/var/www/` 对应目录

### 服务器信息

- **Ubuntu (su5090dd)**: RTX 5090D, miniconda3, conda env `insightagent` (Python 3.11), PostgreSQL (pgdata at ~/pgdata)
- **ECS (ecs)**: 阿里云轻量，nginx, EasyTier, 仅做前端托管和反向代理
- **后端进程**: `conda run -n insightagent uvicorn local_agent_api.main:app --host 0.0.0.0 --port <port>`

## Development Commands

### Backend

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r local_agent_api/requirements.txt
cp local_agent_api/.env.example local_agent_api/.env  # then fill in secrets

# Run (from project root)
uvicorn local_agent_api.main:app --reload
# API: http://127.0.0.1:8000
```

### Frontend

```bash
cd local_agent_frontend
npm install
npm run dev      # http://127.0.0.1:5173
npm run build    # production build
```

### Tests

```bash
pytest -q                              # all tests
pytest test/test_agent_routing.py -q  # single test file
```

### Required environment variables

`DEEPSEEK_API_KEY` and `POSTGRES_URL` (with pgvector extension) are required. All config is validated via Pydantic in `local_agent_api/core/config.py`.

## Architecture

### Request lifecycle

```
Frontend → POST /api/v3/chat/agent
  → Middleware (model binding, context injection)  [core/middleware.py]
  → Agent Service (ReAct loop / PAE dispatch)      [services/agent_service.py]
      → Runtime Engine (routing, complexity)       [runtime/engine.py]
      → Context Builder (dynamic system prompt)    [runtime/context_builder.py]
      → Tool Registry (RAG, web, memory, MCP, PAE) [runtime/tool_registry.py]
  → NDJSON stream back to frontend
```

### Key layers

| Layer | Directory | Purpose |
|-------|-----------|---------|
| API & Schemas | `api/` | FastAPI routes, Pydantic request/response models |
| Core | `core/` | Config, LLM factories, embeddings, memory, middleware |
| Runtime | `runtime/` | Routing engine, context builder, skill loader, MCP client, tool registry, LangGraph workflow |
| Agents | `agents/` | PAE internals: planner, executor, reflection, orchestrator |
| Retrieval | `retrieval/` | RAG pipeline: chunking, embedding, reranking, citations |
| Services | `services/` | Agent service (main loop), session service, asset service |
| Skills | `skills/` | Strategy files injected into system prompt (YAML frontmatter + Markdown) |
| Persona | `persona/` | `AGENTS.md` (system role), `SOUL.md` (tone/style) — editable at runtime |
| Memory | `memory/` | `MEMORY.md` workspace-level long-term memory template |

### Execution strategies

- **ReAct loop**: default for simple queries and direct tool calls. Runs in `agent_service.py` via LangGraph.
- **Plan-and-Execute (PAE)**: invoked as a tool by the ReAct loop for complex multi-step tasks. Internals live in `agents/`.
- **Runtime router** (`runtime/engine.py`): LLM judges whether to enter PAE or use direct tools, and selects applicable skills based on metadata.

### State objects

- `RuntimeRequest` — what the API receives; includes `thread_id`, `user_id`, `model_choice`, `plan_mode`
- `AgentState` — LangGraph state for the ReAct loop; includes `messages`, `conversation_summary`, `runtime_route`
- `OrchestratorState` — extends AgentState for PAE; adds planner/executor/output hints and activated skill names

### Memory model

- **Short-term (thread-level)**: LangGraph `MemorySaver` or `PostgresSaver` — scoped to a conversation
- **Long-term (user-level)**: pgvector + PostgreSQL — persists across sessions
- **Conversation compression**: old messages are auto-summarized to prevent token overflow (`runtime/conversation_memory.py`)

### Skill system

Skills in `skills/` are YAML-frontmatter + Markdown files (same format as Claude SKILL.md). They are **not** executable plugins — they are structured strategy/guidance text injected into the system prompt when the runtime router activates them. Metadata is loaded first; full text only when a skill is matched.

The project also keeps a copy of its MCP configs and SKILL files under `.claude/` (project root), intentionally mirroring Claude-compatible formats. **This directory is shared with Claude Code's own configuration** (settings, hooks, etc.), which causes an unavoidable naming conflict. When reading files under `.claude/`, distinguish between project runtime assets (MCP/SKILL files used by the agent) and Claude Code's own config — do not conflate or overwrite either side.

### Streaming protocol

All agent responses stream as **NDJSON** (one JSON object per line). The frontend (`App.tsx`) parses this incrementally to render tool calls, PAE traces, and final text in real time.

### MCP integration

`.mcp.json` at project root defines MCP servers. Config can be updated at runtime via `/api/v3/runtime/mcp/config` (PUT) and reloaded via `/api/v3/runtime/mcp/reload` (POST) without restarting the server.

### Database

PostgreSQL with pgvector is required (not optional). Tables are created automatically on startup:
- `agent_sessions`, `agent_session_messages` — session history
- `checkpoints*` — LangGraph persistent checkpointer
- pgvector tables — long-term memory embeddings

### API surface (all under `/api/v3`)

| Endpoint | Purpose |
|----------|---------|
| `POST /chat/agent` | Full agent with ReAct + PAE (main endpoint) |
| `POST /chat/stream` | Direct LLM chat, no agent/tools |
| `POST /knowledge/upload` | Ingest documents into RAG |
| `GET/PUT /runtime/assets` | Fetch/save AGENTS.md, SOUL.md, MEMORY.md, skills |
| `GET/PUT /runtime/mcp/config` | MCP server config |
| `POST /runtime/mcp/reload` | Reinitialize MCP without restart |
| `POST /sessions/bootstrap` | Get or create user sessions |
| `GET /sessions/{thread_id}/messages` | Load session history |
| `DELETE /sessions/{thread_id}` | Delete a session |
