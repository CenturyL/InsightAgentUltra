# Local Agent Frontend

一个给 `local_agent_api` 配套的独立前端控制台。

## 功能

- 流式调用 `/api/v1/chat/agent`
- 上传知识库文件到 `/api/v1/knowledge/upload`
- 一键调用 `/api/v1/testing/rebuild`
- 运行：
  - `/api/v1/eval/retrieval`
  - `/api/v1/eval/retrieval/compare`
  - `/api/v1/eval/generation`
  - `/api/v1/eval/benchmark`

## 启动

```bash
cd /Users/century/code/agent/langchainPro/local_agent_frontend
npm install
npm run dev
```

默认打开：

```text
http://127.0.0.1:4173
```

## 后端地址

Vite 开发代理默认把 `/api/*` 转发到：

```text
http://127.0.0.1:8000
```

如果你的 FastAPI 不是跑在这个地址：

1. 可以改 `vite.config.js`
2. 或者直接在页面左侧把 `API Base` 改成你自己的地址，例如：

```text
http://127.0.0.1:8000/api/v1
```

## 说明

- 这个前端是独立目录，不改动现有后端结构
- 默认适合本地联调
- 页面里的 `Thread ID` 会保存在浏览器本地，方便连续测试多轮对话
