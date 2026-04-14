# Frontend

InsightAgentUltra 的 React 前端控制台。

## 启动

```bash
cd frontend
npm install
npm run dev
```

默认打开：`http://127.0.0.1:5173`

## 后端地址

Vite 开发代理默认把 `/api/*` 转发到 `http://127.0.0.1:8000`。

如果你的 FastAPI 不是跑在这个地址，可以改 `vite.config.ts`。
