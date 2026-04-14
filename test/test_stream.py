import httpx
import asyncio
import sys
import pytest

pytestmark = pytest.mark.skip(reason="遗留手工流式联调脚本，依赖本地服务与旧版接口，不纳入自动化 pytest。")

async def test_llm_stream():
    url = "http://127.0.0.1:8000/api/v1/chat/stream"
    payload = {
        "query": "你好，请用一句话介绍你自己。",
        "temperature": 0.7
    }
    
    print(f"🌍 正在连接本地 API ({url})...")
    print(f"💬 问题：{payload['query']}\n")
    print("🤖 Qwen 模型回复: ", end="", flush=True)
    
    try:
        # 使用 httpx 客户端接收流式响应
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status() # 检查 HTTP 错误
                
                # 异步迭代接收每一个数据块(token)
                async for chunk in response.aiter_text():
                    print(chunk, end="", flush=True)
        print("\n\n✅ 测试完成！大模型连通成功！")
    except httpx.ConnectError:
        print("\n❌ 连接失败：请确认主程序 (main.py) 是否已经启动在 8000 端口。")
    except Exception as e:
        print(f"\n❌ 发生其他错误：{str(e)}")
        print("请检查 Windows 端的 Ollama (Easytier 对应 IP) 是否正常运行。")

if __name__ == "__main__":
    asyncio.run(test_llm_stream())
