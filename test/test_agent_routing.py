import httpx
import asyncio
import pytest

pytestmark = pytest.mark.skip(reason="遗留路由联调脚本，依赖旧版接口与手工服务，不纳入自动化 pytest。")

async def test_modern_agent():
    url = "http://127.0.0.1:8000/api/v1/chat/agent"
    
        # 测试用例矩阵：
    # 模拟路由分发与工具能力，如果 Middleware 逻辑命中，“你好” 会用本地Qwen，而其余涉及时间的会被截获并升配到 DeepSeek，同时触发工具
    tests = [
        {"query": "你好啊～", "desc": "【预期：本地 Qwen】因为是简单问候"},
        {"query": "我有个复杂的问题，请帮我分析一下：现在几点了？", "desc": "【预期：云端 DeepSeek】因为包含关键词'分析'或'几点'，并会触发工具"},
        {"query": "根据公司的员工守则，我晚上加班到几点可以吃免费的夜宵？", "desc": "【预期：云端 DeepSeek】因为包含'公司'/'几点'，并触发RAG向量检索"}
    ]
    
    for item in tests:
        query = item["query"]
        desc = item["desc"]
        print(f"\n======================================")
        print(f"👉 测试点：{desc}")
        print(f"👤 用户：{query}")
        print("🤖 Agent：", end="", flush=True)
        
        try:
            # ✅ 把超时时间稍微放宽到 300 秒，防止第一次拉取 Embedding 模型时下载过慢引起 Timeout 断开
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", url, json={"query": query}) as response:
                    if response.status_code != 200:
                        err_text = await response.aread()
                        print(f"\n❌ 请求失败, 代码: {response.status_code}, {err_text}")
                        break
                    
                    async for chunk in response.aiter_text():
                        print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ 连接或执行失败：{str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(test_modern_agent())
