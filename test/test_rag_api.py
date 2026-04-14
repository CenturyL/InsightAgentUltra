import httpx
import asyncio
import pytest

pytestmark = pytest.mark.skip(reason="遗留手工 RAG 接口联调脚本，依赖旧版接口，不纳入自动化 pytest。")

async def test_rag_stream():
    # 注意这里请求的 URL 换成了 /chat/rag
    url = "http://127.0.0.1:8000/api/v1/chat/rag"
    
    # 问我们刚才在文档里写的内容
    tests = [
        {"query": "几点可以吃免费的夜宵？需要带什么凭证？", "temperature": 0.1},
        {"query": "公司下午几点下班？请假怎么请？", "temperature": 0.1},
        {"query": "公司的老板叫什么名字？", "temperature": 0.1} # 应该回答知识库不知道
    ]
    
    for payload in tests:
        print(f"\n======================================")
        print(f"🌍 正在请求 RAG 接口 [{url}]...")
        print(f"💬 你的问题：{payload['query']}")
        print("🤖 RAG 助手回复: \n", end="", flush=True)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        print(f"\n请求失败, 代码: {response.status_code}")
                        break
                    
                    async for chunk in response.aiter_text():
                        print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"\n❌ 连接失败或出错：{str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(test_rag_stream())
