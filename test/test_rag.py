import os
import pytest

pytestmark = pytest.mark.skip(reason="遗留手工联调脚本，依赖已删除的 rag_service，不纳入自动化 pytest。")

from local_agent_api.retrieval.pipeline import process_and_store_document, search_knowledge

def test_rag_ingest():
    # 1. 创建一份测试文档
    test_file = "../local_agent_api/data/company_rules.md"
    os.makedirs("../local_agent_api/data", exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("""
# 凌越科技内部员工守则
1. 公司上班时间为每天上午 9:30 到 下午 18:30。
2. 免费夜宵供应时间为晚上 20:00，凭工卡到一楼食堂领取。
3. 请假需要提前至少 1 天在 OA 系统中由部门主管审批。病假需要附带三甲医院证明。
4. WIFI 访客密码是: LingYue@2024_Guest，内网密码请联系 IT 部门（分机号 666）。
""")
    
    print(f"📄 测试库文档建立成功：{test_file}")

    # 2. 存入 ChromaDB 并在本地计算 Embeddings
    print("⏳ 开始切分文档并进行本地化 Embedding (第一次运行会自动下载模型权重)...")
    chunks_count = process_and_store_document(test_file)
    print(f"✅ 文档处理完毕！总共被切成了 {chunks_count} 个数据块并成功存入本地 ChromaDB 向量库。\n")

    # 3. 模拟 RAG 第一步：问题召回
    queries = [
        "几点可以吃免费的夜宵？",
        "WiFi密码是多少？"
    ]

    for q in queries:
        print(f"❓ 模拟用户提问：{q}")
        results = search_knowledge(q, k=1)
        print(f"💡 本地数据库召回线索：{results[0].page_content}\n")

if __name__ == "__main__":
    test_rag_ingest()
