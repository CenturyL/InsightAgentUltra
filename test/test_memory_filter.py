from backend.core.memory import normalize_memory_fact, rerank_memory_facts
from backend.services.agent_service import _extract_explicit_user_facts


def test_normalize_memory_fact_accepts_structured_user_fact():
    assert normalize_memory_fact("用户姓名：刘世纪") == "用户姓名：刘世纪"


def test_normalize_memory_fact_rejects_ai_and_generic_noise():
    assert normalize_memory_fact("记住了AI的名字刘世纪") is None
    assert normalize_memory_fact("无有效信息") is None
    assert normalize_memory_fact("用户可能需要完成各种任务") is None
    assert normalize_memory_fact("century") is None


def test_extract_explicit_user_facts_handles_suffix_words():
    from langchain_core.messages import HumanMessage

    facts = _extract_explicit_user_facts([HumanMessage(content="我叫刘世纪你记住")])
    assert facts == ["用户姓名：刘世纪"]


def test_rerank_memory_facts_prioritizes_identity_for_name_queries():
    facts = [
        "用户身份：通用智能体平台的使用者",
        "用户偏好：可能需要任务导向、务实简洁的智能助手服务",
        "用户姓名：刘世纪你记住",
        "用户姓名：刘世纪",
    ]
    ranked = rerank_memory_facts("我叫什么", facts, k=3)
    assert ranked[0] == "用户姓名：刘世纪"
    assert "用户身份：通用智能体平台的使用者" in ranked
