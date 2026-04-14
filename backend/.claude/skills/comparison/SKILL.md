---
name: comparison
description: Compare multiple options, documents, or entities across dimensions and summarize differences.
argument-hint: objects or dimensions to compare
user-invocable: true
allowed-tools:
  - rag_search
  - rag_search_uploaded
  - web_search
  - run_plan_and_execute
---

# Comparison Skill

当用户要求比较、对比、差异分析时：
- 明确比较对象
- 提炼维度
- 若问题跨多文档或多来源，优先考虑调用 `run_plan_and_execute`
- 输出时优先给结论、差异点和证据
