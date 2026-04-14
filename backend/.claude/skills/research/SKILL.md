---
name: research
description: Research, reporting, and long-form synthesis across multiple sources.
argument-hint: research topic or report goal
user-invocable: true
allowed-tools:
  - rag_search
  - rag_search_uploaded
  - web_search
  - run_plan_and_execute
---

# Research Skill

当用户需要调研、报告、长总结时：
- 优先判断是否需要联网搜索与计划执行
- 汇总多来源时避免重复
- 输出要包含结论、关键发现和不确定性
