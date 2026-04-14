---
name: extraction
description: Extract structured fields, requirements, or key facts from source material.
argument-hint: fields to extract
user-invocable: true
allowed-tools:
  - rag_search
  - rag_search_uploaded
  - run_plan_and_execute
---

# Extraction Skill

当用户要求字段提取、信息抽取、结构化输出时：
- 先识别字段清单
- 不确定时明确标注缺失
- 尽量给结构化输出
