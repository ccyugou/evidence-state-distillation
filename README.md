TRIBOT ESI Evidence-Locked 


TRIBOT 是一个面向多轮急诊分诊对话的 evidence-locked ESI 管线。项目目标不是训练一个直接从原始文本输出 ESI 1-5 的黑盒模型，而是把临床对话转换为可追踪、可验证、可解释的中间状态，再按照 ESI Handbook 的政策路径确定性合成最终分诊结果。

核心原则：

- 先保证证据合法，再进行临床组合判断。
- 患者陈述、护士问题、护士观察和流程说明必须区分。
- 每个临床结论都应能追溯到 evidence_id、原文片段、来源层和局部语义状态。
- 否定、不确定、历史、问题语境和多轮问答承接不能被整句粗暴处理。
- 异常 measurement、标签、triage 轨迹和 persona 不进入 prediction-facing timeline。
- LLM 只负责受约束的语言语义裁决，不能直接决定最终 ESI。
- 最终 ESI 必须通过可审计的 ESI 政策图生成。
