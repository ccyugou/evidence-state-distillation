
# TRIBOT ESI Evidence-Locked Pipeline

## 1. 项目定位

TRIBOT 是一个面向多轮急诊分诊对话的 evidence-locked ESI 管线。项目目标不是训练一个直接从原始文本输出 ESI 1-5 的黑盒模型，而是把临床对话转换为可追踪、可验证、可解释的中间状态，再按照 ESI Handbook 的政策路径确定性合成最终分诊结果。

核心原则：

- 先保证证据合法，再进行临床组合判断。
- 患者陈述、护士问题、护士观察和流程说明必须区分。
- 每个临床结论都应能追溯到 evidence_id、原文片段、来源层和局部语义状态。
- 否定、不确定、历史、问题语境和多轮问答承接不能被整句粗暴处理。
- 异常 measurement、标签、triage 轨迹和 persona 不进入 prediction-facing timeline。
- LLM 只负责受约束的语言语义裁决，不能直接决定最终 ESI。
- 最终 ESI 必须通过可审计的 ESI 政策图生成。

本项目目前已经完成 00-04 的工程主线。05/06 的最终 reconciliation、评估和校准仍属于后续阶段。

---

## 2. 总体架构

~~~text
raw MIMIC-style JSON transcripts
        |
        v
00  raw dataset governance and measurement audit
        |
        v
01  prediction-safe evidence timeline
        |
        v
02  evidence-locked claim qualification
        |
        +----------------------+
        |                      |
        v                      v
03 deterministic vital       04 resource ordinal
   state / Step D facts         feasibility and learning
        |                      |
        +----------+-----------+
                   v
05 final reconciliation
                   |
                   v
06 deterministic ESI policy mapping and evaluation
~~~

### 2.1 ESI政策路径

- Step A immediate lifesaving intervention：进入 ESI 1 路径。
- Step B high-risk situation、confused/lethargic/disoriented 或 severe pain/distress：进入 ESI 2 候选路径，但必须通过 evidence qualification 和 field policy。
- Step C anticipated resource needs：
  - 0 resource -> ESI 5
  - 1 resource -> ESI 4
  - 2+ resources -> provisional ESI 3
- Step D danger-zone vitals：不是直接 ESI 2。只有在完整 safety gate 满足时，才对 provisional ESI 3 进行 reassessment。

03 的 official_danger_zone_signal_present=true 必须同时满足以下条件才可影响最终路径：

1. 05.provisional_esi3_eligible=true；
2. Step A 未命中；
3. Step B 未形成有效覆盖；
4. resource path 支持 2+；
5. 当前路径不要求人工 review；
6. 对应 vital 字段可评估且 danger-zone signal 存在。

---

## 3. 研究贡献与方法边界

### 3.1 不是规则堆叠，也不是端到端黑盒

00-02 的规则和门禁不是最终算法本身，而是临床语义接口层。它们解决的是传统机器学习中的数据合同、特征定义、缺失值处理和标签边界问题，只是多轮临床对话还必须额外处理：

- 否定与局部否定范围；
- 不确定和诊断性提问；
- 历史与当前状态；
- nurse question 与 patient endorsement；
- QA coreference；
- 多目标数字量表；
- evidence provenance；
- 异常测量值隔离；
- LLM 输出越权和 evidence ID 漂移。

如果跳过这层，后续模型可能把护士问题、异常数值或普通药物表达当成高危临床事实。

### 3.2 真正的算法研究重点

04-06 的目标是从合法证据学习可解释的临床组合关系：

- 哪些证据组合更可能满足 Step B；
- 哪些证据组合对应 0、1、2+ resources；
- 多个 evidence realization 的相对权重；
- 何时输出 uncertain/review；
- 哪些候选资源应保留为 second-resource possibility；
- Step D 对 provisional ESI 3 的安全 reassessment 价值。

推荐的模型族：

- Ridge ordinal；
- Elastic-net ordinal；
- Bayesian ordinal；
- 稀疏逻辑/序数模型；
- GAM 或有限交互模型；
- resource-set ranker；
- 带 evidence-linked explanation 的小型决策模型。

由于数据集只有 541 个独立 case，不能把 688 个 realization 当成 688 个独立训练样本。duplicate realizations 必须进入同一 fold，并优先使用 case-grouped cross-validation。

### 3.3 DPO、GRPO、Hermes 的定位

项目不直接训练 DPO/GRPO/RL agent 来替代 ESI 决策。

我们借鉴的是：

- DPO 的 chosen/rejected preference 思想：将 LLM 原始判断和 validator 修正后的安全判断组成可审计 pair。
- GRPO 的组内相对诊断思想：在同一 case、同一 field 或同一资源候选组内，比较 evidence lock、source policy、status consistency、policy fit 等维度。
- Hermes：负责失败样本归因、回归 fixture、错误模式聚合、修复建议和 release gate；它不直接决定临床标签。

当前 preference-style diagnostics 主要用于：

- 记录 LLM 越权；
- 分离 raw output 与 validated status；
- 发现 field rubric 边界；
- 生成 failure harness；
- 支持后续窄范围校准。

这是一种可解释的闭环工程与研究基础，而不是把临床规则隐藏进不可审计的 RL 参数。

---

## 4. 数据单位与治理

### 4.1 实验单位

当前数据集：

- 688 transcript instances；
- 541 unique cases；
- 147 duplicate case groups；
- instance_id 唯一；
- duplicate realizations 不能跨训练/验证 fold；
- case-level 是主要独立统计单位，instance-level 用于审计不同 realization 的稳定性。

### 4.2 禁止进入预测时间线的内容

以下字段只能用于 audit 或 gold evaluation，不能进入 prediction-facing evidence timeline：

- ground_truth；
- acuity；
- vignette.acuity；
- history[].triage；
- recorded_triage；
- triage_state；
- patient_persona；
- nurse_persona；
- model、pairing、seed 等实验元数据；
- 任何从标签或未来结果反推的字段。

多轮对话中的 turn-level triage 只进入 audit-only sidecar，用于后续 trajectory analysis，不作为预测证据。

### 4.3 共享 measurement contract

00 与 01 共同使用 src/clinical_data_contract.py，而不是让 01 读取 00 的审计输出。

异常值必须在 01 中 quarantine，并显式保留：

- raw_value；
- raw_unit；
- normalized_value；
- canonical_unit；
- value_plausibility；
- quarantine_reason；
- usable_for_clinical_reasoning；
- evidence_use_policy。

当前已审计：

- 25 条 pain=13 异常；
- 3 条 implausible vital 事件；
- 下游 invalid measurement consumed=0。

异常 measurement 可以保留在 audit 中，但不能进入 prompt、candidate、support 或 clinical model feature。

---

## 5. 当前模块状态

### 5.1 00：raw dataset governance

代码：

src/00_scan_dataset.py

最终输出：

outputs/00_dataset_scan_v2_1_shared_scale_contract

主要职责：

- 扫描 688 个 raw JSON；
- 检查 instance/case/run identity；
- 检查 vignette 与 ground truth 一致性；
- 审计 pain、vital、multi-target scale；
- 识别 raw 文本中的标签泄漏字段；
- 输出 audit-only 数据合同和分布统计；
- 不向 01 提供可直接读取的审计结果。

当前状态：FROZEN / PASS（soft freeze）。

00 发现的异常并不意味着删除 raw 数据，而是建立 quarantine 合同，让 01 和 02 不能消费异常测量。

### 5.2 01：prediction-safe evidence timeline

代码：

src/01_build_timeline_v4_5_4_frozen.py

最终输出：

outputs/01_timelines_v4_5_4_narrow_semantic_patch_final

核心输入：

outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl

主要职责：

- 从合法 vignette clinical fields、清洗后的 history 和 system vital reveal 构建 evidence timeline；
- 为事件和 cue 分配 evidence_id；
- 保留 parent event 与 derived clause/span；
- 记录 cue-local polarity、temporality、source layer；
- 区分 patient_reported、nurse_question、nurse_observed_statement、process/instruction；
- 建立 QA dialogue links；
- 支持多轮信息修正的安全运输；
- 将 triage history 写入 audit-only trace，不进入预测时间线；
- 对 invalid pain/vital 做 quarantine；
- 将 patient persona 入口 hard fail。

关键语义策略：

- No chest pain, just pressure 可以分成局部否定与局部阳性；
- No pressure, just sharp pain when I breathe 不被粗暴当作完全无胸部症状；
- don’t have、don’t feel、don’t think 等 target-local denial 受限处理；
- I was thinking about taking pills 与 I took too many pills 分离；
- 普通 take tablets、heart rate 8/10、fatigue/distress scale 不得误映射为 SI 或 severe pain；
- history triage 不成为 evidence。

最终审计结果：

- 688 timelines；
- 541 cases；
- 147 duplicate groups；
- load error=0；
- validation error=0；
- silent denial positive=0；
- invalid measurement 不被下游消费；
- prediction timeline 无 triage/acuity/ground truth/persona 泄漏。

### 5.3 02：evidence-locked claim qualification

当前 full-run 代码：

src/02_expert_step_ab_v3_non_dry_smoke_ready.py

旧版本 src/02_expert_step_ab.py 不作为当前 full entry；它指向过时的输入/输出配置。

当前 full 输出：

outputs/02_claim_qualification_v3_full688_resumable

主要输出：

- 02_qualified_claims.jsonl；
- 02_build_audit.json；
- claim_candidate_catalog.jsonl；
- 02_to_02d_claim_contract.jsonl；
- invalid_measurement_consumption_audit.jsonl；
- upstream_recall_gap_audit.jsonl；
- preference-style audit files。

02 只读取 01 的 prediction-safe evidence_timeline.jsonl。它不读取 raw label、01 audit metadata、triage trace 或 persona。

主要职责：

- 对 claim 做 source/evidence policy qualification；
- 使用 bounded LLM 处理 threshold ambiguity；
- 区分 confirmed_symptom_only、confirmed_step_a_trigger、confirmed_step_b_policy_trigger、uncertain_policy_review、explicit_negative、reject；
- 强制 evidence_id 存在且属于 candidate allowed evidence；
- 防止 question/process/support-only/invalid measurement 变成阳性；
- 进行 validator downgrade；
- 输出 raw status、normalized status、validated status；
- 记录 evidence lock、source policy、question policy、status consistency；
- 记录 LLM fallback、contract conflict 和 preference-style diagnostics。

full 688 结果：

- processed=688；
- claim candidates=7972；
- support bundles=4153；
- LLM eligible/successful=355/355；
- failure fallback=0；
- validator downgrade=13；
- validated Step A positive=0；
- validated Step B positive=2；
- question/process positive=0；
- invalid measurement consumed=0；
- same-evidence conflicts=0；
- preflight failures=0；
- release gate=true。

重要解释：Step B positive 很少不等于管线失败。02 的任务是证据锁定和 bounded qualification，不能把 symptom-only 或普通痛苦表达强行升级为 ESI 2。最终 ESI 分布必须在 04-06 的资源学习、reconciliation 和 policy evaluation 中判断。

### 5.4 03：deterministic vital state and Step D facts

代码：

src/03_build_vital_state_and_danger_zone_signals.py

review：

src/03_review_vital_state_outputs.py

输出：

outputs/03_vital_state_outputs_v1_full688

review：

outputs/03_vital_state_review_v1_full688

主要职责：

- 只处理合法 structured vital；
- canonical measurement 与 dialogue reveal 去重；
- 生成 vital_observations；
- 标记 field_assessability；
- 记录 missing_vital_flags；
- 记录 quarantined_measurements；
- 生成 official danger-zone flags；
- 生成 safety-context flags；
- 不使用 LLM；
- 不输出 final ESI；
- 不把 official danger zone 直接解释成 ESI 2。

当前 full 结果：

- 688 instances；
- 541 cases；
- 147 duplicate groups；
- invalid measurements quarantined=28；
- duplicate reveal counted as clinical measurement=0；
- official danger-zone rows=7，全部为 SpO2；
- invalid measurement consumed=0；
- nonofficial danger trigger=0；
- LLM/model/final ESI=0；
- suspicious review rows=0；
- release gate=true。

年龄策略当前为 conservative unknown：不能从 persona、语言风格、症状或标签推断年龄。SpO2 可按年龄无关规则评估；HR/RR 在当前证据合同下不做年龄依赖的官方判断。

03 对后续只提供：

- vital_observations；
- field_assessability；
- missing_vital_flags；
- quarantined_measurements；
- official_danger_zone_flags；
- official_danger_zone_signal_present；
- safety_context_flags；
- supporting_evidence_ids；
- policy_rule_ids。

### 5.5 04：resource ordinal feasibility and interpretable learning entry

当前代码：

src/04_resource_ordinal_feasibility_audit.py

review：

src/04_review_resource_ordinal_feasibility.py

输出：

outputs/04_resource_ordinal_feasibility_v1_full688

review：

outputs/04_resource_ordinal_feasibility_review_v1_full688

04 当前不是旧版 LLM resource extractor，也不是最终 ESI 预测器。旧代码 src/04_expert_step_c.py 及旧输出 outputs/04_expert_outputs 不作为当前研究主线。

当前 04 的定位是 Evidence-Constrained Resource Ordinal Learning 的 feasibility audit：

- 以 541 unique cases 为主要监督单位；
- 同一 case 的 duplicate realizations 固定在同一 fold；
- 从 01/02/03 构建低维、evidence-linked、label-free feature matrix；
- 仅以 gold ESI 3/4/5 做弱监督 feasibility audit；
- 高危/标签政策冲突病例进入 audit exclusion 或降权候选；
- 暂不拟合最终模型；
- 不读取 LLM 做 resource decision；
- 不输出 final ESI。

当前结果：

- 688 instances；
- 541 cases；
- 147 duplicate groups；
- eligible weak supervision cases=341；
- weak bucket 0=36；
- weak bucket 1=58；
- weak bucket 2+=247；
- excluded high-acuity cases=200；
- interpretable case-level features=91；
- grouped folds=5；
- duplicate split count=0；
- alignment issues=0；
- feature leakage=0；
- model_fit=false；
- release/review gate=true。

下一步应比较：

1. Ridge ordinal；
2. Elastic-net ordinal；
3. Bayesian ordinal；
4. resource-set ranker；
5. 规则/模型融合的 threshold calibration。

评价应至少包括：

- macro-F1；
- balanced accuracy；
- ordinal absolute error；
- calibration；
- case-level bootstrap；
- duplicate-aware grouped cross-validation；
- undertriage/overtriage；
- evidence-linked explanation coverage。

不能仅用 688 instance accuracy 宣称模型有效。

---

## 6. 05-06 计划

05 和 06 尚未完成最终实现，不能把当前 04 feasibility audit 误称为完整 ESI 系统。

### 6.1 05 final reconciliation

05 的职责：

- 合并 02 Step A/B claims；
- 合并 03 official danger-zone/safety context；
- 合并 04 resource ordinal path；
- 处理 provisional ESI 3；
- 应用 fail-closed review；
- 保留 strict 与 calibrated 两种结果；
- 不让 LLM 自由覆盖最终 ESI；
- 对 rescue/promote 规则输出 evidence-linked reason。

推荐输出：

- effective Step A/B state；
- resource bucket/path；
- provisional ESI；
- final candidate path；
- rescue/review flags；
- supporting evidence IDs；
- policy rule IDs；
- chosen/rejected diagnostic records。

### 6.2 06 policy mapping and evaluation

06 的职责：

- 按 ESI Handbook 政策图确定性映射 ESI 1-5；
- 计算 instance-level 和 case-level 结果；
- 同时输出 strict、calibrated、rescue simulation；
- 评估 label-policy-evidence mismatch；
- 评估 undertriage、overtriage、macro-F1、balanced accuracy、ordinal error；
- 分层评估 541 cases 与 688 realizations；
- 做 trajectory 和 multi-realization consistency analysis；
- 不把 unsupported gold label 自动当成上游代码 bug。

对于 gold ESI1/2/3 与可用 evidence 不一致的病例，应单独报告：

- evidence-supported；
- evidence-insufficient；
- label-policy mismatch；
- likely upstream recall gap；
- uncertain boundary。

---

## 7. Hermes failure-loop

Hermes 是项目的失败分析与修复控制层，不是临床决策器。

建议闭环：

~~~text
raw case / evidence
      |
      v
module output and audit
      |
      v
Hermes failure taxonomy
      |
      +--> fixture / counterfactual / preference-style pair
      |
      v
narrow patch or interpretable model calibration
      |
      v
regression gate and release decision
~~~

Hermes 重点检查：

- silent false positive；
- source-policy violation；
- question/process leakage；
- invalid measurement consumption；
- evidence ID hallucination；
- same-field contradiction；
- wrong historical/current state；
- resource bucket collapse；
- Step D misuse；
- multi-realization inconsistency；
- validator downgrade pattern。

停止条件：

- P0 安全问题：立即阻断 release；
- 同一 P1 语义模式重复出现 2-3 次：集中修复；
- 单个低影响、无越权边界：进入 backlog；
- 修复必须包含 fixture、回归测试、审计指标和 lineage hash；
- 不因单个普通 recall gap 无限重写上游。

---

## 8. 运行环境与复现

建议使用 data_engineering 环境。以下命令均在项目根目录执行，PowerShell 可直接运行。

### 8.1 00

~~~powershell
conda run -n data_engineering python src/00_scan_dataset.py --raw-data-dir transcripts/mimic --output-dir outputs/00_dataset_scan_v2_1_shared_scale_contract --quiet
~~~

### 8.2 01

~~~powershell
conda run -n data_engineering python src/01_build_timeline_v4_5_4_frozen.py --raw-data-dir transcripts/mimic --output-dir outputs/01_timelines_v4_5_4_narrow_semantic_patch_final --expected-instance-count 688 --expected-case-count 541 --primary-experiment-unit case --case-aggregation canonical_run --duplicate-weighting equal_case --hard-fail-on-count-mismatch
~~~

### 8.3 02 full

~~~powershell
conda run -n data_engineering python src/02_expert_step_ab_v3_non_dry_smoke_ready.py --input-file outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl --output-dir outputs/02_claim_qualification_v3_full688_resumable --use-default-count-expectations --overwrite
~~~

恢复中断运行时使用 --resume，不要同时使用 --overwrite。

02 review：

~~~powershell
conda run -n data_engineering python src/02_review_step_ab_outputs.py --timeline-file outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl --output-dir outputs/02_claim_qualification_v3_full688_resumable --review-dir outputs/02_claim_qualification_v3_full688_resumable_review --expected-instance-count 688 --expected-case-count 541 --expected-duplicate-case-count 147 --fail-on-hard-gate
~~~

### 8.4 03 full

~~~powershell
conda run -n data_engineering python src/03_build_vital_state_and_danger_zone_signals.py --input-file outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl --output-dir outputs/03_vital_state_outputs_v1_full688 --policy-dir policy --cohort-contract manifests/prediction_safe_cohort_contract.json --count-manifest manifests/instance_count_manifest.json --overwrite
~~~

03 review：

~~~powershell
conda run -n data_engineering python src/03_review_vital_state_outputs.py --output-dir outputs/03_vital_state_outputs_v1_full688 --review-dir outputs/03_vital_state_review_v1_full688
~~~

### 8.5 04 resource feasibility

~~~powershell
conda run -n data_engineering python src/04_resource_ordinal_feasibility_audit.py --timeline-file outputs/01_timelines_v4_5_4_narrow_semantic_patch_final/evidence_timeline.jsonl --claims-file outputs/02_claim_qualification_v3_full688_resumable/02_qualified_claims.jsonl --vitals-file outputs/03_vital_state_outputs_v1_full688/03_vital_state_outputs.jsonl --raw-dir transcripts/mimic --count-manifest manifests/instance_count_manifest.json --output-dir outputs/04_resource_ordinal_feasibility_v1_full688 --folds 5 --overwrite
~~~

04 review：

~~~powershell
conda run -n data_engineering python src/04_review_resource_ordinal_feasibility.py --output-dir outputs/04_resource_ordinal_feasibility_v1_full688 --review-dir outputs/04_resource_ordinal_feasibility_review_v1_full688
~~~

### 8.6 运行前检查

- 确认当前工作目录是 TRIBOT 根目录；
- 确认使用 data_engineering；
- 确认 01 输入输出目录与 02/03/04 命令一致；
- 不要把旧 01_build_timeline.py、旧 02_expert_step_ab.py 或旧 04_expert_step_c.py 当作最终入口；
- 不要把 outputs/04_expert_outputs 当作新版 04 结果；
- full run 前先检查 count manifest 和 artifact lineage；
- 任何 release gate 失败都先进入 Hermes 分类，不要直接调阈值或删记录。

---

## 9. 文件索引

### 当前主代码

- src/clinical_data_contract.py
- src/00_scan_dataset.py
- src/01_build_timeline_v4_5_4_frozen.py
- src/02_expert_step_ab_v3_non_dry_smoke_ready.py
- src/02_review_step_ab_outputs.py
- src/03_build_vital_state_and_danger_zone_signals.py
- src/03_review_vital_state_outputs.py
- src/04_resource_ordinal_feasibility_audit.py
- src/04_review_resource_ordinal_feasibility.py

### 当前主输出

- outputs/00_dataset_scan_v2_1_shared_scale_contract
- outputs/01_timelines_v4_5_4_narrow_semantic_patch_final
- outputs/02_claim_qualification_v3_full688_resumable
- outputs/03_vital_state_outputs_v1_full688
- outputs/03_vital_state_review_v1_full688
- outputs/04_resource_ordinal_feasibility_v1_full688
- outputs/04_resource_ordinal_feasibility_review_v1_full688

### 关键政策与合同

- policy/esi_step_d_policy_manifest.json
- policy/esi_step_d_thresholds.json
- manifests/prediction_safe_cohort_contract.json
- manifests/instance_count_manifest.json

### 历史输出说明

outputs/ 中还存在多轮 smoke、dry-run、旧版 01/02/04 和 review 目录。它们用于回归和历史审计，不能与当前主输出混用。整理时应先保留本 README 列出的主代码、主输出、policy 和 manifest，再由人工确认后删除历史目录。

---

## 10. 当前结论与下一步

当前可以声明：

- 00 数据治理和 measurement contract 已通过；
- 01 prediction-safe evidence timeline 已通过；
- 02 evidence-locked claim qualification full run 已通过工程 gate；
- 03 deterministic vital fact layer 已通过 full review；
- 04 已完成 resource ordinal feasibility audit，具备进入可解释算法比较的条件。

当前不能声明：

- 最终 ESI accuracy 已经达到目标；
- 04 已经训练并验证了最终 resource model；
- 05/06 全链路已经完成；
- 02 的 2 条 Step B positive 代表最终 ESI 2 数量；
- 旧版 04 resource extractor 仍是研究主线。

下一阶段优先顺序：

1. 固化 04 label-policy-evidence feasibility audit；
2. 建立 case-grouped resource ordinal baseline；
3. 比较 Ridge、Elastic-net、Bayesian ordinal；
4. 设计 05 strict/reconciled interface；
5. 实现 06 deterministic ESI policy mapping；
6. 用 Hermes 做 failure taxonomy、counterfactual fixture 和多 realization consistency；
7. 在不破坏 evidence lock 的前提下评估 DPO/GRPO-style preference diagnostics；
8. 最后再报告 case-level accuracy、macro-F1、undertriage、overtriage、calibration 和标签-证据不一致。

任何最终性能结论，都必须基于同一 artifact lineage、541 case 主单位和 688 realization 辅助统计。


## 11. 03-04的最新进展
目前项目已经从“规则管线调试”进入“可解释资源决策模型研究”阶段。
03：生命体征事实层
当前状态：基本冻结，通过。
03的职责已经收窄为：
消费01提供的合法生命体征；
区分有效、缺失、异常和不可评估；
输出官方Step D danger-zone信号；
输出普通safety context；
不预测最终ESI；
不把危险生命体征直接转换为ESI 2。
已确认的问题和改进：
28条异常measurement被隔离，未进入03有效特征；
canonical vital与dialogue reveal没有重复计数；
Step D官方信号与普通安全上下文分离；
年龄合同暂未启用，因此HR/RR不做不合理年龄推断；
03官方信号目前主要来自SpO₂，数量较少但符合严格政策。
03后续只在以下情况才重开：
01 schema或measurement合同变化；
vital阈值或Step D政策变化；
年龄规则启用；
发现异常值或reveal重复进入下游。
否则不再继续给03增加临床规则。
04：资源数量学习层
04已经拆成三部分。
04a：监督可行性审计
已完成：
使用541个case作为独立实验单位；
688个instance中的重复realization被正确分组；
296个case进入资源建模；
388条instance作为输入；
弱资源标签为：0资源：30
1资源：53
2+资源：213

其中153个case属于相对可信的supported supervision；
143个case属于partial或弱监督。
关键结论是：ESI标签不能直接当作干净的资源真值。ESI 3 -> 2+ resources只能作为弱监督代理，而不是严格ground truth。
04b：prediction-safe特征平面
目前40维特征矩阵已经完成安全隔离：
主要来自02可信claim和合法vital context；
排除了Step D官方危险信号；
异常measurement没有进入特征；
canonical measurement不会因dialogue reveal重复计算；
不使用ground truth、acuity或turn-level triage；
每个特征可以追溯到evidence。
04b目前可以冻结。
04c：资源序数模型
04c初始Ridge模型暴露了真正问题：
原始argmax几乎全部预测为2+；
0资源和1资源召回接近0；
但概率排序并非完全失效：AUC(R>=1)约0.72至0.75；
AUC(R>=2)约0.67至0.70；
期望资源分数整体仍呈0 < 1 < 2+的顺序。

因此不是“特征完全没有信息”，而是：
类别严重失衡；
资源监督存在support-target confounding；
比例优势结构过强；
默认argmax不适合当前资源任务；
1-resource与2+之间缺乏足够可分性。
B2分层阈值实验
B2已经证明模型特定的层级阈值比直接argmax更合适。
Conditional ORTHO B2a
优点：
Macro-F1最高，约0.495；
R1 recall明显改善；
更偏性能。
问题：
2+ recall下降到约0.75；
不满足安全主线要求。
Dual ORTHO B2b
优点：
2+ recall约0.906；
2+ -> 0较低；
QWK和MAOE较好；
安全性更强。
问题：
R1 recall约0.094；
只有3/5 folds出现非零R1召回；
仍然存在1-resource塌缩。
因此目前应保留两个参考点：
Conditional ORTHO B2a：性能参考；
Dual ORTHO B2b：安全参考。
它们都还不是最终可发布模型，05/06暂时不能接入。
当前04的下一步
下一步是受控运行 Fused Conditional Ridge，而不是继续增加规则。
目标：
第一边界学习 R>=1；
第二边界学习 R>=2 | R>=1；
两个head共享部分斜率结构；
保留不同intercept；
通过融合惩罚减少fold间不稳定；
不进行事后系数平均。
首轮只研究：
rho = 0, 0.1, 0.3, 1, 3, 10
固定：
296 case；
相同grouped folds；
natural weighting；
partial weight = 0.5；
ORTHO23作为主实验；
FULL40作为敏感性实验；
B2a和B2b分别评估。
必须保证：
rho=0严格复现Conditional ORTHO；
R1 recall在至少4/5 folds非零；
2+ recall维持约0.87至0.90以上；
2+ -> 0不明显恶化；
概率质量和Boundary 2稳定性不下降；
不能只因Macro-F1提高就直接放行。
如果融合失败，下一步才考虑：
两个独立累计模型；
partial supervision重新分层；
resource ontology兼容性特征；
支持集合与资源数量的结构化建模。
05：ESI政策协调与最终候选生成
05不负责训练模型，也不允许LLM自由决定最终ESI。
它的职责是把：
02可信Step A/B claims
+ 03生命体征事实
+ 04资源数量概率/分支
通过ESI Handbook政策图进行确定性协调。
建议保留两条输出路径：
Strict path
只使用：
已确认的Step A；
已确认的Step B；
已通过资源模型和政策门禁的resource path；
不接受未经验证的rescue。
Calibrated path
只允许受控的校准策略，例如：
高质量uncertain second resource promotion；
03 danger-zone safety rescue；
明确的资源层级阈值；
必须保留原始路径和promotion原因。
05应输出：
provisional ESI；
final candidate ESI；
触发的Step A/B/C/D路径；
resource probability和threshold；
safety override；
uncertainty/review状态；
supporting evidence IDs；
是否满足人工复核条件。
05不应读取raw turn-level triage，也不能用gold label反向修正预测。
06：评估、故障归因与反馈闭环
06应同时承担评估和诊断，而不是只输出一个accuracy。
评估单位
必须同时报告：
688个instance级结果；
541个case级结果；
147组duplicate realization的一致性；
所有fold按case分组，不能让同一case跨fold。
核心指标
包括：
accuracy；
macro-F1；
balanced accuracy；
QWK；
MAE；
每个ESI类别recall；
undertriage与overtriage；
高危类别的安全错误；
resource boundary AUC；
概率log loss、Brier和校准；
多realization一致性。
归因分析
需要区分：
Step A漏召回；
Step B边界过宽或过窄；
03 Step D缺失或冲突；
04资源候选不足；
04资源模型排序失败；
阈值导致R1压缩；
weak label与文本证据不一致；
ESI gold与evidence policy不一致。
Hermes反馈
Hermes应把错误分成：
工程错误；
证据错误；
policy错误；
模型排序错误；
标签/证据不一致；
可修复的边界样本；
不应强行拟合的病例。
它可以形成preference-style diagnostics和反事实比较，但不应直接把最终标签硬编码回01/02。
最终路线
当前最合理的顺序是：
03冻结
→ 04a/04b冻结
→ 04c Fused Conditional Ridge
→ B2a/B2b双路径评估
→ 04资源模型放行判断
→ 05 strict/calibrated reconciliation
→ 06 case-level evaluation + Hermes diagnosis
当前明确不应做：
不再给01/02无限添加关键词；
不把Step D直接当作ESI 2；
不让LLM直接输出最终ESI；
不把弱ESI标签当作干净资源真值；
不用单次accuracy决定模型是否成功；
不在融合模型前同时加入class weight、soft label和feature fusion。
一句话总结：
03已经是稳定的生命体征事实层；04已经从规则资源抽取转向可解释资源序数学习，但04c仍处于模型选择阶段。下一步应验证Fused Conditional Ridge能否同时保住2+安全性并恢复1-resource召回；只有资源层稳定后，05/06才应正式接入。
