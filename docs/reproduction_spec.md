# JITMEM ALFWorld 复现规格

本文是实现和评测的可核查规格。主要来源是 [JITMEM v1 PDF](https://arxiv.org/pdf/2609.27334) §3、§4、Appendix A/B；judge 和 split 补充来源是 [SkillOS v1 PDF](https://arxiv.org/pdf/2605.06614) Appendix A.4 Figure 13、Appendix C。页码均采用 PDF 印刷页码，从 1 开始。设置的证据等级分为：原文明确、继承来源、工程选择、待核实。

## 当前复现范围

按用户要求，连接参数通过环境变量指定：通用 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`，默认 OpenAI-compatible `https://api.openai.com/v1/chat/completions`。可用 `JITMEM_EXECUTOR_*`、`JITMEM_CURATOR_*` 覆盖角色；环境变量优先于可选 TOML 值，未单独配置的 curator 继承 executor。运行 metadata 保存解析后的 URL/model 和密钥变量名称，不保存密钥值。

当前实现训练自由的 read-time memory pipeline。curator 与 executor 可以分别配置用户提供的模型 API；judge 固定复用 executor 的同一个 client、模型与生成配置，不支持独立 `judge` 配置。这对应论文的 JITMEM-base 或强模型 prompted-curator 变体；如果 API 提供的是普通 pretrained 模型，不能称为论文 RL-trained JITMEM，也不能承诺复现其 77.4% 等训练后结果。

没有找到可验证的官方 JITMEM 代码或 trained-curator checkpoint 链接。检查了论文全文、arXiv abstract 页面、[由作者提交的 Hugging Face paper 页面](https://huggingface.co/papers/2609.27334)及针对 Salesforce/GitHub/Hugging Face 的搜索；作者 paper 页面当前显示没有关联模型。检索未发现不等于不存在，后续可接入作者发布的 curator API。

## 推理协议

一次 episode 的输入是任务描述 `x`、环境 initial observation 和当前 memory snapshot。memory entry 持久保存任务和完整的 observation/action trace。轨迹应保留终止 action 后的最后 observation；不能因 executor 只看三步历史而裁剪持久轨迹。executor 的原始模型回答可保存到运行审计日志，但 raw memory 的论文核心数据是环境 observations 与实际 actions。

1. 对 snapshot 中每条 memory 的**任务描述**建立 BM25，查询当前任务描述，取最多 3 条，按分数排序。不能把 trajectory 内容、task type、环境路径或 ground-truth label 混入检索文本。
2. 将当前任务与检索到的完整轨迹送入 curator 一次，产生自然语言 payload。该 payload 在本 episode 全程复用，不在每个环境 step 重新生成。
3. 每步 executor 接收任务、payload、最近 3 个 observation/action 对、当前 observation、当前 admissible actions。生成动作并严格解析一个 `<action>…</action>` 中的合法命令。当前实现最多 30 次 executor decisions；格式错误或非法动作也消耗一次 decision budget，产生显式错误反馈，但不对环境执行。分别统计 `steps`（决策数）和 `environment_steps`（实际执行命令数）。论文没有披露非法动作的解析/恢复细节，因此这一处理是工程选择。
4. episode 结束后，executor 模型另作一次 judge 调用，仅观察任务与完整轨迹。judge 成功则将 raw trajectory 加入 pending writes。payload 不作为持久 memory。
5. 原生环境 verifier 的成功结果用于评测日志和指标，**不**参与默认存储 gate，也不送给 curator 或 judge。

论文不说明空库时是否还调用 curator。当前 `jitmem` 和 `write-summary` 在空库时仍调用 curator，并显式提供空 retrieved context；prompt 此时允许一般操作指导。这是 cold-start 工程选择，可以产生来自模型参数知识的 hints。不能把空库任务误标为有经验支持的 memory gain。

## Batched streaming 和评估

每个 run 开始时初始化独立空 memory bank，训练轨迹不带入。按指定 order seed 打乱完整任务列表；每 10 个任务组成一个 batch。batch 内全部任务使用同一个 bank snapshot，只有 batch 完成后才提交 pending successful trajectories；可以串行执行环境来减少资源消耗，仍保持相同的逻辑语义。最后不足 10 个任务的 batch 也按相同规则处理。跨 run 不共享 bank、结果和随机数状态。

原文 ALFWorld 设置为 140 test tasks、3 个不同随机任务顺序 run，报告 SR 的均值及标准差。JITMEM 未披露随机种子值、完整 task manifests、环境版本和精确 split 名。SkillOS Appendix C 明确测试使用 140 `valid_seen`、训练 3553，且 JITMEM 表中的多个 baseline 数值逐项相同，因此采用 `valid_seen`/`eval_in_distribution` 是有证据的继承推断。应检查真实数据目录的 manifest；若发现 134 `valid_unseen`，它属于不同 split，不能直接与论文 140-task 数字比较。保存每个 run 的确切有序 task IDs、数据路径和数据文件 hash，以供复查。

本地 task discovery 已确认：按官方 text-eligible filters，`valid_seen` 为 140（Pick35、Look13、Clean27、Heat16、Cool25、Pick2 24），六类数量逐项匹配 SkillOS 的评测表；`valid_unseen` 为 134。这里核实的是任务覆盖和 split 选择，尚不是 API agent 的评估结果。

## 已核实设置

| 项目 | 设置 | 来源 / 等级 |
| --- | --- | --- |
| 持久 memory | 完整未压缩 task + observation/action trajectory | §3，第 4 页，原文明确 |
| 默认写入 | executor-as-judge 判为成功 | §3，第 4 页，原文明确 |
| 检索 | BM25，只索引 task description | §3，第 4 页，原文明确 |
| 检索数 | k=3 | Appendix A，第 18 页，原文明确 |
| payload | 自然语言、面向当前任务、一次 episode 一个 | §3，第 4 页，原文明确 |
| 评估 bank | 每个 sequence/run 空库开始 | §3，第 5 页，原文明确 |
| ALFWorld test count | 140 | §4，第 5 页，原文明确 |
| ALFWorld test split | valid_seen / eval_in_distribution | SkillOS Appendix C，第 25 页；对 JITMEM 是继承推断 |
| 评估 batch size | 10，batch 后更新 bank | Appendix A，第 18 页，原文明确 |
| 评估重复 | 3 个随机 task-order runs | Table 1，第 6 页，原文明确 |
| executor history | 最近 3 步 | Appendix A，第 18 页，原文明确 |
| episode step budget | 论文 30 turns；代码 30 decisions，非法输出也消耗 | Appendix A，第 18 页 + 公开工程解释 |
| executor temperature | 1.0 | Appendix A，第 18 页，原文明确 |
| executor max output | 4096 tokens / call | Appendix A，第 18 页，原文明确 |
| Qwen3-8B executor eval | ALFWorld thinking enabled | Appendix A，第 18 页，原文明确 |
| Qwen3-8B curator | non-thinking，temperature .6，top-p .95，top-k 20 | Appendix A，第 18 页，原文明确 |
| GPT-5.4 / Gemini-2.5-Pro curator | temperature 1.0 | Appendix A，第 18 页，原文明确 |
| curator inference max output | 代码默认 8192；原文未披露 | 工程选择，参考训练 rollout 上限，不能冒充作者测试设置 |
| executor eval top-p / top-k | 原文未独立披露 | Table 7 的 .95 / 20 明确描述训练 executor |
| order seeds | 代码默认 0、1、2；原文未披露值 | 工程选择，可配置并保存有序 manifest |
| BM25 tokenizer and parameters | 代码 regex `[a-z0-9]+`、lowercase、k1=1.5、b=.75、positive Robertson IDF | 工程选择，原文只说明 BM25 |
| judge generation settings | 未披露 | 与 executor 一致属于工程选择 |
| standard deviation convention | 代码 sample std，ddof=1；原文未披露 | 工程选择，summary 公开定义 |

Appendix A 指定 vLLM 最大模型长度 40960；这是作者服务配置，不能简单转成任意 API 的可用上下文。API 超过上下文时应显式失败或使用已记录的限制模式，不能静默删掉 raw memory 后继续称为 faithful reproduction。

## Prompt 的语义规格

实现采用语义改写并公开工程补充，**不是逐字原 prompt**。原文 curator 见 JITMEM 第 14 页，executor 见第 15 页，distillation 见第 17 页；ALFWorld judge 原文来源是 SkillOS 第 22 页。运行 fingerprint 包含源代码和 game file 内容 hash，避免 prompt 或数据变化后混用旧 checkpoint；这些改写仍会影响与论文数字的可比性。

### Curator

角色是 ALFWorld household-task memory curator。输入区分 current question 与编号 retrieved memories，每条包含 past question 和完整 trajectory。要求输出一个简短可执行 briefing：指出最有用的 episodes，提炼寻物方式和动作顺序，最后给当前任务的具体操作建议。原 prompt 只要求简洁，没有硬性字数限制，也没有规定必须输出 JSON 或固定三级标题。当前代码使用中性的 retrieved experience 措辞，避免全量存储消融时把失败轨迹称为成功；两组共享相同system prompt，全量组为每条检索轨迹附上executor judge success/failure标签。代码另外提醒验证旧 episode 的地点/object numbers，并在无经验或不相关时提供一般操作指导；这些 caution/cold-start 指令是工程补充，原文没有明确要求。`task_adaptive=false` 只隐藏 user prompt 中的 current question，保留相同 curator system 并补充当前 query 未知的说明；retriever 仍使用真实 query。

建议语义改写：`Prepare a compact briefing for a household agent. Review the current objective and the supplied episodes. Explain which episodes help, recover effective object-search and action-order patterns, and turn those patterns into concrete advice for this objective. Keep the briefing short enough for the agent to reuse during execution.`

### Executor

输入明确包含 `task_description`、`retrieved_context`（此处实际是 payload）、`step_count`、`history_length`、`action_history`、`current_step`、`current_observation`、`admissible_actions`。要求先推理当前情境，参考相关 past experience，再从合法动作中选一个并包在 action tags。JITMEM 去掉了 SkillOS 强制 `<think>` 格式指令；不得因此擅自关掉 ALFWorld Qwen executor 的 thinking mode。

建议语义改写：`Act in the ALFWorld household environment to complete the objective below. Use the memory briefing when relevant. Review the recent observations and executed actions, then consider the current observation and available commands. Reason about the next step and return exactly one available command enclosed in <action> and </action>.`

### Self judge

输入包含任务与完整交互轨迹；不包含原生 success、reward、won 标量，也不包含 curator 对完成的断言。检查最终状态所有条件，并验证先变换再放置等前置过程；只认环境 observation 证据。局部完成、歧义、耗尽预算但未完成、循环和反复非法动作判为失败。返回 `success` 布尔值、`rationale` 简短说明、`evidence_step`（失败为 -1）。解析失败应显式记录 judge error，保守不入库。

真实 API pilot 暴露 Look 类的 judge 误判：3 个 native 成功都被要求额外 examine 或放置动作而拒绝。正式评测前，在 judge system prompt 补充公开的 ALFWorld Look 规则：持有目标、到达已激活目标灯的 receptacle 即满足，无须独立 examine/放置，也允许先开灯再拿物体。这是环境语义补充，不是原文 prompt；依据 [goal_library.py](https://github.com/alfworld/alfworld/blob/master/alfworld/gen/goal_library.py#L164-L191)。输入仍不包含 native success/reward 或 walkthrough。保留原 pilot 原始结果，另存诊断性重新判定，正式实验从空库重新开始。

建议语义改写：`Evaluate whether the observed interaction accomplished every requirement of the household objective. Base the verdict on simulator observations; planned actions and agent claims are insufficient. Required transformations must appear before the final placement or interaction. Treat uncertain, partial, stuck, or budget-exhausted attempts as failures. Return JSON with success, rationale, and evidence_step; use -1 for a failure.`

### Distillation ablation

`write-summary` 在 write time 接收已完成任务和轨迹，生成至多 3 个互不重复的 reusable memory items。每项含简短 title、一句 description 和 1–3 句 actionable content。强调寻物或动作顺序的可迁移规则，不保留具体 object/location 名作为泛化结论。持久 memory 删除 raw trace，只存这份 fixed summary；完整执行审计仍保留于 episode artifact。read-time curator 再消费这些 fixed items。它是论文 `w/o raw traj.` 的简化消融实现，**不能**宣称实现了完整 ReasoningBank 或 SkillOS，尤其不包括它们的全部 write/update/组织规则。

## 必要对照和指标

首轮跑相同模型与 matched task orders 的 `no-memory`、`jitmem`（prompted）以及直接注入 retrieved raw trajectories 的 `raw-memory`。后者是工程诊断对照，不是论文表格中的 named baseline。进一步跑 `jitmem` 配合 `task_adaptive=false`（curator 不见当前 query）、`jitmem` 配合 `store_policy="all"`（全部写入并展示 judge labels），以及 `write-summary`（write-time 固定摘要再 read-time curate）。`retrieval_k=0` 可以做无 retrieved context 的诊断，但不训练时不可称为论文 RL 消融的复现。

CLI 的 `evaluate --method` 接受的值只有 `jitmem`、`no-memory`、`raw-memory`、`write-summary`；task adaptivity、store policy、retrieval count 在 TOML `[experiment]` 中配置。baseline 对照必须显式采用同一 split、同一 task list、同一 seeds 和 executor 参数。`compare` 命令显示已完成 summary，不能代替逐任务 paired outcomes 的统计分析。

质量过滤与带标签全量存储的专用配对配置、Table9原文结果、分析入口见[存储消融协议](storage_ablation.md)。该消融固定 `method="jitmem"`、两组冷启动和完整任务集；全量组不会用原生标签修正自评误判。分析器拒绝未标注暖启动与不完整结果。本次只检验API prompted变体，不复现本地RL训练。

生成预算耗尽/过滤的处理是公开工程选择：API 返回文本和 `finish_reason`，不当作 transport failure 重试。executor 不执行截断输出，消耗一次 decision budget；curator 省略截断 payload，judge 拒绝截断 verdict，distiller 不存截断 summary。任务仍参与 SR，逐调用保存 incomplete 标志。API 省略 usage 时保存未知值 `null`，受影响的 token 汇总/均值也为未知，不以零代替；其他完整提供用量的角色仍可独立统计。

至少输出每 run SR、每 task 类型 SR、平均 decisions/环境 steps，并将 executor、curator、judge、distiller 的 input/output tokens 分开统计，避免把论文的 executor-only token 减少解释为全系统成本减少。实际费用仍需根据用户 API 的计费规则另行计算。记录 native success 与 judge success 的 2×2 计数和 judge parse errors，检验 memory gate。episode artifacts 保留 bank size、retrieved IDs、payload 和完整 calls，可据此分析 early/late batches 的 SR 与 cold start。模型故障和环境错误会中止实验并保留 error/checkpoint，不能将部分完成 run 当作完整 benchmark，也不能从 denominator 静默删除失败任务。

配置 loader 对未指定的 curator 字段继承 executor 设置，并默认改用 8192 output tokens。Qwen3 ALFWorld 使用者必须显式为 curator 设置 non-thinking，而为 executor 设置 thinking，避免继承 `extra_body` 后两角色模式相同；`.6/.95/20` 仅适用于论文 Qwen curator，GPT/Gemini curator 应使用 temperature1。API 是否接受这些 provider-specific 参数需要 `api-check` 及 endpoint 文档确认。模型名为空时可以检查环境，不能进行实际 API evaluation。

论文 Table 1 的 Qwen no-memory 47.9 是 SkillOS reported 数值；Appendix Table 6 的作者 own reproduction 为 thinking42.1、non-thinking34.5。不能把47.9当作本实现必须达到的直接 API validation threshold。优先比较本仓库相同配置下的 paired no-memory 和 JITMEM runs。

## 训练路径（当前不执行）

训练 bank 用 no-curator executor 跑训练集一次，按原生 ground-truth success 留下成功轨迹，再冻结 bank。为采样的 training task 检索 3 条，生成 8 个 payload，让冻结 executor 分别执行同一任务得到 binary native reward，curator 使用 group-centered advantage，不除以 group 标准差。训练 task 不能与测试轨迹混合。

Table 7 规定 Qwen3-8B non-thinking curator，100 RL steps，learning rate 1e-6，5-step warmup，prompt batch 与 mini-batch 都 32，group size 8，KL reward disabled、low-variance KL loss coefficient .001，clip .2/.2，token-mean aggregation，max prompt 32768，ALFWorld max response 8192，train sampling temperature 1.0。训练 executor 为 Qwen3-8B non-thinking，temperature1/top-p.95/top-k20，4096 max new tokens，30 env steps。当前 API pipeline 留出角色替换入口；接入 trained curator 后需要重新验证生成设置与 benchmark protocol。

## 尚待核实

作者的精确 task list/order seeds、数据版本、BM25 实现和 tokenization、judge inference 参数、空库处理、invalid-action 恢复行为、episode termination 方法、curator evaluation output limit、轨迹格式/长度截断规则均没有在 JITMEM 正文和 Appendix 给足细节。上述本仓库 choices 应公开，不能通过猜测填成作者设置。Appendix B 还将 validation 称为 subsampled test stream；未来进行训练/checkpoint selection 时需要核实是否存在 evaluation reuse，当前无训练范围不涉及这一流程。

并行执行采用 `multiprocessing` spawn 与独立进程内的环境/API client；各 batch 的 memory snapshot 不变，父进程按预定 task order 提交结果和记忆。`workers` 是公开工程参数，默认 1，完整对照使用 10；不会把完成先后顺序变成记忆写入顺序。
