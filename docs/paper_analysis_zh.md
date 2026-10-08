# Just-in-Time Memory：中文阅读与实现分析

阅读对象：[Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents](https://arxiv.org/pdf/2609.27334)，arXiv:2609.27334v1，Salesforce AI Research，2026-09-23。本文完整阅读正文和 Appendix 后撰写；数字和设置的核查表见 [reproduction_spec.md](reproduction_spec.md)。文字与 prompts 采用解释性改写，不逐字转载。

## 1. 研究问题与重要性

研究问题是：跨任务 agent memory 应在什么时候决定一段经验中哪些信息有用？现有流程通常在任务完成时把轨迹压成固定 reflection、strategy 或 skill，等未来任务到来再检索。这要求模型在不知道未来用途时就决定丢掉什么。JITMEM 将决定延迟到未来任务已经明确之后；原始轨迹保留，使用时再解释它。

这对 ALFWorld 特别直观。一段完成冷却并放置物品的轨迹，可能帮助未来的温度变换任务，也可能帮助普通搬运任务。如果只写入一种摘要，另一类用途可能丢失。本文的核心 claim 是 read-time task conditioning 本身改善经验复用，训练 curator 又进一步改善其有效性。

## 2. 前人工作与不足

ReasoningBank 将经历变成可检索的 reasoning strategies；MemP 组织不同粒度的 procedural memory；SkillOS 用 GRPO 学会维护 skill repository。它们对经验抽象有价值，但抽象在 future query 到来前已经固定。本文认为信息损失不可逆，单一摘要也难以兼顾不同用途。

与本文最接近的训练路线是 [SkillOS](https://arxiv.org/pdf/2605.06614)。其 write-time curator 必须利用后续相关任务的效果评价较早的存储选择，因此通过相关 task grouping 形成训练信号。本文改用同一任务直接评价刚产生的 payload。相关并发工作 MemHarness 也在 read time 重构记忆；JITMEM 的区别是独立 curator 与 frozen executor，使训练后的 memory module 可以更换执行模型。这并不意味着 read-time processing 或 trajectory retrieval 首次出现，论文 Related Work 已列出 Synapse、SkillTTA 等更早路线。

## 3. 重建作者的思考路径

下面是基于已有失败模式的推断，并非作者公开的研究记录。先考虑只检索 raw demonstrations：信息完整，但 executor 要承担比较、归纳和执行三种工作，context 会变长。再考虑提前压缩：执行上下文短，但同一经历的多种用途只能预先猜测。由此可以把职责分开，保留 episodes，在 current task 明确时让单独模型只负责生成使用指南。

这还使学习目标变得局部。既然执行器立即消费这份指南，当前任务的成功就是指南是否有效的证据，不必等多个后续任务才能知道之前的摘要是否值得保存。这个推理也解释了为什么不需要任务分组和复杂 storage-operation rewards。

## 4. 核心 intuition

过去经历的意义取决于现在要做什么。先保存事实和操作过程，等问题出现再从中提取相关线索，能够让同一个 episode 服务不同任务。让生成指南的模型与执行模型分开，指南是否有效就可以直接用当前任务是否完成来评价。

## 5. 方法与完整 pipeline

以当前任务需要清洁并放置一个盘子为例，bank 可能含有一个盘子冷却放置 episode、一个番茄清洁放置 episode、一个皂条清洁 episode。检索只比较它们的 task descriptions，而不把轨迹长度当相关性。curator 可以组合第一条的物品操作线索和后两条的清洁顺序，为当前任务形成具体 briefing；它不需要把任何完整旧轨迹直接发给 executor。

完整路径如下，具体语义见论文 §3：

1. bank 存完整 task、observations 和实际 actions，不在写入时生成固定摘要。
2. BM25 用 current task 检索 top 3 descriptions 对应的 trajectories。
3. curator 同时读 current task 与 retrieved trajectories，输出一个 concise natural-language payload。
4. frozen executor 每步读取 payload、当前 observation、admissible actions 与短 history，发出下一条合法动作。
5. episode 结束后，同一个 executor 模型作为 judge 判断轨迹是否完成任务；通过 gate 的 raw trajectory 在 batch boundary 入库，payload 随任务结束丢弃。

当前实现不做本地训练，curator 与 executor 分别调用模型 API，judge 复用同一个 executor client。因此它验证的是 prompted read-time curation pipeline；若后续提供 trained-curator API，可以替换 curator 而复用 executor/environment/evaluation。CLI 的 `jitmem` 是这条流程，`no-memory` 是执行器对照，`raw-memory` 直接注入检索轨迹，`write-summary` 是先固定摘要再 read-time curate 的简化信息损失消融；后两者不能等同于完整 ReasoningBank、MemP 或 SkillOS baseline。

## 6. 数学含义与训练

设当前任务为 $x_t$、bank 为 $M_t$、检索轨迹为 $\hat\xi_t$。retriever 产生 $\hat\xi_t=R(x_t,M_t)$；curator 产生 $p_t=\pi_\phi(x_t,\hat\xi_t)$；冻结 executor 产生 trajectory 与 reward：$(\xi_t,r_t)=\pi_L(x_t,p_t)$。只在 judge 接受时将 $\xi_t$ 加入 bank。

训练先用 base executor 跑 training split，按原生 success 建立固定 training bank，防止 bank 随训练随机改变，让 payload 质量成为主要变量。每个任务生成 $G=8$ 个候选 payload，分别用同一 frozen executor 执行。组内 advantage 为 $\hat A_i=r_i-\frac{1}{G}\sum_jr_j$，不除以标准差：超过组平均的 payload 得正信号，低于平均的得负信号。

论文给出简化目标 $L=-\frac1G\sum_i\hat A_i\log\pi_\phi(p_i\mid x,\hat\xi)$。实际训练还使用 clipping 与 KL loss，参数在 Appendix Table 7。该公式说明 reward 与当前 payload 的关系，并不证明 credit assignment 全部被消除：executor 的采样随机性、检索质量、环境难度依然影响 reward。它解决的是 write-time 决策和未来使用之间的时间间隔。

## 7. 实验设计与结论

第一组实验问 read-time curation 是否只依赖更强的 curator。设计是在相同 curator backbone 下比较 prompted read-time 与 prompted write-time methods。ALFWorld 的 Qwen3-8B 组合中，JITMEM-base 优于 ReasoningBank 和 SkillOS-base，说明无需 RL 也能出现收益。

第二组问同样训练 curator 后是否仍有优势。作者使用 Qwen3-8B curator 训练并和 RL SkillOS 比较；ALFWorld 主表显示 JITMEM 超过最强 baseline。第三组将同一个训练 curator 转移到更强 executor，结果表明 curator 的收益不只绑定训练时的执行模型。上述数字属于作者指定模型与训练设置；本仓库使用用户配置的 API 进行独立 prompted 对照，实际记录见 [validation.md](validation.md)，不能直接当作上述原模型/训练结果的复跑。

消融分别隐藏 curator 的 current task、存入所有带 judge labels 的轨迹、把 raw traces 换成 write-time 固定摘要，以及训练后移除 retrieved traces。它们验证 task conditioning、quality gate、保留原始细节和经验来源各自是否有贡献。另有 bank refresh 与 warm start，收益有限。评估采用 batched streaming，task order 会影响 bank，因此报告多次乱序均值。executor token 与 steps 的下降并不直接等价于全系统 API 费用下降，因为增加的 curator/judge calls 需要另外计算。

## 8. Take-aways

五句话：保存经验和解释经验可以是两个独立阶段。未来任务提供了解释旧轨迹的用途。完整 episodes 允许多种任务条件下的重构。当前任务的成功让 curator training 获得直接信号。prompted 与 trained 版本都应通过成对评测验证，而不能混用它们的结果。

三句话：不要过早丢弃未来可能有用的交互细节。使用时再生成相关指南能够减轻 executor 的比较和归纳负担。学习改善的是这次使用的质量，而不是一个用途未知的永久摘要。

一句话：让当前任务决定旧经验该怎样被解释。

## 9. 最脆弱的假设

最关键假设是：检索到的过去成功轨迹中存在能迁移到当前任务的可用经验，而且 curator 能从表面相似中找出真正有效的因果操作规律。如果 BM25 只找到了词汇相似、机制不同的 episodes，后续 read-time synthesis 可能把旧任务的成功条件错误搬到新任务。

论文的 task adaptivity 与 raw trace 消融支持“当前任务和保留细节有帮助”，但没有系统操纵表面相似度与因果相关性。ALFWorld 的模型、环境规则和任务族较稳定，不能据此保证这种迁移在环境规则改变时仍成立。这是基于实验范围的判断，而非作者已证实的失败案例。

## 10. 一周最小复现实验

先建立能在真实 ALFWorld TextWorld environment reset、step 和判定 native success 的适配器，然后选完整 `valid_seen` 任务集。固定 API 模型、history3、decision budget30、batch10，使用相同 order seeds 0、1、2 跑 `no-memory` 和 prompted `jitmem`。这三个种子是复现选择，论文未披露种子值；非法输出消耗 decision budget 也是公开工程选择。测 per-run SR、paired task outcomes、decisions 与实际环境 steps、所有角色 token usage、bank growth 与 judge/native confusion matrix。

这个实验验证最核心且无需训练的主张：task-conditioned read-time briefing 是否能在相同执行条件下提高成功率。支持证据应是 matched orders 下稳定的正差和可解释的 trajectory changes；一次随机顺序正差不足以验证。若无提升，应先检查 environment/action parsing 与质量 gate，再报告结果，不以调整 seed 或过滤失败任务追求论文数字。

## 11. 最强反例设计

构造表面词汇匹配、操作规则冲突的 episodes：旧环境某种物品变换成功，新环境的设备效果或操作前提不同，同时 query 文本保持相似。控制 retrieved traces 数量与长度，比较 no-memory、raw retrieval、task-conditioned curation，检查 curator 是否把旧规则表述成确定事实并诱导失败。

更接近现有 benchmark 的测试可将 memory bank 分成真实相关、词汇相关但 task-type 不兼容、随机三组，并保留同一任务顺序和 native verifier。若 read-time curation 的收益来自一般性提醒而不是有效经验综合，那么随机 bank 也可能获得相近结果；若不兼容 bank 导致显著负迁移，则暴露了 relevance 与可迁移性之间的差距。两种结果都是对原 claim 的实质检验，不能通过为某个模型特调 prompt 隐藏。

## 12. Follow-up research idea

一个更有价值的新 framing 是把 memory 定义为“带适用条件的可检验经验”，目标同时包含收益和负迁移风险。检索返回的不只是一个相似旧任务，而是一个操作假设及其成立的环境条件；系统遇到不确定条件时，可以先用低成本 observation/action 验证，再决定是否使用该经验。这借鉴 causal transportability 与 decision-theoretic value of information，改变的是复用目标，而不是给现有 curator 再加一个摘要模块。

第一个实验是上述规则变化环境：固定 API executor，比较相似度驱动复用和条件验证驱动复用，测 task success、错误复用率、验证开销以及何时应放弃 memory。这里提出的是研究设想，论文尚未实现；其新颖性还需要针对 causal memory、conditional skill applicability 和 safe transfer 文献作进一步核查。以重要问题、明确反例和可判别实验为标准，当前优先级高于扩展更多数据集或仅更换 embedding retriever。

## 来源与边界

方法、实验和训练参数来自 JITMEM 全文；ALFWorld judge semantic origin 与 `valid_seen` 定位来自 SkillOS Appendix。代码使用改写 prompt，并加入旧地点/object numbers 应核查、空库时可提供一般指导的工程指令；BM25 tokenization/positive IDF、seeds0/1/2、curator output8192 和非法动作预算语义也不是作者明确设置。完整清单见复现规格。论文数字不是本仓库测得数字；真实 environment smoke test 或 offline scripted mock 也不是模型质量评估。论文 main-table no-memory 数字部分直接继承 SkillOS，而 Appendix 的作者 own reproduction 更低，因此本仓库应以自己同配置的成对运行进行判断。
