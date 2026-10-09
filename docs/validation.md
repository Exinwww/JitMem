# 实现验证记录

验证日期：2026-10-08（Asia/Shanghai）。本记录区分软件正确性、真实模拟器可用性和模型能力评测。

从下方早期验证到“存储消融验证（2026-10-09）”的模型协议记录均属于历史 `legacy-paraphrase`，保留原数字和实现行为。当前正式 `paper-v1` 的验证见后文“原文协议切换验证”和“仅curator模型切换验证”；旧截断处理、严格动作拒绝和Look补充不适用于新协议。

版本管理中的 [评测结果报告](results/alfworld_valid_seen_2026-10-08.md) 与 [结构化结果](results/alfworld_valid_seen_2026-10-08.json) 可在新克隆中查看。本页指向 `outputs/` 的链接对应本地生成的审计和原始记录，目录不提交到 Git。

评测后进行了客户端凭证保护加固；[密钥安全审计](security_audit.md)记录检查范围与修复。下述模型结果和原实验哈希对应初始提交`3ef61bb`，安全修复后不重写这些历史记录。

## 软件正确性

完整测试：`93 passed, 89 subtests passed`。Ruff 检查、21 个文件的格式检查及 Shell syntax check 均通过。

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
bash -n scripts/setup_alfworld.sh
```

测试覆盖 OpenAI-compatible HTTP 请求、环境变量优先级、角色模型与凭证继承、密钥不进入配置、重试和鉴权错误、模型生成截断处理、空响应错误、缺失用量标为未知、BM25 仅索引任务描述、完整轨迹持久化、严格动作解析、非法输出预算、judge 与原生 success 独立、batch snapshot、各轮冷启动、checkpoint 续跑和协议变化拒绝续跑。

HTTP 集成测试使用 `127.0.0.1` 的临时 scripted server，经过真实 HTTP 序列化和 ChatClient，不访问外部模型 API。完整测试已在允许本机端口监听的执行权限下通过；普通受限制的沙箱可能拒绝 bind。

## ALFWorld 原生交互

安装在项目 `.venv` 中：Python 3.11.16、ALFWorld 0.4.2、TextWorld 1.6.2、fast-downward-textworld 20.6.4，macOS arm64。完整依赖快照为 [requirements-lock.txt](../requirements-lock.txt)。

`doctor` 确认 train 3553、valid_train 200、valid_seen 140、valid_unseen 134 个官方筛选后的文本游戏。`env-smoke` 对真实 valid_seen 游戏成功执行 reset 和 `look`，返回正常 observation 和 admissible commands。

```bash
.venv/bin/python scripts/verify_native_environment.py
```

该独立验证使用官方 walkthrough，seen/unseen 各六类任务，共 12 个游戏。12/12 触发 native `won`，所有 walkthrough actions 合法。输出 [native_environment_smoke.json](../outputs/native_environment_smoke.json) 标注 `expert_only=true`、`is_model_benchmark=false`，含轨迹、版本、game hashes 和数据来源。

这只证明 simulator 和 adapter 可交互、可判定成功。expert actions 不被 agent pipeline、记忆库或模型评测入口读取。未进行模型训练。

完整默认评估 split 也已验证：

```bash
.venv/bin/python scripts/verify_native_environment.py --all --splits valid_seen \
  --output outputs/native_valid_seen_full.json
```

140/140 个游戏全部触发 native `won`，无非法参考动作；报告 task IDs 与官方筛选后的完整 `valid_seen` manifest 逐项相同。[完整环境检查报告](../outputs/native_valid_seen_full.json) 保持 `expert_only=true`、`is_model_benchmark=false`，这仍不是模型成功率。

10 进程 spawn 并发也已验证：[parallel_environment_smoke.json](../outputs/parallel_environment_smoke.json)。10/10 独立游戏触发 native won，10 个 worker PID，最大 10 个 episode 重叠；外部数据共 18,416 个文件的完整内容、大小、修改时间 fingerprint 在验证前后相同。该报告仍为 expert-only 环境测试，不计入模型评测。

## 完整流程与续跑

```bash
.venv/bin/python -m jitmem smoke --resume
```

独立 fixture 中的 3 个 episode 各用 5 个动作完成。batch size 2 时 bank size before 依次为 0、0、2，第三个 episode 检索到前一批经验。memory 不含 payload；审计 artifact 保留 payload。相同协议续跑返回一致结果，无新调用。[mock_smoke/summary.json](../outputs/mock_smoke/summary.json) 明确 `benchmark=false`，不能当作 ALFWorld 模型分数。

故障注入测试在第二批中模拟 API 故障，保留完整第一批 checkpoint；恢复后重跑未提交批次，不重复计入已提交任务。配置、源代码、game bytes、warm-start bank 内容或关键 runtime 版本变化会拒绝沿用旧实验。未提交批次重试可能消耗额外 API 用量，失败调用不被混入完整 benchmark 的 token 统计。

模型输出达到生成 token 上限属于模型行为，不能当作网络故障中止并反复重试同一任务。此处记录历史legacy协议：executor截断输出消耗一次决策预算且不执行任何部分动作；curator截断指导被省略，judge截断结果被拒绝，distiller截断摘要不入库。所有情况记录 `finish_reason` 和 `generation_failures`，任务仍计入评测。缺失 API usage 时，相应用量及受影响汇总为 `null`，`usage.complete` / summary `usage_complete` 为 false，不能据此宣称 token 减少或零成本。新paper-v1的返回文本消费规则见末节。

## 完整模型评测结果

用户环境变量中的 API 已连通，配置模型名为 `gpt-5.5`，curator/executor 相同；默认 OpenAI-compatible chat completions。初次 GROUP_NOT_ALLOWED 后，按用户要求重试已成功。

原始 5-task pilot 位于 [api_pilot](../outputs/api_pilot/summary.json)：native 3/5 成功，judge 对这 3 条 Look 成功轨迹均误判，导致 bank 为 0。原始输出保留。基于公开 ALFWorld 语义修正 judge prompt 后，独立重新判定的 5 条轨迹均匹配 native label：[api_pilot_judge_audit.json](../outputs/api_pilot_judge_audit.json)。native 标量未提供给 judge；这份诊断不能作为重新运行的 agent 分数或正式对照结果。

更新后的 [并行 pilot](../outputs/api_parallel_pilot/summary.json) 完成：native 3/5 成功，0 次非法动作、0 次生成截断；memory size before 为 0、0、2、2、4，第三个任务检索到前批两条经验。judge 为 3 TP、1 FP、1 TN：有一条失败 CD 轨迹被 judge 错误忽略“回到灯的位置”条件而入库。该模型误判保留记录并在正式报告中量化，未用 native label 替换 gate、未再次调 prompt 来筛选结果。pilot 仍是单类小样本功能检查，不能解释为收益。

正式对照已全部完成：完整 `valid_seen` 140-task / batch10 / seeds 0,1,2 / workers10，每组 420 次真实 episode，共 840 次。两组冻结相同的 11 个实现源码哈希、executor 配置及配对 task order；每轮从空库开始，无 warm start。`scripts/analyze_results.py` 已校验完整任务集、summary/results 一致性、种子确定的顺序、每批冻结记忆与整批后的有序增长，并生成 [完整报告](../outputs/comparison/comparison.md) 和 [机器可读比较](../outputs/comparison/comparison.json)。

| Seed | no-memory 成功数 / 140 | prompted JITMEM 成功数 / 140 | SR 差（百分点） |
| --- | ---: | ---: | ---: |
| 0 | 111 | 121 | +7.14 |
| 1 | 107 | 121 | +10.00 |
| 2 | 109 | 122 | +9.29 |
| SR mean ± sample std | 77.86 ± 1.43% | 86.67 ± 0.41% | +8.81 ± 1.49 |

均值和样本标准差基于三轮，ddof=1。同一 140 个任务重复评测，不能将 420 个 episode 当作独立任务样本。模型名为 API 配置中的 `gpt-5.5`，prompt 使用公开语义改写和上述 Look 补充；这是用户提供模型上的训练自由 prompted 变体，不是作者原模型或 RL-trained 数字的等价复跑。

平均决策次数由 14.85 降至 11.23（减少 24.35%），平均实际环境步数由 14.79 降至 11.21。所有角色的 input+output tokens 从 3,021,642 增至 4,623,861（增加 53.02%），usage 全部完整；角色用量见完整报告，未推算 API 费用。非法动作计数分别为 26 和 9；两组均无生成截断，完整运行没有基础设施错误。

JITMEM 三轮最终 bank 各 118 条，judge/native 的重复 episode 计数为 353 TP、1 FP、11 FN、55 TN；其中 1 条原生失败轨迹被写入 bank，保留原始 gate 行为。原生结果决定 SR，judge 只决定记忆写入。下述数据语义例外说明 native 与可观察轨迹的分歧不能全部归因于 judge 推理错误。

完整评估结束后再次只读检查全部 18,416 个外部数据文件，内容、大小和修改时间的 SHA256 fingerprint 与运行前一致：[final_dataset_integrity.json](../outputs/final_dataset_integrity.json)。

独立最终审计覆盖全部 840 条 episode 的任务身份、配对顺序、记录/summary/checkpoint 一致性、140 个游戏哈希、11 个源码哈希、批内冻结 bank 与整批后提交、无 warm start、请求 messages 和角色用量，并独立重算 comparison 数字；[final_results_audit.json](../outputs/final_results_audit.json) 为 `passed=true`、`errors=[]`。此次审计不调用 API、不重放环境、不修改实现或结果。

独立首批审计重放两方法 seed0 的前20个任务，共40条真实轨迹，初始与每步 observation、合法动作、native won/reward/done 和描述均与记录一致；逐调用 messages 可由公开 observation/action 与前批 raw bank 重新构建，无 expert/PDDL/native label 输入。[审计报告](../outputs/first_batch_replay_audit.json) 不调用模型 API。

其中 `trial_T20190908_184242_348366`（cool mug to coffeemachine）环境在冷却后直接 won，judge 因无最后放置拒绝。审计发现原始 game 初始 PDDL 同时将该 mug 关联到 countertop 与 coffeemachine；从 countertop 拿取未删除另一个关系，使冷却动作满足原生目标。这属于继承数据/环境行为。保留 native score 和独立 judge gate，解释结果时不把所有分歧都归因于 judge 推理错误。

## 存储消融验证（2026-10-09，legacy-paraphrase）

新增质量过滤与带标签全量存储的成对配置、协议测试和完整结果分析器，见[消融协议](storage_ablation.md)。当前完整软件验证为 **150 passed、108 subtests passed**，Ruff与24个Python文件的格式检查通过；11个pipeline源文件未改变。新测试覆盖curator标签差分、executor/judge消息一致、失败及非法/截断决策的完整保存、跨批标签读取、确定性写入、配对配置、完整三轮/六类覆盖、checkpoint/bank/raw轨迹一致性、基础设施错误拒绝，以及逐调用usage与汇总一致、未知用量保留null。

独立重放两组seed0前两批，共40条真实交互轨迹，全部通过：初始/每步observation、admissible actions、实际执行决策、最终native reward/won/done、BM25与三角色messages均与记录一致。全量组第二批curator实际读取13个failure-label引用，标签全部来自对应executor judge；过滤组无显式标签。审计不调用模型API、不修改数据或原始结果。逐步native标量来自重放，原始日志只有最终标量可独立比对；此审计不证明judge语义判断正确或后续批次模型能力。

两组同一cool-mug数据语义例外在重放中再次出现：native成功而judge失败，过滤组不保存，全量组保留并显示failure。保持原数据与gate，不用native标签改写judge。独立报告位于本机 `outputs/storage_ablation_first_batch_audit.json`。

正式评测两组均重新运行并全部完成，使用安全加固后版本的相同源码哈希，140任务×seeds0/1/2，每个seed空库。一次执行中断时，两组各已完整提交40个任务；检查没有残留评测进程后，从相同配置checkpoint续跑。已提交记录不重跑、不重写，未提交请求可能产生额外调用成本；报告用量仅覆盖已提交记录，不能倒推完整账单。

| Seed | 过滤成功 / 140 | 全量成功 / 140 | 过滤减全量（百分点） |
| --- | ---: | ---: | ---: |
| 0 | 120 | 122 | −1.43 |
| 1 | 117 | 125 | −5.71 |
| 2 | 123 | 125 | −1.43 |
| Mean ± sample std | 85.71 ± 2.14% | 88.57 ± 1.24% | −2.86 ± 2.47 |

独立最终审计 `outputs/storage_ablation_final_audit.json` 为 `passed=true`、`errors=[]`、`complete_840_episode_audit=true`、`comparison_validated=true`。覆盖全部840条episode、84批、11,112次已记录模型调用，重建全部三角色messages，独立核对BM25排名、judge gate、原始轨迹、bank/checkpoint和汇总指标；11个源码文件和140个游戏hash均一致。审计不调用API；原生环境重放范围仍为前述40条。

再次只读核对外部18,416个文件，其相对路径、大小、修改时间与完整内容的fingerprint同前：`ea156e972ba076f73968c690ae732c5dc7a3978f3c6a39270cc5aa1abadad278`。没有修改数据或用原生标签替换judge。

本次未复现论文“过滤更优”的方向；模型与提示词差异及统计边界见[存储消融结果](results/alfworld_storage_ablation_2026-10-09.md)。版本管理只保存经过字段白名单导出的[结构化汇总](results/alfworld_storage_ablation_2026-10-09.json)，完整比较、请求与审计保留在本机被忽略的 `outputs/`。

## 原文协议切换验证（2026-10-09，paper-v1）

上面的840条存储消融属于legacy-paraphrase协议。按用户要求，新正式配置使用原文资产，差异与未披露细节见[对齐说明](paper_fidelity.md)，旧结果和checkpoint保留，不混入新运行。

完整软件验证为 **226 passed、108 subtests passed**；Ruff检查、29个Python文件格式检查和diff检查通过。新增测试覆盖固定源包/成员/模板的完整性链、缺失或篡改不回退、不安全解包拒绝、离线准备不触发下载、原文模板渲染、原生动作反馈、bounded输出文本消费、原文judge与native分离、模板fingerprint和成对审计的消息/输出一致性。没有模型API调用。

五份原文模板已由固定版本源包成功提取并核对SHA256；模型运行只加载本地资产。judge源图框没有原始runtime字符串，换行规范化与哈希明确记录，不宣称字节完全等同作者未发布代码。

真实ALFWorld脚本检查执行3次命令，包括一个非admissible命令和一个缺action标签但可解析为look的返回；均实际交给环境。决策与原生交互均为3，没有人工Invalid decision反馈，模型动作列表排除help。`outputs/paper_native_protocol_check.json`标为 `is_model_benchmark=false`、`scripted_commands=true`、`model_api_calls=0`，不是模型能力分数，也没有使用expert/walkthrough。

新成对模型评测已在 `outputs/storage_filtered_paper_v1/` 与 `outputs/storage_all_paper_v1/` 完成，共140任务×3轮×2组、840条真实episode、84批、10,867次已记录模型调用。两组冻结相同的12个核心源码文件、5份模板、runtime、140个游戏和配对task order；每轮空库，没有warm start。curator/executor配置模型名均为`gpt-5.5`，judge复用executor。

| Seed | 过滤成功 / 140 | 全量成功 / 140 | 过滤减全量（百分点） |
| --- | ---: | ---: | ---: |
| 0 | 120 | 126 | −4.29 |
| 1 | 125 | 122 | +2.14 |
| 2 | 127 | 123 | +2.86 |
| Mean ± sample std | 88.57 ± 2.58% | 88.33 ± 1.49% | +0.24 ± 3.93 |

三轮合计372与371次成功，差异方向随seed翻转，不能支持稳定的过滤优势。主指标为native SR；效率仅计executor，每任务input K为8.84±0.43与8.93±0.22、output K为0.77±0.05与0.78±0.01、交互次数为10.90±0.70与10.98±0.30，全部任务参与分母；全角色成本单列诊断，不混入Table4口径。

运行中两组遇到一次HTTP429中断，已提交记录分别为90与110条。确认没有残留评测进程后，保持各组workers10、同一配置和源码，先后执行`--resume`完成全部三轮。独立核对中断前200条记录的内容SHA256、大小和纳秒修改时间，均未改变；已提交任务没有重跑或重写。用量只覆盖已提交记录，未提交请求与重试可能额外计费，不能由报告倒推完整账单。

最终独立审计 `outputs/paper_storage_ablation_final_audit.json` 为`passed=true`、`errors=[]`、`complete_840_episode_audit=true`、`comparison_validated=true`。重新构建全部840条episode的原文messages、动作解析、描述BM25排序、judge gate、完整raw轨迹、批内快照与bank/checkpoint，独立重算主指标和分角色用量；核对源包与模板溯源、12个源码指纹和中断记录保全。审计不调用模型API，也不修改原始实验。

原生环境重放范围是两组seed0前20任务，共40条。初始与每步observation、admissible actions、实际提交命令以及最终native reward/won/done均一致；证据文件绑定本次新记录的hash。该检查不调用模型，不代表全部840条均经过环境重放。

最终审计再次只读核对外部全部18,416个文件的内容、大小和修改时间，fingerprint仍为`ea156e972ba076f73968c690ae732c5dc7a3978f3c6a39270cc5aa1abadad278`。没有读取expert/walkthrough来指导模型，也没有用native标签纠正judge。完整原始证据保留在本机被忽略的`outputs/`；公开[结果报告](results/alfworld_storage_ablation_paper_v1_2026-10-09.md)和[结构化汇总](results/alfworld_storage_ablation_paper_v1_2026-10-09.json)只包含白名单字段。

## 仅curator模型切换验证（2026-10-09，paper-v1）

质量过滤下仅将curator从`gpt-5.5`切换为`gpt-6.1-sol`；executor和judge仍为`gpt-5.5`。最终解析配置只有`curator.model`与`experiment.output_dir`不同，12个核心源码文件、5份原文模板、runtime、140个游戏、任务和每seed顺序保持相同，未训练或调用蒸馏。固定协议与命令见[模型对照协议](curator_model_comparison.md)。

新评测420条真实episode，与既有质量过滤基线420条记录配对，共840条；没有重跑基线。新curator各seed成功126/125/127，基线120/125/127；native SR **90.00±0.71% vs 88.57±2.58%**，新减旧为**+1.43±2.47个百分点**。净增6次成功均在seed0，另两轮成功数相同，不宣称稳定或显著优势。

新增分析器的21项测试通过，包含成对协议、完整记录、差值方向、未知usage以及输出路径保护；相关回归合计112项通过。Ruff与新增Python文件格式检查通过。此前完整226项测试属于上节的实现验证，不冒充本次全量重跑。分析器只读取记录，不调用API或环境。

`outputs/curator_gpt61_final_audit.json`为`passed=true`、`errors=[]`、`comparison_validated=true`，完整新420与配对840两个覆盖标志均为true。独立重建全部消息、动作解析、BM25、judge gate、raw bank、checkpoint及汇总，独立重算各seed指标和角色usage，绑定当前分析文件hash。审计覆盖10,706次已记录模型调用，其中5,417次属于历史基线，5,289次属于本次新评测；新组executor调用/原生交互4,449次。分析和审计均0模型API调用。

仅新组seed0前20条在原生环境完整重放，初始与每步observation、admissible actions、解析命令以及最终native reward/won/done均一致；证据绑定新记录hash。其余400条新记录没有完整环境重放；未将旧协议的重放证明替代本次证据。

新组seed0提交10条后遭遇一次transport URLError，4次尝试后退出。确认无残留进程后以原配置和workers10续跑；最终核对中断前10条的SHA256、大小和纳秒mtime完全不变，没有重写或重跑已提交记录。用量仅含已提交episode返回的usage；另有一次非benchmark连接探针与一次只读模型目录核对，不计入评测。未提交请求和重试账单未知。目录列出`gpt-6.1-sol`仅证明ID声明，不独立验证权重。

最终只读扫描确认基线全部437文件（420条episode）的内容SHA256、大小和纳秒mtime与前置快照相同；外部18,416数据文件的完整fingerprint仍为`ea156e972ba076f73968c690ae732c5dc7a3978f3c6a39270cc5aa1abadad278`。公开[结果报告](results/alfworld_curator_gpt61_paper_v1_2026-10-09.md)与[结构化汇总](results/alfworld_curator_gpt61_paper_v1_2026-10-09.json)经白名单导出，完整证据留在被忽略的`outputs/`。

## gpt-6.1-sol curator的全量存储验证（2026-10-09，paper-v1）

在上节质量过滤组的基础上，单独完成全量组140任务×seeds0/1/2，共420条新模型episode；curator仍为`gpt-6.1-sol`，executor/judge仍为`gpt-5.5`。配置仅改变`experiment.store_policy`与`experiment.output_dir`，12个核心源码、5份原文模板、runtime、数据和任务顺序一致；没有训练、蒸馏或warm start。旧过滤组420条直接配对，未重新运行或评分。

| Seed | 过滤成功 / 140 | 全量成功 / 140 | 过滤减全量（百分点） |
| --- | ---: | ---: | ---: |
| 0 | 126 | 124 | +1.43 |
| 1 | 125 | 128 | −2.14 |
| 2 | 127 | 124 | +2.14 |
| Mean ± sample std | 90.00 ± 0.71% | 89.52 ± 1.65% | +0.48 ± 2.30 |

过滤净多2次成功，逐轮方向翻转，不宣称稳定或显著优势。沿用既有存储分析器，无需修改pipeline或统计代码；executor-only input K/task为8.006±0.101与8.068±0.187，output K/task为0.766±0.004与0.772±0.033，交互/task为10.59±0.17与10.56±0.30。

严格独立审计 `outputs/storage_gpt61_final_audit.json` 为`passed=true`、`errors=[]`、`comparison_validated=true`，新full420及完整配对840覆盖标志均为true。核对6个run、84冻结batch、全部messages/动作解析/标签/BM25/gate/raw bank/结果文件/checkpoint/summary/usage，绑定完整分析SHA256并独立重算数字；共10,563次已记录模型调用，5,289次为历史过滤组，5,274次为本次新全量组（其中4,434次executor原生交互）。各seed全量bank均140；curator实际请求共216次judge-failure标签引用，标签全部来自对应judge，未用native标签纠正gate。

本次新full组seed0前20条独立原生环境重放通过，逐步observation、admissible actions、实际命令和最终native won/reward/done与记录一致；证据绑定这20条新记录的hash。其余400条新full记录没有完整环境重放，未将旧filtered20或其他协议40条重放作为本次证明。

本轮运行完整退出，无基础设施中断/续跑，未新增连接探针或模型目录GET；历史目录声明只作请求模型ID的证据，不验证服务端权重。用量来自已提交记录，不倒推内部重试或完整账单。数据前后完整扫描再次确认18,416文件指纹仍为`ea156e972ba076f73968c690ae732c5dc7a3978f3c6a39270cc5aa1abadad278`，过滤baseline全部437文件的SHA256、大小和纳秒mtime与新前置快照完全一致。

新审计工具的11项负向守卫检查、新公开导出工具的38项局部自检及Ruff/编译检查通过；这些是本地工具验证，0模型API。此前pipeline测试记录仍见前节，本轮未改核心代码。公开[结果报告](results/alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.md)与[结构化汇总](results/alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.json)只含白名单字段，完整请求和证据继续留在被忽略的`outputs/`。
