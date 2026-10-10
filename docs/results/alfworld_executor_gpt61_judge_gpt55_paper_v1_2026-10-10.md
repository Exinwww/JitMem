# ALFWorld Executor模型对照（paper-v1，full storage，固定judge）

评测日期：2026-10-10，Asia/Shanghai。curator固定gpt-6.1-sol，judge固定gpt-5.5，仅将executor从gpt-5.5改为gpt-6.1-sol。Candidate减baseline的native SR差为 **4.76 ± 2.89 个百分点**，按三轮报告样本标准差；这是描述性结果，不预设更换模型改善性能。

结构化汇总见[alfworld_executor_gpt61_judge_gpt55_paper_v1_2026-10-10.json](alfworld_executor_gpt61_judge_gpt55_paper_v1_2026-10-10.json)。复用420条历史全量存储baseline，新增candidate420条，共比较840条记录，不是新增840条模型评测。两组full storage均展示judge标签，task-adaptive，每seed空库；valid_seen140任务、seeds0/1/2、batch10/workers10 spawn、BM25 top3、history3、最多30次环境交互。

有效配置仅executor.model与输出目录不同，curator与judge参数均保持一致；源码不完全相同。独立judge路由改变cli.py、config.py、evaluation.py、pipeline.py四份模块，分别冻结旧/新12份指纹。历史源码绑定cc289e2提交，候选源码以预检冻结的文件hash标识，不虚构候选commit。

兼容证明使用记录的API返回与环境反馈回放历史420条，分别验证旧默认、新默认、新显式同模型judge配置的messages/actions/storage/usage/batches；其0模型API、0原生环境步。该离线回放不是新增模型采样、不是原生环境重放，也不能证明全部未见状态的软件等价。

原论文judge通常复用executor，本轮将judge固定为独立gpt-5.5以控制变量，因此这是当前全量流水线的executor模型消融扩展。使用API、未训练curator，没有本地训练；distillation仅冻结资产，本次未调用。请求ID不独立验证权重；复用旧目录中gpt-6.1-sol被列出的历史证据，本轮新增探针和目录GET均0。

## Native SR 与 Table4 executor-only效率

| Arm / executor | SR mean ± sample std | 成功 / 420 | Executor input K / task | Executor output K / task | Executor turns / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / gpt-5.5 | 89.52 ± 1.65% | 376 / 420 | 8.068 ± 0.187 | 0.772 ± 0.033 | 10.56 ± 0.30 |
| candidate / gpt-6.1-sol | 94.29 ± 1.24% | 396 / 420 | 7.185 ± 0.111 | 0.611 ± 0.009 | 9.23 ± 0.19 |

K=1000，所有任务参与分母，累计每任务全部executor调用；curator/judge另列用量，不混入论文效率列。API缺失usage保留null，不补0；隐藏推理token及tokenizer口径由提供方决定。

| Seed | Baseline成功 / 140 | Candidate成功 / 140 | Δ pp | 两者成功 | 仅candidate | 仅baseline | 两者失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 124 | 133 | +6.43 | 123 | 10 | 1 | 6 |
| 1 | 128 | 130 | +1.43 | 126 | 4 | 2 | 8 |
| 2 | 124 | 133 | +6.43 | 124 | 9 | 0 | 7 |

同一140任务重复三轮，streaming记忆使轮内结果相关；420对记录不视作420个独立任务样本。Order seed不固定服务端采样，三轮std不是置信区间，不做pooled显著性检验，也不仅凭均值差宣称稳定优势。

## 分任务类与记忆诊断

| 任务类型 | 任务 / 轮 | Baseline SR mean ± std | Candidate SR mean ± std | Δ pp mean ± std |
| --- | ---: | ---: | ---: | ---: |
| look_at_obj_in_light | 13 | 94.87 ± 8.88% | 97.44 ± 4.44% | 2.56 ± 11.75 |
| pick_and_place_simple | 35 | 94.29 ± 2.86% | 99.05 ± 1.65% | 4.76 ± 1.65 |
| pick_clean_then_place_in_recep | 27 | 96.30 ± 0.00% | 98.77 ± 2.14% | 2.47 ± 2.14 |
| pick_cool_then_place_in_recep | 25 | 88.00 ± 4.00% | 88.00 ± 0.00% | 0.00 ± 4.00 |
| pick_heat_then_place_in_recep | 16 | 91.67 ± 9.55% | 100.00 ± 0.00% | 8.33 ± 9.55 |
| pick_two_obj_and_place | 24 | 72.22 ± 8.67% | 83.33 ± 7.22% | 11.11 ± 14.63 |

| Seed / arm | 最终bank | Native失败入库 | Judge失败入库 | 检索引用 | Native失败引用 | Judge失败引用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / baseline | 140 | 16 | 29 | 390 | 43 | 84 |
| 0 / candidate | 140 | 7 | 17 | 390 | 28 | 59 |
| 1 / baseline | 140 | 12 | 23 | 390 | 61 | 75 |
| 1 / candidate | 140 | 10 | 21 | 390 | 40 | 71 |
| 2 / baseline | 140 | 16 | 27 | 390 | 37 | 57 |
| 2 / candidate | 140 | 7 | 12 | 390 | 24 | 27 |

两组保存所有轨迹，judge标签不决定入库。固定judge仍可能对不同executor轨迹给出不同标签；executor改变会经轨迹、标签、检索和curator payload反馈影响后续任务，不能单独隔离这些中介作用。Native verifier决定SR，不用native标签修正judge。

## 已提交episode的API返回用量

| Arm / role | Calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| baseline / curator | 420 | 862,251 | 84,289 |
| baseline / executor | 4,434 | 3,388,547 | 324,115 |
| baseline / judge | 420 | 425,614 | 28,274 |
| candidate / curator | 420 | 809,469 | 83,366 |
| candidate / executor | 3,878 | 3,017,893 | 256,511 |
| candidate / judge | 420 | 405,126 | 33,565 |

Baseline用量来自历史420条记录，candidate用量来自本次新420条已提交记录。本轮探针0、目录GET0；旧探针/目录证据、重试与未提交请求的账单不混入上表，也不从已提交usage倒推完整账单。

## 审计范围与解释边界

本轮记录1次传输中断（URLError，4次尝试后退出）。当时seed1已提交90条，加其他完成seed合计230条；这些已提交记录的SHA256、大小及纳秒mtime未变，checkpoint和raw bank前缀保留。确认没有残留进程后，使用原配置续跑并跳过已提交记录；传输中断不计作native任务失败，重试与未提交请求的额外账单未知。完整独立结构审计覆盖840条（420历史+420新增）、消息/动作解析、BM25、judge标签、raw bank、冻结批次、checkpoint、指标与usage。全部437历史文件的SHA256、大小和纳秒mtime保持原样；数据集18,416文件前后完整指纹不变。

本次candidate首20条在原生环境重放通过，其余400条未完整环境重放；这20条新证据与历史full20、filtered20或更早40条证明独立。来源兼容bridge的420×3回放使用记录的响应/反馈，不能计作原生重放或新增模型benchmark。分析和审计均0模型API。

原文词语已对齐，但作者runtime空白、轨迹序列化、BM25、精确任务/顺序、环境版本和推理细节仍部分未公开；full标签位置/格式及本次独立judge路由是明示的实现选择。相对于原论文，不能保证只剩模型差异，也不能把所有论文性能差距归因于executor。详见[对齐说明](../paper_fidelity.md)。

完整请求、配置和轨迹保留于被忽略的outputs；公开文件只含白名单aggregate与指纹，不含endpoint、密钥变量名、个人绝对路径、原始消息或完整prompt。
