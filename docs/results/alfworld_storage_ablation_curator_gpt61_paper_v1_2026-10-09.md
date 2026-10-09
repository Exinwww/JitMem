# ALFWorld 存储消融（paper-v1，gpt-6.1-sol curator）

评测日期：2026-10-09，Asia/Shanghai。两组curator固定为gpt-6.1-sol，executor/judge固定为gpt-5.5。Quality-filtered减label-annotated full storage的native SR差为 **0.48 ± 2.30 个百分点**，按三轮计算样本标准差。该结果描述这套API与streaming协议下的存储策略对照，不预设过滤优于全量存储。

结构化汇总见[alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.json](alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.json)。复用420条历史quality-filtered episode，新增full-storage420条，共比较840条记录；本次没有重新评测过滤组，也不是新增840条模型评测。

两组均为原文paper-v1模板、task-adaptive、每轮空库、valid_seen140任务、seeds0/1/2、batch10/workers10 spawn、history3、最多30次环境交互、任务描述BM25 top3。除了存储策略与输出目录，其余配置、角色模型、源码、资产、runtime、任务和顺序经过匹配验证。Native verifier决定SR，同一个固定executor模型judge决定过滤入库；full组保存全部轨迹，并向curator展示executor judge的success/failure标签。

使用模型API，没有本地训练。Distillation模板只冻结为可复用资产，本task-adaptive消融未调用蒸馏，不代表原文RL-trained结果。请求模型名不代表已独立验证服务端权重。复用上轮只读网关模型目录证据：目录列出gpt-6.1-sol，但不验证权重；本轮没有新增连接探针或目录GET，它们不计入benchmark与用量。

## Native SR 与 Table4 executor-only 效率

| Arm | SR mean ± sample std | 成功 / 420 | Executor input K / task | Executor output K / task | Executor turns / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| filtered | 90.00 ± 0.71% | 378 / 420 | 8.006 ± 0.101 | 0.766 ± 0.004 | 10.59 ± 0.17 |
| full | 89.52 ± 1.65% | 376 / 420 | 8.068 ± 0.187 | 0.772 ± 0.033 | 10.56 ± 0.30 |

K=1000，所有任务参与分母，累计每任务全部executor调用；curator/judge不混入论文效率列。API缺失usage保留null并显示未知，不补0，tokenizer和隐藏推理token口径取决于服务提供方。

| Seed | Filtered成功 / 140 | Full成功 / 140 | Δ pp | 两者成功 | 仅filtered | 仅full | 两者失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 126 | 124 | +1.43 | 121 | 5 | 3 | 11 |
| 1 | 125 | 128 | -2.14 | 122 | 3 | 6 | 9 |
| 2 | 127 | 124 | +2.14 | 121 | 6 | 3 | 10 |

同一140任务重复三轮，streaming记忆使轮内结果相关；420对记录不视为420个独立任务样本。Order seed不固定服务端生成采样，三轮std不是置信区间，不宣称统计显著，也不做pooled显著性检验。均值及逐seed差值需一起解释，不能仅凭平均差宣称稳定过滤优势。

## 分任务类及记忆诊断

| 任务类型 | 任务 / 轮 | Filtered SR mean ± std | Full SR mean ± std | Δ pp mean ± std |
| --- | ---: | ---: | ---: | ---: |
| look_at_obj_in_light | 13 | 94.87 ± 4.44% | 94.87 ± 8.88% | 0.00 ± 13.32 |
| pick_and_place_simple | 35 | 96.19 ± 1.65% | 94.29 ± 2.86% | 1.90 ± 3.30 |
| pick_clean_then_place_in_recep | 27 | 96.30 ± 3.70% | 96.30 ± 0.00% | 0.00 ± 3.70 |
| pick_cool_then_place_in_recep | 25 | 88.00 ± 0.00% | 88.00 ± 4.00% | 0.00 ± 4.00 |
| pick_heat_then_place_in_recep | 16 | 89.58 ± 3.61% | 91.67 ± 9.55% | -2.08 ± 9.55 |
| pick_two_obj_and_place | 24 | 73.61 ± 6.36% | 72.22 ± 8.67% | 1.39 ± 13.39 |

| Seed / arm | 最终bank | Native失败入库 | Judge失败入库 | 检索引用 | Native失败引用 | Judge失败引用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / filtered | 115 | 0 | 0 | 390 | 0 | 0 |
| 0 / full | 140 | 16 | 29 | 390 | 43 | 84 |
| 1 / filtered | 114 | 0 | 0 | 390 | 0 | 0 |
| 1 / full | 140 | 12 | 23 | 390 | 61 | 75 |
| 2 / filtered | 115 | 0 | 0 | 390 | 0 | 0 |
| 2 / full | 140 | 16 | 27 | 390 | 37 | 57 |

| Arm | Judge TP / FP / FN / TN | Judge解析异常 | 非法命令 | 达交互预算的episode |
| --- | --- | ---: | ---: | ---: |
| filtered | 344 / 0 / 34 / 42 | 0 | 8 | 42 |
| full | 339 / 2 / 37 / 42 | 0 | 8 | 44 |

全量存储同时改变失败案例可用性、可见judge标签、bank规模及检索；此对照不能单独识别标签或质量过滤的独立因果贡献。judge gate的误判也不由native标签修正。

## 已提交episode的API返回用量

| Arm / role | Calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| filtered / curator | 420 | 718,387 | 80,634 |
| filtered / executor | 4,449 | 3,362,563 | 321,809 |
| filtered / judge | 420 | 427,236 | 28,415 |
| full / curator | 420 | 862,251 | 84,289 |
| full / executor | 4,434 | 3,388,547 | 324,115 |
| full / judge | 420 | 425,614 | 28,274 |

Filtered用量是历史记录，full用量来自本次新420条已提交episode。本轮新增连接探针0次、模型目录GET0次；旧探针/目录证据不混入此表。不推算retry attempts或未提交请求的账单与额外消耗。

## 本轮执行与完整性审计

本轮事件账本没有记录中断；不会沿用上轮URLError/10条保存记录或更早429/200条记录作为本轮执行证据。

冻结核心实现提交 `cc289e285da3753ec679c9cfe3219892330f3c6f`，JSON保存12份src文件/5份原文模板hash及源包指纹。新独立审计核对全部840条对照记录（420历史filtered+420新full）、每次消息/动作解析、BM25排名、native标量记录、judge gate、bank/checkpoint、批次与指标，并绑定当前分析SHA256。历史过滤组由上轮curator对照审计与分析独立绑定，其全部437文件（含420条episode）的SHA256、大小和纳秒mtime保持原样。

新增full组首20条轨迹已在原生环境重放通过，其余400条full没有完整环境重放；这不是上轮filtered20或更早40条重放证明。数据集18,416文件的前后指纹不变，独立最终扫描已验证。分析和审计均0模型API调用。

原文词语已对齐，但作者runtime空白、轨迹序列化、BM25、精确任务/顺序、环境版本和部分推理细节仍未充分披露。原文没有公开full组独立模板，这里保留原curator指令、显式添加executor judge标签。历史过滤组也未与新full组同时重跑。相对于原论文，不能保证只剩模型差异；这轮结果只描述当前固定角色配置下的存储策略比较。详见[原文对齐说明](../paper_fidelity.md)。

本机完整记录与原文资产保留于被忽略的outputs。公开文件只含白名单aggregate与指纹，不含endpoint、密钥/变量名、个人绝对路径、原始请求或完整提示词。
