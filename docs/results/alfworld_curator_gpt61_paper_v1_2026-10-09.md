# ALFWorld Curator 模型对照（paper-v1，quality-filtered）

评测日期：2026-10-09，Asia/Shanghai。固定executor/judge为gpt-5.5，仅将curator从gpt-5.5替换为gpt-6.1-sol。Candidate减baseline的native SR差为 **1.43 ± 2.47 个百分点**，按三轮计算样本标准差。结果是这套API与streaming协议下的描述性比较，不预设替换模型带来改善。

结构化汇总见[alfworld_curator_gpt61_paper_v1_2026-10-09.json](alfworld_curator_gpt61_paper_v1_2026-10-09.json)。复用420条历史baseline episode，新评测candidate420条，共比较840条记录；本次没有重新评测baseline，也不是新增840条模型评测。

两组都是judge质量过滤、原文paper-v1模板、task-adaptive、每轮空库、valid_seen140任务、seeds0/1/2、batch10/workers10 spawn、history3、最多30次环境交互、描述BM25 top3。除了curator请求模型与输出目录，其余配置、源码、资产、runtime、数据、任务和顺序经过匹配验证。Native verifier决定SR，同一个固定executor模型judge决定入库，curator看不到结果标签。

使用模型API，没有本地训练；请求模型名不代表已独立验证服务端权重。Distillation模板只冻结为可复用资产，本task-adaptive消融未调用蒸馏，不代表原文RL-trained结果。
另有1次只读网关模型目录核对，目录列出了请求ID `gpt-6.1-sol`；这仅证明目录声明了该ID，不验证实际权重，不属于benchmark或生成token用量。对应证据文件SHA256保存在JSON中。

## Native SR 与 Table4 executor-only 效率

| Arm / curator | SR mean ± sample std | 成功 / 420 | Executor input K / task | Executor output K / task | Executor turns / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline / gpt-5.5 | 88.57 ± 2.58% | 372 / 420 | 8.837 ± 0.434 | 0.772 ± 0.052 | 10.90 ± 0.70 |
| candidate / gpt-6.1-sol | 90.00 ± 0.71% | 378 / 420 | 8.006 ± 0.101 | 0.766 ± 0.004 | 10.59 ± 0.17 |

K=1000，所有任务参与分母，累计每任务全部executor调用；curator/judge不混入论文效率列。API缺失usage保留null并显示未知，不补0，tokenizer和隐藏推理token口径取决于服务提供方。

| Seed | Baseline成功 / 140 | Candidate成功 / 140 | Δ pp | 两者成功 | 仅candidate | 仅baseline | 两者失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 126 | +4.29 | 115 | 11 | 5 | 9 |
| 1 | 125 | 125 | +0.00 | 121 | 4 | 4 | 11 |
| 2 | 127 | 127 | +0.00 | 122 | 5 | 5 | 8 |

同一140任务重复三轮，streaming记忆使轮内结果相关；420对记录不视为420个独立任务样本。Order seed不固定服务端生成采样，三轮std不是置信区间，不宣称统计显著，也不做pooled显著性检验。

## 分任务类及记忆诊断

| 任务类型 | 任务 / 轮 | Baseline SR mean ± std | Candidate SR mean ± std | Δ pp mean ± std |
| --- | ---: | ---: | ---: | ---: |
| look_at_obj_in_light | 13 | 87.18 ± 8.88% | 94.87 ± 4.44% | 7.69 ± 7.69 |
| pick_and_place_simple | 35 | 94.29 ± 4.95% | 96.19 ± 1.65% | 1.90 ± 3.30 |
| pick_clean_then_place_in_recep | 27 | 93.83 ± 2.14% | 96.30 ± 3.70% | 2.47 ± 2.14 |
| pick_cool_then_place_in_recep | 25 | 86.67 ± 2.31% | 88.00 ± 0.00% | 1.33 ± 2.31 |
| pick_heat_then_place_in_recep | 16 | 89.58 ± 3.61% | 89.58 ± 3.61% | 0.00 ± 6.25 |
| pick_two_obj_and_place | 24 | 76.39 ± 4.81% | 73.61 ± 6.36% | -2.78 ± 10.49 |

| Seed / arm | 最终bank | Native失败入库 | Judge失败入库 | 检索引用 | Native失败引用 | Judge失败引用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / baseline | 111 | 1 | 0 | 390 | 3 | 0 |
| 0 / candidate | 115 | 0 | 0 | 390 | 0 | 0 |
| 1 / baseline | 116 | 2 | 0 | 390 | 13 | 0 |
| 1 / candidate | 114 | 0 | 0 | 390 | 0 | 0 |
| 2 / baseline | 116 | 2 | 0 | 390 | 1 | 0 |
| 2 / candidate | 115 | 0 | 0 | 390 | 0 | 0 |

| Arm | Judge TP / FP / FN / TN | Judge解析异常 | 非法命令 | 达交互预算的episode |
| --- | --- | ---: | ---: | ---: |
| baseline | 338 / 5 / 34 / 43 | 0 | 5 | 48 |
| candidate | 344 / 0 / 34 / 42 | 0 | 8 | 42 |

更换curator会改变payload、executor轨迹、judge gate和后续bank/检索。这些是流水线的反馈与中介作用，此对照不能单独识别其中某个因素，也不用native标签修正judge gate。

## 已提交episode的API返回用量

| Arm / role | Calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| baseline / curator | 420 | 718,957 | 102,546 |
| baseline / executor | 4,577 | 3,711,365 | 324,049 |
| baseline / judge | 420 | 432,217 | 28,564 |
| candidate / curator | 420 | 718,387 | 80,634 |
| candidate / executor | 4,449 | 3,362,563 | 321,809 |
| candidate / judge | 420 | 427,236 | 28,415 |

Baseline用量是历史记录，candidate用量来自本次新420条已提交episode。另有1次gpt-6.1-sol连接检查，它是非benchmark探针，未计入评测与上表用量；不推算探针、transport retries或未提交请求的账单与额外消耗。

## 本次中断与恢复

Candidate曾在seed0完成10条后遭遇URLError，4次尝试后退出。确认无残留评测进程后，保持相同配置、源码和原文资产，从checkpoint恢复；workers10/batch10不变。最终审计绑定本次原始错误、事件与中断快照指纹，确认先前10条记录的内容SHA256、字节数和纳秒mtime未变，已提交记录未重跑；未提交请求的额外账单未知。本次证据没有用旧存储消融的429/200条保存记录代替。

## 完整性审计与解释边界

冻结核心实现提交 `cc289e285da3753ec679c9cfe3219892330f3c6f`，JSON保存12份src文件/5份原文模板hash及源包指纹。新独立审计核对全部840条对照记录（420历史baseline+420新candidate）、每次消息/动作解析、BM25排名、native标量记录、judge gate、bank/checkpoint、批次与指标，并绑定当前分析SHA256。原baseline所在的旧840条存储对照审计也独立绑定；baseline全部437文件（含420条episode）的SHA256、大小和纳秒mtime均保持原样。

新candidate首20条轨迹已在原生环境重放通过，其余400条candidate没有完整环境重放；该新20证明没有冒充历史40条重放。数据集18,416文件的前后指纹不变，独立最终扫描已验证。分析和审计均0模型API调用。

原文词语已对齐，但作者runtime空白、轨迹序列化、BM25、精确任务/顺序、环境版本和部分推理细节仍未充分披露。模型请求名称不验证权重，历史baseline也未与candidate同时重跑。因此只能描述固定executor/judge、质量过滤协议下更换curator的观测效果，不能确认原文全部性能差距来自模型智力；相对于原论文，不能保证只剩模型差异。详见[原文对齐说明](../paper_fidelity.md)。

本机完整记录/原文资产保留于被忽略的outputs。公开文件只含白名单aggregate与指纹，不含endpoint、密钥/变量名、个人绝对路径、原始请求或完整提示词。
