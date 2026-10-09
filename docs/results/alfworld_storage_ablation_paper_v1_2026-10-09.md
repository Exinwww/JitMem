# ALFWorld 原文提示词存储消融（paper-v1）

评测日期：2026-10-09，Asia/Shanghai。三轮合计每组420次评测，质量过滤成功372次，全量存储成功371次。质量过滤仅多1次成功。差异方向随seed翻转，不能据此支持稳定的过滤优势。过滤减全量为 **+0.24 ± 3.93 个百分点**，按三轮计算样本标准差。

完整结构化汇总见[alfworld_storage_ablation_paper_v1_2026-10-09.json](alfworld_storage_ablation_paper_v1_2026-10-09.json)，协议与原文Table9见[存储消融规格](../storage_ablation.md)。本报告仅包含新paper-v1的840条真实episode评测，共9,187次executor原生交互，不拼接或重判历史legacy实验。

Executor配置模型为 `gpt-5.5`，curator为 `gpt-5.5`；judge复用executor。使用模型API，没有本地训练，不独立验证服务端权重，不代表原文Qwen3-8B curator或RL-trained结果。

使用固定版本原文curator/executor模板和SkillOS Figure13 judge；distillation模板仅冻结为可复用资产，本task_adaptive=true消融未调用蒸馏。取消旧额外指导、Look规则和人工Invalid decision反馈。两组均为valid_seen140任务、seeds0/1/2、batch10、workers10 spawn、history3、最多30次原生交互、描述BM25 top3、每轮空库。两组只改变存储策略及对应judge标签展示，native verifier决定SR，模型judge决定入库。

## 主指标与Table4效率口径

| 策略 | Native SR mean ± sample std | 成功 / 420 | Executor input K / task | Executor output K / task | Executor turns / task |
| --- | ---: | ---: | ---: | ---: | ---: |
| 质量过滤 | 88.57 ± 2.58% | 372 / 420 | 8.84 ± 0.43 | 0.77 ± 0.05 | 10.90 ± 0.70 |
| 全量存储附标签 | 88.33 ± 1.49% | 371 / 420 | 8.93 ± 0.22 | 0.78 ± 0.01 | 10.98 ± 0.30 |

K=1000。效率仅计算executor每任务全部调用，全部任务参与分母；curator/judge不混入论文效率列。API未返回usage时保留null；API token统计与模型tokenizer/隐藏推理token仍受提供方实现影响。

| Seed | 过滤成功 / 140 | 全量成功 / 140 | Δ pp | 两者成功 | 仅过滤成功 | 仅全量成功 | 两者失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 126 | -4.29 | 117 | 3 | 9 | 11 |
| 1 | 125 | 122 | +2.14 | 119 | 6 | 3 | 12 |
| 2 | 127 | 123 | +2.86 | 118 | 9 | 5 | 8 |

同一140任务重复三轮，记忆使轮内结果相关，420条记录不视为420个独立任务样本。均值、样本std和配对转移为描述性结果，不宣称统计显著；order seed不固定服务端生成采样。

## 诊断汇总

| 任务类型 | 任务 / 轮 | 过滤 SR mean ± std | 全量 SR mean ± std | Δ pp mean ± std |
| --- | ---: | ---: | ---: | ---: |
| look_at_obj_in_light | 13 | 87.18 ± 8.88% | 82.05 ± 11.75% | 5.13 ± 16.01 |
| pick_and_place_simple | 35 | 94.29 ± 4.95% | 93.33 ± 1.65% | 0.95 ± 3.30 |
| pick_clean_then_place_in_recep | 27 | 93.83 ± 2.14% | 92.59 ± 0.00% | 1.23 ± 2.14 |
| pick_cool_then_place_in_recep | 25 | 86.67 ± 2.31% | 90.67 ± 2.31% | -4.00 ± 0.00 |
| pick_heat_then_place_in_recep | 16 | 89.58 ± 3.61% | 93.75 ± 6.25% | -4.17 ± 3.61 |
| pick_two_obj_and_place | 24 | 76.39 ± 4.81% | 73.61 ± 6.36% | 2.78 ± 10.49 |

| Seed / 策略 | 最终bank | 原生失败入库 | judge失败入库 | 检索引用 | 原生失败引用 | judge失败引用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 / filtered | 111 | 1 | 0 | 390 | 3 | 0 |
| 0 / full | 140 | 14 | 28 | 390 | 50 | 84 |
| 1 / filtered | 116 | 2 | 0 | 390 | 13 | 0 |
| 1 / full | 140 | 18 | 30 | 390 | 72 | 93 |
| 2 / filtered | 116 | 2 | 0 | 390 | 1 | 0 |
| 2 / full | 140 | 17 | 27 | 390 | 46 | 70 |

| 策略 | Judge TP / FP / FN / TN | Judge解析异常 | 非法命令 | 达到交互预算的episode |
| --- | --- | ---: | ---: | ---: |
| filtered | 338 / 5 / 34 / 43 | 0 | 5 | 48 |
| full | 334 / 1 / 37 / 48 | 0 | 10 | 49 |

失败经验、显式标签、索引规模及后续检索同时随策略变化；此对照不能单独识别其中某项的因果作用。judge与native分歧完整保留，不用原生标签修正模型gate，也不将全部分歧解释为judge推理错误。

| 策略 / 角色 | Calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| filtered / curator | 420 | 718,957 | 102,546 |
| filtered / executor | 4,577 | 3,711,365 | 324,049 |
| filtered / judge | 420 | 432,217 | 28,564 |
| full / curator | 420 | 887,401 | 105,705 |
| full / executor | 4,610 | 3,750,036 | 328,031 |
| full / judge | 420 | 431,652 | 28,470 |

该成本诊断仅涵盖840条已提交episode的API返回用量，不推算实际账单、transport retries或未提交请求的额外用量。

## 执行中断与续跑

两组最初并行运行时发生HTTP 429，用完内置重试后退出。后续保持原配置、源码和原文资产，从最后完整batch checkpoint续跑，已提交记录不重跑；未提交请求可能额外计费，账单影响未知。跨组改为依次执行以降低总并发，各组仍采用workers10和batch10，批内固定memory快照与批后有序写入协议不变。
最终独立审计绑定原始中断快照与当前事件指纹，确认中断前已提交的200条记录（过滤90、全量110）内容SHA256、字节数与纳秒修改时间均未变化。

| 事件 | Seed | 过滤已提交任务 | 全量已提交任务 | 后续方式 |
| --- | ---: | ---: | ---: | --- |
| HTTP 429 | 0 | 90 | 110 | 两组依次续跑，各组workers10 |

## 完整性审计与复现边界

冻结实现提交 `cc289e285da3753ec679c9cfe3219892330f3c6f`；公开JSON保存12个实现文件与5份模板的SHA256，固定源包/成员指纹和两组manifest指纹。独立最终审计核对全部840条记录、原文资产、每次消息、BM25排名、judge gate、bank/checkpoint、批次快照、任务顺序及指标，并与分析文件SHA256绑定。新paper-v1首40条轨迹已原生重放通过；其余800条没有进行完整环境重放。审计没有模型API调用。

外部18,416个数据文件的相对路径、大小、修改时间和完整内容fingerprint未变化。完整逐任务记录与原文全文资产保存在本机被忽略的outputs；公开报告不含API endpoint、密钥、密钥变量名、个人绝对路径或原始模型请求。

原文模板的文字已对齐，但judge图框空白、完整轨迹序列化、全存储label格式，以及BM25细节、精确任务/顺序、环境与数据版本和部分推理配置未由作者充分披露。GiGPO继承实现也不能证明作者使用相同fork；因此不声称只剩模型差异，详见[原文对齐说明](../paper_fidelity.md)。
