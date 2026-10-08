# ALFWorld API 版 JITMEM 评测结果

评测日期：2026-10-08，Asia/Shanghai。本文保存可随 Git 仓库查看的结果；详细数值、运行用量和原始实现哈希见 [结构化结果](alfworld_valid_seen_2026-10-08.json)。

本次使用 API 配置模型 `gpt-5.5` 作为 executor 和 curator，复现 [JITMEM](https://arxiv.org/pdf/2609.27334) 的 prompted read-time curation 流程，没有本地模型训练。模型名来自 API 配置，未独立验证服务端权重。提示词使用公开方法的语义改写，并在正式实验前补充 ALFWorld Look 的公开目标语义；与作者原模型及 RL-trained 设置不同。具体来源和工程选择见 [复现规格](../reproduction_spec.md)。

两方法使用相同 executor 配置、完整 `valid_seen` 任务集和配对任务顺序。140 个唯一任务、seeds 0/1/2、batch10、workers10（spawn）、history3、最多30次决策；JITMEM description-only BM25 top3、judge gate、每轮空库、无 warm start。批内共享冻结记忆，完成整批后按预定任务顺序写入；native verifier 决定评测标签，judge 只决定记忆写入。

## 成功率与交互次数

| 方法 | 成功率 mean ± sample std | 成功 episode / 全部 episode | 平均决策次数 | 平均实际环境步数 |
| --- | ---: | ---: | ---: | ---: |
| no-memory | 77.86 ± 1.43% | 327 / 420 | 14.85 | 14.79 |
| prompted JITMEM | 86.67 ± 0.41% | 364 / 420 | 11.23 | 11.21 |

平均成功率差为 **+8.81 ± 1.49 个百分点**，平均决策次数减少 24.35%。均值和样本标准差按三轮计算，ddof=1。同一任务在不同种子中重复出现，且流式记忆使任务相互关联；这些是描述性结果，没有将420次执行当作独立任务样本，也没有计算 pooled p-value。

| Seed | no-memory 成功数 / 140 | JITMEM 成功数 / 140 | SR 差（百分点） | 两者成功 | 仅 baseline 成功 | 仅 JITMEM 成功 | 两者失败 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 111 | 121 | +7.14 | 108 | 3 | 13 | 16 |
| 1 | 107 | 121 | +10.00 | 99 | 8 | 22 | 11 |
| 2 | 109 | 122 | +9.29 | 104 | 5 | 18 | 13 |

## 分任务类型

| 任务类型 | 每轮任务数 | no-memory SR mean ± std | JITMEM SR mean ± std | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| Look | 13 | 76.92 ± 0.00% | 76.92 ± 0.00% | +0.00 |
| Pick | 35 | 83.81 ± 1.65% | 96.19 ± 3.30% | +12.38 |
| Clean | 27 | 82.72 ± 5.66% | 92.59 ± 0.00% | +9.88 |
| Cool | 25 | 77.33 ± 4.62% | 84.00 ± 8.00% | +6.67 |
| Heat | 16 | 83.33 ± 3.61% | 91.67 ± 3.61% | +8.33 |
| Pick2 | 24 | 61.11 ± 2.41% | 70.83 ± 7.22% | +9.72 |

## API 用量与记忆 gate

以下为每方法三轮合计的 API 返回用量，均完整。它们不包括独立的功能 pilot；没有根据 token 数推算费用。

| 方法 / 角色 | Calls | Input tokens | Output tokens |
| --- | ---: | ---: | ---: |
| no-memory / executor | 6,237 | 2,923,895 | 97,747 |
| JITMEM / curator | 420 | 746,587 | 94,489 |
| JITMEM / executor | 4,718 | 3,299,202 | 75,675 |
| JITMEM / judge | 420 | 385,888 | 22,020 |

全角色 input+output tokens 由 **3,021,642** 增至 **4,623,861**，增加 **53.02%**；更少的决策没有带来本次全系统 token 节省。curator 在空库的首批任务也会被调用并允许提供一般指导，因此首批表现不能归因于历史经验检索。

两方法非法动作计数分别为26和9，均无生成截断或未解决基础设施错误。JITMEM 三轮最终 bank 均为118条；相对 native label 的 judge 计数为353 TP、1 FP、11 FN、55 TN。1条原生失败轨迹被存入后续记忆，保留原始 gate 行为。

judge/native 分歧并不全部等于 judge 推理错误。已确认一个原始游戏将 mug 同时关联到 countertop 和 coffeemachine，冷却后 native 即判成功，但可观察轨迹没有最后放置，judge 拒绝写入。该继承数据行为及 pilot 修正历史见 [验证记录](../validation.md)。

## 验证与原始记录

软件检查为93项测试、89项子测试通过，Ruff、格式和安装脚本语法检查通过。独立最终审计验证全部840个 episode 的覆盖、顺序、summary/results/checkpoint 一致性、140个游戏哈希、11个源码哈希、冻结记忆与有序提交、模型输入边界和比较数值，未发现新增异常。外部数据18,416个文件的内容、大小和修改时间指纹在评测前后一致。

原始 `outputs/` 保留在执行本次评估的本机，包含完整轨迹、API messages、manifest、checkpoint、每轮summary、逐任务comparison和独立审计。该目录不提交到Git；本报告和结构化结果不包含API endpoint、密钥或个人绝对路径。重新运行和生成逐任务比较的命令见 [README](../../README.md)。
