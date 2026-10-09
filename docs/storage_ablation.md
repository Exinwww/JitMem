# 消融：质量过滤与带标签的全量存储

本实验检验 [JITMEM v1](https://arxiv.org/pdf/2609.27334) 第8页 §4.2、Figure 2a 的存储消融。质量过滤组只保存 executor-as-judge 判为成功的完整轨迹；全量组保存所有完整轨迹，并在检索后将每条轨迹的自评成功/失败标签展示给 curator。标签来自模型 judge，不能替换为原生环境的真实成功标量。

原论文第21页 Table 9 的 ALFWorld 结果如下；curator 均为未训练的 Qwen3-8B，三轮随机任务顺序：

| Executor | 质量过滤 SR mean ± std | 全量存储附标签 SR mean ± std | 全量减过滤（百分点） |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 60.5 ± 2.6% | 59.0 ± 2.2% | −1.5 |
| Gemini-2.5-Pro | 80.0 ± 1.5% | 77.1 ± 3.1% | −2.9 |
| GPT-5.4 | 79.3 ± 3.6% | 77.6 ± 1.8% | −1.7 |

这是 JITMEM-base 的已报告消融。当前项目按用户要求使用模型 API、不训练 curator；本机使用环境变量配置的模型，不能将本次数字当作原论文模型或 RL-trained JITMEM 的等价复跑。

curator/executor均为gpt-5.5的paper-v1原文模板与对齐的继承runtime已重新完成140任务×3轮×2组：过滤组 **88.57 ± 2.58%**，全量组 **88.33 ± 1.49%**，过滤减全量 **+0.24 ± 3.93 个百分点**。合计成功372与371次，差异方向随seed翻转，不能支持稳定的过滤优势；见[原文协议报告](results/alfworld_storage_ablation_paper_v1_2026-10-09.md)和[结构化结果](results/alfworld_storage_ablation_paper_v1_2026-10-09.json)。全部840条记录通过独立审计，另有40条原生环境重放。

旧legacy-paraphrase协议结果为过滤组 **85.71 ± 2.14%**、全量组 **88.57 ± 1.24%**，差值 **−2.86 ± 2.47 个百分点**，见[历史报告](results/alfworld_storage_ablation_2026-10-09.md)和[结构化结果](results/alfworld_storage_ablation_2026-10-09.json)。旧协议和checkpoint保持独立；前后还改变了交互与输出处理，不能把结果变化单独归因于prompt。

## gpt-6.1-sol curator 的存储对照

仅切换curator后，质量过滤组已完成140任务×seeds0/1/2：curator为`gpt-6.1-sol`，executor/judge仍为`gpt-5.5`，native SR **90.00±0.71%**、378/420成功，见[模型对照结果](results/alfworld_curator_gpt61_paper_v1_2026-10-09.md)。对应全量组采用[独立模板](../configs/storage_all_curator_gpt61.example.toml)，与该过滤组配置仅有`store_policy`和`output_dir`不同，完整评测420条；复用过滤组历史420条进行配对，不重新打分或改写基线。

全量组已完成，native SR **89.52±1.65%**、376/420成功，过滤减全量为**+0.48±2.30个百分点**。三轮的方向翻转，不能支持稳定或显著的过滤优势。完整[结果报告](results/alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.md)与[结构化汇总](results/alfworld_storage_ablation_curator_gpt61_paper_v1_2026-10-09.json)独立保存。

| Seed | 过滤成功 / 140 | 全量成功 / 140 | 过滤减全量（百分点） |
| --- | ---: | ---: | ---: |
| 0 | 126 | 124 | +1.43 |
| 1 | 125 | 128 | −2.14 |
| 2 | 127 | 124 | +2.14 |

Table4口径executor-only input K/task为8.006±0.101与8.068±0.187，output K/task为0.766±0.004与0.772±0.033，交互/task为10.59±0.17与10.56±0.30。全角色API返回tokens为4,939,044与5,113,090，另作诊断，不混入executor效率列或推算账单。

全量组从每seed空库开始，使用相同原文模板、源码、runtime、数据和任务顺序；保留judge判失败的轨迹并展示judge标签。模型参数与指标口径沿用下述协议。连接参数继续通过环境变量提供，运行时明确固定角色模型：

```bash
JITMEM_EXECUTOR_MODEL=gpt-5.5 JITMEM_CURATOR_MODEL=gpt-6.1-sol \
  .venv/bin/python -m jitmem evaluate \
  --config configs/storage_all_curator_gpt61.local.toml
.venv/bin/python scripts/analyze_storage_ablation.py \
  outputs/storage_filtered_curator_gpt61_paper_v1 outputs/storage_all_curator_gpt61_paper_v1 \
  --output-dir outputs/storage_gpt61_ablation_comparison
```

evaluate会产生真实API调用；分析仅读取本地记录。只在新全量420条完整评测与独立审计通过后发布存储差值，定义仍为过滤减全量。完整分析含本机配置与逐任务证据，只保存于被忽略的`outputs/`，不直接作为公开JSON。

## 配对协议

| 项目 | 固定设置 |
| --- | --- |
| 环境 | ALFWorld 文本环境，`valid_seen` 全部140个任务，六类齐全，无 limit |
| 两组方法 | `method="jitmem"`，相同 curator/executor/judge 模型与采样参数 |
| 顺序与重复 | seeds 0、1、2；每组420条episode，相同 seed 的任务顺序一致 |
| 初始记忆 | 每个 seed 独立空库，无 warm start |
| Streaming | batch10，每批10个独立进程共享固定快照；整批完成后按预定任务顺序写入 |
| 检索 | 描述 BM25、top3；全量组的失败轨迹也参与索引和排序 |
| Curator | 每个任务一次，面向当前任务；完整 observation/action 轨迹输入 |
| Executor | 原文相同模板，最近3步编号历史，最多30次原生交互 |
| Judge | 同一 executor client，仅观察任务和完整轨迹；每条轨迹均自评 |
| 输出预算 | executor/judge 4096、curator8192；temperature1，其他采样设置相同 |
| 主指标 | 原生环境 SR；按三轮报告均值和样本标准差，差值为过滤减全量 |

`configs/storage_filtered.example.toml` 与 `configs/storage_all.example.toml` 除 `store_policy` 和 `output_dir` 外完全相同。模型连接参数与密钥通过现有环境变量提供。两组使用相同的12个实现源码哈希、原文资产指纹、游戏哈希和runtime。gpt-5.5原文协议重评测在同一冻结实现下重新运行两组；gpt-6.1-sol curator存储对照则复用此前同协议过滤组420条，与新增全量组420条配对。两项对照都与旧legacy协议记录保持独立。

两组使用同一份原文curator system模板；全量组额外展示 `Executor judge label: success/failure`。原文没有公布这个消融专用模板或标签格式，因此不自行改写system中的成功经验措辞；这一位置与格式的实现选择仍须披露。executor只接收curator payload，judge只接收当前任务与完整轨迹，二者不直接读取检索标签或native success/reward标量。[原文对齐说明](paper_fidelity.md)记录模板指纹、继承history/parser/actions格式和未公开细节。

存储策略改变后，两组后续生成的轨迹、BM25索引、检索结果与 bank 大小会随之分化；不强制两组使用同一批历史轨迹。该实验检验完整 streaming 存储策略，不能单独区分过滤、显式标签、索引规模与失败经验暴露的各自作用。

## 运行与审计

复制两份模板为对应的 `*.local.toml`，仅将 `data_root` 改为实际 ALFWorld 路径。确认两组均没有 `limit` 或 `warm_start`，连接参数沿用 `OPENAI_*` 或角色环境变量。

```bash
.venv/bin/python scripts/prepare_paper_prompts.py
.venv/bin/python -m jitmem doctor --config configs/storage_filtered.local.toml
.venv/bin/python -m jitmem evaluate --config configs/storage_filtered.local.toml
.venv/bin/python -m jitmem evaluate --config configs/storage_all.local.toml
.venv/bin/python scripts/analyze_storage_ablation.py \
  outputs/storage_filtered_paper_v1 outputs/storage_all_paper_v1 --output-dir outputs/storage_ablation_comparison_paper_v1
```

模型 API 已配置时，这些 evaluate 命令会产生真实调用。可分别运行两组，或在不同终端同时运行；不要修改运行中的 `src/jitmem/` 或采样配置。出现基础设施错误后，只能在协议未变时对对应 evaluate 命令追加 `--resume`，从最后完整 batch checkpoint 继续；未提交批次会重跑并可能额外计费。已有输出目录默认拒绝覆盖。

分析器要求完整配对的140任务×3轮，验证配置仅有上述差异、source/runtime/game/order一致、每批快照和有序写入、judge gate、全量组标签与实际存储轨迹相符。还核对逐任务记录、checkpoint、最终bank和模型请求，不接受未完成/故障run。

主SR按Table9，效率按Table4报告每任务executor-only input/output K与交互次数，均按三轮计算mean/sample std。另报告各seed/任务类SR、配对成功/失败转移、分角色tokens、judge/native混淆矩阵、bank增长、存储和检索中的失败比例作为诊断。judge判失败和原生失败分别统计，避免把模型误判当作环境事实。API不提供usage时保留 `null`，不能宣称零成本。

## 解释边界

Order seed只控制任务顺序，不固定服务端token sampling。三轮共享同一组140任务，streaming经验在轮内相关；420条记录不作为420个独立任务样本做显著性检验。配对转移与每轮差值为描述性指标，三轮std也不是置信区间。

paper-v1正常解析judge返回文本，包括API报告length但已有完整JSON的输出；只有格式不符合布尔JSON要求时保守拒绝。生成异常单列为诊断，不能冒充有效语义判断；非法命令均交给环境并消耗交互预算。API/环境故障中止run，不作为失败任务混入SR。主SR与Table4效率口径保持一致，额外诊断不替代论文指标。

“过滤更好”是待检验的假设。结果必须据完整运行如实报告，即使全量组持平或更高，也不删任务、不修改judge、不重试选择有利采样。最终结果应限定为当前模型、prompt和streaming设置下的存储策略对照；仅凭失败经验暴露与SR同时变化，不能证明噪声是唯一原因。

完整请求与逐任务证据保存在被Git忽略的本机 `outputs/`；公开仓库只保存去除endpoint、凭证、个人路径与原始请求的汇总报告。
