# 质量过滤下的 curator 模型对照

本实验在完成的paper-v1质量过滤基线上，仅将curator从`gpt-5.5`替换为`gpt-6.1-sol`。executor和executor-as-judge保持`gpt-5.5`；使用现有OpenAI-compatible模型API，不训练模型。

| 角色 | 已完成基线 | 新实验 |
| --- | --- | --- |
| Curator | gpt-5.5 | gpt-6.1-sol |
| Executor | gpt-5.5 | gpt-5.5 |
| Judge | 复用executor | 复用executor |

基线为[原文协议质量过滤组](results/alfworld_storage_ablation_paper_v1_2026-10-09.md)，372/420成功，native SR **88.57 ± 2.58%**。新实验使用独立目录，每轮从空记忆库开始；不沿用基线bank或checkpoint，不重新运行或重新打分基线。

## 固定协议

两组采用相同的140个`valid_seen`任务、seeds0/1/2、batch10、workers10 spawn、history3、描述BM25 top3、最多30次原生交互、`task_adaptive=true`及`store_policy="judge"`。使用相同的12个核心源码文件、5份原文模板、游戏与runtime指纹。两组均不调用distillation。

executor/judge的temperature1、输出预算4096，以及curator的temperature1、输出预算8192、其余API参数与连接配置保持一致。预检要求最终解析配置仅有`curator.model`与`experiment.output_dir`不同；不因新模型表现调整prompt、gate、参数或任务集。

主指标为全部140任务三轮native SR的mean/sample std，差值定义为新curator减基线curator。效率沿用Table4口径：每任务executor-only input/output K及交互次数；curator/judge用量、记忆增长与误判另作诊断。缺失API usage保留null。只有完整420条新评测与最终审计通过后才发布分数。

## 运行

复制[配置模板](../configs/storage_filtered_curator_gpt61.example.toml)为`configs/storage_filtered_curator_gpt61.local.toml`，设置本机`data_root`；endpoint与key继续通过已有环境变量提供。角色环境变量优先于共同`OPENAI_MODEL`和TOML，执行时明确固定两个角色：

```bash
JITMEM_EXECUTOR_MODEL=gpt-5.5 JITMEM_CURATOR_MODEL=gpt-6.1-sol \
  .venv/bin/python -m jitmem api-check \
  --config configs/storage_filtered_curator_gpt61.local.toml
JITMEM_EXECUTOR_MODEL=gpt-5.5 JITMEM_CURATOR_MODEL=gpt-6.1-sol \
  .venv/bin/python -m jitmem evaluate \
  --config configs/storage_filtered_curator_gpt61.local.toml
.venv/bin/python scripts/analyze_curator_comparison.py \
  outputs/storage_filtered_paper_v1 outputs/storage_filtered_curator_gpt61_paper_v1 \
  --output-dir outputs/curator_gpt61_comparison
```

这些命令会产生实际API调用。只有基础设施中断且协议未改变时，给evaluate追加`--resume`，从最后完整batch继续；已提交记录保留，未提交请求可能额外计费。分析与最终审计使用独立证据目录，不覆盖旧实验报告。

分析命令仅读取本地完整记录，不调用API或环境；校验原文messages、judge gate、raw bank、checkpoint、各seed任务顺序和逐调用用量。其完整输出含本机配置与逐任务证据，只能保存在被忽略的`outputs/`，不能直接复制为公开报告。

## 解释范围

该对照检验固定`gpt-5.5` executor/judge时，curator模型替换对完整streaming流程的影响。curator输出会改变动作、轨迹、后续记忆与检索，因此不能把观察到的变化仅归为某一个任务上的指导能力。

它不能单独确认与原论文GPT-5.4结果的全部差距都来自“模型智力”：原文使用Qwen3-8B curator，executor也不同，另有[未公开实现细节](paper_fidelity.md)。模型名由API请求配置记录，不独立验证服务端权重；order seeds只控制任务顺序，API生成采样未固定，三轮及轮内记忆相关性限制显著性和普遍性解释。
