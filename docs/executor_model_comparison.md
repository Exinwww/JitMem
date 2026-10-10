# 全量存储下的 executor 模型对照

在已完成的 `gpt-6.1-sol` curator 全量组基础上，把 executor 从 `gpt-5.5` 切换为 `gpt-6.1-sol`，judge 保持 `gpt-5.5`。两组使用同一 OpenAI-compatible API 接口，不训练模型。

| 角色/协议 | 历史基线 | 新 executor 组 |
|---|---|---|
| curator | gpt-6.1-sol | gpt-6.1-sol |
| executor | gpt-5.5 | gpt-6.1-sol |
| judge | gpt-5.5，复用 executor | gpt-5.5，独立 client |
| 存储 | 全量 raw 轨迹及 judge 标签 | 相同 |
| 提示词 | paper-v1 原文模板 | 相同 |
| 任务 | valid_seen 全部 140 任务 | 相同 |
| 任务顺序 | seeds 0/1/2，三轮各自空库 | 相同 |
| batch / workers | 10 / 10，spawn | 相同 |
| max_steps / history / retrieval | 30 / 3 / top-3 BM25 | 相同 |
| 生成设置 | temperature=1；executor/judge 4096、curator 8192 | 相同 |

主指标沿用原论文口径：每轮 native 成功数除以 140，报告三轮均值及样本标准差（ddof=1）；差值定义为新 executor 减旧 executor。效率为每任务 executor-only input/output K tokens 与交互次数。judge 标签只参与记忆注释，native verifier 决定成功率，不按 judge 结论修正任务得分。

新组单独执行 420 条模型 episode，历史基线的 420 条直接配对。原有 bank、checkpoint、原始记录与报告保持完整。两组完整流式演进会产生不同轨迹、标签及后续检索，即使 curator 和 judge 模型相同，其实际输入与输出也会随 executor 改变。

## 独立 judge 与兼容验证

默认未指定 judge 时仍复用 executor client。显式 `[judge]` 或非空 `JITMEM_JUDGE_BASE_URL` / `JITMEM_JUDGE_MODEL` / `JITMEM_JUDGE_API_KEY` 启用独立角色。角色环境变量优先于普通 `OPENAI_*`，后者优先于 TOML；独立 judge 未指定的字段继承最终解析的 executor 参数。因此已有 `OPENAI_MODEL` 时，应设置 `JITMEM_JUDGE_MODEL=gpt-5.5`，固定本次 judge。

本轮固定独立 judge，是为了控制 executor 消融变量而作的扩展。原文通常由 executor 兼任 judge；本轮不对应同时替换 executor 和 judge 的原模型实验。

本轮新增独立路由，历史与新组源码有四处差异：`config.py`、`pipeline.py`、`cli.py`、`evaluation.py`。其余八个核心模块、五份模板、runtime 与 140 个游戏固定。预检比较有效角色配置：历史隐式 judge 展开为旧 executor 配置，与新显式 judge 逐字段相等；允许的有效配置差异只有 `executor.model` 和输出目录。

正式运行前，以历史基线全部 420 条记录重放三种实现：旧源码默认路由、新源码默认路由、新源码显式同模型 judge。响应和环境状态来自历史记录，对照全部 messages、动作、gate、raw bank、批次顺序与 usage；这属于离线兼容 fixture，模型 API 调用和 native 环境步数均为零。软件测试另用本地 HTTP 和真实 spawn 子进程检查请求模型、预算及密钥变量路由。历史记录重放不计入新模型评测，也不能替代新组的原生环境重放。

新组完成后，独立核对完整配对 840 条记录并在原生 ALFWorld 重放新组前 20 条轨迹；再次核对全数据文件和历史基线文件的内容、大小及 mtime。完整评测及审计通过后才发布分数。

## 运行

先准备原文模板与交互环境，参见 [README](../README.md)。复制 [配置模板](../configs/storage_all_executor_gpt61_judge_gpt55.example.toml)，填写数据根目录，连接信息与密钥通过环境变量配置。

```bash
cp configs/storage_all_executor_gpt61_judge_gpt55.example.toml \
   configs/storage_all_executor_gpt61_judge_gpt55.local.toml
export JITMEM_CURATOR_MODEL=gpt-6.1-sol
export JITMEM_EXECUTOR_MODEL=gpt-6.1-sol
export JITMEM_JUDGE_MODEL=gpt-5.5
.venv/bin/python -m jitmem evaluate \
  --config configs/storage_all_executor_gpt61_judge_gpt55.local.toml
```

上述 `evaluate` 命令不会自动生成严格分析所需的兼容重放与预检证据。分析必须使用启动前冻结、与同一历史基线绑定的证据，不能在评测结束后补造预检。

严格分析器只读取完整记录与冻结证据，不调用模型或环境：

```bash
.venv/bin/python scripts/analyze_executor_model_comparison.py \
  outputs/storage_all_curator_gpt61_paper_v1 \
  outputs/storage_all_executor_gpt61_judge_gpt55_paper_v1 \
  --bridge-proof outputs/executor_gpt61_source_behavior_bridge.json \
  --preflight outputs/executor_gpt61_preflight.json \
  --output-dir outputs/executor_gpt61_comparison
```

本实验能检验固定 curator/judge 与全量存储时，executor 模型替换对完整流程的影响。三轮 order seed 只控制任务排列，不能固定服务端随机采样或确认实际模型权重；结果不能直接证明原论文与本复现差距来自模型智力。原论文使用 Qwen3-8B curator，与本次角色组合不同；仍未公开的协议细节见 [原文对齐说明](paper_fidelity.md)。

## 完整评测结果（2026-10-10）

新组420条原生评测完成，历史全量组420条直接配对，三轮各140任务、各自空库，最终bank均为140条。

| Seed | 旧 executor 成功 / 140 | 新 executor 成功 / 140 | 新减旧（百分点） |
|---|---:|---:|---:|
| 0 | 124 | 133 | +6.43 |
| 1 | 128 | 130 | +1.43 |
| 2 | 124 | 133 | +6.43 |
| Mean ± sample std | 89.52 ± 1.65% | 94.29 ± 1.24% | +4.76 ± 2.89 |

新executor净多20次成功，三轮方向均为正；这是三个任务排列的描述性结果，不宣称统计显著。executor-only input K/task从8.068±0.187降至7.185±0.111，output K/task从0.772±0.033降至0.611±0.009，原生交互/task从10.56±0.30降至9.23±0.19。全角色用量与审计范围见[公开结果报告](results/alfworld_executor_gpt61_judge_gpt55_paper_v1_2026-10-10.md)和[结构化汇总](results/alfworld_executor_gpt61_judge_gpt55_paper_v1_2026-10-10.json)。
