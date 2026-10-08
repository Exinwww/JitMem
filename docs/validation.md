# 实现验证记录

验证日期：2026-10-08（Asia/Shanghai）。本记录区分软件正确性、真实模拟器可用性和模型能力评测。

版本管理中的 [评测结果报告](results/alfworld_valid_seen_2026-10-08.md) 与 [结构化结果](results/alfworld_valid_seen_2026-10-08.json) 可在新克隆中查看。本页指向 `outputs/` 的链接对应本地生成的审计和原始记录，目录不提交到 Git。

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

模型输出达到生成 token 上限属于模型行为，不能当作网络故障中止并反复重试同一任务。现在 executor 截断输出消耗一次决策预算且不执行任何部分动作；curator 截断指导被省略，judge 截断结果被拒绝，distiller 截断摘要不入库。所有情况记录 `finish_reason` 和 `generation_failures`，任务仍计入评测。缺失 API usage 时，相应用量及受影响汇总为 `null`，`usage.complete` / summary `usage_complete` 为 false，不能据此宣称 token 减少或零成本。

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
