# 原文提示词与 ALFWorld 协议对齐

当前正式配置使用 `paper-v1`。旧的 `legacy-paraphrase` 只用于历史实验和离线兼容测试；旧840条存储消融结果不是本协议的评测结果。

## 原文资产与可核查来源

执行以下准备命令，不调用模型API：

```bash
uv pip install --python .venv/bin/python -e '.[paper]'
.venv/bin/python scripts/prepare_paper_prompts.py
```

普通pip环境可使用 `.venv/bin/python -m pip install -e '.[paper]'`。准备程序下载固定版本源包，校验源包、具体成员和每个模板的SHA256，然后保存到被Git忽略的 `outputs/paper_prompts/v1/`。运行模型时只读取本地资产，不访问论文网站；缺失、篡改或溯源不一致均提前失败，不回退到改写模板。

| 角色 | 原始来源 | 实际处理 |
| --- | --- | --- |
| Curator | [JITMEM v1](https://arxiv.org/pdf/2609.27334v1)，第14页；TeX `sections/appendix_prompt.tex` 第0个listing | 保留原文词语和内部空白；去掉展示用角色标记，按system/user拆分；展开编号memory示例 |
| Executor | 同源第15页，第3个listing | 使用原文模板，替换原有占位符；没有新增system指令、寻物规则或强制think标签 |
| Distillation | 同源第17页，第7个listing | 使用原文Markdown输出要求，取消原实现的JSON要求 |
| Executor-as-judge | [SkillOS v1](https://arxiv.org/pdf/2605.06614v1)第22页，§A.4 Figure13；源包 `figures/alfworld_judge.pdf` | 提取system/input图框文字；保留词语，恢复标题/列表换行与输入中的换行转义；取消自行添加的Look规则 |

JITMEM明确说明ALFWorld executor/judge沿用SkillOS；它自己的executor模板优先。judge仅提供排版PDF图框，没有可核查的作者运行时字符串，因此不能声称其空白字节与作者运行时完全相同。准备命令固定 `pypdf==6.10.0` 和规范化模板哈希，使本项目提取结果可复核。

原始模板全文和PDF/TeX留在本地资产目录，公开仓库提供提取代码与指纹。manifest保存公开源URL、固定版本和模板指纹，运行fingerprint也包含这些资产。修改源码、模板或配置后拒绝沿用旧checkpoint。

## 已修正的额外行为

原文未发布完整runtime。未给出具体字符串的history/actions字段和动作恢复路径，参考SkillOS引用的[GiGPO实现](https://github.com/langfengQ/verl-agent/tree/20bd331bdbc9026a5668e11362178e10ab7400c8)，固定commit `20bd331bdbc9026a5668e11362178e10ab7400c8`。这是可审计的继承选择，不能标为作者实际fork已验证。

| 项目 | `paper-v1` 行为 |
| --- | --- |
| 空库 | 原文curator模板接收空的retrieved memory区；payload/history为空时不加入额外指导语或占位说明 |
| Curator指导 | 去掉验证旧地点/object编号等自行添加的指令；两存储组使用同一份原文system模板 |
| History | 最近3条原始pre-observation和解析动作，保留绝对步编号；不附模型推理 |
| 可用动作 | 排除 `help`；单引号包围命令，以换行和空格连接，填入模板原有方括号；日志仍保留完整原生动作池 |
| 动作解析 | 输出小写化，提取第一个action block；缺标签时采用返回文本末30字符。解析动作交给原生环境，不进行自造反馈或动作替换 |
| 预算 | 最多30次executor调用/原生交互，含非法命令；每次决策均提交环境，接收原生feedback |
| 达到生成上限 | 正常消费已返回文本，不自行清空payload或拒绝一个已完整输出的JSON判定；记录finish_reason及incomplete诊断 |
| 存储 | judge-positive保留完整raw observation/action轨迹；全量组保留全部轨迹并展示同一judge的success/failure标签 |
| 评分 | native verifier binary success，全部任务参与分母；judge只决定入库，不用native标签纠正judge |
| Streaming | 每轮空库；batch10共享冻结快照；整批完成后按预定任务顺序写入 |

JITMEM只公布普通curator模板，未单列全存储组的修改模板或label格式。本项目使用共同原文模板并附明确judge标签，不自行把原文的successful-experience措辞改为中性措辞。标签位置与轨迹序列化仍是公开实现选择。

## 指标口径与剩余不确定性

主指标按Table9报告native SR三轮均值与标准差。若报告效率，按Table4采用每任务所有executor调用的input/output tokens之和，单位K=1000，以及executor交互次数。curator/judge、全角色成本、解析异常、分任务类结果与混淆矩阵保留为诊断，不能替代论文主指标。API不返回usage时保留null，不补零；服务端隐藏推理token和不同模型tokenizer的差异仍需按API实际返回解释。

明确对齐的设置包括140任务、六类任务、每轮冷启动、batch10、描述BM25 top3、history3、30turns、executor temperature1/max4096、GPT类curator temperature1、native评分和executor同模型judge。

以下细节原文没有充分公开，当前实现不能保证只剩模型差异：

- 精确任务ID/order seeds、环境和数据版本、标准差的ddof定义。
- BM25具体实现、IDF、分词器、参数和tie-break；保留现有description-only BM25，并在manifest披露。
- curator评价输出预算、judge采样参数、空库是否仍调用curator、完整trajectory字符串、label格式和消息role封装。
- 作者实际动作解析fork和终止/恢复实现；本项目采用以上固定继承选择。

当前种子0/1/2控制任务顺序；std为样本标准差ddof=1。curator预算8192和judge继承executor设置仍为明确工程选择，不将Table7的训练rollout预算冒充作者披露的评价设置。参数不会按结果调整；修改后的成对实验重新运行到新目录，不拼接或重新打分旧轨迹来充当新结果。

当前配置只调用用户模型API，不训练模型或curator，仍不能代表原文RL-trained结果。
