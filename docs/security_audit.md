# 密钥安全审计

审计日期：2026-10-08，Asia/Shanghai。按用户最新要求，GitHub 仓库保持公开。本次检查没有调用外部模型 API，也不在报告或扫描输出中显示密钥值。

## 已发布内容与本地内容

对 Git 全部提交历史及当前文件使用 Gitleaks 8.30.1 默认规则和递归解码扫描；另从 GitHub 独立下载 bare clone，检查实际已发布的分支历史。对本地原始运行记录额外扫描约107 MB内容。这些检查没有发现真实凭证。

在本机读取当前模型 API 凭证和 GitHub 登录凭证，只进行内存内的逐字比对。检查 literal UTF-8、URL编码、JSON转义和base64形式；首轮覆盖964个项目/运行记录文件、35个Git blob（包含不可达对象）以及Git配置和HEAD日志，均无匹配。环境依赖、下载的Python、扫描器二进制和缓存不属于该文件扫描范围。

Gitleaks 唯一类型的误报是原实验 `api.py` 的公开源码SHA256，被通用API key规则识别。该值已与初始提交中的文件内容核对；`.gitleaks.toml`只允许这一准确、固定的哈希，保留全部默认检测规则。不忽略整个文件、目录或提交，也不忽略任意看起来像哈希的字符串。测试中已有凭证字符串是显式假数据；`.env.example`中的实际key字段为空。

`.env`及其变体、`.envrc`、本机配置、私钥文件、输出日志、虚拟环境和扫描器二进制均不被Git跟踪。版本管理中的评测结果不包含API endpoint、Authorization header或密钥。原实验源码哈希与结果保留，依据提交`3ef61bbb693d5b0c48183840008dbc399063765d`；安全修复后的新运行使用新源码哈希。

## 发现并修复的运行时路径

检查发现默认HTTP重定向可能将Authorization转发到其他origin。现在客户端拒绝所有重定向，3xx作为provider错误，不重试。测试在本机使用假key覆盖同origin和跨origin重定向，确认后续地址没有收到请求。

含换行的key可能使HTTP header校验异常回显完整Authorization值，随后被CLI或error日志记录。现在构造客户端时拒绝空白、控制字符或非ASCII凭证，错误只说明环境变量要求；测试检查异常文本不包含凭证。

`extra_body`原先可以嵌套`api_key`、Authorization、access-token或client-secret字段，随`asdict`进入manifest。现在配置构造阶段拒绝这些凭证字段，并测试大小写、连字符、嵌套字典和列表。认证通过环境变量，metadata只记录密钥变量名；URL中的用户名、密码、query和fragment继续被拒绝。

没有证据显示上述路径已导致实际密钥外泄。本次未重跑付费模型评测，也未修改历史结果或Git历史。

## 持续保护

本机启用`.githooks/pre-commit`，通过Gitleaks扫描暂存区，隐藏命中的secret值。负向验证使用随机生成、未关联服务的假key：即使工作区已清除，暂存区中的key仍使hook失败；清理并重新暂存后通过。扫描器缺失时hook也失败。

`.githooks/pre-push`额外扫描完整历史，防止将已经从当前文件删除、仍存在旧提交中的凭证推送到GitHub。两种hook都通过同一个固定配置的扫描入口，不输出secret值。

Gitleaks CI模板位于`configs/secret-scan.workflow.example.yml`，配置push/pull_request完整历史扫描。Action使用固定提交SHA，扫描器固定版本并校验官方SHA256；权限只有`contents: read`，checkout不持久化凭证。当前登录令牌没有`workflow`权限，因此模板未部署为工作流；本次没有申请扩大令牌权限。

仓库同时启用GitHub原生密钥扫描和推送保护，并核对服务端状态。其保护范围是GitHub支持的凭证类型；项目Gitleaks提供额外检测。[GitHub密钥扫描说明](https://docs.github.com/en/code-security/how-tos/secure-your-secrets/detect-secret-leaks/enable-secret-scanning)。

完整本地脱敏扫描记录位于`outputs/security_audit/`，不提交。新克隆的hook启用和手动扫描方式见[README](../README.md)。审计结论对应已检查的内容和当前配置；后续提交继续经过上述扫描。

修复后的完整软件验证：`105 passed, 108 subtests passed`；Ruff检查、21个文件格式检查、hook和安装脚本语法检查均通过。配置的凭证字段拒绝、非法header凭证拒绝、10种同/跨origin重定向与提交hook负向验证均使用假数据和本机环境。

## 存储消融发布检查（2026-10-09）

新增报告采用字段白名单，只公开实验协议、汇总指标、版本与源码指纹，不公开endpoint、凭证值或变量名、个人绝对路径、逐任务请求与响应。原始记录、checkpoint、本机配置与审计工具继续留在被Git忽略的 `outputs/` 或 `*.local.toml`；提交前扫描暂存区，推送前扫描完整历史。

本机内存中读取的2个不同凭证，经literal、URL编码、JSON转义和base64比对，覆盖1,879个项目/运行记录文件、63个Git blob（包括不可达对象）、Git配置与HEAD日志，均无匹配。检查不输出或持久化凭证值，不调用模型API。依赖、下载的Python、扫描器二进制和缓存仍不在这个文件扫描范围。

Gitleaks扫描公开文档无命中；另检查两组新原始记录共约126.75 MB。每组只命中一处manifest中的新 `api.py` 源码SHA256，均已与当前公开源文件完整内容独立核对，是指纹误报。未为此增加白名单，也未扩大任何目录/文件例外；公开汇总将源码指纹写为独立的 `file` / `sha256` 字段。

再次核对GitHub：仓库为public，原生secret scanning与push protection均enabled，开放密钥告警数量为0。公开汇总的独立复核、暂存区和完整历史扫描均通过；检查结论限于本次内容及上述范围。本地脱敏证据位于 `outputs/security_audit/storage_ablation_*`。

## 原文协议重评测发布检查（2026-10-09，paper-v1）

新paper-v1报告单独发布，保留旧报告，不公开完整prompt资产、原始请求、endpoint、凭证及其变量名或个人绝对路径。导出前要求完整840条评测和最终独立审计通过，核对冻结实现、原文资产、分析文件、40条新协议原生重放和200条中断前记录保全证据；公开JSON采用逐字段白名单，指标经独立复核与本地分析一致。

本轮发布前的逐字凭证检查在内存中读取2个不同凭证，覆盖2,862个项目/运行记录文件、95个Git blob（包括不可达对象）、Git配置和HEAD日志；literal UTF-8、URL编码、JSON转义和base64均无匹配。该范围包括新完整评测记录和新公开报告；依赖、下载的Python、扫描器二进制和缓存仍排除。检查不显示或保存凭证，不调用模型API，脱敏记录为本机 `outputs/security_audit/paper_v1_exact_credentials.json`。

Gitleaks对当前全部公开文档约230 KB的扫描通过，既有源码指纹的唯一固定例外未扩大。发布继续经过暂存区和完整历史扫描，推送后再次核对仓库public、GitHub原生secret scanning与push protection启用、开放密钥告警为0，并对包含新提交的Git对象再次做上述凭证比对。检查结论对应本次实际扫描范围，不将扫描通过解释为对未知凭证或未来提交的保证。

## 仅curator模型切换发布检查（2026-10-09，paper-v1）

新模型对照报告在完整新420条、配对840条与严格独立最终审计通过后才生成。公开MD/JSON采用字段白名单，逐项核对本地分析指标和审计证据，不发布endpoint、密钥/变量名、个人绝对路径、原始请求、完整prompt或逐任务配对记录。旧报告保留，所有新原始记录与本机配置仍被Git忽略。

首次发布前的逐字检查在内存中读取2个不同凭证，覆盖3,332个项目/运行记录文件、106个Git blob（包括不可达对象）、Git配置及HEAD日志；literal UTF-8、URL编码、JSON转义和base64均无匹配，不显示或持久化凭证。扫描范围包含新完整420条记录和公开报告；依赖、下载的Python、扫描器二进制和缓存仍排除。脱敏证据保存在本机 `outputs/security_audit/curator_gpt61_exact_credentials.json`。

Gitleaks对新增后的公开文档扫描通过。另扫描新组全部原始评测记录约66.20 MB，仅有一处manifest中的`api.py`源码SHA256命中通用API-key规则，已逐行与当前源码完整内容hash独立核对，确认是源码指纹。没有新增白名单或目录例外；公开汇总继续使用独立`file`/`sha256`字段。暂存区与全部提交历史继续由既有hook检查，推送后再比对含新提交的Git对象。

本轮核对GitHub仓库仍为public，原生secret scanning与push protection均enabled，开放密钥告警数量为0。上述检查均不调用模型API，结论限于已检查的内容和当前凭证；未知凭证仍由模式扫描与GitHub保护补充检查。

## gpt-6.1-sol curator全量组发布检查（2026-10-09，paper-v1）

新全量组420条与历史过滤420条的完整配对审计通过后，单独生成存储对照公开报告，保留此前所有结果。公开MD/JSON只含逐字段白名单汇总与指纹，不含endpoint、密钥/变量名、个人绝对路径、原始请求、完整prompt或逐任务配对内容；本机配置、全量记录和审计工具仍被Git忽略。

首次发布前的内存凭证比对读取2个不同凭证，覆盖3,802个项目/运行记录文件、116个Git blob（包括不可达对象）、Git配置及HEAD日志。literal UTF-8、URL编码、JSON转义和base64均无匹配；不显示或保存凭证，不调用模型API。范围包括新完整420条全量记录和公开报告，依赖、下载的Python、扫描器二进制和缓存仍排除。前置脱敏证据单独保留为本机 `outputs/security_audit/storage_gpt61_exact_credentials_prepublication.json`，发布后再次比对新增Git对象。

Gitleaks对新增后的公开文档扫描通过；另扫描新全量原始记录约71.78 MB，唯一命中为manifest中的`api.py`源码SHA256，已逐行与当前源码完整内容hash核对为指纹误报。没有扩大固定白名单、文件或目录例外。提交和推送仍经过暂存区与全部历史扫描，GitHub仓库保持public，原生secret scanning与push protection均enabled，开放密钥告警为0。检查结论限于本次已扫描内容及当前凭证，不将模式扫描解释为未知凭证的完全保证。
