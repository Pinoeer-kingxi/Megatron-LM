# GDN GPU 验收脚本

适用于本机现有的 A6000、容器、依赖和本地 Qwen 权重。这里只整理验收入口，不修改三个 PR 的代码。脚本于 2026-10-15 完成 CPU 检查，2026-10-16 开始实际 GPU 验收；GPU 结果与限制见文末。

当前状态：本次约定的本地验收已完成，汇总为 `passed_current_sources`。共享调优缓存的竞争已复现并在 Qwen 验收入口修复；14 项 CPU 自测、两卡及四卡空缓存三组训练对照均通过。验收使用上表固定提交，不代表上游 GPU CI 已获批准或已执行。

## 固定版本

| 源码 | 提交号 | 用途 |
| --- | --- | --- |
| 上游基线 | `9131593673d037d360d252904f9198d36bc0c246` | 原始 atomic gate 性能基线 |
| #8243 | `af9569270f476dc201484296315a9c30dcaa3feb` | 共享 offload manager |
| #8244 | `9d54b8729c9277f1e032110c5e1b149e9654bcc5` | gate backward replay 和性能 |
| #7908 | `8a2d19894288a8eb890dfdf20c9d54683035de80` | 包含两个前置修复的 GDN 集成 |

以 `source-after.json` 为机器可读版本记录。默认源码目录为 `/home/songbinbin/gdn-offload-artifacts/split-pr-20261015/{base,shared,gate,feature}`。运行前和每个任务边界检查提交号及工作区；版本不符或有未提交文件即停止，不自动 checkout、rebase 或拉取代码。

四卡 Qwen 使用 Megatron-Bridge `c860f8a5bc5fddd78690f32baa0b8696774308b4`，以及同目录 `bridge-compatibility.patch` 中已有的 canonical import 兼容补丁；预检逐字核对该差异。补丁 SHA256 为 `d485a7b8fa3566c78ff602a34582778869c28ba7bcb339cfb51edba740ce3739`。

## 运行

先预留 GPU。下列 `0` 和 `0,1,2,3` 仅为卡号示例，也可填完整 GPU UUID。脚本按 compute process 检查占用，发现已有进程就退出；它不预留 GPU，也不停止其他人的进程。运行期间需保持设备独占。

```bash
cd /home/songbinbin/gdn-offload-artifacts/gpu-acceptance-20261015

# 只检查版本并打印命令；不创建容器、不初始化 CUDA、不创建结果目录。
./run.sh single --gpus 0 --output ../gpu-acceptance-results-single --dry-run
./run.sh four --gpus 0,1,2,3 --output ../gpu-acceptance-results-four --dry-run

# 确认卡已预留且空闲后执行。
./run.sh single --gpus 0 --output ../gpu-acceptance-results-single
./run.sh four --gpus 0,1,2,3 --output ../gpu-acceptance-results-four
```

两个阶段可分开运行，完整验收需要两份结果。每次必须使用新的 `--output` 目录，失败重跑也不覆盖旧日志。可先看 `./run.sh --help`；默认每个任务超时 3600 秒，`--timeout` 可调整。四卡默认使用：

```text
--weights /home/songbinbin/gdn-offload-artifacts/models/Qwen3.5-0.8B
--data /home/songbinbin/gdn-offload-artifacts/training-data/ks-item-sft.jsonl
```

可通过同名参数指定同一型号的其他本地副本。验收锁定完整 24 层文本解码器，不用于任意 Qwen 型号；脚本不下载模型或安装依赖。

如果 `shared-groups` 和 `shared-models` 已有同一提交、同一组 GPU 的独立通过记录，可用 `four --training-only` 只补原生训练和全部 Qwen 对照。manifest 和离线校验输出会明确列出 `omitted_jobs`；这两项不会在新运行中被计为通过，完整覆盖需另行核对原始 rank 报告。该选项只适用于四卡阶段。

## 验收内容

| 阶段 | 任务 | 检查内容 |
| --- | --- | --- |
| 单卡 | `shared-races`，4 项 | 延迟 D2H、增长的 microbatch、fraction=0/1、复用 pinned buffer 时的 H2D/D2H 同步 |
| 单卡 | `gate-replay`，16 项 | BF16/FP32、尾块、非连续上游梯度、独立 stream、竞争负载下逐位重放及 Torch 数值参考 |
| 单卡 | `gdn-scopes`，73 项 | 配置边界、saved-tensor scope、anchor/alias、prefetch、norm recomputation 与 fusion 等现有回归 |
| 单卡 | `gate-bench-*`，24 个进程 | 8 个形状，每个形状 3 次独立进程，比较原始与修复后的完整 gate backward wrapper |
| 四卡 | `shared-groups`，每 rank 16 项 | 共享 manager 的其余非模型回归，包括 warmup 和重复 storage 统计 |
| 四卡 | `shared-models`，每 rank 24 项 | 现有 GPT/MoE、EP A2A overlap、CUDA graph 正确性和显存检查 |
| 四卡 | `native-training`，每 rank 96 项 | 现有缩小规模完整模型训练矩阵，包括 TP/SP/CP/PP/VPP、fraction、norm 与 fusion 组合 |
| 四卡 | `qwen-*`，8 个独立进程组 | 本地 Qwen3.5-0.8B 文本模型，TP2+SP+PP2，fusion × norm recomputation × offload 开关 |

测试数量按每个 rank 记录，不将四个 rank 当成四倍独立测试。六组 pytest 的预期数量已在当前提交上以 `--collect-only` 核对，收集不等于执行通过。

Qwen 使用 sequence length 128、4 个 microbatch、1 次 warmup 和 2 次后续 optimizer step。每个配置臂单独 torchrun，共保存 32 个 rank JSON、96 份 rank/step 状态，进行 48 次 baseline/offload 配对比较。逐位比较已记录的 loss、全部参数梯度、更新后权重、Adam 与 master 参数状态；要求有效梯度、实际选中的 offload 字节，以及现有训练入口在 manager reset 前执行的 pool 释放检查。它覆盖完整文本训练路径，不训练视觉模块，也不据此宣称长序列性能通过。

gate 性能形状为 `batch={1,4} × sequence={2048,4096} × value_heads={16,32}`，head dimension 128、BF16；使用生产调用中的预分配 projection-gradient buffer。每个进程先校验数值并预热 20 次，再进行 3 组交替顺序的配对测量，每组 100 次调用。每个形状汇总 9 对样本的耗时变化和额外 peak allocated bytes；计时不包含编译，offload 关闭。

## 结果与判定

结果目录保存：

- `manifest.json`：准确提交号、GPU UUID/驱动、容器命令、每项命令、退出码、测试统计及最终状态。
- `environment.json`：实际 GPU 和 PyTorch/CUDA/TE/FLA 等依赖版本；Qwen 另在 manifest 记录数据与权重文件 SHA256。
- `<任务>/launcher.log` 和 `<任务>/ranks/**/{stdout,stderr}.log`：启动日志及各 rank 原始输出。
- `<pytest 任务>/reports/rank-*.json`：每 rank 的收集数量、执行结果、失败和 skip 原因。
- `<Qwen 任务>/result.rank*.json`、`<性能任务>/result.json`：原始校验及测量结果。
- `scripts/`：本次使用的脚本快照；worker 和 benchmark 实际执行该快照。
- `SHA256SUMS`：结果文件校验和，排除 `cache/` 下的编译缓存。

脚本仅清理本次创建且名称唯一的容器。测试失败、超时、Ctrl-C、SIGTERM 或 SIGHUP 均进入清理流程并保留失败证据。SIGKILL 或宿主机故障无法执行清理；必要时按 manifest 的 `container_id` 清理该容器。

可以在无 GPU 的环境复核导出的结果：

```bash
/usr/bin/python3 -S verify_results.py ../gpu-acceptance-results-single
/usr/bin/python3 -S verify_results.py ../gpu-acceptance-results-four

# 也可直接使用结果包中的校验器，无需访问原工作区。
/usr/bin/python3 -S ../gpu-acceptance-results-four/scripts/verify_results.py ../gpu-acceptance-results-four
```

状态和退出码：

- `automated_checks_passed`，退出 0：全部计划任务完成，规定的数值/状态校验通过。
- `incomplete`，退出非 0：既有 `shared-models` 测试存在 skip，保留原因并继续其余任务，不能算全部通过。
- `failed`，退出非 0：其他 skip、缺少 rank、数量不符、数值不一致、命令失败、超时或清理失败。

**退出 0 不等于性能或合并批准。** 单卡结果的 `performance_review` 保持 `pending`，需要审阅 8 个形状的实测差异，出现明显退化时先定位。这里不擅自设定性能阈值。Qwen `--check-state` 采集状态会干扰计时，不作为端到端性能证据。

历史显存/运行时间数据保留原提交归属：unfused 完整 Qwen 数据对应 `4e18562c6bcf894795d23378a8f1699c79e4b797`，fixed-fused 完整 Qwen 数据对应 `2f1ee4f2e3dd6d909d08981338f5879632131139`。这些是 offload 开关对照，不是本次 gate kernel 修改前后的性能对照。本脚本不会将旧结果改标为当前提交通过。已有 post-rebase CPU 证据见 [固定版本的公开记录](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/251ae748bd5566fe57c15ba4e8431dfbd5483675/rebase-20261015)。

## 本机环境与本次检查

运行器沿用本机已验证的手工设备及驱动库映射，固定镜像 ID 为 `sha256:061e8fda67a4dd1c1c635129d4ba0442b7a093cc6fd737423e9589eb6be22034`，复用 `/home/songbinbin/.venvs/megatron-gdn-offload` 和现有依赖 overlay。本机没有 NVIDIA Docker runtime，因此没有使用 `--gpus all`。源码和依赖只读挂载，仅结果目录可写。单卡 `CUDA_DEVICE_MAX_CONNECTIONS=8`，四卡为 `1`；这是针对当前机器的脚本包，迁往其他机器需核对这些路径和容器环境。

脚本准备阶段（2026-10-15）做了以下 CPU 验证，记录在 `validation/`：

- 准备阶段 11 项脚本自测（本次修复后扩至 14 项）：缺 rank、teardown 失败、skip、脏源码、GPU 已占用、Qwen 状态差异/非有限梯度/空转移、校验和及未完成任务等失败检查。
- 6 组现有测试仅收集，不执行 CUDA 用例。
- 单卡和四卡命令 dry-run，不创建 GPU 容器。
- 未映射 GPU 的独立容器 smoke test：`torch.cuda.device_count()==0`、依赖导入、只读源码访问、Qwen 全 24 层配置与数据预检。
- pytest report 插件的真实 CPU smoke test：故意产生 skip 和 teardown 失败，确认报告保留原因且校验器拒绝失败结果；`validation/pytest-report-smoke/broken/` 中的失败是该检查的预期输入。
- Shell 语法、Python 编译、isort、Black 和 Ruff 检查。

CPU 自测入口：

```bash
/usr/bin/python3 -S -m unittest discover -s . -p test_acceptance.py -v
```

## 2026-10-16 GPU 运行记录

首轮四卡使用物理卡 `0,1,3,6`，最后的冷缓存隔离补测使用 `1,2,3,6`；每份 manifest 按实际 GPU UUID 记录。源码仍是上表四个固定提交，未修改 PR 代码。原始输出保存在相邻目录：

| 运行目录 | 结果 |
| --- | --- |
| `gpu-results-20261016-single-3` | 4 项传输竞态、16 项 replay 通过；按要求转向四卡，scope 阶段被主动中断 |
| `gpu-results-20261016-four-0136` | 四个 rank 各 16 项 shared manager、24 项既有模型回归全部通过、零 skip；另一位用户在 6 卡启动进程，任务边界检查停止后续运行 |
| `gpu-results-20261016-four-training-0136` | 四个 rank 各 96 项原生训练全部通过、零 skip；8 个 Qwen 配置臂全部执行完成，原始矩阵因首组第 2 步不一致而判失败 |
| `gpu-results-20261016-qwen-baseline-repeat-0136` | 同一组四卡、相同权重/数据/参数，复制已生成的 Triton cache，在新容器中重复两次关闭 offload 的基线；四个 rank 的两次完整状态彼此一致，并与原 offload 臂一致 |
| `gpu-results-20261016-single-0` | 4+16+73 项测试、24 个 gate benchmark 命令全部退出 0；汇总器误将 CUDA 的裸 UUID 与 SMI 的 `GPU-` 前缀格式判为不同设备，原始 manifest 保留失败；修复边界比较后，原始数据离线复核通过 |
| `gpu-results-20261016-qwen-matched-pair-0136` | 新容器中直接运行相同已填充缓存的 baseline/offload；四 rank、三步完整状态逐位一致 |
| `gpu-results-20261016-qwen-cold-isolated-1236` | 物理卡 1、2、3、6，TP2+SP+PP2；从空缓存启动，baseline / baseline / offload 的四 rank、三步完整训练状态逐位一致；48 项实际调优选择与 rank 自有缓存一致，后两组缓存未改变，pool reset 前检查通过 |
| `gpu-results-20261016-qwen-cold-isolated-16` | 物理卡 1、6，TP2+SP、PP1；从空缓存启动，baseline / baseline / offload 三组的两 rank、三步完整训练状态逐位一致；24 项实际调优选择均与 rank 自有缓存一致，后两组缓存未改变 |
| `gpu-results-20261016-qwen-cold-isolated-13` | 两卡 TP2+SP、PP1 的空缓存隔离对照；两次基线完整状态逐位一致，随后 3 卡出现其他用户进程，第三组未启动，保持失败 |
| `gpu-results-20261016-qwen-cold-shared-0136` | 新容器、空共享缓存，运行 baseline / baseline / offload；前两组基线不同，后两组完整状态一致。最后一组期间 0 卡被其他用户占用，原始运行保持失败，仅用于诊断 |

Qwen 原始矩阵的 fusion/norm 四组中，三组直接逐位通过；`fusion=0, norm=0` 的首次基线在第 0、1 步与 offload 完全一致，第 2 步不同。两次关闭 offload 的重复基线均与原 offload 臂逐位一致，且均不同于首次基线。数据 SHA256 没有变化。

这证明基线本身存在不可复现的结果，不能把首次差异直接归因于 offload。随后用空共享缓存复现了调优缓存竞争，详情如下。**原始失败矩阵保持失败状态，不将其改写为无条件通过**。使用明确记录的重复基线引用，四组可完成 48 次 rank/step 的完整状态配对检查；该检查与原始矩阵结果分别报告。

离线核对脚本为相邻的 `validate-gpu-results-20261016.py`，输出 `gpu-validation-20261016.json`，包含原始 manifest 校验和、每 rank 数量、重复基线结果、引用映射及上述限制。它通过临时目录映射复用既有 verifier，不修改任何原始 JSON。


## 首次基线漂移：共享调优缓存的诊断与修复

FLA 0.5.1 默认启用 Triton 调优结果持久化。本机 Triton 3.6.0 的缓存键包含后端架构、kernel、调优键和候选配置，不包含 GPU UUID 或 rank。冷启动时，多个 rank 可以分别选出不同配置；磁盘文件最后一次写入的配置，并不一定等于各 rank 在本进程中继续使用的配置。后续新进程读取这个共享文件时，算术配置就可能发生变化。

在 `qwen-cold-shared-0136` 的首组 rank 日志中，六项实际选中的配置与首组结束后的磁盘赢家不同，涉及卷积、cumsum 和 GDN backward 等 kernel。两次关闭 offload 的基线在第 0 步的梯度范数分别为 `55.65458679199219`、`55.65776824951172`；第二次基线与 offload 的三步完整状态一致。该实验复现了共享缓存使比较两侧调优条件不同的问题。原始首次失败运行没有记录这些选择，无法事后确定当时具体是哪项配置改变。

修复只调整 Qwen 的验收入口；现有 pytest 调用保持原样：`rank_env.py` 在启动目标 Python 进程前设置 `TRITON_CACHE_DIR=<本次结果>/cache/triton/rank-<RANK>`。各 rank 独占缓存，后续 baseline/offload 复用同一 rank 的选择；不修改 PR、FLA 或 Triton kernel，不放宽状态比较。每个 Qwen 臂另保存 `autotune-cache.json`，记录调优键、选中配置和缓存文件校验和，便于复核比较条件。

新增 CPU 回归通过真实子进程检查环境在目标启动前设置、argv 保持原值、不同 rank 使用不同目录，并检查配置证据保留实际赢家。当前 14 项 CPU 自测通过。两卡 GPU 冷启动隔离对照已完成：完整 24 层文本模型、TP2+SP、PP1，三个独立进程组的全部记录状态一致，每 rank 的选择与磁盘记录一致。随后在四卡 1、2、3、6 上完成独立的 TP2+SP+PP2 同项复验，四 rank、三组、三步的完整状态一致，48 项实际选择与各自缓存一致。两卡和四卡有各自独立的原始记录。

该修复保证对照两侧使用同一组调优配置，不承诺每次完全独立的空缓存调优都产生逐位相同的训练结果。不同合法 kernel 配置可能改变浮点归约顺序，这与 offload 是否改变同一输入的梯度是两个独立条件。

诊断复核：

```bash
/usr/bin/python3 -S ../analyze-qwen-cache-20261016.py
```

输出为 `../qwen-cache-diagnosis-20261016.json`。原始失败、其他用户占用、两卡诊断与四卡验收分别保留，不互相改标。

单卡 gate 微基准的 8 个形状，每个形状 9 对样本，当前修复相对原始 atomic wrapper 的耗时中位数下降约 5.9%–8.7%；额外 peak allocated 增加约 2–32 KiB。数值参考检查均通过。它仅说明已测 gate wrapper 没有明显回归，不能代替完整模型的 offload 显存/运行时间测量；历史端到端数据仍保留前述原提交号。


## 四卡补测与完整证据包

2026-10-16 在 1、2、3、6 卡完成 `fusion=0, norm=0` 首组冷缓存对照：baseline、重复 baseline、offload。三个独立进程组均退出 0，全部状态逐位相同，容器清理退出 0，四卡已释放。其他三组 Qwen 及每 rank 96 项原生训练来自同一提交的既有执行记录，不将它们表述为本次重新运行。

本次实际执行命令如下；复现时须使用新的输出目录：

```bash
/usr/bin/python3 -S ../qwen-baseline-repeat-20261016.py --isolated --gpus 1,2,3,6 \
  --output ../gpu-results-20261016-qwen-cold-isolated-1236
/usr/bin/python3 -S ../analyze-qwen-cache-20261016.py
/usr/bin/python3 -S ../validate-gpu-results-20261016.py
```

诊断入口同样拒绝覆盖旧输出；失败后可用 `--output` 指定新目录，并在汇总脚本中明确引用新运行。完整汇总在四卡冷缓存对照未通过时返回 1；两卡通过、历史热缓存对照与四卡验收状态分别记录。本次汇总已退出 0：首组的两侧均引用新四卡直接对照，其他三组保留原矩阵引用，共完成 48 次 rank/step 配对、96 份记录状态的校验。原始失败矩阵不改写。每份运行的 GPU UUID 在 `gpus_by_run` 和原始 manifest 中分别列出。

`gpu-evidence-20261016.tar.gz` 汇集本次脚本包、原始成功/失败运行、各 rank 日志、准确源码提交号、命令及校验输出；排除编译缓存、模型权重、训练数据和 Python 字节码。解压到同一目录后，可运行上述分析/汇总脚本复核；复核不需要 GPU。本地证据包另有 [公开副本](https://github.com/Pinoeer-kingxi/Megatron-LM/tree/evidence/gdn-offload-split-20261015/gpu-20261016)，其中保留实际使用的脚本快照及各 rank 原始日志。
