# VISTA-SPM

[English](README.md) · 中文

**让多模态大模型看见、记住、回看并测量扫描隧道显微镜给它的每一幅图。**

VISTA-SPM 是一个以图像为主的视觉 harness，供前沿多模态大模型操作扫描隧道显微镜（STM）。它把何恺明组的 VISTA（Han, Hu, Qiu, Wu, He，*VISTA: A Visual Harness for Reasoning in an Interactive World*，[arXiv 2610.02200](https://arxiv.org/abs/2610.02200)）从 ARC-AGI-3 游戏移植到 [STM-Bench](https://github.com/HamsterPark/STM-Bench) 的论文复现场景。STM-Bench 是一台模拟的 STM，配有按证据判分的裁判。

它要回答的问题是：应该让模型自己看像素，并给它每一帧的无损记忆、一把放大镜和精确读数；还是先用视觉模型把每幅图转成文字？

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green) ![Snapshot](https://img.shields.io/badge/snapshot-2026--10--13-lightgrey)

**状态：精简的公开演示版。**
- 维护中的 harness 在 MAST 里。MAST 是实验室的 STM 自动化工具包，本身是私有仓库；[MAST-public](https://github.com/HamsterPark/MAST-public) 是它的有限公开版。
- [dsh-spm](https://github.com/HamsterPark/dsh-spm) 把 harness 移植成了 TypeScript。
- 本仓库保留一份能独立运行的快照：harness、282 项测试、静态感知基准和第一版消融结果。
- 所有结果都来自模拟器，没有在真实仪器上跑过。
- 视觉记忆、测量工具和静态基准只需要 STM-Bench；闭环回合还需要 MAST，它的公开版 MAST-public 可以替代（[详见下文](#不装-mast-能跑什么)）。

## 为什么要给显微镜套一个 ARC-AGI-3 的 harness

### STM 简介

扫描隧道显微镜让一根尖锐的金属针尖停在导电表面上方几埃处，一边扫描一边记录隧穿电流，图像能分辨出单个原子。同一根针尖还能测谱，也能把单个原子在表面上拖动。仪器给出的一切都要经过这根针尖，而针尖顶端的原子结构谁也看不见，用着用着还会变。[MAST-public 的 README](https://github.com/HamsterPark/MAST-public/blob/main/README.zh.md) 对 STM、以及它为什么难以自主操作，讲得更详细。

### 同一类问题

ARC-AGI-3 把智能体放进规则和目标都隐藏的游戏里。智能体要自己动手，看画面怎么变，从帧里推出规则；评分看它比人少用多少步。操作 STM 是同一类问题，只是多了物理：

| | ARC-AGI-3 游戏 | STM 实验（STM-Bench 的一个场景） |
|---|---|---|
| 智能体看到什么 | 每次动作后一帧 64×64、16 色的画面 | 每次仪器动作后的扫描帧（形貌、电流，正反扫）和谱 |
| 什么是隐藏的 | 规则与目标 | 针尖状态、漂移、样品，以及按种子抽取的场景数值 |
| 怎么行动 | 每次一个游戏动作 | 每次一个仪器动作：扫描、设偏压、移针、脉冲、轻戳、测谱、拖原子 |
| 动作的代价 | 用的动作越少，分越高 | 仪器时间和动作数；一次针尖脉冲、一次拖原子都无法撤回 |
| 证据在哪里 | 动作前后画面的变化 | 帧与帧的差别：原子动了没有？针尖变了没有？样品漂了没有？ |
| 怎样算成功 | 通关 | 复现论文的结论，并给出判分要求的测量 |

两者的关键证据往往都来自跨时间的比较：动作前的一帧对比动作后的一帧。VISTA 的帧档案和放大镜给模型的正是这种能力。

在[调研笔记](docs/notes/2026-10-09-research-and-plan.zh-CN.md)梳理的工作中，前沿大模型在真实 STM 上的智能体报告只有 quailbot（[arXiv 2609.27302](https://arxiv.org/abs/2609.27302)）一篇。它的失败案例都是人对比动作前后的帧诊断出来的。

VISTA 没有训练任何视觉模型，增益全部来自 harness。它在 ARC-AGI-3 的 25 个公开游戏上做的消融里：
- 把官方的文本网格换成图像，RHAE 从 13.33 升到 47.32；
- 再经过几步，其中笔记 +4.2、无损记忆加 `inspect` +24.1、`read_pixels` +4.9，最后到 99.00。

### 需要改什么

STM 的帧不是 ARC 的帧：
- 里面是带物理单位的连续高度和电流，有噪声、漂移和针尖造成的模糊；
- 要紧的特征可能只是 512 像素的帧里几个像素宽的一个原子；
- 谱是曲线，不是图片。

所以移植保留了 VISTA 的全套工具（图像、笔记、全部帧的档案、放大镜 `inspect`、精确读数和方格纸），改造其中两样，再加上物理学家会伸手去拿的东西：

- 精确读数 `read_values` 给出 pm、pA 或 V 的数值，而不是调色板编号；
- 方格纸改用纳米单位，格距 1、2 或 5 × 10ⁿ nm，代替 ARC 的单元格线；
- 五个只测量、不下判断的测量工具：`line_profile`、`fft_peaks`、`detect_blobs`、`diff` 和 `stats`；
- 谱按需绘制，也可以按数值读取；
- 仪器时钟只在动作执行时走，模型思考不消耗仪器时间。

MAST 原来的智能体通过微调过的视觉模型 VIGIL 看 STM 图像，VIGIL 把每幅图转成文字。本仓库在 STM-Bench 模拟器上用 VISTA 的路线与之对照，并保留 VIGIL 文本作为对照臂。

## 第一版结果（2026-10-10）

实验设置：

- **模型：** claude-opus-5-5，effort high。
- **动作：** V0 原语动作集。
- **场景：** P1（Au(111) 鱼骨重构）、P4（在 Cu(111) 上搬运一个 Fe 原子）、P5（qPlus 力谱与 Sader–Jarvis 反演）。
- **种子：** 0–2。
- **图像：** 当时还没有方格纸，它是 2026-10-12 加的。
- **判分：** STM-Bench 官方判分。

| 臂 | 观测与工具 | 结论通过比例 | 完整复现 |
|---|---|---|---|
| mem | 图像、笔记、任意帧回看放大 | 0.83 | 6/9 |
| full | mem 加精确读值和五个测量工具 | 0.72 | 5/9 |
| img | 只有最新一幅图像 | 0.43 | 2/9 |
| vigil-text | VIGIL 文本代替图像（谱为数值表）、笔记 | 0.25 | 1/9 |

下表把两个臂在同一场景、同一种子上的回合配成一组比较：

| 比较 | 胜 | 平 | 负 |
|---|---|---|---|
| mem 对 img | 5 | 4 | 0 |
| mem 对 vigil-text | 6 | 3 | 0 |
| full 对 mem | 0 | 8 | 1 |

每格只有 3 个种子，区分不了 full 和 mem。另外，每一步都不止改了一样东西：

- mem 比 img 多了三样：笔记、帧档案和 `inspect`。
- full 比 mem 多了两类工具：精确读数和五个测量工具。

两臂在回合里用不同的工具做同一件事：mem 平均每回合放大 7.8 次，full 平均测量 15 次、只放大 2 次。

**结论通过比例没有做机会校正。** 2026-10-12 对 STM-Bench 的审计发现，在证据要求满足的前提下，只报固定常数就能通过这些比例的结论：

- P1 约 0.26；
- P5 约 0.41；
- P4 为 0。

同一组常数的完整复现率：P1 约 2–3%，P5 为 13%，P4 为 0。读上表时要对照这些地板，并结合完整复现数一起看。

**静态感知基准（STM-BabyVision），37 道难题：**

| 答题方式 | 答对 |
|---|---|
| Claude，全工具 | 31 |
| Claude，只看图（没有任何工具） | 23 |
| 经典算法基线 | 30 |

两个 Claude 臂的差距几乎全部来自鱼骨条纹周期，这类题需要精确测量。

**这些数字是什么。** 它们都是存档记录：
- **来源。** 这些运行用的是 MAST 和 2026-10-09/10 的 STM-Bench，早于它公开的规则 v1 和之后的模拟器改动。冻结代码的文件哈希见 [results/m1/code_manifest.json](results/m1/code_manifest.json)。只用公开组件无法重跑这些回合。
- **标签能复现什么。** 标签 `m1-surface` 加 `--grid off` 能复现这些回合看到的系统提示、工具界面和工具行为。2026-10-13 修参考实现时又改了部分工具（[说明](docs/notes/2026-10-13-reference-fixes.zh-CN.md)）。
- **细节在哪里。** 细节与局限见 [docs/notes/2026-10-10-first-results.zh-CN.md](docs/notes/2026-10-10-first-results.zh-CN.md)，表格在 [results/](results/)。

## 工作方式

```
python -m vista_spm run
 ├─ STM-Bench RuntimeHost：MAST CoreRuntime 跑在 stmsim 模拟器上
 ├─ VistaHost：按臂组装工具、帧档案、笔记、仪器时钟、预算、判分账本
 ├─ RpcServer（回环地址，带令牌）
 └─ ClaudeCodeRunner：claude -p 分段运行，内置工具全部关闭，stream-json
       claude ── stdio MCP ──► mcp_bridge.py（只用标准库）──► RpcServer ──► VistaHost
```

- **观测：** 原始帧，按整数倍最近邻缩放。标签和数字放在图像旁边的文字里。
- **方格纸：** 扫描图上唯一画的东西，相当于 VISTA 在 ARC 帧上画的格线。
  - 它是一层淡琥珀色的纳米方格，格距取 1、2 或 5 × 10ⁿ nm，从整帧左上角起算。
  - 每个视野最多 16 格，所以放大后格子会变细。
  - 格距 `grid_nm` 写在每张图的标签里，`--grid off` 可以关掉。
- **视觉记忆：** 每一帧、每个通道、每个扫描方向和每次中途快照都无损归档，各有固定 id。
  - `inspect` 可以对任一已归档区域、通道或去背景方式裁剪放大。
  - `read_values` 返回精确数值。
- **测量：** `line_profile`、`fft_peaks`、`detect_blobs`、`diff`、`stats`。工具只测量，不下判断。
- **笔记：** GUIDE.md 和 WORKING.md。Claude Code 压缩上下文时，运行器按 VISTA 的做法通过 WORKING.md 交接。
- **仪器：** `act` 通过 MAST 技能执行一个 V0 原语：扫描、设偏压与设定点、移动针尖、脉冲、轻戳、STS、Δf(z)、PLL、搬原子、漂移补偿、等待、退针和进针。仪器时钟只在动作执行时走。
- **报告：** `report_result` 和 `report_tip_state` 交给 STM-Bench 判分，判分要求数值与采集证据都满足。

各臂的工具：

| 臂 | 观测 | act 和两个报告工具之外的工具 |
|---|---|---|
| full | 图像 | 历史、笔记、inspect、read_values、测量工具 |
| read | 图像 | 历史、笔记、inspect、read_values（VISTA 的工具全集） |
| mem | 图像 | 历史、笔记、inspect |
| notes | 图像 | 历史、笔记 |
| img | 图像 | 无 |
| text | 数值文本网格 | 历史、笔记、read_values |
| vigil-text | VIGIL 文本；谱为数值表 | 历史、笔记 |
| advisor | 图像 | full 的全部工具，另可按需调用 MAST 检测器 |

设计规格见 [docs/DESIGN.md](docs/DESIGN.md)，静态基准说明见 [docs/staticbench.md](docs/staticbench.md)。

## 工程实现

| 问题 | 做法 | 位置 |
|---|---|---|
| 只让模型通过 harness 行动 | Claude Code 关闭全部内置工具。它唯一的 MCP 服务器是一个只用标准库的桥，接到带令牌校验的回环 RPC 服务器。 | [mcp_bridge.py](vista_spm/mcp_bridge.py)、[rpc.py](vista_spm/rpc.py)、[claude_runner.py](vista_spm/claude_runner.py) |
| 一动一观 | 上一个动作的结果送到模型之前，`act` 一律拒绝执行。模拟器时钟只在动作执行时走。 | [host.py](vista_spm/host.py) |
| 无损视觉记忆 | 每一帧、通道、方向和中途快照都带固定 id 归档。任何区域都能重新裁剪放大，或按精确值读取。像素坐标与扫描坐标精确互换，含亚像素。 | [archive.py](vista_spm/archive.py)、[viewtools.py](vista_spm/viewtools.py)、[frames.py](vista_spm/frames.py) |
| 只测量、不下判断 | 带台阶候选的剖面、FFT 峰、椭圆高斯斑点拟合、亚像素帧配准和统计。回复里只有数字，没有"双针尖"这类判断。 | [measure.py](vista_spm/measure.py) |
| 把模型看到的界面当作实验参数 | 每个臂的系统提示和工具 schema 都用哈希钉住，覆盖有无方格纸、有无 qPlus。第一轮矩阵可以从标签复现。 | [test_surface.py](tests/test_surface.py) |
| 真值不进观测 | 工具回复、图像和动作描述不得带出隐藏的场景数值、容差或只给判分用的数据。真值只留在账本里，供判分和回放使用。 | [AGENTS.md](AGENTS.md)（不变量 1） |
| 长回合 | 采用 VISTA 的压缩协议：Claude Code 压缩上下文之前，模型先写 WORKING.md，再由新的分段根据笔记和动作历史接着跑。 | [compact_hook.py](vista_spm/compact_hook.py)、[claude_runner.py](vista_spm/claude_runner.py) |
| 计费安全 | 运行器从 Claude Code 子进程中去掉 API 凭据。会话初始化时如果报告的凭据来源是 API key，立即中止。 | [claude_runner.py](vista_spm/claude_runner.py) |
| 可审计的消融 | 冻结代码快照并附哈希清单；按种子顺序运行矩阵，遇到用量上限就暂停；找出失败的回合重跑，已完成的回合可以离线重判。 | [scripts/](scripts/) |

## 不装 MAST 能跑什么

| 只需 STM-Bench（Python 3.13、STM-Bench 84fb385 或更新、PyPI 上的包） | 需要 MAST 或 MAST-public |
|---|---|
| 测试：282 项通过，隔离的端到端回合测试跳过 | 闭环回合（`run`）：`act` 要靠 MAST 的运行时和技能在模拟器上执行 |
| `tools`：每个臂的系统提示和工具 schema | `advisor` 和 `vigil-text` 两个臂：要用 MAST 的检测器和 VIGIL 视觉模型 |
| 视觉记忆和测量工具，作用于 `.sxm` / `.dat` 文件 | 静态基准里的原子分辨题，它们的真值来自 MAST 的判定 |
| 静态基准：`generate`、经典基线（`answer`）、`score`、`report`，以及用 Claude Code 的 `qa` | |
| 为已记录的回合生成 `replay` 回放页面 | |

**用 MAST-public 代替 MAST。** 2026-10-13 在装好 `MASTv2/requirements-v2.txt` 的环境里，用 MAST-public 9552b27 核对过：

- 隔离的端到端回合测试通过；
- `act` 背后的 22 个技能全部注册（MAST-public 共注册 481 个技能）；
- 静态基准的原子分辨题可以生成。

MAST-public 不含 VIGIL 的权重，所以 `vigil-text` 臂和 `advisor` 臂里的 VIGIL 读数用不了；advisor 的经典检测器都在。还没有在 MAST-public 上跑过由模型驱动的回合。

## 环境

1. **Python 3.13。**
2. **STM-Bench**，提交 84fb385 或更新。它提供 `RuntimeHost(controller_timeout_s=...)` 和 `.sxm`/`.dat` 读取器：

   ```bash
   pip install "stm-bench @ git+https://github.com/HamsterPark/STM-Bench@84fb385b990dd867580933c72b9ad9b5f6547d76"
   ```

   开发时改为安装本地目录：`pip install -e <STM-Bench 目录>`。
3. **本仓库：** `pip install -e ".[dev]"`，或直接在仓库根目录运行。
4. **MAST 或 [MAST-public](https://github.com/HamsterPark/MAST-public)**，只有闭环回合需要。安装它的 `MASTv2/requirements-v2.txt`，把 `MASTv2` 目录加到 `PYTHONPATH`，并把 `MAST_ROOT` 设为仓库目录。
5. **Claude Code** 命令行，用 Claude 订阅登录；回合和静态基准的 Claude 运行需要它。

Windows 上还要设置 `PYTHONUTF8=1`。

## 计费

回合只走 Claude 订阅。STM-Bench 的 RuntimeHost 会把 MAST 的模型服务商密钥导出到进程环境变量里，早期回合因此记在了 API key 上。

`RunnerConfig.auth = "subscription"` 是默认值，它做两件事：

- 从 Claude Code 子进程中去掉 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_AUTH_TOKEN`；
- 会话初始化时如果报告的认证来源是 API key，立即中止该会话。

请保持这个默认值。

## 用法

```bash
python -m vista_spm preflight                          # 检查 Claude Code 与 MCP 桥，一次低成本调用
python -m vista_spm tools --arm full [--qplus]         # 打印某个臂向模型展示的工具
python -m vista_spm run --scenario P4_atom --seeds 0 --arm mem [--out DIR] [--run ID] [--grid off]
python -m vista_spm replay <回合目录> --single-file    # 为完成的回合生成 STM-Bench 回放页面

python -m vista_spm.staticbench generate --out tmp/staticbench/v1 --types all --seeds 0-9
python -m vista_spm.staticbench answer tmp/staticbench/v1 --mode baseline --out tmp/staticbench/baseline.jsonl
python -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 --out tmp/staticbench/run1 ...
```

输出默认写到 `STM_BENCH_DATA`（`~/stm_bench`）。`--grid off` 不画方格纸，回合目录名记为 `<臂>-nogrid`；`tools`、`preflight`、`run_matrix.py` 和静态基准的 `qa` 也都支持这个选项。

跑消融矩阵前先冻结代码，再从快照目录运行，见 [scripts/README.md](scripts/README.md)：

```bash
python scripts/snapshot.py tmp/code/m2
python scripts/run_matrix.py --cwd tmp/code/m2 --out <目录> --run-id m2 --arms full,mem,img --scenarios P1_barth1990_au111,P4_atom_positioning_cu111 --seeds 0,1,2
python scripts/aggregate.py <目录> --run-id m2 --csv rows.csv --md summary.md
```

## 测试

```bash
python -m pytest -q -m "not isolated"     # 单元与集成测试，不需要 MAST
python -m pytest -q -m isolated           # 端到端脚本化回合（需要 MAST；每个进程只能有一个 RuntimeHost）
```

在 STM-Bench 84fb385 上不装 MAST：282 项通过、4 项跳过、1 项未选中。装上 MAST 或 MAST-public 9552b27，隔离的端到端回合也能通过。
- **跳过的：** `tests/test_runner.py` 里调用真实 Claude Code 的测试，只有设置 `VISTA_LIVE_CLAUDE=1` 时才运行。
- **未选中的：** 隔离的端到端回合。

`tests/test_repo_hygiene.py` 检查三件事：

- 不出现机器路径；
- 只有一处读取 `STM_BENCH_*` 和 `MAST_ROOT`；
- 控制器厂商名只以 MAST 标识符的形式出现。

## 目录

| 路径 | 内容 |
|---|---|
| `vista_spm/` | harness 本体 |
| `vista_spm/staticbench/` | STM-BabyVision：题目生成、题目、评分、答题宿主、经典基线 |
| `tests/` | 测试；`fakes.py` 和 `fake_claude.py` 分别替代仪器和 Claude Code 命令行 |
| `scripts/` | 矩阵工具：快照、运行、汇总、找出需重跑的回合、离线重判 |
| `results/` | 历次运行的汇总结果（存档，不再重新生成） |
| `docs/DESIGN.md`、`docs/staticbench.md` | 设计规格、静态基准说明 |
| `docs/notes/` | 调研、计划、第一版结果与参考实现修复（中文）。公开版删去了 MAST 的内部细节、未公开的模型评测和本地路径。 |

## 相关仓库

| 仓库 | 可见性 | 关系 |
|---|---|---|
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | 公开 | 模拟器（`stmsim`）、`.sxm`/`.dat` 读取器、场景与判分（`stmbench`） |
| MAST | 私有 | `act` 背后的仪器运行时与技能、检测器和 VIGIL；本 harness 的维护版本所在 |
| [MAST-public](https://github.com/HamsterPark/MAST-public) | 公开 | MAST 6.5.0 的有限公开源码版；在这里可以替代 MAST，但没有 VIGIL（[详见上文](#不装-mast-能跑什么)） |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | 公开 | 视觉记忆与测量工具的 TypeScript 移植 |

dsh-spm 的一致性金样由 `vista_spm/frames.py`、`archive.py`、`viewtools.py` 和 `measure.py` 生成。

## 许可

[MIT](LICENSE)。源自 VISTA 的代码在上游同样采用 MIT 许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
