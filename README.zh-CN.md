# VISTA-SPM

[English](README.md) · 中文

VISTA-SPM 是一个以图像为主的视觉 harness，让多模态大模型操作扫描探针显微镜。

它把何恺明组的 VISTA（Han, Hu, Qiu, Wu, He, "VISTA: A Visual Harness for Reasoning in an Interactive World", arXiv 2610.02200）从 ARC-AGI-3 游戏移植到 [STM-Bench](https://github.com/HamsterPark/STM-Bench) 的论文复现场景。在一个回合里，模型可以：

- 以图像形式看到仪器产生的帧；
- 无损保存所有帧，随时回看任意一帧并做测量；
- 写自己的笔记。

模型通过一个 MCP 服务器操作仪器，每轮只执行一个仪器动作。

**状态：** 研究原型，私有仓库。第一版结果出自 2026-10-10，见下文。仓库里有 harness、消融矩阵工具、静态感知基准和汇总结果；回合账本与对话记录不进仓库。

## 为什么做

MAST 原来的智能体通过微调过的视觉模型 VIGIL 看 STM 图像，VIGIL 把每幅图转成文字。VISTA 在 ARC-AGI-3 上的消融指向另一条路：把像素直接交给前沿多模态模型，无损保存它看过的每一帧，再给它一把放大镜（`inspect`：取出任意历史帧、裁剪、放大）和一个精确读数工具（`read_pixels`）。VISTA 的消融里，加笔记得 4.2 分，加无损记忆和 `inspect` 得 24.1 分，再加 `read_pixels` 得 4.9 分。

本仓库在 STM-Bench 模拟器上检验这条路，并保留 VIGIL 文本作为对照臂。

## 第一版结果（2026-10-10）

实验设置：

- **模型：** claude-opus-5-5，effort high。
- **动作：** V0 原语动作集。
- **场景：** P1（Au(111) 鱼骨重构）、P4（在 Cu(111) 上搬运一个 Fe 原子）、P5（qPlus 力谱与 Sader–Jarvis 反演）。
- **种子：** 0–2。
- **图像：** 当时还没有格线。下面的方格纸是 2026-10-12 加的。复现这些回合要用标签 `m1-surface` 并加 `--grid off`：2026-10-13 修参考实现时改了部分工具（[说明](docs/notes/2026-10-13-reference-fixes.zh-CN.md)）。
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

细节、局限和待拍板事项见 [docs/notes/2026-10-10-first-results.zh-CN.md](docs/notes/2026-10-10-first-results.zh-CN.md)；表格在 [results/](results/)。

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
- **方格纸：** 扫描图上唯一画的东西，相当于 VISTA 在 ARC 帧上画的格线。它是一层淡琥珀色的纳米方格，格距取 1、2 或 5 × 10ⁿ nm，从整帧左上角起算；每个视野最多 16 格，所以放大后格子会变细。格距 `grid_nm` 写在每张图的标签里，`--grid off` 可以关掉。
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

## 环境

1. **Python 3.13。**
2. **STM-Bench**，以可编辑方式安装：`pip install -e <STM-Bench 目录>`。它必须支持 `RuntimeHost(controller_timeout_s=...)`（[HamsterPark/STM-Bench#1](https://github.com/HamsterPark/STM-Bench/pull/1)），因为回合要把 MAST 的控制器回复超时提高到 60 s。
3. **MAST**（私有），跑完整回合时需要。把它的 `MASTv2` 目录加到 `PYTHONPATH`，并把 `MAST_ROOT` 设为 MAST 仓库目录。
4. **本仓库：** 在仓库根目录运行，或执行 `pip install -e . --no-deps`。
5. **Claude Code** 命令行，用 Claude 订阅登录。

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
python -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 --out tmp/staticbench/run1 ...
```

输出默认写到 `STM_BENCH_DATA`（`~/stm_bench`）。`--grid off` 不画方格纸，回合目录名记为 `<臂>-nogrid`；`tools`、`preflight`、`run_matrix.py` 和静态基准的 `qa` 都支持这个选项。跑消融矩阵前先冻结代码，再从快照目录运行，见 [scripts/README.md](scripts/README.md)：

```bash
python scripts/snapshot.py tmp/code/m2
python scripts/run_matrix.py --cwd tmp/code/m2 --out <目录> --run-id m2 --arms full,mem,img --scenarios P1_barth1990_au111,P4_atom_positioning_cu111 --seeds 0,1,2
python scripts/aggregate.py <目录> --run-id m2 --csv rows.csv --md summary.md
```

## 测试

```bash
python -m pytest -q -m "not isolated"     # 单元与集成测试；没有 MAST 时相关测试跳过
python -m pytest -q -m isolated           # 端到端脚本化回合（每个进程只能有一个 RuntimeHost）
```

`tests/test_runner.py` 里调用真实 Claude Code 的测试，只有设置 `VISTA_LIVE_CLAUDE=1` 时才运行。

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
| `docs/notes/` | 调研、计划与第一版结果（中文，原样收录） |

## 相关仓库

| 仓库 | 可见性 | 关系 |
|---|---|---|
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | 公开 | 模拟器（`stmsim`）、场景与判分（`stmbench`） |
| MAST | 私有 | `act` 背后的仪器运行时与技能，以及 VIGIL |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | 公开 | 视觉记忆的 TypeScript 移植 |

dsh-spm 的一致性金样由 `vista_spm/frames.py`、`archive.py` 和 `viewtools.py` 生成。

## 许可

尚未选定许可证，仓库目前私有。源自 VISTA 的代码在上游采用 MIT 许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
