# VISTA-SPM

[English](README.md) · 中文

**面向自主扫描探针显微实验的视觉推理层：观察、记忆、回看、测量与行动。**

原子尺度实验需要智能体能够跨时间地利用图像推理。关键证据可能是当前扫描中的晶格、早先图像中的一个缺陷，也可能是针尖修整前后很细微的变化。[VISTA](https://arxiv.org/abs/2610.02200) 为此提供了有力的出发点：让多模态模型直接观察图像，保留无损的视觉档案，并在理解不断深入时主动检索和检查证据。

**VISTA-SPM 将这一思路发展为面向扫描探针显微实验（SPM）的通用框架与可复用模块，以 STM 为主要应用场景。** 我们把视觉记忆、标定坐标、物理测量、实验笔记和动作—观测闭环组织在一起。这套框架服务于我们的 MAST 自动化工具包和 [dsh-spm](https://github.com/HamsterPark/dsh-spm)，其 Python 参考模块也可独立用于图像分析、智能体开发与基准评测。[STM-Bench](https://github.com/HamsterPark/STM-Bench) 则提供模拟仪器和基于证据的评估环境。

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green) ![参考实现](https://img.shields.io/badge/reference_implementation-2026--10--13-lightgrey)

[科学动机](#为什么在-stm-中参照-vista) · [可复用模块](#通用框架与可复用模块) · [实验结果](#第一版结果2026-10-10) · [环境配置](#环境) · [相关仓库](#相关仓库)

## 为什么在 STM 中参照 VISTA

**Representation is part of reasoning.**

**表征本身就是推理的一部分。**

我们将这一主张作为自主显微实验的设计原则。图像如何呈现、历史如何保留、空间关系如何组织，以及观测如何连接到物理量，都会影响智能体能够提出什么假设、比较哪些证据，以及选择怎样的实验。观察、回看、放大、配准和测量，本身就在参与推理。**为自主实验设计视觉表征与证据接口，就是在设计自主实验系统进行科学推理的条件。**

### 让视觉证据参与整个推理过程

Han 等人的 *VISTA: A Visual Harness for Reasoning in an Interactive World* 将持久保存、主动查阅视觉证据放在了核心位置。智能体直接看图，检索历史观测，放大相关区域，读取精确像素，并维护可修订的笔记，从而让已有多模态模型能够在长程交互中持续进行视觉推理。论文在 25 个公开 ARC-AGI-3 游戏上，以 GPT-5.6 Sol、max 推理强度进行的逐步消融中，加入无损视觉记忆和 `inspect` 后，相对人类动作效率（RHAE）从 70.05 提升至 94.10；再加入精确像素读取后达到 99.00。这些结果为我们选择这一架构提供了依据。[VISTA 消融研究](https://arxiv.org/html/2610.02200v1#S4.SS3) · [上游实现](https://github.com/joshhhhhan/VISTA)。

这一原则在 STM 中具有直接的科学意义。把一幅图概括成“干净的台面”或“针尖状态较差”，实际上已经从丰富的测量信息中选择了一种解释。随着实验推进，先前忽略的缺陷、重复特征或扫描线突变，可能成为判断的关键。保留原始观测，使智能体能够重新检查证据、比较不同解释，并选择足以消除歧义的下一项测量。**视觉记忆由此成为实验方法的一部分。**

### STM 的感知与控制紧密相连

扫描隧道显微镜利用原子级尖锐的针尖探测导电表面。隧穿电流对针尖—样品间距和电子结构高度敏感；在恒流成像中，反馈系统记录维持设定电流所需的针尖高度。因此，原子尺度的图像衬度同时承载样品、针尖和测量条件的信息。隧穿电流与局域电子态之间的联系是 STM 理论的核心内容（[Tersoff 与 Hamann，1985](https://doi.org/10.1103/PhysRevB.31.805)）。解释一个看似凸起的特征，需要结合偏压、空间环境和采集历史。

操作者还必须从研究样品所用的同一组数据中，推断仪器自身的状态。针尖顶端发生变化，可能改变整幅图的衬度；热漂移和压电蠕变改变指令坐标与实际位置的对应关系；反馈参数和扫描方向影响特征的呈现方式；一次电压脉冲可能在改善针尖的同时改变局部表面。[MAST-public 对 STM 控制难点的介绍](https://github.com/HamsterPark/MAST-public/blob/main/README.zh.md) 展开了这些实际问题：自主 STM 本质上需要处理**部分可观测、随时间变化的系统，以及连续的实验设计与决策**。

自主 SPM 研究已经展示了在这一闭环中引入学习方法的价值。DeepSPM 将区域选择、图像质量评估和用于针尖修整的强化学习结合起来（[Krull 等，2020](https://doi.org/10.1038/s42005-020-0317-3)）；强化学习也被用于针尖条件不确定且会发生变化的原子操纵（[Chen 等，2022](https://doi.org/10.1038/s41467-022-35149-w)）。VISTA-SPM 沿着这一实验研究脉络，提供统一的接口，让通用多模态模型能够查阅证据、调用定量工具，并在不同任务中持续修订实验方案。

## STM 与 ARC-AGI-3 的相似之处

### 图像都包含需要推理的空间结构

ARC 的视觉任务依赖物体、重复、对称性、边界和相对位置之间的关系。STM 图像同样包含丰富的空间结构：周期晶格、重构畴、台阶边缘、孤立吸附物、空位，以及重复或畸变的特征。关键信息可能横跨整个台面，也可能只占几个像素。理解这些信息，需要在全局视野与局部细节之间切换，同时保留坐标关系。

这正是 VISTA 的图像访问与 `inspect` 操作适合 STM 的原因。模型可以先观察整个重构图案，再放大一个缺陷，随后回到历史扫描中的同一位置。VISTA-SPM 为这一过程加入扫描几何与物理单位，使视觉关系能够进一步转化为关于周期、位移、高度或谱学响应的可检验陈述。

### 实验都需要在交互中增进理解

[ARC-AGI-3](https://arxiv.org/abs/2603.24621) 在陌生的交互环境中评估智能体，要求它通过探索发现动作如何改变画面、怎样推进任务。STM 有已知的物理规律和明确的科学目标，但每次实验面对的针尖—样品系统仍然只有部分状态已知。两者可迁移的推理结构是：提出假设，选择有信息量的动作，检查变化，再更新当前认识。

| 推理需求 | ARC-AGI-3 | STM 实验 |
|---|---|---|
| 理解空间场景 | 物体、重复图案、边界及其相互关系 | 晶格、台阶、吸附物、缺陷与扫描伪影 |
| 推断隐藏状态 | 发现动作效果和环境规则 | 推断针尖状态、漂移、局部样品状态与测量稳定性 |
| 获取有用证据 | 用探索动作区分可能的规则 | 改变偏压、重新扫描、比较正反扫、测谱或修整针尖 |
| 追踪时间变化 | 比较动作前后的画面 | 区分原子移动、视野漂移和针尖衬度变化 |
| 维护当前认识 | 在连续交互中保留已经发现的规律 | 记录稳定参照物、测量条件、已检验假设和待完成的对照 |
| 有效使用动作 | 相对人类以较少动作完成任务 | 有效利用仪器时间，并考虑操作对针尖或样品的影响 |
| 验证任务进展 | 达到环境的完成条件 | 用相应的采集证据支撑科学结论 |

例如，一个原子在操纵后看起来发生了位移。可信的判断需要把它与邻近参照物比较，检查整幅图是否漂移，并观察其他特征的形状是否因针尖变化而改变。相关证据跨越多幅图像和多个尺度。帧档案、局部检查、图像配准与标定坐标，使这一判断过程能够明确地执行和复核。

STM 为这种共同结构增加了连续信号、噪声、物理单位，以及依赖采集条件的图像衬度。**VISTA-SPM 将 VISTA 的视觉交互方法发展为理解仪器与测量语境的实验工作流。**

## 通用框架与可复用模块

我们的构想是围绕持续保存的视觉记录来组织自主显微实验：

```text
观察 → 归档证据 → 提出或修订假设
 ↑                         ↓
核验结果 ← 执行动作 ← 回看、比较与测量

实验笔记贯穿整个循环，保存当前认识与实验计划。
```

这套框架整合了四项能力：

1. **持续保存、随时回看的证据。** 将扫描通道、正反扫、中途扫描、谱和派生测量以稳定的帧 ID 归档。新的问题出现时，模型可以重新查阅相关证据。
2. **以计量为基础的视觉推理。** 将局部检查与像素—扫描坐标换算、纳米网格、数值读取、剖面、傅里叶分析、特征定位和配准差分结合起来。工具给出测量，模型将测量与物理假设联系起来。
3. **长程实验中的连续性。** `GUIDE.md` 保存智能体可修订的实验认识；`WORKING.md` 在上下文更替时保留当前计划与状态；动作历史连接每项观测与产生它的干预。
4. **对模型接口进行可控评估。** 在相同场景中，分别配置图像、笔记、记忆、数值读出和额外测量工具，把观测与工具接口作为明确的实验变量。

这一贡献具有独立的适用范围：显微实验普遍需要把图像、测量与不断更新的实验认识联系起来。这些模块解决的正是科学数据与智能体推理之间的共性需求。MAST 提供了一种执行环境；dsh-spm 则展示了如何将同一套视觉记忆和测量设计带入另一种语言与智能体宿主。

| 模块 | 提供的能力 | 复用入口 |
|---|---|---|
| 图像与坐标 | 读取 `.sxm` / `.dat`，统一通道方向，渲染视图，换算显示位置与扫描坐标 | [frames.py](vista_spm/frames.py) |
| 视觉档案 | 保存采集数组与元数据，检索任意帧，重新渲染所选区域 | [archive.py](vista_spm/archive.py) |
| 检查与读值 | 用 `inspect` 检查视觉细节；用 `read_values` 读取 pm、pA、V 或 Hz 数值 | [viewtools.py](vista_spm/viewtools.py) |
| 定量工具 | `line_profile`、`fft_peaks`、`detect_blobs`、`diff` 和 `stats` | [measure.py](vista_spm/measure.py) |
| 笔记与工具协议 | 持久实验笔记，以及供智能体宿主使用的结构化文字与图像回复 | [notes.py](vista_spm/notes.py)、[protocol.py](vista_spm/protocol.py) |
| 评估 | STM-BabyVision 静态感知任务与可配置的闭环消融 | [静态基准](docs/staticbench.md)、[设计规格](docs/DESIGN.md) |

例如，视觉模块可以独立于 MAST 运行时，对已有扫描文件进行检查：

```python
from vista_spm.archive import Archive
from vista_spm.viewtools import inspect
from vista_spm.measure import stats

archive = Archive("tmp/visual-memory")
frame = archive.add_scan(turn=0, sxm_path="scan.sxm")
view = inspect(archive, {
    "question": "Examine the terrace and its defects.",
    "views": [{"frame": frame.fid, "label": "Terrace overview"}],
})
measurement = stats(archive, {"frame": frame.fid})
# view.images contains labelled PNGs; measurement.text contains numerical JSON.
```

可复用的视觉模块使用 STM-Bench 的文件读取器和科学计算 Python 依赖。本仓库的闭环执行使用 MAST 的运行时与技能，配套运行器通过 Claude Code 承载模型。持续维护的 MAST 集成版和 [dsh-spm](https://github.com/HamsterPark/dsh-spm) 中的 TypeScript 实现，将这一设计应用于各自的系统。本仓库保留 Python 参考实现、测试、静态基准和首轮模拟器消融实验。[依赖详情](#独立使用与运行时集成)。

## 第一版结果（2026-10-10）

**首轮模拟器研究展示了保留与回看视觉证据的价值。** `mem` 臂在 9 个回合中完整复现了 6 个，只有最新图像的臂为 2 个；在 37 道静态感知题中，Claude 使用工具时答对 31 道，只看图时答对 23 道。两组实验分别从连续实验交互与单项视觉测量两个层面考察了这一框架。

闭环实验设置：

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

在这轮矩阵中，配对结果支持 `mem` 相对 `img` 和 `vigil-text` 的优势。这里，`mem` 同时增加了笔记、帧档案和 `inspect`；`full` 又增加了数值读出和五个测量工具。每个场景使用三个种子，这轮实验刻画的是这些组合的表现，各组件的独立贡献需要进一步的针对性消融。两个较完整的臂也采用了不同策略：`mem` 平均每回合放大 7.8 次；`full` 平均测量 15 次、放大 2 次。

**判分背景。** 结论通过比例未做机会校正。2026-10-12 的审计给出了以下固定常数基线，前提是满足采集证据要求：

| 场景 | 固定常数通过的结论比例 | 同一组常数的完整复现率 |
|---|---|---|
| P1 | 约 0.26 | 约 2–3% |
| P4 | 0 | 0 |
| P5 | 约 0.41 | 约 13% |

这些基线为结合配对结果、完整复现数和结论通过比例来理解实验表现提供了参照。

**静态感知基准（STM-BabyVision），37 道难题：**

| 答题方式 | 答对 |
|---|---|
| Claude，全工具 | 31 |
| Claude，只看图（没有任何工具） | 23 |
| 经典算法基线 | 30 |

两个 Claude 臂的差距几乎全部来自鱼骨条纹周期，这类题需要精确测量。

**实验范围与版本记录。** 上述结果均来自模拟器实验。存档运行使用 MAST 和 2026-10-09/10 版本的 STM-Bench，早于公开规则 v1 及后续模拟器改动。精确重跑历史实验需要当时的依赖快照，其中包含私有组件；冻结文件的哈希见 [results/m1/code_manifest.json](results/m1/code_manifest.json)。`m1-surface` 标签配合 `--grid off` 保留了当时面向模型的提示词、工具 schema 和工具行为；当前工具已纳入 [2026-10-13 参考实现修复](docs/notes/2026-10-13-reference-fixes.zh-CN.md)。实验细节见[首轮结果报告](docs/notes/2026-10-10-first-results.zh-CN.md)，汇总表见 [results/](results/)。

## 工作方式

```
python -m vista_spm run
 ├─ STM-Bench RuntimeHost：MAST CoreRuntime 跑在 stmsim 模拟器上
 ├─ VistaHost：按臂组装工具、帧档案、笔记、仪器时钟、预算、判分账本
 ├─ RpcServer（回环地址，带令牌）
 └─ ClaudeCodeRunner：claude -p 分段运行，内置工具全部关闭，stream-json
       claude ── stdio MCP ──► mcp_bridge.py（只用标准库）──► RpcServer ──► VistaHost
```

- **观测：** 从采集数组渲染视图，放大采用最近邻方式，超大图像采用分块均值缩小。标签和数字放在图像旁边的文字里；归档数组始终可用于细节检查和数值测量。
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
| 把模型看到的界面当作实验参数 | 每个臂的系统提示和工具 schema 都用哈希钉住，覆盖有无方格纸、有无 qPlus。标签保留首轮矩阵面向模型的接口与工具行为。 | [test_surface.py](tests/test_surface.py) |
| 真值不进观测 | 工具回复、图像和动作描述不得带出隐藏的场景数值、容差或只给判分用的数据。真值只留在账本里，供判分和回放使用。 | [AGENTS.md](AGENTS.md)（不变量 1） |
| 长回合 | 采用 VISTA 的压缩协议：Claude Code 压缩上下文之前，模型先写 WORKING.md，再由新的分段根据笔记和动作历史接着跑。 | [compact_hook.py](vista_spm/compact_hook.py)、[claude_runner.py](vista_spm/claude_runner.py) |
| 计费安全 | 运行器从 Claude Code 子进程中去掉 API 凭据。会话初始化时如果报告的凭据来源是 API key，立即中止。 | [claude_runner.py](vista_spm/claude_runner.py) |
| 可审计的消融 | 冻结代码快照并附哈希清单；按种子顺序运行矩阵，遇到用量上限就暂停；找出失败的回合重跑，已完成的回合可以离线重判。 | [scripts/](scripts/) |

## 独立使用与运行时集成

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

运行器默认使用 Claude 订阅认证（`RunnerConfig.auth = "subscription"`）。为在 MAST 将模型服务商凭据导出到环境变量后继续保持这一选择，运行器执行两项检查：

- 从 Claude Code 子进程中去掉 `ANTHROPIC_API_KEY` 和 `ANTHROPIC_AUTH_TOKEN`；
- 会话初始化时如果报告的认证来源是 API key，立即中止该会话。

使用订阅运行时，保持这一默认配置。

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
| `docs/notes/` | 调研、计划、第一版结果与参考实现修复（中文） |

## 相关仓库

VISTA-SPM 在我们的 SPM 研究软件中承担视觉推理层的角色。相关项目将仪器操作、实验评估与科学解释连接起来：

| 仓库 | 在研究流程中的作用 | 与 VISTA-SPM 的联系 |
|---|---|---|
| [MAST-public](https://github.com/HamsterPark/MAST-public) | STM 自动化：仪器运行时、技能、状态反馈与实验工作流 | MAST 的有限公开版可执行本仓库的闭环动作；其 README 系统介绍自主 STM 的科学动机；持续维护的视觉 harness 位于 MAST 中。 |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | DeepSeek Harness 中的 SPM 工具与运行时集成 | 将视觉记忆、标定读出和测量工具移植到 TypeScript，并用本 Python 实现导出的参考结果检验一致性。 |
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | 模拟显微镜、科学场景、采集记录与基于证据的判分 | 为 VISTA-SPM 消融提供仪器与评估环境，同时提供 `.sxm` / `.dat` 读取器。 |
| [AFS3D](https://github.com/HamsterPark/AFS3D) | 三维原子力谱：漂移校正、力重建与实验—理论对照 | 互补的科学分析流程，体现空间配准、定量信号和可追溯比较在 SPM 中的价值。 |
| [dft-spm-toolkit](https://github.com/HamsterPark/dft-spm-toolkit) | 基于电子结构计算和探针粒子模型的 STM/nc-AFM 图像模拟 | 将图像衬度与原子结构、探针响应和成像机制联系起来的互补研究工具。 |

MAST 运行时提供动作，VISTA-SPM 组织视觉证据，STM-Bench 评估由此形成的实验行为。在 dsh-spm 中，[运行时集成](https://github.com/HamsterPark/dsh-spm/blob/master/packages/bundle/dsh-spm/src/visual-harness.ts) 将 TypeScript 档案、观测传递和动作门控连接起来；其一致性参考数据由本仓库的 `frames.py`、`archive.py`、`viewtools.py` 和 `measure.py` 生成。这给出了从共同科学设计到不同智能体系统实现的具体复用途径。

## 许可

[MIT](LICENSE)。源自 VISTA 的代码在上游同样采用 MIT 许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
