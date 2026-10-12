# VISTA → STM：调研结论与实施计划

> 迁入 VISTA-SPM 仓库时（2026-10-12）收录。公开版删去了 MAST 的内部细节、未公开的模型评测、本地路径与数据清单，以及只对内的待办事项。文中的 `stmbench/vista/` 即现在的 `vista_spm/`，`python -m stmbench.vista` 即 `python -m vista_spm`，`tmp/vista_lead/` 里的脚本在 `scripts/`，汇总结果在 `results/`。

日期：2026-10-09 · 范围：MAST / STM-Bench · 状态：§7 已定；Phase 0–2 第一版完成（结果见 §8.5 与《VISTA-STM 第一版结果》）；2026-10-12 代码迁入独立仓库 VISTA-SPM（包名 `vista_spm`）

---

## 0. 一句话结论

何恺明组的 VISTA（arXiv 2610.02200）证明：**不训练任何视觉模型**，只给前沿多模态模型一个"视觉 harness"——原始图像观测 + 无损视觉记忆（可随时取回任意历史帧并放大）+ 精确像素读取 + 两份自写笔记——就能把 Claude Opus 5.0 在 ARC-AGI-3 上从 40.68 分拉到 100 分。消融里两个最大增益项是"图像替代文本网格"（+34）和"无损视觉记忆"（+24）。

对照 MAST 现状：模型从来**不直接看**完整的 STM 图像，而是读视觉模型 VIGIL 的输出经 LLM 压成的一两句文字摘要；图像只偶尔以"最近 2 张缩略图"的形式送达，无历史检索、无放大、无像素读取。**"编码器→文字摘要"作为主感知通道是弯路**；但 VIGIL 与经典检测器不必丢，它们应降级为"模型按需调用的测量工具"与"后台安全看门狗"。

建议的实验：在 STM-Bench 模拟器上实现一个 STM 版 VISTA（下称 **VISTA-STM**），以 Claude Code + MCP 为宿主（与论文同款路径），用 P1/P4/P5 三个论文场景 × 3 个 seed 做 VISTA 式消融，对照现有 mode A 与人类 mode H。预计 Phase 0–2 约 2–3 周可出第一张结果表。

文献调研（§3.3）显示：截至 2026-10，没有任何闭环 STM 代理具备"原生视觉 + 无损图像记忆"，也没有人做过"编码器→文本→LLM"对"原图→VLM"的对照实验；唯一的真机前沿 LLM-STM 论文（Boston College quailbot）的三个失败案例都是人靠前后帧比对才诊断出来的。

---

## 1. VISTA：我们要移植的是什么

### 1.1 机制（论文 + 代码核对）

| 组件 | VISTA 的做法 | 关键数字 |
|---|---|---|
| 观测 | 64×64 网格 → 8× 最近邻放大为 512×512 PNG，一像素网格线；图内不写字，标签放在相邻文本 `<current_visual turn= frame= frame_count= size=>` | 512 px ≈ 308 image tokens；文本网格 ≈ 4000 tokens |
| 动作 | `play(action, x, y)` 唯一改变环境；**一动一观**：上一个 play 的结果进入模型上下文之前拒绝下一个 play | 25 局 7302 动作 vs 人类 17135 |
| 视觉记忆 | 每个 play 的**所有中间动画帧**全部落盘 `turn_NNN_frame_MM.png`，永不淘汰；只把最后一帧发给模型，元数据告知归档了几帧 | — |
| `inspect` | `views[≤16]{label, turn, frame?, region{x,y,w,h}}`：取回任意历史帧/中间帧，按区域裁剪后 NEAREST 放大到最长边 512；每个视图单独一个 image block（不拼图） | 消融 +24.05 |
| `read_pixels` | `views[≤64]{region, rows, cols}`：等分 bin 中心采样，返回调色板 + 每行符号串；"只量像素，不解释" | 消融 +4.9 |
| `history` | `events`（逐 turn 的动作/结果记录）/ `attempts`（本关 RESET→GAME_OVER 摘要） | — |
| 笔记 | `GUIDE.md`（跨关卡、跨会话；≤64 KiB）、`WORKING.md`（当前关卡；RESET 不丢，过关清空；≤16 KiB） | 消融 +4.2 |
| 压缩 | Claude Code 的 PreCompact 钩子阻断原生压缩 → 强制 `write_working` → 结束 segment → 新会话以 GUIDE+WORKING+动作历史+当前帧重启 | 200K→100K 窗口只损 0.4 |
| 提示词 | 全文 5 句：尽量少动作；维护紧凑可修订的游戏模型；每次 play 前说预期、后说所有可见变化；把理解写进 GUIDE/WORKING | — |
| 宿主 | `claude -p --input-format stream-json --tools "" --mcp-config … --allowedTools mcp__game__*`，Docker 内运行，禁用全部内置工具；MCP 结果 = text(JSON) + 每图 [text 标签, image(base64)] | Claude Code 2.1.220 固定版 |

### 1.2 消融（GPT-5.6 Sol，ARC-AGI-3 公开 25 局，RHAE）

| 配置 | RHAE |
|---|---|
| 官方文本网格接口 | 13.33 |
| (i) 换成 PNG 图像 | 47.32 |
| (ii) 放宽动作/时间预算 | 51.66 |
| (iii) 连续对话 + 原生压缩 | 65.82 |
| (iv) + GUIDE/WORKING 笔记 | 70.05 |
| (v) + 无损视觉记忆 + inspect | 94.10 |
| (vi) + read_pixels（完整 VISTA） | 99.00 |

其他：图像尺度 4×(256 px) 与 8× 相当而 image tokens 少 75%，2× 变差，16× 无益；开源 GLM-5.3 Flash 用完整 VISTA 只到 66.93（harness 只能"激发"基座已有能力）；BabyVision 41.0→63.2；GameWorld/AI GameStore 也超基线。

### 1.3 代码库（github.com/joshhhhhan/VISTA，MIT，2026-10-04）

- 可直接复用（MIT）：`core/tools.py`（工具 JSON schema）、`core/session.py`（inspect/read_pixels 实现、checkpoint 状态机）、`backends/claude/mcp_bridge.py`（205 行 stdio MCP 桥）、`compact_hook.py`、`lifecycle.py`（segment/恢复循环）、`arc3/claude/prompt.md`。
- 通用环境接口 `TaskSession{task(objective, context, observation), tools[ToolBinding(ToolSpec, handler)], submit(), close()}`；`TaskRuntime` 自动加 inspect/read_pixels/history/notes、校验 schema、计预算、存图、打标签。**新环境只需实现动作工具 + 观测打包**。
- 代码里论文没写的：preflight（"Reply exactly READY" 并校验工具面只含 mcp__game__*）、凭证清洗、决策租约 120 min、15 次无效动作后重播种、图像数 600/会话上限、多账号配额调度、逐 segment 全量日志（stream-json、bridge、hook、用量 manifest）。
- 环境：Python 3.12 固定（因 arc-agi 依赖）；我们用 3.13 自写即可，不必装 vista-arc3 包。

---

## 2. MAST 现状诊断

### 2.1 当前感知链路

模型不直接看完整的 STM 图像。每帧先经视觉模型 VIGIL 与一组经典检测器处理，结果汇总成状态，再由 LLM 压成一两句文字交给仪器智能体；技能渲染的图片只偶尔以最近两张缩略图附在消息里，不进历史，不能检索，也不能放大。

### 2.2 与 VISTA 的差距

| 维度 | VISTA | MAST 现状 |
|---|---|---|
| 主感知通道 | 原始图像，模型自己看 | 视觉模型 → 文本摘要 |
| 历史 | 每帧永久归档，可随时 inspect | 只带最近 2 张；旧帧只剩文本 |
| 细节 | inspect 放大 + read_pixels 精确读 | 无 |
| 记忆 | agent 自写 GUIDE/WORKING，压缩前强制保存 | 模型外的摘要与记忆库召回 |
| 动作节奏 | 一动一观，预期→报告 | 多步技能，一次工具调用可跑完一个闭环 |

### 2.3 不是全盘否定

- 非视觉模型（DeepSeek/GLM/Qwen）仍需文本通道；毫秒级安全事件（扫描中途换针、急停）需要本地检测器而非每次问 VLM。
- VIGIL 与经典检测器（双针尖、针尖突变、晶格标定、帧间漂移、鱼骨条纹等）改造成**按需调用的测量工具**与**后台看门狗事件**，而不是"替模型看图"。本计划把"检测器作为工具"单独设为一个消融臂，用数据回答它还值不值。

---

## 3. 相关工作可借鉴的结论

### 3.1 视觉工具使用与视觉记忆（2024–2026）

- 对前沿闭源模型，**纯提示**的裁剪/放大（Visual Sketchpad、ZoomEye、Zoom-Refine）就带来 3–35 分增益；RL 主要用于教小模型学会调用工具（DeepEyes、Mini-o3 等）。Mini-o3 的"轮数 4→32 持续涨分"与 VISTA 的 inspect 消融一致：**让模型反复再看，是最大增益来源**。
- 厂商已原生支持 zoom（o3 thinking-with-images、Claude computer-use 的 `zoom`、Gemini 3 Flash Agentic Vision）；harness 应暴露高分辨率原图，而不是替模型预处理。
- 历史截图保留数量：Mano（arXiv 2509.17336）0/1/2/3/4 张 → 29.6/31.5/32.7/32.6/32.7；Anthropic 建议保留最近 3 张、批量裁剪以保护缓存；Claude 5.x 应用服务端 tool-result 清理而非客户端删图。**结论：上下文里留 2–3 张近帧 + 全量可检索归档**，与 VISTA 一致。
- 对照 ARC-AGI-3 的程序合成派（Tycho 100 分/$2986、Retrodict 99.86/$654、Schema 98.98）：VISTA 的论点是视觉越复杂，显式程序世界模型越难写；STM 图像显然属于"视觉复杂"一侧。

### 3.2 VLM 的已知弱点及对 STM 的含义

| 弱点 | 证据 | 对 STM harness 的含义 |
|---|---|---|
| 低层几何、计数 | VLMs are blind（BlindTest 58%）、VLMs are Biased（反事实计数 17%）、VLMCountBench | 原子/分子计数要靠 `detect_blobs` 工具 |
| 精确位置与尺寸 | ScreenSpot-Pro、MeasureBench（会认比例尺但算不对） | 坐标由 harness 换算（像素↔nm），距离/周期由工具算 |
| 细微纹理/噪声变化 | VDiff-Bench（噪声/纹理变化 8.7–33%）、DiffSpot（最佳召回 40.7%） | 变化检测要作为工具（对齐 + 差分图 + 热图） |
| 拼图并排 | VLM-SubtleBench：两图拼成一张在 9/10 领域降准确率 | inspect 多视图**分开**送、各带标签 |
| 小图 | Anthropic：<200 px 易错 | 永远整数 NN 放大到 ≥512 px |
| 有效手段 | Set-of-Mark、SCAFFOLD 点阵坐标（+9.6/+13.3）、分块多尺度 | 可选叠加层（比例尺/坐标网格/编号标记），保留干净图做对照 |

显微专门基准：μ-Bench、MicroVQA、SPM-Bench（arXiv 2602.22971，PhD 级 SPM 多模态题）。周期晶格检测没有 VLM 基准（空白，我们的静态基准可以填）。

### 3.3 自主 SPM 的 LLM/VLM 工作（2023–2026）

要点：

- **真机上的前沿 LLM-STM 代理只有一篇**：Cheng/Xu/Candelora/Zeljkovic（Boston College）"quailbot"，arXiv 2609.27302。Nanonis 经 CLI，4 个工具（cli_get/set/ramp/action），gpt-5.6-sol 与 claude-opus-5，FeTe 4 K 上端到端修针到原子分辨，专家用 BraggZ 指标（一阶 Bragg 峰超出背景 MAD 的倍数）事后核验：起点 20.7–41.6 → 终点 79.5–295.4。论文没有说明图像是否以像素进入模型。三个失败案例（弯针被误判为第二个尖、压电步进器卡死约 6.5 nm 被误归因于胶、代理仍报告完成）全是**人用前后帧比对杂质位置**诊断出来的——正是 VISTA 的 `inspect` 能做而现有代理不能做的事。
- **其余 SPM 代理全部给 LLM 喂文本/描述子，不喂像素**：Kalinin 谱系（arXiv 2609.04015，Claude Sonnet 4.6 做结构化决策，"像素阵列从不进入状态"，代理写分析代码、不选坐标）；AILA/AFMBench（Nat. Commun. 16, 9104, 2025，图像转 numpy 由 LLM 生成 Python 分析，GPT-4o 分析题 33.3%，失败模式"梦游"与提示脆弱）；大阪 Abe 组（arXiv 2602.20669，LoRA 微调 Phi-4 出结构化命令，纯文本，多模态是计划中的 Stage iii）；EPFL agentic AFM（arXiv 2608.26198，三个 MCP 代理，Pilot 评图质量达专家水平，输入格式未说明）；ZEISS 基准（arXiv 2608.05266，1949 次运行，只看文本/JSON，已见任务 AUC 0.78、未见任务 0.53–0.55）。
- **唯一把原图喂给 VLM 并做跨帧比对的仪器代理是 X 射线显微镜上的 EAA**（Argonne，arXiv 2602.15294，GPT-5）：工具渲染图 base64 注入；相位相关漂移配准 + VLM 重叠判断；标记定位误差 GPT-5/Gemini ≤10 px、GPT-4o 约 32 px；结论"VLM 在定量图像分析上落后人类"。这与 §3.2 一致：测量归工具。
- **基准**：MicroVQA（o1 52.8 vs 人 50.3）、MaCBench（AFM 图像尺寸题 0.24；同样信息给文本比给图像好 35 个百分点）、MatVQA（数衍射斑/晶格条纹差）、MatCha（人 88.9 vs GPT-4o 59.1）、SPM-Bench（752 道 STM 题，但标签由 gemini-3-flash 生成未经人工核验，且没有针尖/双尖/漂移/噪声题）。DISSECT（arXiv 2604.06250）是唯一的"先描述再推理"对照：它只对弱开源模型有益，闭源前沿模型无差——对我们的含义是，**对前沿模型，文本化中间表示不是增益**。
- **编码器**：DINO/DINOv2/v3 在 STM/AFM 上没有公开论文；CIMP（arXiv 2604.24909）用图像-元数据对比预训练，线性探针恢复 7 个采集参数，提示"针尖状态作为元数据"的预训练路线；APL-SAM 报告原版 SAM 在 SPM 图上显著退化。
- **修针/质量基线**：Rashidi & Wolkow 2018（97%）、Gordon 2019（AUC 0.94–0.98）、DeepSPM 2020、Zhu…Sun JACS 2024（48 h 无人）、Barker 2024（无 ML 模板匹配）、Barker 2025（STS 1D-CNN 86% vs 确定性方法 82%）、Scanbot（Monash，JOSS 2024，刻意不用 ML）。可直接复用的自动终点指标：BraggZ、Scanbot 压痕对称性、Barker 互相关、Reddy–Kalinin 逐行诊断。
- 纠错：Scanbot 是 Monash 而非 Nottingham；DeepSPM 作者是 Krull/Hirsch/Rother/Schiffrin/Krull；"Zhu 2024 针尖分类器"是上海大学 Zhu…Sun，JACS 146, 29199。
- **文献空白**：① 没有"原生视觉 + 无损图像记忆"的闭环 STM 代理；② 没有"编码器→文本→LLM"对"原图→VLM"的对照实验；③ 没有 STM 操作感知基准（针尖/双尖/中途换针/漂移/噪声/台阶/晶格常数）；④ 前后帧隐状态推断；⑤ 确定性校验 + 守护执行 + VLM 感知的混合未被测试；⑥ 用编码器嵌入作视觉记忆检索键（"这种针尖状态见过吗"）。

---

## 4. 实验设计：VISTA-STM

### 4.1 映射

| ARC-AGI-3 / VISTA | STM-Bench / VISTA-STM |
|---|---|
| 一局游戏，隐藏规则与目标 | 一个论文场景（P1 鱼骨周期与旋转畴、P4 单原子搬运、P5 Δf(z) 力谱…），隐藏真值按 seed 抽，开局针尖状态（good/blunt/double/unstable/dirty）不提示 |
| 64×64 彩色网格帧 | `.sxm` 帧：Z / Current / dI/dV 通道，正反扫，128–512 px，浮点物理量 |
| 中间动画帧 | 扫描进行中的部分帧（`Scan_FrameDataGrab` 实时缓冲，仿 scan_monitor 每 1/8 抓一次）、反向扫描、其他通道 |
| `play(action, x, y)` | `act(name, params)`：一次仪器动作（扫一帧、改偏压/设定点、移针、脉冲、戳针、点谱、搬原子…） |
| 关卡推进 | 一条 claim 被验证（`claims_verified`） |
| RHAE（动作效率 vs 人） | 复现率 + 仪器时间 + 动作数 + token，对照 mode H 的人类记录 |
| 25 局 | P1/P4/P5 × seed 0–2（配对设计），后扩 P2/P3 与 B1–B9 回归家族 |

### 4.2 观测渲染规范

- 每个动作返回一张"当前观测"：默认 Z 通道正扫，平面校正（视野 ≤30 nm 逐行去中位偏置，仿 `gallery/render.py`），0.5% 分位裁剪，灰度，**整数最近邻放大到 512 px**（128→4×，256→2×，512→1×；非整除时 NEAREST 缩放并在元数据注明 scale）。不在图上写字。
- 相邻文本元数据：`{turn, scan_id, channel, direction, size_nm, center_nm, nm_per_px, display_scale, bias_v, setpoint_a, acquired_rows/total, rec_time_sim, lo/hi(pm), channels_available, frames_archived}`。
- 坐标系：模型用显示像素坐标（0..511）指位置；harness 用 `stmbench.human.render.pixel_to_scan_nm` 换算为扫描坐标 nm，并在动作回显里同时给出两套坐标。
- 可选叠加层（消融项，不默认）：比例尺、10×10 坐标网格、检测特征编号（Set-of-Mark）。
- 尺度消融：256 / 512 / 1024 px（Claude 计费 ⌈w/28⌉² → 361 / 1369 tokens；标准档 ≤1092 px 不被缩放）。

### 4.3 工具面（MCP，全部 JSON schema + 注解）

**改变环境（唯一）**
- `act(name, params)`：V0 原语集约 12 个（`set_bias`、`set_setpoint`、`scan{center, size, pixels, angle, speed}`、`stop_scan`、`move_tip`、`pulse`、`poke`、`sts{x,y,v_range}`、`move_atom{from,to,R,v}`、`withdraw`/`approach`、`set_feedback`），全部经 MAST `ExecutionContext`（SafetyGate、参数校验、仪器锁）执行；V1 = `run_skill(name, params)` + `find_skills` + `skill_card` 暴露完整注册表。一动一观：上一动作的观测进入上下文前拒绝下一动作。
- `report_result` / `report_tip_state`：复用 STM-Bench 的裁判通道。

**只读：视觉记忆**
- `inspect(question, views[≤16]{label, turn, frame?, channel?, direction?, region?})`：任意历史帧/中间帧/通道/反扫，区域裁剪后 NEAREST 放大到 512；每视图一个 image block。
- `read_values(question, views[≤64]{turn, channel, region, rows, cols})`：返回**物理量**（pm / pA）采样矩阵 + 对应 nm 坐标；VISTA `read_pixels` 的 STM 版。
- `history(view=events|attempts, start_turn, end_turn, limit)`：动作、参数、仪器状态回显、结果摘要。

**只读：测量（VISTA 没有、STM 必需；全部"只量不解释"）**
- `line_profile(turn, channel, p1, p2)`：剖面值 + 台阶检测。
- `fft_peaks(turn, channel, region)`：周期（nm）与角度列表（复用 `lattice_calibration` / `herringbone` 的纯计算部分）。
- `detect_blobs(turn, channel, region, min_size)`：质心（px 与 nm）、计数。
- `diff(turn_a, turn_b, region, align=true)`：互相关对齐（复用 `frame_drift`）→ 位移 nm + 差分图 + 变化热图 + 文字摘要。
- `stats(turn, channel, region)`：RMS 粗糙度、正反扫相关、直方图。

**可选消融臂：advisor**
- `advisor(turn)`：返回 VIGIL / 经典检测器（double_tip、tip_quality、tip_change）的判语。默认关闭。

**笔记**
- `read_guide` / `write_guide`（GUIDE.md，跨回合：仪器脾气、针尖配方、样品晶格常数）；`read_working` / `write_working`（WORKING.md，本回合）；`save_compact_checkpoint`。

### 4.4 记忆与压缩

- 归档：每个 turn 的全部通道、正反扫、部分帧，原始 `.npz` + 渲染 PNG，`archive/turn_{t:04d}/frame_{i:02d}_{channel}_{dir}.{npz,png}`，永不淘汰；`events.jsonl` 追加写。
- 上下文：只保留最近 2–3 张观测图；其余经 `inspect` 取回。
- 压缩：照搬 VISTA 的 Claude Code 钩子（PreCompact 阻断 → 只允许 read/write_working → 写完结束 segment → 新会话以 `环境记录{当前观测, 上一动作结果, 本回合 attempts}` + `笔记{guide, working}` + 当前帧重启）。Phase 1 先用大窗口跑通，Phase 1b 接钩子。

### 4.5 协议与提示词

系统提示词沿用 VISTA 的 5 句结构，只换名词：

> 用尽量少的仪器动作完成任务。建立并维护一个紧凑、可修订的"仪器-样品-针尖"模型，随证据更新。每次 `act` 前简述你预期看到什么；之后简述所有可见变化，无论是否符合预期。把耐久的理解写进 `GUIDE.md`，用 `WORKING.md` 做草稿。

任务题面 = STM-Bench 场景 YAML 的 `task` 字段原文 + 预算；不提示针尖状态与故障。禁用 Claude Code 全部内置工具（`--tools ""`），模型只能通过 MCP 看图与行动。

### 4.6 宿主与架构

```
Claude Code (claude -p, stream-json, --tools "", --mcp-config)      ← 论文同款路径；本机 2.1.295，不用 Docker
   │ MCP stdio (text + image blocks)
stmvista MCP server (Python 3.13, 标准库 + numpy/PIL)
   │ in-process
stmvista host: 渲染 / 归档 / inspect / read_values / measure / 笔记 / 一动一观闸门 / 时钟暂停
   │ ExecutionContext（SafetyGate、参数校验、仪器锁）
STM-Bench RuntimeHost（MAST CoreRuntime 跑在 stmsim 上，4 个本地端口，隐藏真值，判分，账本）
```

- **新模式 V**：在 `stmbench/harness/modes.py` 增加 `V0`/`V1`，driver 是外部 Claude Code 进程而非 `ChatModelPort`；账本仍写 `episode.json / driver.json(带 sim_s) / events.json / tip_timeline.json / session/*.sxm`，使 `stmbench.report` 与 `stmbench.replay` 直接可用。
- **时钟协议**：与 `ClockPausedPort` 等价——工具返回即 `clock.pause()`，下一个工具进入即 `resume()`；模型思考不消耗仪器时间。
- **真机路径（Phase 3）**：同一 MCP server 换后端为 MAST 的外部接口 `/api/ext/v1`（需要支持 PNG 帧与 MCP 图像结果），SAFE 模式 + HITL 不变。
- 模型：Claude Code 订阅下的当前 Opus 级模型（VISTA 用 Opus 5.0 xhigh）。Codex CLI 可作第二基座（VISTA 的 Codex 路径）。

### 4.7 评测与消融矩阵

场景 P1（测量型：FFT 周期/取向）、P4（操纵型：搬原子，最"视觉"）、P5（谱学型）；seed 0–2 配对；每臂 9 回合。

| 臂 | 观测 | 笔记 | inspect | read_values/measure | advisor | 对应 VISTA 消融 |
|---|---|---|---|---|---|---|
| A0 现状 mode A | VIGIL 文本 + 最近 2 张缩略图 | 记忆库 | 否 | 否 | 是 | 官方接口 |
| V-text | 数值文本网格（64×64 降采样 z 值） | 是 | 否 | 否 | 否 | 文本网格 |
| V-img | 图像 | 否 | 否 | 否 | 否 | (i) |
| V-notes | 图像 | 是 | 否 | 否 | 否 | (iv) |
| V-mem | 图像 | 是 | 是 | 否 | 否 | (v) |
| V-full | 图像 | 是 | 是 | 是 | 否 | (vi) |
| V-full+adv | 图像 | 是 | 是 | 是 | 是 | — |

优先跑 A0 / V-img / V-mem / V-full（36 回合），其余按预算补。指标：`claims_verified` 复现率、诊断准确率（ReportTipState vs 真值）、仿真仪器时间、动作数、token 与墙钟、三 seed 方差；人类参考 = mode H 记录。单回合墙钟估计 1–2 h（工具时间 ≤9 min，其余是模型思考），可多进程并行（每进程一个 RuntimeHost）。

两个借自文献的补充对照：**V-describe 臂**（DISSECT 的"模型先把图写成文字描述，再只凭描述决策"）——这是用同一个 VLM 复现"DINO→文本"路线的最干净对照；**无图对照**（MicroVQA 的 no-image 消融）——A0 去掉缩略图中间件，检验现有文本通道到底承载了多少信息。评测协议借 ZEISS 的全程 trace 日志、Kalinin 的重复一致性（同一决策 10 次重复）、AILA 的"梦游/提示脆弱"检查，并报告每模型延迟。

### 4.8 静态感知基准（STM-BabyVision）

用模拟器生成带真值的帧，几分钟出几百题，先于回合实验校准渲染与工具选择：
1. 针尖状态五分类（hidden tip condition）；
2. 鱼骨条纹周期与取向（P1 真值）；
3. 原子计数与位置（P4 adatoms）；
4. 两帧间原子是否移动、移动了多少（adatom_hop 事件）；
5. 台阶高度与漂移量（已知漂移）；
6. quailbot 式隐状态题：给定步进/搬运前后两帧，判断样品实际位移、步进器是否卡死、原子是否被针尖带走（真值来自模拟器事件日志）；
7. 原子分辨判定：用 BraggZ 式 FFT 指标作为自动真值，比较模型目测与工具辅助的判断。

每题允许 inspect / read_values / measure 的不同组合，比较 256/512/1024 px、平场方式、叠加层开关。实验室的真机 `.sxm` 数据没有真值，只用于定性检查与渲染参数核对。

---

## 5. 分阶段计划

| 阶段 | 内容 | 交付物 / 验收 | 工作日 |
|---|---|---|---|
| **P0 环境** | uv 建 Python 3.13；装 MAST `requirements-v2.txt`（CPU torch 即可）与 STM-Bench `.[dev,bench]`；在本机跑通 `stmbench.cli run --mode C --scenario P1 --seeds 0`；渲染一张 `.sxm` 为 512 px PNG；用一个只返回图片的玩具 MCP 验证 `claude -p --tools ""` 在 Windows 上能看图 | mode C 账本 + 一张 PNG + preflight 通过 | 1–2 |
| **P1 harness** | `stmvista/`：`render.py`、`archive.py`、`tools.py`（schema，vendoring VISTA 的 MIT 代码）、`host.py`（RuntimeHost + 一动一观 + 时钟暂停 + 归档）、`mcp_server.py`、`runner.py`（segment 循环、日志、manifest）、`mode_v.py`（账本兼容）；V0 原语集；inspect/read_values/history/notes | P4 seed 0 一回合端到端跑通并出账本与回放页 | 4–6 |
| **P1b 压缩** | PreCompact/Stop 钩子、恢复提示、图像计数上限 | 人为设小窗口触发一次恢复，回合继续完成 | 1–2 |
| **P1c 测量工具** | line_profile / fft_peaks / detect_blobs / diff / stats；advisor 臂 | 静态基准上各工具有单测与真值对照 | 2–3 |
| **P2 实验** | 静态基准 → 选定渲染；36 回合主矩阵 → 补臂；结果表、回放、成本表 | 《VISTA-STM 第一版结果》含 7 臂表、三 seed 方差、token/时间成本 | 5–7 |
| **P3 真机** | MAST ext API 加 PNG 帧与 MCP 图像返回；影子模式（模型只看只提议，人执行）→ 有人陪跑 → SAFE 自主 | 真机影子回合记录；6.5.0 ext API 的首次硬件验证 | 待仪器 |
| **P4 延伸** | 检测器作为工具 / 看门狗的量化结论；STM 操作感知基准公开化（填补文献空白③） | 文稿 | 后续 |

P0–P2 合计约 13–20 个工作日；假期内可完成 P0–P1c 与主矩阵的前半。

---

## 6. 风险与应对

| 风险 | 应对 |
|---|---|
| STM 帧是连续灰度 + 噪声，不是 16 色网格；VLM 对细微纹理/计数/测距不可靠 | 测量工具与 diff 工具承担数值；静态基准先量化 |
| 像素↔nm、漂移、朝向约定出错 | 只用 MAST `sxm_oriented_frames` 与 STM-Bench `pixel_to_scan_nm`（有测试钉住），动作回显双坐标 |
| Claude Code 版本差异（VISTA 固定 2.1.220；本机 2.1.295，2.1.283 起 MCP 图片原件落盘） | 放宽 preflight 版本断言；`--tools ""` 下模型读不到落盘原件，不影响 |
| Windows 无 Docker 沙箱 | 模拟器阶段不需要；真机阶段靠 MAST SafetyGate/SAFE/HITL，不靠容器 |
| 订阅限额与回合时长 | 每回合 1–2 h，先跑 36 回合；100K 窗口即可（VISTA 只损 0.4） |
| 模拟器保真度与迁移 | STM-Bench 自己的声明：sim 结果不等于真机；P3 用影子模式验证 |
| 公开论文在训练数据内 | 真值按 seed 抽，背文献值无用（STM-Bench 设计原则） |
| 一次工具调用跑完闭环（V1 的多步技能）削弱"一动一观" | V0 原语集为主实验；V1 作对照 |

---

## 7. 设计选择

计划阶段的四个选择，2026-10-09 确定（见 §8.1）：

1. 宿主：先走 Claude Code 订阅 + MCP（论文同款，最快），不直接做进 MAST 内部回路（各家模型服务商都能用，但工程量大）；内部回路放到 P3 之后。
2. 动作面：主实验用 V0 原语集（约 12 个动作），V1 完整技能栈只作对照。
3. 模型：Opus 级为主，不跑 Codex CLI 第二基座。
4. VIGIL 默认关闭，只作 advisor 消融臂。

---

## 8. 第一阶段进展（2026-10-09 更新）

### 8.1 已确定

- **宿主：** Claude Code 订阅 + MCP。MAST 内部回路放到 P3 之后。
- **动作面：** V0 原语集做主实验，V1 完整技能栈只作对照。
- **基座：** 不跑 Codex 第二基座。
- **VIGIL：** 默认关闭，只出现在两个消融臂里。advisor 臂按需调用 MAST 检测器，vigil-text 臂把图像换成 VIGIL 文本。

### 8.2 已完成

- **harness 代码**原在 STM-Bench 的 `stmbench/vista/`（从未提交）；2026-10-12 迁入独立仓库 VISTA-SPM，包名改为 `vista_spm`。
  - 观测层：帧加载与渲染、无损帧档案、inspect、read_values。
  - 测量工具：line_profile、fft_peaks、detect_blobs、diff、stats。
  - 笔记、MCP 桥、Claude Code 运行器（含压缩检查点与恢复）。
  - 宿主（一动一观、时钟暂停、预算）、回合入口、7 个消融臂、回放兼容层。
- **测试：** 全部视觉 harness 测试 232 项通过、4 项跳过，隔离的端到端回合测试通过。
- **静态感知基准 v1：** 230 题。经典算法基线得分 92.6%，oracle 100%，全弃权 0%。
- **dsh-spm：** 文档 PR #1 已开（未合并）。视觉记忆 TS 包在 `feat/visual-memory` 分支，本地已有 10 个提交，正在对齐 Python 现行的模型可见命名约定，之后推送为 PR #2。

### 8.3 首轮真实回合

full 臂，claude-opus-5-5，effort high，seed 0：

| 场景 | 结论通过比例 | 针尖诊断 | 仪器时间用量 | 估算费用 | 备注 |
|---|---|---|---|---|---|
| P4 搬原子 | 0.50 | 错 | 23% | $0.94 | 原子停在离目标一个 Cu 格点处，且搬动后没有 ≤0.1 nm/px 的帧覆盖它 |
| P1 鱼骨条纹 | 0.75 | 错 | 14% | $0.53 | 周期报 6.1 nm，真值 6.36 nm，超出 0.15 nm 容差 |
| P5 力谱 | 1.00 | 判分为 0 | 23% | $2.75 | 没有专用工具，模型自己完成 Sader–Jarvis 反演，三个量误差都小于 3% |

### 8.4 发现

1. **模拟器只沿快扫描方向模糊。** 钝针尖时特征只在 x 方向变宽，另一个方向仍然很窄。P1 和 P4 抽到的都是钝针尖，模型都判为尖锐；VIGIL 对 P4 的图也给出 good。在决定是否改成各向同性之前，针尖诊断分不可信。
2. **单幅图内漂移很大。** 静态基准统计的中位数是视场的 8.4%。P1 的模型只扫了两幅图、没开漂移补偿，周期因此偏短约 4%。
3. **P4 判分只用帧起始时刻的漂移换算坐标。** 长帧、未补偿漂移时，正确的报告也可能匹配失败（静态基准报告第 9 节第 6 条）。
4. **原定的 A0（现状 mode A）臂跑不了。** MAST 的内部回路没有 claude-opus-5-5 的配置。改用 vigil-text 臂做同模型、同 harness 的“编码器→文本”对照。
5. **Claude Code 会在工具结果文本里写出存图路径。** 这个路径由子进程的工作目录决定，原先会带出场景名、种子和臂名。现在子进程改在中性临时目录里运行。
6. **订阅额度是实际瓶颈。** 一天的重度开发加三个回合就用尽了五小时窗口。矩阵脚本遇到上限会暂停，到重置时间后续跑；七天窗口使用率到 70% 就不再派发新回合。

### 8.5 主矩阵结果（2026-10-10，详见《VISTA-STM 第一版结果》）

36 回合全部有效：4 个臂 × P1、P4、P5 × seed 0–2，官方判分如下。

| 臂 | 结论通过比例 | 完整复现 | 平均动作数 | 平均估算费用 |
|---|---|---|---|---|
| mem | 0.83 | 6/9 | 18.7 | $1.24 |
| full | 0.72 | 5/9 | 15.8 | $1.46 |
| img | 0.43 | 2/9 | 24.4 | $1.01 |
| vigil-text | 0.25 | 1/9 | 190.7 | $3.18 |

- 图像加无损视觉记忆（mem、full）明显强于只看图（img）和编码器文本（vigil-text），方向与 VISTA 的消融一致。
- full 与 mem 在 3 个种子下分不出高下。P4 按宽松证据口径，full 为 3/3。
- 期间修正了五个基础设施问题，最重要的是计费：此前回合误走了 API key 计费，约 $52；现在只走订阅。受影响的回合均已在最终代码上重跑。

### 8.6 当时的开放问题

1. stmsim 的针尖模糊是否改为各向同性。改的话所有帧都会变，静态基准要重新生成。
2. P4 判分是否改用逐行漂移换算坐标。
3. 静态基准默认用核心 190 题还是全部 230 题。

（STM-Bench 之后改成了二维针尖模糊和逐行漂移，见其仓库。）

---

## 附录：主要参考

VISTA arXiv 2610.02200；VARC 2511.14761；NAT-ARC（ECCV 2026，无 arXiv id）；ARC-AGI-3 2603.24621；Tycho 2607.28287；Mini-o3 2509.07969；ZoomEye 2411.16044；Visual Sketchpad 2406.09403；VLMs are blind 2407.06581；BabyVision 2601.06521；VDiff-Bench 2609.06245；DiffSpot 2605.29615；VLM-SubtleBench 2603.07888；Set-of-Mark 2310.11441；SCAFFOLD 2402.12058；Mano 2509.17336；SPM-Bench 2602.22971；Cheng…Zeljkovic（quailbot）2609.27302；Diao…Abe 2602.20669；Kalinin 等 2609.04015；AILA/AFMBench Nat. Commun. 16, 9104 (2025)；EPFL agentic AFM 2608.26198；ZEISS 基准 2608.05266；EAA 2602.15294；DISSECT 2604.06250；MaCBench Nat. Comput. Sci. 5, 952 (2025)；MatCha 2509.09307；CIMP 2604.24909；Scanbot JOSS 9(99), 6028 (2024)；DeepSPM Commun. Phys. 3, 54 (2020)；Barker 2024 ACS Nano 18, 2384。

