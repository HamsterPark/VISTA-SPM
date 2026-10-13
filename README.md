# VISTA-SPM

English · [中文](README.zh-CN.md)

**A visual reasoning layer for autonomous scanning probe microscopy: see, remember, revisit, measure, and act.**

Atomic-scale experiments demand an agent that can reason with images over time. A useful observation may be a lattice in the current scan, a defect in an earlier frame, or a subtle change after a tip-conditioning pulse. [VISTA](https://arxiv.org/abs/2610.02200) offers a compelling starting point: give a multimodal model direct visual observations, a lossless archive, and tools to retrieve and inspect the evidence as its understanding develops.

**VISTA-SPM develops this principle into a general framework and reusable modules for scanning probe microscopy (SPM), with STM as its primary setting.** We combine visual memory with calibrated coordinates, physical measurements, experimental notes, and an action–observation loop. The framework serves our MAST automation toolkit and [dsh-spm](https://github.com/HamsterPark/dsh-spm), while its Python reference modules also support independent image analysis, agent development, and benchmarking. [STM-Bench](https://github.com/HamsterPark/STM-Bench) supplies the simulated instrument and evidence-based evaluation.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green) ![Reference implementation](https://img.shields.io/badge/reference_implementation-2026--10--13-lightgrey)

[Scientific motivation](#why-vista-for-stm) · [Reusable modules](#a-general-framework-with-reusable-modules) · [Results](#first-results-2026-10-10) · [Setup](#setup) · [Related repositories](#related-repositories)

## Why VISTA for STM

**Representation is part of reasoning.**

We adopt this as a design principle for autonomous microscopy. How images are presented, history is retained, spatial relationships are organised, and observations are connected to physical quantities shapes the hypotheses an agent can formulate, the evidence it can compare, and the experiments it can choose. Viewing, revisiting, magnifying, registering, and measuring all participate in that reasoning process. **Designing visual representations and evidence interfaces means designing the conditions under which an autonomous experimental system can reason scientifically.**

### Visual access is part of the reasoning method

The central contribution of Han et al.'s *VISTA: A Visual Harness for Reasoning in an Interactive World* is to make visual evidence persist and remain available for active inspection. The agent sees images directly, retrieves earlier observations, enlarges relevant regions, reads exact pixels, and maintains revisable notes. This gives an existing multimodal model a practical way to sustain visual reasoning through a long interaction. In the paper's sequential ablation with GPT-5.6 Sol at max reasoning effort on 25 public ARC-AGI-3 games, adding lossless visual memory and `inspect` raises Relative Human Action Efficiency (RHAE) from 70.05 to 94.10; adding exact pixel readout raises it to 99.00. These results motivate our choice of architecture. [VISTA ablation study](https://arxiv.org/html/2610.02200v1#S4.SS3) · [upstream implementation](https://github.com/joshhhhhan/VISTA).

For STM, this principle has a direct scientific role. Describing an image as “a clean terrace” or “a poor tip” selects one interpretation of a much richer measurement. Later evidence may make a previously overlooked defect, duplicated feature, or scan-line discontinuity decisive. Preserving the original observations lets the agent revisit that evidence, compare competing explanations, and choose a measurement that resolves the ambiguity. **Visual memory becomes part of the experimental method.**

### STM makes perception and control inseparable

A scanning tunnelling microscope probes a conducting surface with an atomically sharp tip. The tunnelling current depends strongly on tip–sample separation and electronic structure; in constant-current imaging, the feedback records the tip height needed to maintain the current setpoint. Atomic-scale contrast therefore carries information about the sample, the tip, and the measurement conditions. The connection between tunnelling current and local electronic states is central to STM theory ([Tersoff and Hamann, 1985](https://doi.org/10.1103/PhysRevB.31.805)). Interpreting an apparent protrusion requires its bias, spatial context, and acquisition history.

The operator must also infer the state of the instrument from the same data used to study the sample. Tip-apex changes can alter contrast across an entire frame; thermal drift and piezoelectric creep change the relation between commanded and actual position; feedback settings and scan direction affect how features appear. A voltage pulse may improve the tip and simultaneously change the local surface. The practical challenges developed in [MAST-public's STM introduction](https://github.com/HamsterPark/MAST-public#what-is-stm) make autonomous STM a problem of **partial observability, changing dynamics, and sequential experimental design**.

Autonomous SPM research already demonstrates the value of learning within this loop. DeepSPM combines area selection, image-quality assessment, and reinforcement learning for tip conditioning ([Krull et al., 2020](https://doi.org/10.1038/s42005-020-0317-3)); reinforcement learning has also enabled atomic manipulation under uncertain and changing tip conditions ([Chen et al., 2022](https://doi.org/10.1038/s41467-022-35149-w)). VISTA-SPM builds on this experimental perspective by providing a shared interface through which general-purpose multimodal models can inspect evidence, use quantitative tools, and revise an experiment across different tasks.

## What STM shares with ARC-AGI-3

### Images encode spatial structure

ARC's visual tasks make objects, repetition, symmetry, boundaries, and relative position central to reasoning. STM images are similarly rich in spatial relationships: periodic lattices, reconstruction domains, step edges, isolated adsorbates, vacancies, and repeated or distorted features. The informative pattern may span a whole terrace or occupy only a few pixels. Recognising it requires moving between the full field of view and local detail, preserving coordinates as well as appearance.

This is why VISTA's image access and `inspect` operation are especially relevant. A model can examine a reconstruction globally, magnify a defect, and then return to the same location in an earlier scan. VISTA-SPM extends that workflow with scan geometry and physical units, so a visual relationship can become a testable statement about a period, displacement, height, or spectral response.

### Experiments require learning through interaction

[ARC-AGI-3](https://arxiv.org/abs/2603.24621) evaluates agents in unfamiliar interactive environments, where exploration reveals how actions affect the scene and how to make progress. STM has established physical principles and explicit scientific objectives, yet each experiment presents an incompletely known tip–sample system. The transferable reasoning pattern is to form a hypothesis, choose an informative action, inspect the resulting change, and update the working model.

| Reasoning demand | ARC-AGI-3 | STM experiment |
|---|---|---|
| Parse a spatial scene | Objects, repeated patterns, boundaries, and their relations | Lattices, steps, adsorbates, defects, and scan artefacts |
| Infer hidden state | Discover action effects and environment rules | Infer tip condition, drift, local sample state, and measurement stability |
| Gather useful evidence | Explore with actions that distinguish possible rules | Change bias, rescan, compare directions, acquire a spectrum, or condition the tip |
| Track change over time | Compare frames before and after a move | Separate atom motion, field-of-view drift, and a change in tip contrast |
| Maintain a working model | Carry discoveries across a sequence of interactions | Retain stable landmarks, measurement conditions, tested hypotheses, and unfinished controls |
| Use actions efficiently | Solve with few actions relative to human participants | Use instrument time well and account for interventions that can alter the tip or sample |
| Verify progress | Reach the environment's completion condition | Support a scientific conclusion with the required acquisition evidence |

Consider an atom that appears displaced after manipulation. A convincing diagnosis compares the atom with neighbouring landmarks, checks for whole-frame drift, and examines whether the tip changed the appearance of other features. The relevant evidence spans several images and scales. An archive, region inspection, frame registration, and calibrated coordinates turn this into an explicit reasoning procedure.

STM adds continuous signals, noise, physical units, and acquisition-dependent contrast to this shared structure. **VISTA-SPM turns VISTA's visual interaction pattern into an instrument-aware experimental workflow.**

## A general framework with reusable modules

Our proposal is to organise autonomous microscopy around a persistent visual record:

```text
Observe → archive the evidence → form or revise a hypothesis
   ↑                                      ↓
Check the outcome ← act ← inspect, compare, and measure

Experimental notes preserve the working model throughout the loop.
```

The framework brings together four capabilities:

1. **Persistent, revisitable evidence.** Archive scan channels, forward/backward directions, partial scans, spectra, and derived measurements under stable frame IDs. The model can return to the evidence whenever a new question arises.
2. **Visual reasoning grounded in metrology.** Couple region inspection with pixel-to-scan coordinate conversion, metric grids, numerical readout, line profiles, Fourier analysis, feature localisation, and registered differences. Tools return measurements; the model relates them to physical hypotheses.
3. **Continuity across long experiments.** `GUIDE.md` holds the agent's revisable understanding; `WORKING.md` preserves the current plan and state across context replacement. Action history connects observations to the interventions that produced them.
4. **Controlled evaluation of the interface.** The same scenarios can be run with images, notes, memory, numerical readout, or additional measurement tools. This makes the observation and tool interface an explicit experimental variable.

The contribution has an independent scope: every microscope workflow needs to connect its images and measurements to an evolving account of the experiment. These modules address that recurring need at the boundary between scientific data and agent reasoning. MAST provides one execution environment; dsh-spm demonstrates how the same visual-memory and measurement design can be carried into another language and agent host.

| Module | What it provides | Reuse entry point |
|---|---|---|
| Frames and coordinates | Read `.sxm` / `.dat` data, orient channels, render views, and map display positions to scan coordinates | [frames.py](vista_spm/frames.py) |
| Visual archive | Preserve acquired arrays and metadata, retrieve any frame, and re-render selected regions | [archive.py](vista_spm/archive.py) |
| Inspection and readout | `inspect` for visual detail; `read_values` for values in pm, pA, V, or Hz | [viewtools.py](vista_spm/viewtools.py) |
| Quantitative tools | `line_profile`, `fft_peaks`, `detect_blobs`, `diff`, and `stats` | [measure.py](vista_spm/measure.py) |
| Notes and tool protocol | Persistent experimental notes and typed text/image replies for an agent host | [notes.py](vista_spm/notes.py), [protocol.py](vista_spm/protocol.py) |
| Evaluation | STM-BabyVision static perception tasks and configurable closed-loop ablations | [staticbench](docs/staticbench.md), [design specification](docs/DESIGN.md) |

For example, the visual modules can inspect an existing scan independently of the MAST runtime:

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

The reusable visual modules use STM-Bench's file reader and the scientific Python dependencies. Closed-loop execution in this repository uses MAST's runtime and skills; the supplied runner hosts the model through Claude Code. The maintained MAST integration and the TypeScript implementation in [dsh-spm](https://github.com/HamsterPark/dsh-spm) apply the design in their respective systems. This repository preserves the Python reference implementation, tests, static benchmark, and initial simulator ablations. [Dependency details](#standalone-use-and-runtime-integration).

## First results (2026-10-10)

**The initial simulator study shows the value of retaining and revisiting visual evidence.** The `mem` arm fully reproduced 6 of 9 episodes, compared with 2 of 9 for the latest-image-only arm. On the 37-item static perception set, Claude answered 31 correctly with tools and 23 with images alone. Together, these experiments examine the framework at two levels: extended experimental interaction and individual visual measurements.

Closed-loop setup:

- **Model:** claude-opus-5-5, effort high.
- **Actions:** the V0 primitive actions.
- **Scenarios:** P1 (Au(111) herringbone), P4 (positioning an Fe atom on Cu(111)) and P5 (qPlus force spectroscopy with Sader–Jarvis inversion).
- **Seeds:** 0–2.
- **Images:** no graph paper yet; it was added on 2026-10-12.
- **Scoring:** the official STM-Bench judge.

| Arm | Observation and tools | Claims verified | Fully reproduced |
|---|---|---|---|
| mem | images, notes, look again at any frame | 0.83 | 6/9 |
| full | mem plus exact values and five measurement tools | 0.72 | 5/9 |
| img | the latest image only | 0.43 | 2/9 |
| vigil-text | VIGIL text instead of images (spectra as numbers), notes | 0.25 | 1/9 |

Each comparison below pairs the two arms' episodes on the same scenario and seed:

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| mem vs img | 5 | 4 | 0 |
| mem vs vigil-text | 6 | 3 | 0 |
| full vs mem | 0 | 8 | 1 |

The paired results favour `mem` over `img` and `vigil-text` in this matrix. Here, `mem` adds notes, the archive, and `inspect` together; `full` adds numerical readout and five measurement tools. With three seeds per scenario, the matrix characterises these bundles and leaves the individual contributions for more targeted ablations. The two richer arms also used different strategies: `mem` magnified frames 7.8 times per episode, while `full` measured 15 times and magnified twice.

**Scoring context.** Claim fractions are reported without chance correction. The 2026-10-12 audit measured the following fixed-constant baselines, conditional on meeting acquisition-evidence requirements:

| Scenario | Share of claims passed by fixed constants | Full reproduction with the same constants |
|---|---|---|
| P1 | about 0.26 | about 2–3% |
| P4 | 0 | 0 |
| P5 | about 0.41 | about 13% |

These baselines provide the reference for reading claim fractions alongside paired outcomes and full-reproduction counts.

**Static perception benchmark (STM-BabyVision), 37 hard items:**

| Answerer | Correct |
|---|---|
| Claude with tools | 31 |
| Claude, images only (no tools at all) | 23 |
| Classical baseline | 30 |

The gap between the two Claude arms comes almost entirely from herringbone periods, which need measurement.

**Experimental scope and provenance.** All reported results are simulator experiments. The archived runs used MAST and the 2026-10-09/10 STM-Bench revision, before public rules v1 and subsequent simulator changes. Exact historical replay requires the original dependency snapshot, including private components; the frozen file hashes are in [results/m1/code_manifest.json](results/m1/code_manifest.json). The `m1-surface` tag with `--grid off` preserves the model-facing prompt, tool schemas, and tool behaviour. Current tools incorporate the [2026-10-13 reference fixes](docs/notes/2026-10-13-reference-fixes.zh-CN.md). Experiment details are in the [first-results report](docs/notes/2026-10-10-first-results.zh-CN.md), and aggregated tables are in [results/](results/).

## How it works

```
python -m vista_spm run
 ├─ STM-Bench RuntimeHost: MAST CoreRuntime on the stmsim simulator
 ├─ VistaHost: tools per arm, frame archive, notes, instrument clock, budgets, the judge's ledger
 ├─ RpcServer (loopback, token)
 └─ ClaudeCodeRunner: claude -p segments, all built-in tools off, stream-json
       claude ── stdio MCP ──► mcp_bridge.py (stdlib only) ──► RpcServer ──► VistaHost
```

- **Observation:** views rendered from acquired arrays, with nearest-neighbour enlargement and block-mean reduction for oversized frames. Labels and numbers travel in text next to the image; archived arrays remain available for detailed inspection and numerical measurement.
- **Graph paper:** the only thing drawn on a scan image, as VISTA draws cell lines on its ARC frames.
  - It is a faint amber metric grid of 1, 2 or 5 × 10ⁿ nm squares, counted from the frame's top-left corner.
  - A view has at most 16 cells across, so zoomed views get finer squares.
  - The spacing `grid_nm` is in each image's label. `--grid off` removes the grid.
- **Visual memory:** every frame, channel, direction and partial snapshot is archived losslessly, with a stable id.
  - `inspect` crops and enlarges any archived region, channel or flattening.
  - `read_values` returns exact numbers.
- **Measurement:** `line_profile`, `fft_peaks`, `detect_blobs`, `diff` and `stats`. They measure and never interpret.
- **Notes:** GUIDE.md and WORKING.md. When Claude Code compacts its context, the runner hands over through WORKING.md, as in VISTA.
- **Instrument:** `act` runs one V0 primitive through MAST's skills: scan, bias and setpoint, tip move, pulse, poke, STS, Δf(z), PLL, atom move, drift compensation, wait, withdraw and approach. The instrument clock runs only while an action executes.
- **Reporting:** `report_result` and `report_tip_state` feed STM-Bench's judge, which needs both the value and the acquisition evidence.

Each arm has these tools:

| Arm | Observation | Tools besides act and the two report tools |
|---|---|---|
| full | image | history, notes, inspect, read_values, measurement |
| read | image | history, notes, inspect, read_values (VISTA's tool set) |
| mem | image | history, notes, inspect |
| notes | image | history, notes |
| img | image | none |
| text | numeric text grid | history, notes, read_values |
| vigil-text | VIGIL text; spectra as numbers | history, notes |
| advisor | image | everything in full, plus MAST's detectors on request |

[docs/DESIGN.md](docs/DESIGN.md) is the design specification. [docs/staticbench.md](docs/staticbench.md) describes the static benchmark.

## Engineering

| Problem | Mechanism | Where |
|---|---|---|
| Let the model act only through the harness | Claude Code runs with every built-in tool off. Its one MCP server is a standard-library bridge to a loopback RPC server that checks a token. | [mcp_bridge.py](vista_spm/mcp_bridge.py), [rpc.py](vista_spm/rpc.py), [claude_runner.py](vista_spm/claude_runner.py) |
| One action, one observation | `act` is refused until the previous action's result has reached the model. The simulator clock is paused except while an action executes. | [host.py](vista_spm/host.py) |
| A lossless visual memory | Every frame, channel, direction and partial snapshot is archived with a stable id. Any region can be cropped and enlarged again or read as exact values. Pixel and scan coordinates convert exactly, sub-pixel included. | [archive.py](vista_spm/archive.py), [viewtools.py](vista_spm/viewtools.py), [frames.py](vista_spm/frames.py) |
| Measurement without interpretation | Line profiles with step candidates, FFT peaks, elliptical blob fits, sub-pixel frame registration and statistics. Replies carry numbers, never verdicts such as "double tip". | [measure.py](vista_spm/measure.py) |
| Treat what the model sees as an experiment parameter | Every arm's system prompt and tool schemas are pinned by hash, with and without graph paper and qPlus. A tag preserves the first matrix's model-facing interface and tool behaviour. | [test_surface.py](tests/test_surface.py) |
| Keep the truth out of observations | Tool replies, images and action descriptions must not carry hidden scenario values, tolerances or judge-only data. The ledger keeps the truth for the judge and for replays. | [AGENTS.md](AGENTS.md) (invariant 1) |
| Long episodes | VISTA's compaction protocol: before Claude Code compacts, the model writes WORKING.md, and a new segment resumes from the notes and the action history. | [compact_hook.py](vista_spm/compact_hook.py), [claude_runner.py](vista_spm/claude_runner.py) |
| Billing safety | The runner strips API credentials from the Claude Code child process. It aborts a session whose init reports an API key as the credential source. | [claude_runner.py](vista_spm/claude_runner.py) |
| Ablations that can be audited | Frozen code snapshots with hash manifests, and a seed-major matrix runner that pauses at usage limits. Failed episodes are found and rerun, and finished ones can be re-judged offline. | [scripts/](scripts/) |

## Standalone use and runtime integration

| Runs with STM-Bench only (Python 3.13, STM-Bench 84fb385 or later, packages from PyPI) | Needs MAST or MAST-public |
|---|---|
| the test suite: 282 tests pass; the isolated end-to-end episode is skipped | closed-loop episodes (`run`): MAST's runtime and skills execute `act` on the simulator |
| `tools`: every arm's system prompt and tool schemas | the `advisor` and `vigil-text` arms: MAST's detectors and the VIGIL vision model |
| the visual memory and the measurement tools on `.sxm` / `.dat` files | the static benchmark's atomic-resolution items, whose truth is MAST's judge |
| the static benchmark: `generate`, the classical baseline (`answer`), `score`, `report`, and `qa` with Claude Code | |
| `replay` pages for recorded episodes | |

**MAST-public instead of MAST.** This was checked on 2026-10-13 with MAST-public 9552b27, in an environment with its `MASTv2/requirements-v2.txt` installed:

- the isolated end-to-end episode passes;
- every one of the 22 skills behind `act` is registered (MAST-public registers 481 skills);
- the static benchmark's atomic-resolution items are generated.

MAST-public leaves out VIGIL's weights, so the `vigil-text` arm and the VIGIL readings of the `advisor` arm do not work with it. The advisor's classical detectors are included. No model-driven episode has been run on MAST-public yet.

## Setup

1. **Python 3.13.**
2. **STM-Bench** at commit 84fb385 or later. It provides `RuntimeHost(controller_timeout_s=...)` and the `.sxm`/`.dat` reader:

   ```bash
   pip install "stm-bench @ git+https://github.com/HamsterPark/STM-Bench@84fb385b990dd867580933c72b9ad9b5f6547d76"
   ```

   For development, install a checkout instead: `pip install -e <STM-Bench checkout>`.
3. **This repository:** `pip install -e ".[dev]"`, or run from its root.
4. **MAST or [MAST-public](https://github.com/HamsterPark/MAST-public)**, for closed-loop episodes only. Install its `MASTv2/requirements-v2.txt`, put its `MASTv2` directory on `PYTHONPATH` and set `MAST_ROOT` to the checkout.
5. **Claude Code** CLI, logged in with a Claude subscription, for episodes and the static benchmark's Claude runs.

On Windows also set `PYTHONUTF8=1`.

## Billing

The runner uses Claude subscription authentication by default (`RunnerConfig.auth = "subscription"`). To preserve that choice when MAST exports provider credentials into the environment, it applies two checks:

- it removes `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the Claude Code child process;
- it aborts any session whose init message reports an API key source.

Use this default for subscription-based runs.

## Usage

```bash
python -m vista_spm preflight                          # Claude Code + MCP bridge, one cheap call
python -m vista_spm tools --arm full [--qplus]         # the tool surface an arm shows the model
python -m vista_spm run --scenario P4_atom --seeds 0 --arm mem [--out DIR] [--run ID] [--grid off]
python -m vista_spm replay <run-dir> --single-file     # STM-Bench replay page for a finished run

python -m vista_spm.staticbench generate --out tmp/staticbench/v1 --types all --seeds 0-9
python -m vista_spm.staticbench answer tmp/staticbench/v1 --mode baseline --out tmp/staticbench/baseline.jsonl
python -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 --out tmp/staticbench/run1 ...
```

Run outputs default to `STM_BENCH_DATA` (`~/stm_bench`). `--grid off` renders without the graph paper and files the runs under `<arm>-nogrid`. The option also exists on `tools`, `preflight`, `run_matrix.py` and the static benchmark's `qa`.

For an ablation matrix, freeze the code first and run from the snapshot ([scripts/README.md](scripts/README.md)):

```bash
python scripts/snapshot.py tmp/code/m2
python scripts/run_matrix.py --cwd tmp/code/m2 --out <dir> --run-id m2 --arms full,mem,img --scenarios P1_barth1990_au111,P4_atom_positioning_cu111 --seeds 0,1,2
python scripts/aggregate.py <dir> --run-id m2 --csv rows.csv --md summary.md
```

## Tests

```bash
python -m pytest -q -m "not isolated"     # unit and integration; needs no MAST
python -m pytest -q -m isolated           # end-to-end scripted episode (needs MAST; one RuntimeHost per process)
```

On STM-Bench 84fb385 without MAST: 282 passed, 4 skipped, 1 deselected. With MAST, or with MAST-public 9552b27, the isolated episode passes as well.
- **Skipped:** the live Claude Code tests in `tests/test_runner.py`, which run only with `VISTA_LIVE_CLAUDE=1`.
- **Deselected:** the isolated end-to-end episode.

`tests/test_repo_hygiene.py` guards three things:

- no machine paths;
- no second readers of `STM_BENCH_*` or `MAST_ROOT`;
- the controller vendor's name only as MAST's identifiers.

## Layout

| Path | Contents |
|---|---|
| `vista_spm/` | the harness |
| `vista_spm/staticbench/` | STM-BabyVision: generator, questions, scorer, QA host, classical baseline |
| `tests/` | tests; `fakes.py` and `fake_claude.py` stand in for the instrument and the CLI |
| `scripts/` | matrix tooling: snapshot, runner, aggregation, rerun finder, offline re-judge |
| `results/` | aggregated results of past runs (records, not regenerated) |
| `docs/DESIGN.md`, `docs/staticbench.md` | design specification, static benchmark |
| `docs/notes/` | research plan, literature review, first results, and reference implementation fixes (Chinese) |

## Related repositories

VISTA-SPM sits at the visual reasoning layer of our SPM research software. The surrounding projects connect instrument operation, experimental evaluation, and scientific interpretation:

| Repository | Role in the research workflow | Connection to VISTA-SPM |
|---|---|---|
| [MAST-public](https://github.com/HamsterPark/MAST-public) | STM automation: instrument runtime, skills, state feedback, and experimental workflows | The limited public edition of MAST can execute this repository's closed-loop actions. Its README develops the scientific motivation for autonomous STM; the maintained visual harness lives in MAST. |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | SPM tools and runtime integration for DeepSeek Harness | Carries the visual memory, calibrated readout, and measurement tools into TypeScript, with reference outputs exported from this Python implementation. |
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | Simulated microscope, scientific scenarios, acquisition records, and evidence-based judging | Supplies the instrument and evaluation environment for VISTA-SPM ablations, together with the `.sxm` / `.dat` reader. |

The MAST runtime supplies the actions; VISTA-SPM organises the visual evidence; STM-Bench evaluates the resulting experimental behaviour. In dsh-spm, [runtime integration](https://github.com/HamsterPark/dsh-spm/blob/master/packages/bundle/dsh-spm/src/visual-harness.ts) connects the TypeScript archive, observation delivery, and action gate. Its parity fixtures are generated from this repository's `frames.py`, `archive.py`, `viewtools.py`, and `measure.py`. This provides a concrete path from a shared scientific design to implementations in different agent systems.

## License

[MIT](LICENSE). Code derived from VISTA is MIT-licensed upstream; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
