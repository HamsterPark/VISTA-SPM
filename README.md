# VISTA-SPM

English · [中文](README.zh-CN.md)

**Letting a multimodal model see, remember, re-inspect and measure what a scanning tunnelling microscope shows it.**

VISTA-SPM is an image-first visual harness for frontier multimodal models that operate a scanning tunnelling microscope (STM). It ports VISTA (Han, Hu, Qiu, Wu, He, *VISTA: A Visual Harness for Reasoning in an Interactive World*, [arXiv 2610.02200](https://arxiv.org/abs/2610.02200)) from the ARC-AGI-3 games to the paper-reproduction scenarios of [STM-Bench](https://github.com/HamsterPark/STM-Bench), a simulated STM with an evidence-based judge.

Its question: should the model look at the pixels itself, with a lossless memory of every frame, a magnifier and an exact readout? Or should a vision model turn each frame into text first?

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![MIT license](https://img.shields.io/badge/license-MIT-green) ![Snapshot](https://img.shields.io/badge/snapshot-2026--10--13-lightgrey)

**Status: a slim public demonstration.**
- The maintained harness lives inside MAST, the lab's STM automation toolkit. MAST is private; [MAST-public](https://github.com/HamsterPark/MAST-public) is its limited public edition.
- [dsh-spm](https://github.com/HamsterPark/dsh-spm) ports the harness to TypeScript.
- This repository keeps a self-contained snapshot: the harness, its 282 tests, the static perception benchmark and the first ablation results.
- All results come from the simulator. Nothing here has run on a real instrument.
- The visual memory, the measurement tools and the static benchmark need only STM-Bench. Closed-loop episodes also need MAST, and its public edition MAST-public can stand in for it ([details](#what-runs-without-mast)).

## Why an ARC-AGI-3 harness for a microscope

### STM in brief

A scanning tunnelling microscope holds a sharp metal tip a few ångströms above a conducting surface and records the tunnelling current while it scans, so its images resolve single atoms. The same tip takes spectra and can drag single atoms across the surface. Everything the instrument shows passes through that tip, whose atomic structure nobody can see and which changes as it is used. [MAST-public's README](https://github.com/HamsterPark/MAST-public#what-is-stm) explains STM, and why it is hard for autonomous agents, at more length.

### The same kind of problem

ARC-AGI-3 drops an agent into games whose rules and goals are hidden. The agent must act, watch what changes and infer the rules from the frames, and it is scored on how few actions it needs compared with people. Operating an STM is the same kind of problem, with physics in it:

| | ARC-AGI-3 game | STM experiment (an STM-Bench scenario) |
|---|---|---|
| What the agent sees | a 64×64 frame of 16 colours after each action | scan frames (topography, current; forward and backward) and spectra after each instrument action |
| What is hidden | the rules and the goal | the tip's state, the drift, the sample and the scenario's values, drawn per seed |
| How it acts | one game action at a time | one instrument action at a time: scan, set the bias, move the tip, pulse, poke, take a spectrum, drag an atom |
| What actions cost | the score rewards solving in few actions | instrument time and actions; a tip pulse or a dragged atom cannot be undone |
| Where the evidence is | in how a frame changes after an action | in how frames differ: did the atom move, did the tip change, did the sample drift? |
| What counts as success | the level is solved | a paper's claim is reproduced, with the measurements the judge asks for |

In both, the decisive evidence is often a comparison across time: a frame before an action against a frame after it. VISTA's archive and magnifier give the model exactly that.

Among the agents surveyed in the [research notes](docs/notes/2026-10-09-research-and-plan.zh-CN.md), quailbot ([arXiv 2609.27302](https://arxiv.org/abs/2609.27302)) is the only report of a frontier-LLM agent on a real STM. People diagnosed its failures by comparing frames from before and after an action.

VISTA trains no vision model; its gains come from the harness. In its ablation on the 25 public ARC-AGI-3 games:
- replacing the official text grid with images took RHAE from 13.33 to 47.32;
- further steps, among them notes (+4.2), the lossless memory with `inspect` (+24.1) and `read_pixels` (+4.9), took it to 99.00.

### What has to change

STM frames are not ARC frames:
- they hold continuous heights and currents in physical units, with noise, drift and the tip's blur;
- the feature that matters may be one atom a few pixels wide in a frame of 512;
- spectra are curves, not pictures.

So the port keeps VISTA's whole kit (images, notes, an archive of every frame, the `inspect` magnifier, an exact readout and graph paper). It adapts two of the tools and adds what a physicist reaches for:

- the exact readout, `read_values`, returns pm, pA or V instead of palette indices;
- the graph paper is metric, in squares of 1, 2 or 5 × 10ⁿ nm instead of ARC's cells;
- five measurement tools measure and never interpret: `line_profile`, `fft_peaks`, `detect_blobs`, `diff` and `stats`;
- spectra are plotted on request and can be read as numbers;
- the instrument clock runs only while an action executes, so thinking costs no instrument time.

MAST's own agents saw STM frames through a fine-tuned vision model, VIGIL, which turned each image into text. This repository tests VISTA's route against that one on the STM-Bench simulator, and keeps VIGIL's text as a control arm.

## First results (2026-10-10)

Setup:

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

Three seeds per cell do not separate full from mem. Each step also changes more than one thing:

- mem adds three things over img: notes, the frame archive and `inspect`.
- full adds two kinds of tools over mem: the exact readout and five measurement tools.

In the episodes the two arms used different tools for the same job. mem magnified frames 7.8 times per episode; full measured 15 times and magnified twice.

**The claim fractions are not chance-corrected.** An audit of STM-Bench on 2026-10-12 found that reporting fixed constants, with the evidence requirements met, passes these shares of claims:

- about 0.26 in P1;
- about 0.41 in P5;
- none in P4.

The same constants fully reproduce about 2–3% of P1 episodes, 13% of P5 episodes and no P4 episodes. Read the fractions against these floors and together with the full-reproduction counts.

**Static perception benchmark (STM-BabyVision), 37 hard items:**

| Answerer | Correct |
|---|---|
| Claude with tools | 31 |
| Claude, images only (no tools at all) | 23 |
| Classical baseline | 30 |

The gap between the two Claude arms comes almost entirely from herringbone periods, which need measurement.

**What these numbers are.** All of them are a record:
- **What produced them.** The runs used MAST and the STM-Bench of 2026-10-09/10, from before its public rules (v1) and its later simulator changes. The frozen code's file hashes are in [results/m1/code_manifest.json](results/m1/code_manifest.json). These runs cannot be repeated from public components alone.
- **What the tag reproduces.** The tag `m1-surface` with `--grid off` reproduces the system prompt, tool surface and tool behaviour that the episodes saw. The reference fixes of 2026-10-13 changed some tools since ([notes](docs/notes/2026-10-13-reference-fixes.zh-CN.md)).
- **Where the details are.** Details and caveats are in [docs/notes/2026-10-10-first-results.zh-CN.md](docs/notes/2026-10-10-first-results.zh-CN.md) (Chinese); the tables are in [results/](results/).

## How it works

```
python -m vista_spm run
 ├─ STM-Bench RuntimeHost: MAST CoreRuntime on the stmsim simulator
 ├─ VistaHost: tools per arm, frame archive, notes, instrument clock, budgets, the judge's ledger
 ├─ RpcServer (loopback, token)
 └─ ClaudeCodeRunner: claude -p segments, all built-in tools off, stream-json
       claude ── stdio MCP ──► mcp_bridge.py (stdlib only) ──► RpcServer ──► VistaHost
```

- **Observation:** raw frames with whole-pixel nearest-neighbour scaling. Labels and numbers travel in text next to the image.
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
| Treat what the model sees as an experiment parameter | Every arm's system prompt and tool schemas are pinned by hash, with and without graph paper and qPlus. The first matrix stays reproducible from a tag. | [test_surface.py](tests/test_surface.py) |
| Keep the truth out of observations | Tool replies, images and action descriptions must not carry hidden scenario values, tolerances or judge-only data. The ledger keeps the truth for the judge and for replays. | [AGENTS.md](AGENTS.md) (invariant 1) |
| Long episodes | VISTA's compaction protocol: before Claude Code compacts, the model writes WORKING.md, and a new segment resumes from the notes and the action history. | [compact_hook.py](vista_spm/compact_hook.py), [claude_runner.py](vista_spm/claude_runner.py) |
| Billing safety | The runner strips API credentials from the Claude Code child process. It aborts a session whose init reports an API key as the credential source. | [claude_runner.py](vista_spm/claude_runner.py) |
| Ablations that can be audited | Frozen code snapshots with hash manifests, and a seed-major matrix runner that pauses at usage limits. Failed episodes are found and rerun, and finished ones can be re-judged offline. | [scripts/](scripts/) |

## What runs without MAST

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

Episodes run on the Claude subscription only. STM-Bench's RuntimeHost exports MAST's provider keys into the process environment, and early episodes were billed to an API key because of it.

`RunnerConfig.auth = "subscription"` is the default. It does two things:

- it removes `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the Claude Code child process;
- it aborts any session whose init message reports an API key source.

Keep it.

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
| `docs/notes/` | research plan, literature review, first results and the reference fixes (Chinese). The public copies leave out MAST internals, unpublished model evaluations and local paths. |

## Related repositories

| Repository | Visibility | Relation |
|---|---|---|
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | public | the simulator (`stmsim`), the `.sxm`/`.dat` reader, the scenarios and the judge (`stmbench`) |
| MAST | private | the instrument runtime and skills behind `act`, the detectors and VIGIL; the maintained home of this harness |
| [MAST-public](https://github.com/HamsterPark/MAST-public) | public | MAST 6.5.0's limited public source edition; it can stand in for MAST here, without VIGIL ([details](#what-runs-without-mast)) |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | public | a TypeScript port of the visual memory and the measurement tools |

dsh-spm's parity goldens are generated from `vista_spm/frames.py`, `archive.py`, `viewtools.py` and `measure.py`.

## License

[MIT](LICENSE). Code derived from VISTA is MIT-licensed upstream; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
