# VISTA-SPM

English · [中文](README.zh-CN.md)

VISTA-SPM is an image-first visual harness for multimodal models that operate a scanning probe microscope.

It ports VISTA (Han, Hu, Qiu, Wu, He, "VISTA: A Visual Harness for Reasoning in an Interactive World", arXiv 2610.02200) from ARC-AGI-3 games to the paper-reproduction scenarios of [STM-Bench](https://github.com/HamsterPark/STM-Bench). In each episode the model:

- sees the frames the instrument produced as images;
- keeps a lossless visual memory of all of them;
- looks at any frame again, and measures it;
- writes its own notes.

It acts through one MCP server, one instrument action per turn.

**Status:** research prototype in a private repository. First results are from 2026-10-10 (below). This repository holds the harness, the matrix tooling, the static perception benchmark and aggregated results. Episode ledgers and transcripts stay outside it.

## Why

MAST's agents saw STM frames through a fine-tuned vision model (VIGIL) that turned each image into text. VISTA's ablations on ARC-AGI-3 point the other way: give a frontier multimodal model the pixels, a lossless memory and tools to look again.

This repository tests that route on the STM-Bench simulator. VIGIL's text is kept as a control arm.

## First results (2026-10-10)

Setup:

- **Model:** claude-opus-5-5, effort high.
- **Actions:** the V0 primitive actions.
- **Scenarios:** P1 (Au(111) herringbone), P4 (positioning an Fe atom on Cu(111)) and P5 (qPlus force spectroscopy with Sader–Jarvis inversion).
- **Seeds:** 0–2.
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

Three seeds per cell do not separate full from mem.

**The claim fractions are not chance-corrected.** An audit of STM-Bench on 2026-10-12 found that reporting fixed constants, with the evidence requirements met, passes these shares of claims:

- about 0.26 in P1;
- about 0.41 in P5;
- none in P4.

The same constants fully reproduce about 2–3% of P1 episodes, 13% of P5 episodes and no P4 episodes. Read the fractions against these floors and together with the full-reproduction counts.

**Static perception benchmark (STM-BabyVision), 37 hard items:**

| Answerer | Correct |
|---|---|
| Claude with tools | 31 |
| Claude, images only | 23 |
| Classical baseline | 30 |

The gap between the two Claude arms comes almost entirely from herringbone periods, which need measurement.

Details, caveats and open decisions are in [docs/notes/2026-10-10-first-results.zh-CN.md](docs/notes/2026-10-10-first-results.zh-CN.md) (Chinese). The tables are in [results/](results/).

## How it works

```
python -m vista_spm run
 ├─ STM-Bench RuntimeHost: MAST CoreRuntime on the stmsim simulator
 ├─ VistaHost: tools per arm, frame archive, notes, instrument clock, budgets, the judge's ledger
 ├─ RpcServer (loopback, token)
 └─ ClaudeCodeRunner: claude -p segments, all built-in tools off, stream-json
       claude ── stdio MCP ──► mcp_bridge.py (stdlib only) ──► RpcServer ──► VistaHost
```

- **Observation:** raw frames with whole-pixel nearest-neighbour scaling. Labels and numbers travel in text next to the image, never drawn inside it.
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
| mem | image | history, notes, inspect |
| notes | image | history, notes |
| img | image | none |
| text | numeric text grid | history, notes, read_values |
| vigil-text | VIGIL text; spectra as numbers | history, notes |
| advisor | image | everything in full, plus MAST's detectors on request |

[docs/DESIGN.md](docs/DESIGN.md) is the design specification. [docs/staticbench.md](docs/staticbench.md) describes the static benchmark.

## Setup

1. **Python 3.13.**
2. **STM-Bench**, installed editable: `pip install -e <STM-Bench checkout>`. It must provide `RuntimeHost(controller_timeout_s=...)` ([HamsterPark/STM-Bench#1](https://github.com/HamsterPark/STM-Bench/pull/1)), because episodes raise MAST's controller reply timeout to 60 s.
3. **MAST** (private) for full episodes. Put its `MASTv2` directory on `PYTHONPATH` and set `MAST_ROOT` to the checkout.
4. **This repository:** run from its root, or `pip install -e . --no-deps`.
5. **Claude Code** CLI, logged in with a Claude subscription.

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
python -m vista_spm run --scenario P4_atom --seeds 0 --arm mem [--out DIR] [--run ID]
python -m vista_spm replay <run-dir> --single-file     # STM-Bench replay page for a finished run

python -m vista_spm.staticbench generate --out tmp/staticbench/v1 --types all --seeds 0-9
python -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 --out tmp/staticbench/run1 ...
```

Run outputs default to `STM_BENCH_DATA` (`~/stm_bench`). For an ablation matrix, freeze the code first and run from the snapshot ([scripts/README.md](scripts/README.md)):

```bash
python scripts/snapshot.py tmp/code/m2
python scripts/run_matrix.py --cwd tmp/code/m2 --out <dir> --run-id m2 --arms full,mem,img --scenarios P1_barth1990_au111,P4_atom_positioning_cu111 --seeds 0,1,2
python scripts/aggregate.py <dir> --run-id m2 --csv rows.csv --md summary.md
```

## Tests

```bash
python -m pytest -q -m "not isolated"     # unit and integration; MAST tests skip without MAST
python -m pytest -q -m isolated           # end-to-end scripted episode (one RuntimeHost per process)
```

The live Claude Code tests in `tests/test_runner.py` run only with `VISTA_LIVE_CLAUDE=1`.

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
| `docs/notes/` | research plan, literature review and first results (Chinese, kept verbatim) |

## Related repositories

| Repository | Visibility | Relation |
|---|---|---|
| [STM-Bench](https://github.com/HamsterPark/STM-Bench) | public | the simulator (`stmsim`), the scenarios and the judge (`stmbench`) |
| MAST | private | the instrument runtime and skills behind `act`, and VIGIL |
| [dsh-spm](https://github.com/HamsterPark/dsh-spm) | public | a TypeScript port of the visual memory |

dsh-spm's parity golden is generated from `vista_spm/frames.py`, `archive.py` and `viewtools.py`.

## License

No license has been chosen yet; the repository is private. Code derived from VISTA is MIT-licensed upstream; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
