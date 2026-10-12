# STM-BabyVision — a static perception benchmark on the STM-Bench simulator

VISTA-SPM work package D (`docs/DESIGN.md` §9). Frames with known truth are
generated on the simulator, each paired with a question in the BabyVision style; a model
answers through the same visual-memory and measurement tools a VISTA episode has, with **no
instrument action**. It calibrates rendering and tool choices before full episodes and
measures STM *operational perception*: is the tip good, how many atoms are there and where,
did one move, how far did the sample drift, what is the herringbone period.

## Quick start

```bash
# from the VISTA-SPM root, with the environment of docs/DESIGN.md §4
$PY -m vista_spm.staticbench generate --out tmp/staticbench/v1 --types all --seeds 0-9 --jobs 8 --overlays
$PY -m vista_spm.staticbench validate tmp/staticbench/v1
$PY -m vista_spm.staticbench answer   tmp/staticbench/v1 --mode baseline          # classical reference row
$PY -m vista_spm.staticbench score    tmp/staticbench/v1 tmp/staticbench/v1/answers_baseline.jsonl
$PY -m vista_spm.staticbench report   tmp/staticbench/v1 --run baseline=tmp/staticbench/v1/answers_baseline.jsonl --out tmp/staticbench/v1/report
$PY -m vista_spm.staticbench qa       tmp/staticbench/v1 --item <id>              # the QA host's first message + tools
```

`--types` defaults to the core set `tip,adatoms,move,drift,herringbone` (19 items per seed);
`all` adds `step,tipchange,atomic` (23 per seed). `answer --mode oracle` answers with the truth
(scorer sanity check: 100 %), `--mode abstain` abstains everywhere.

## Layout of a dataset

```
<out>/items.jsonl        one item per line (schema: items.py)
<out>/frames/*.sxm|.dat  the frames the items refer to; neutral names (a hash), and an .sxm header's SCAN_FILE
                         is rewritten to that name: nothing in a name or header says how it was made or
                         what the item's answer is (no session path, no scenario id, empty comment)
<out>/manifest.json      counts per subtype, class balance, difficulty, failures, wall time, versions
<out>/_overlays/*.png    (with --overlays) every frame with its truth drawn on it — for people, never for a model
```

An item: `{id, type, subtype, question, frames, answer_format {kind, choices?, unit?}, truth
{value, accept?, optional?, ...}, tolerance {...}, provenance {scenario, seed, world_seed,
tip_condition, tip_params, hidden_draws, frames [geometry, timing, drift per frame], sim_s,
...}, difficulty}`. The question names its data *frame 1, frame 2, … / spectrum k* in list order.
Positions are **scan-frame nm** (the frame header's coordinates: centre = scan offset, x right,
y up, rotated by the scan angle) — the frame `report_result` and the measurement tools use.

## Item types

| type.subtype | data | answer | truth (from the simulator) | correct when |
|---|---|---|---|---|
| `tip.condition` | 20 nm Cu(111)+Fe or 30 nm Au(111)+adatoms topograph, plus a dI/dV spectrum on bare terrace | choice: good / blunt / double / unstable / dirty | the condition forced on the tip (`Tip.apply_condition`, scale and angle drawn per item); an *unstable* item always has ≥1 tip change inside the topograph, every other class has none in the whole recipe | equal |
| `adatoms.count` | 15–30 nm Cu(111)+Fe topograph; one frame per seed with a good tip, one with a double or blunt tip | count | atoms whose apparent centre is ≥0.4 nm inside the frame; atoms within ±0.4 nm of an edge are optional | `min ≤ a ≤ max`; also reported: within ±1 |
| `adatoms.positions` | same frame | `[[x, y], …]` nm | apparent position of every atom in this frame (per-row drift solution, below) | Hungarian match within 0.3 nm, no miss, no spurious point; score = F1 |
| `move.detect` | two frames of one 15–20 nm window, before / after the `MoveAtomTo` sequence (or a matched wait) | bool | did any atom's lattice site change (balanced 50/50) | equal |
| `move.which` | same pair (moved only) | `[x, y]` nm | where the moved atom appears in frame 1 | within 0.3 nm |
| `move.distance` | same pair (moved only) | number, nm | the displacement as frame 2 shows it; frame 1's and the true surface distance are accepted too | within max(0.25 nm, 15 %) |
| `drift.vector` | two frames of one 20–40 nm window, 0–10 min apart (one pair per even seed with partial drift compensation) | `[dx, dy]` nm | apparent displacement of fixed sample points from frame 1 to frame 2 (median over a grid of points) | within max(0.3 nm, 20 %) |
| `herringbone.period` | 80–150 nm Au(111) frame inside one rotational domain, step-free | number, nm | stripe period along the arm (`period·cos α`, what an FFT sees), with the drift distortion of this frame applied; either arm accepted | within 0.15 nm |
| `herringbone.orientation` | same frame | number, deg mod 180 | stripe direction in image coordinates (CCW from +x); either arm or their mean accepted | within 6° mod 180 |
| `step.height` | 50–80 nm frame across monatomic steps, points A and B in the question | number, pm | (level(B) − level(A)) × material step (Au 235.4 pm, Cu 208 pm) | within max(30 pm, 10 %) |
| `tipchange.row` | 256-row topograph with exactly one spontaneous tip change (|dz| ≥ 30 pm, ≥12 rows from either edge; some scans upward) | number, row | first image row below the discontinuity (0 = top) | within ±2 rows |
| `atomic.resolved` | 5 nm frame, apex σ 0.035–0.065 nm (yes) or 0.12–0.25 nm (no) | bool | **reference metric**: MAST `assess_atomic_phase` on the forward Z (expected period = first-order lattice period) — the B4 judge's rule, not ground truth | equal |

Abstaining (`{"abstain": true}`) is never correct and is counted separately from wrong
answers; a malformed value is *invalid*; an item without any answer is *missing*.

## How the truth is computed

* **No MAST in the loop.** `_sim.py` builds the same world a scenario builds
  (`Scenario.build_world`, templates derived from the P4 and P1 scenario files), freezes the
  world clock (`wall() ≡ 0`, sim time set explicitly) and calls the world's own verbs — the ones
  the controller modules call for `Scan.FrameSet/Action/Save`, `BiasSpectr.Start`,
  `FolMe.XYPosSet` — so frames come from the episode's renderer and `.sxm`/`.dat` writers, and
  generation is deterministic (identical items from the same `(types, seeds)`; a test pins
  it). A MAST-driven run is not deterministic: its sim time follows the wall clock of the
  polling loop. `MoveAtomTo` is re-enacted (`Bench.drag_atom`: start behind the atom, lower the
  resistance, drag along ≤0.1 nm waypoints, raise the setpoint before leaving).
* **Positions under drift.** The renderer images sample point `scan_xy(pixel) + o(t_row)`, one
  drift offset per row. `FrameRec.sample_to_scan` solves `p + o(t(row(p))) = a` with the offsets
  recorded right after each frame was rendered. Checked against blob centroids: < 0.06 nm (a
  constant +0.03–0.05 nm along +x is the feedback loop's lag). `claims.to_sample_frame`'s
  one-offset-per-frame (frame start) mapping would put the 158 atoms of the v1 adatom frames off
  by 0.94 nm median, 2.1 nm p90, 3.1 nm max.
* **Herringbone.** The arm wavevectors `(2π/T)(e_u ± tan α e_v)` mapped through the frame's
  rotation and its drift gradient (`k' = R(θ)ᵀk + (0, k·g)`); the drift moves a period by up to
  ~0.2 nm in a slow frame, more than the tolerance — the undistorted value is in the truth too.
* **Drift.** Not `−(o(t2) − o(t1))` at one row: a feature drifting down the frame is imaged
  some rows later in frame 2, and that extra drift adds ~5 % (0.3 nm on a 6 nm shift); the truth
  is the per-point solution, which a correlation of the two images reproduces to 0.04 nm.

## The QA host (for the runner)

`qa_host.QAHost(item, dataset_dir, work_dir, arm="full")` implements `protocol.RunnerHost`:
the item's frames are archived (`archive.Archive`, turn 0; frame k = `host.frame_ids[k]`) and
shown in the first message as their default observation images (the `text` arm gets
`text_grid`s instead); tools are the arm's `inspect` / `read_values` (`viewtools.VIEW_TOOLS`) and
`line_profile`, `fft_peaks`, `detect_blobs`, `diff`, `stats` (`measure.MEASURE_TOOLS`) plus
`answer(value | abstain, confidence?, note?)`, validated against the item's answer format. No
`act`, no notes, no history; a tool-call budget (60) after which only `answer` runs.
`host.result()` is one answers-file line. `qa_host.run_items(dataset, out, drive,
work_root=...)` loops over items with a caller-supplied `drive(host)` (e.g. a
`ClaudeCodeRunner` around the host), appends each result as it comes and resumes after a crash.
`qa_host.runner_settings()` returns the `RunnerConfig` fields a QA session needs (no act to
acknowledge, `answer` as the progress tool, native compaction, a QA system prompt).

### Answering with Claude Code (`claude_qa.py`)

```bash
$PY -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 --out tmp/staticbench/pilot1 \
    --pilot --tools full --model claude-opus-5-5 --effort high --budget-usd 15
```

One fresh `QAHost` per item (`<out>/<tools>/work/<id>/session<k>/`), an `rpc.RpcServer` on it and a
`claude_runner.ClaudeCodeRunner` with `runner_settings()`; the run ends as soon as a valid `answer`
makes the host terminal (outcome `done`). `--tools full` = inspect, read_values, the measurement
tools and answer; `--tools image` = answer only (the images-only arm: the first message's images
are all the model gets, and the message spells out the display-pixel → scan-nm conversion).
Both arms draw the metric graph paper on the frames since 2026-10-12 (`docs/DESIGN.md` §5.2);
`--grid off` shows them as the pilot1 and hard1 runs saw them and writes `<out>/<tools>-nogrid/`.
Every item's `host.result()` plus the runner's usage (`driver.usage.cost_usd`, tokens, turns, tool
counts, outcome) is appended to `<out>/<tools>/answers.jsonl` at once; reruns skip recorded items
(`--retry-unanswered` re-runs null answers). No new item starts once the dollars recorded under
`<out>` (all arms, preflights included) reach `--budget-usd`, and each Claude Code process gets
`--max-budget-usd` = min(`--segment-cap-usd`, what is left / segment limit). At the end every arm is
scored (`<arm>/scores.jsonl`) and `<out>/report/` compares them with the classical baseline on the
same items. `--pilot` picks one normal-difficulty item per subtype; `--items id,id` names them.
Paths are made absolute before they reach the runner (Claude Code runs in its own cwd).

## Baseline (`baseline.py`)

Plain image processing from the frame files and the question text only (never `truth` or
`provenance`): median-background blob detection with intensity centroids and a ghost-offset
detector, column-median row-jump detection, a running-median dI/dV peak test, phase
correlation refined by matched atoms, a Hann-windowed FFT for the herringbone, a tilt-robust
line profile for steps, and an FFT peak-to-background test for atomic resolution.

## Known caveats (simulator fidelity and item design)

* **Fast-axis-only tip blur.** The renderer smears the smooth surface with the tip-radius
  kernel along the scan line only (`scanner.Renderer.render`, `gaussian_filter1d`), so every
  compact feature is elongated along x — aspect ~1.3–1.5 with a 1 nm tip, ~2 with a 3–8 nm
  ("blunt") tip. A real blunt tip broadens isotropically. The horizontally stretched Fe adatoms
  in the P4 seed-0 survey frame are this: that tip was drawn `blunt` (R = 5.1 nm), not double.
* **Double tips** image every feature a second time, offset by −(apex offset), fainter by
  `w₂/w₁·exp(−2κ|Δz|)` (10–60 %); truth positions are the primary apex's (the bright copy). With
  the second apex < 1.2 nm away or a weight < 0.15 the item is marked `hard`.
* **Dirty tips** change only the spectrum (a tip-state peak between −0.25 and +0.45 V) and
  slightly lower the barrier; the topograph alone cannot show them, hence the spectrum in every
  tip item. Spectra are the material templates (no standing waves in these scenarios).
* **Unstable tips** change at a Poisson rate; a spectrum jump alone can sit where the current
  crosses zero and read as a slope change, so unstable items require a change in the topograph.
  Tip-condition and tip-change frames are step-free: a step running along the fast axis looks
  like a tip change.
* **Drift distortion is large.** At ~1 nm/min a 256-line, 15 nm frame (150 s) is stretched or
  sheared by up to ~15 %; truths are what the images show, and move distances accept the true
  surface value as well.
* **Atomic resolution** truth is MAST's judge: it flips between apex σ 0.08 and 0.10 nm (an
  imaged corrugation of ~1–2 pm under a 2 pm noise floor). The item classes avoid that band;
  a "no" frame is noise-like, a "yes" frame a clear lattice.
* **Step heights** are fixed per material; the step item asks for the height difference between
  two given points, so counting the steps between them is part of the task.
* The simulator images are cleaner than real ones (no tip-shape convolution beyond the above, no
  thermal noise beyond the Z floor, no multiple scattering): the classical baseline is strong,
  which makes it a useful floor for a model that sees the same pixels.

## Reference numbers (v1, 2026-10-09)

`generate --types all --seeds 0-9 --jobs 8`: 230 items, 260 frames, 48–59 s wall, no failure,
regenerated byte-identical (items and frame files). Baseline: 92.6 % overall (tip 92 %, adatom count 80 % exact / 85 %
within ±1, positions 65 % exact with mean F1 0.91, move / drift / herringbone / tip-change /
atomic 100 %, step 80 %); oracle 100 %, abstain 0 %. The WP-A measurement tools on the same
data: `fft_peaks` (6 peaks) lists a 4.5–9 nm peak on 18 of 20 herringbone frames, whose period
matches an arm to 0.014 nm (median) and whose k direction matches to 0.11° (stripe direction =
angle + 90°); `detect_blobs` (default parameters) positions agree with
the truth to 0.032 nm (median), recall 0.96, precision 0.53 (0.67 on good-tip frames: ghosts,
step edges and noise are listed too); `diff` now correlates high-passed frames (3 nm median
background, ±40 % search) and is within 0.2 nm on all 20 drift pairs (median 0.041 nm, worst
0.166 nm); the earlier plane-flattened version missed 5 pairs by 1.9–9.3 nm where a step or a
creep bow dominated the correlation.
