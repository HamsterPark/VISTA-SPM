# VISTA-SPM harness: design and work split

Status: Phase 1 build spec, 2026-10-09. The harness was built as a subpackage of STM-Bench's
`stmbench` and moved to this repository on 2026-10-12. The project plan, the literature
review and the first results are in `docs/notes/` (Chinese).
Reference implementation: VISTA, arXiv 2610.02200, code github.com/joshhhhhan/VISTA (MIT).

## 1. What we are building

An STM version of VISTA. A frontier multimodal model, run as Claude Code in print mode with
**every built-in tool disabled**, operates the STM-Bench simulator through one MCP server.
It sees rendered frames, keeps a **lossless visual memory** of every frame the instrument
produced, can **re-inspect** any of them (crop, enlarge, other channel, backward scan,
different flattening), **read exact values**, **measure** (profiles, FFT peaks, blobs,
frame differences, statistics) and **keep notes** (GUIDE.md, WORKING.md). MAST supplies
the instrument skills and its safety stack (`ExecutionContext`); `stmsim` supplies the
instrument and hidden truth; STM-Bench's judge scores the episode.

Principles carried over from VISTA:

1. Images are the observation. All labels and numbers travel in adjacent text blocks.
   VISTA's ARC renderer draws faint cell grid lines (every 8 px darkened to 3/4), a sheet
   of graph paper; this port draws nothing inside an image. The STM counterpart of that
   grid, a faint nm grid or scale bar, has not been tested.
2. Lossless memory: every frame, channel, direction and partial snapshot is archived and
   addressable forever; only the current observation is pushed into context automatically.
3. One action, one observation: an `act` executes only after the previous `act`'s result
   has reached the model's context.
4. Tools only measure; they never interpret ("tip is double", "atomic resolution reached"
   are the model's judgements, not a tool's).
5. Minimal prompt: expect-before / report-after each action; notes in GUIDE/WORKING.
6. Sim time runs only while an instrument action executes; the model's thinking is free.

## 2. Processes and threads

```
runner process  (python -m vista_spm run ...)
 ├─ RuntimeHost (MAST CoreRuntime on stmsim, 4 loopback wire ports)       [existing]
 ├─ VistaHost   (tools, archive, notes, sinks, clock, budgets, gate)      [lead]
 ├─ RpcServer   (127.0.0.1:<port>, token, JSON lines)                     [WP-B]
 └─ ClaudeCodeRunner (spawns `claude -p` segments, reads stream-json)     [WP-B]
        │ stdin/stdout stream-json
        ▼
   claude.exe  ──stdio MCP──►  mcp_bridge.py (stdlib only) ──TCP──► RpcServer → VistaHost.call
                               (spawned by claude from --mcp-config)
   hooks (PreCompact/Stop) ──► compact_hook.py ──TCP or marker files──► VistaHost
```

## 3. Package layout and ownership

The owners are the Phase 1 work packages (lead, WP-A, WP-B, WP-D), kept for provenance. The
harness imports STM-Bench and MAST and never edits their files. Tests go to
`tests/test_<area>.py`.

| Path (under `vista_spm/`) | Owner | Contents |
|---|---|---|
| `__init__.py`, `protocol.py` | lead | shared types (read `protocol.py` first) |
| `frames.py` | WP-A | load .sxm/.dat, oriented arrays, geometry, display scale, coordinate transforms, rendering |
| `archive.py` | WP-A | per-episode lossless store, frame ids, index, observation images |
| `viewtools.py` | WP-A | `inspect`, `read_values` ToolDefs + implementations |
| `measure.py` | WP-A | `line_profile`, `fft_peaks`, `detect_blobs`, `diff`, `stats` ToolDefs + implementations |
| `notes.py` | WP-B | GUIDE.md / WORKING.md store + ToolDefs |
| `rpc.py` | WP-B | host-side TCP JSON-lines server |
| `mcp_bridge.py` | WP-B | stdio MCP server, stdlib only, forwards to RpcServer |
| `compact_hook.py` | WP-B | Claude Code hook script |
| `claude_runner.py` | WP-B | segment life cycle, stream-json I/O, preflight, logging, compaction, recovery |
| `staticbench/` | WP-D | static perception benchmark: generator, truth, questions, scorer, QA host |
| `host.py`, `actions.py`, `arms.py`, `advisor.py`, `prompts.py`, `episode.py`, `replay_compat.py`, `cli.py`, `__main__.py` | lead | instrument actions, tool assembly per arm, MAST's detectors as a tool or text observation, episode + ledger, replay conversion, CLI |

## 4. Environment

Python 3.13 with STM-Bench installed editable, MAST importable (`MASTv2` on `PYTHONPATH`,
`MAST_ROOT` set) and this repository on the import path (run from its root, or
`pip install -e . --no-deps`); `PYTHONUTF8=1` on Windows. STM-Bench must have
`RuntimeHost(controller_timeout_s=...)` (HamsterPark/STM-Bench#1); episodes set it to 60 s.
Machine-specific commands and the locations of local example frames are kept out of the
repository, in `tmp/local_notes.md` (git-ignored).

Unit tests must not depend on local files or personal paths: synthesize arrays or write a small
.sxm in `tmp_path`; a test that needs a real frame reads its path from an environment variable
and skips when it is unset. Real instrument data never enter the repository. Anything importing
`mast` is marked `requires_mast` (see `tests/conftest.py`). `tests/test_repo_hygiene.py`
forbids drive-rooted path literals and second readers of `STM_BENCH_*` / `MAST_ROOT` in
package source and scripts, and the vendor name outside MAST's identifiers.

Readers to reuse: `mast.io.nanonis_files.read_sxm`, `read_sxm_header`, `sxm_oriented_frames`,
`read_dat`; `stmbench.human.render.frame_geometry`, `pixel_to_scan_nm`, `flatten`,
`frame_array`, `render_dat`. The display-coordinate convention is
`protocol.DISPLAY_CONVENTION`; the judge (`stmbench/trackB/claims.py`) compares reported
positions in scan-frame nm with tolerances down to 0.16 nm (P4), so the transform must be
exact, sub-pixel included.

## 5. Observations and the archive (WP-A)

### 5.1 Frames

A *frame* is one thing the instrument produced, with id `t<turn:04d>.<kind><i>`
(`protocol.frame_id`): `s` saved scan (.sxm), `p` partial snapshot of a scan in progress,
`d` spectrum (.dat), `m` derived image (e.g. a diff map). Turn 0 is the starting
observation; every `act` increments the turn.

A scan frame stores, losslessly, every channel in the file, forward and backward, as
oriented float32 arrays in SI units (`np.savez_compressed`, NaN for unacquired rows), plus
its geometry `{cx_nm, cy_nm, w_nm, h_nm, angle_deg, nx, ny}`, `nm_per_px`, bias, setpoint,
scan direction, acquired rows, record time and the sim time if the host supplies it.

### 5.2 Rendering

Default observation of a scan: channel `Z`, forward, plane-subtracted, 0.5 % / 99.5 %
percentile contrast, grey colour map, NaN rows in a fixed dark colour, integer
nearest-neighbour scaling per `protocol.DISPLAY_CONVENTION` with `display_max` from the
arm (default 768). Lossless PNG. Supported `flatten`: `none`, `plane`, `line` (per-row
median offset), `poly2` (2nd-order surface), `highpass` (subtract a median background of
a given width in nm, default 3 nm). Constant-height frames (Z range below 1 pm) default to
the `Current` channel. A spectrum renders as a line plot (x = sweep column, y = chosen
column; axes and units allowed on plots because a plot is not a camera image — keep fonts
large and the plot ≥ 600 px wide).

Model-facing numbers (WP-A naming rules, guarded by `test_model_facing_json_vocabulary`): one
unit per quantity, carried in the key suffix (m → `_pm` for heights, A → `_pa`, V → `_v`,
Hz → `_hz`, N → `_pn`; lengths `_nm`, angles `_deg`, pixels `_px`, time `_s`); no `unit`,
`type` or `visual_kind` fields; numbers are JSON numbers, never strings; American spelling
(center, color, gray). Values are rounded to the channel's `decimals`, chosen to resolve about
1/100 of the channel's value range (1..6 decimals, at least 0.1 of the unit); the V-text grid
gives Z as integer pm.

Every image carries a label text block before it, e.g.
`<visual kind="current" frame="t0007.s0" channel="Z" direction="forward" flatten="plane" scale="2" size="768x768" region="0,0,768,768" z_range_pm="-23.1..41.7"/>`.

### 5.3 `Archive` (archive.py)

```python
class Archive:
    def __init__(self, root: Path, *, display_max: int = DISPLAY_MAX_DEFAULT): ...
    def add_scan(self, turn: int, sxm_path: str | Path, *, sim_s: float | None = None) -> FrameEntry
    def add_partial(self, turn: int, arrays: dict[str, np.ndarray], *, geometry: dict,
                    meta: dict, sim_s: float | None = None) -> FrameEntry    # arrays: {"Z/forward": 2-D, ...}
    def add_spectrum(self, turn: int, dat_path: str | Path, *, sim_s: float | None = None) -> FrameEntry
    def add_derived(self, turn: int, png: bytes, *, meta: dict) -> FrameEntry
    def get(self, fid: str) -> FrameEntry               # KeyError with a helpful message
    def frames(self, turn: int | None = None, kind: str | None = None) -> list[FrameEntry]
    def latest(self, kind: str = "s") -> FrameEntry | None
    def array(self, fid: str, channel: str, direction: str = "forward") -> np.ndarray
    def render(self, fid: str, *, channel: str | None = None, direction: str = "forward",
               flatten: str = "plane", clip_pct: float = 0.5, region: dict | None = None,
               display_max: int | None = None) -> tuple[bytes, dict]   # (png, render meta incl. label)
    def observation(self, fid: str) -> list[ImageBlock]  # the default "current" image of a frame
    def text_grid(self, fid: str, n: int = 64) -> str    # V-text arm: n x n integers (pm), header with units
    def index(self) -> list[dict]                        # every frame's summary, for history/recovery
```

`FrameEntry` (dataclass): `fid, turn, kind, index, source_name, channels, directions, nx, ny,
geometry, nm_per_px, display_scale, display_size, bias_v, setpoint_a, scan_dir,
acquired_rows, rec_time, sim_s`, and `summary() -> dict` (compact, JSON-safe). The summary is
model-facing: `frame`, `turn`, `kind` ∈ {`scan`, `partial`, `spectrum`, `derived`}; for scans
`source, channels, directions, size_px, field_nm, center_nm, angle_deg, nm_per_px,
display_scale` (a number; `display_scale_frac` such as `"1/2"` only when not an integer),
`display_size_px, default_channel, feedback`, and `rows_acquired` + `rows_total` only when
rows are missing; for spectra `source, columns, n_points, position_nm, experiment`; for derived
frames `tool, frames, description, image_size_px`; optional `bias_v`, `setpoint_pa` (or
`setpoint_hz`), `scan_dir`, `rec_time`, `sim_s`. The attribute `fid` and the kind letter stay
internal.

On disk: `<root>/index.jsonl` (append-only), `<root>/t0007.s0.npz`, `<root>/t0007.s0.png`
(default rendering), spectra keep the original `.dat` copy plus parsed `.npz`. Re-opening an
existing root restores the index (crash recovery).

Coordinate helpers in `frames.py`: `display_to_scan_nm(entry, x, y, display_max=None)`,
`scan_nm_to_display(entry, X, Y, display_max=None)`, both exact inverses (property test).
Validate once against the simulator: a blob centroid in a P4 survey frame converted to nm
must land within 0.05 nm of where `stmbench.human.render.pixel_to_scan_nm` puts that
pixel, and the frame centre must map to `(cx, cy)`.

## 6. Tool surface

Tool names as the model sees them: `mcp__stm__<name>`. Arms (lead, `arms.py`) select
subsets. All tools return `protocol.ToolReply`; implementations never raise.

### 6.1 Instrument (lead)

* `act(action, params, note?)` — the only tool that changes the instrument. One action per
  call; refused until the previous act result was acknowledged. V0 actions (each a fixed
  MAST skill sequence through `ExecutionContext`): `scan`, `set_bias`, `set_setpoint`,
  `set_feedback`, `move_tip`, `pulse`, `poke`, `sts`, `df_curve`, `move_atom`,
  `drift_compensation`, `wait`, `withdraw`, `approach`, `pll` (qPlus only). Positions are
  scan-frame nm, or `{"frame": "t0007.s0", "x": 412, "y": 233}` in that frame's display
  pixels. The reply: action result (success / refusal / error text), instrument state
  echo, new frame ids, budget used, and the new observation image(s).
* `report_result(claim_id, value, unit?, x_nm?, y_nm?, sigma?, note?)` and
  `report_tip_state(junction, atomic_resolution, tip_state, reason)` — STM-Bench's judge
  channels (`stmbench/harness/results.py`, `ic_driver.REPORT_SCHEMA`); never touch the
  instrument.

### 6.2 Visual memory (WP-A, `viewtools.py`)

* `inspect(question, views[1..16])`; view = `{label, frame, channel?, direction?,
  region?: {x, y, width, height} (display px of the full frame), flatten?, clip_pct?,
  highpass_nm?}`. Each view is re-rendered from the archived arrays (not from the PNG),
  cropped to whole scan pixels, and enlarged by the largest whole number of image pixels per
  scan pixel that fits `display_max` (block-averaged when the region holds more scan pixels
  than `display_max`), so its longest side lies between about `display_max`/2 and
  `display_max`. Non-integer nearest-neighbour scaling is avoided because alternating 1- and
  2-pixel columns draw moiré on lattices. Never smoothed; each view is returned as its own
  image block in request order (no collage); the reply gives `magnification`,
  `image_px_per_scan_px` and the snapped `region_px`.
  The reply text lists, per view, the frame, the region in display px AND in scan nm
  (corners), the effective magnification and the value range. `frame` may be a spectrum
  id; then the view is a plot (optional `x_col`, `y_col`, `x_range`).
* `read_values(question, views[1..64])`; view = `{label, frame, channel?, direction?,
  region (display px), rows, columns}`; samples the archived physical values at bin
  centres (same rule as VISTA's read_pixels: `x + ((2c+1) w) // (2 C)`), at most 4096
  samples per call. Reply: per view `values_<u>` as a list of rows with the unit in the key
  (`pm` for Z, `pa` for current, `hz` for Δf, `v`), rounded to the view's `decimals`, the
  sample positions (`sample_x_px`, `sample_y_px`, `first_sample_nm`, `last_sample_nm`), NaN as
  null. For a spectrum the request uses `x_col`, `y_cols`, `x_range`, `n` (because `columns`
  is the grid count of scan views); the reply gives `column_names`, `column_units`,
  `column_decimals` and `values` per column.

### 6.3 Measurement (WP-A, `measure.py`) — measure, never interpret

* `line_profile(frame, p1, p2, channel?, direction?, width_px?, flatten?, highpass_nm?)`: p1/p2 in display
  px or `{"x_nm","y_nm"}`; returns distance (nm) and values; step candidates (position nm,
  height in the channel unit) from a simple change-point fit; one plot image.
* `fft_peaks(frame, region?, channel?, direction?, n_peaks?, flatten?, highpass_nm?)`: Hann window, magnitude
  spectrum, local maxima excluding DC; per peak: wavevector (nm⁻¹), real-space period (nm),
  angle (deg, scan frame; the wavevector direction, so stripes run at angle + 90°), relative strength (in MAD units over the background), plus the
  log-magnitude image with peaks numbered (marks drawn only on this derived image).
* `detect_blobs(frame, region?, channel?, direction?, polarity="bright", min_height?,
  min_sep_nm?, max_n=50, background_nm?)`: local background subtraction (dark blobs report
  negative heights), local maxima above threshold;
  per blob: id, centroid (display px and scan nm), height, FWHM x/y (nm), elongation and
  axis angle; plus a derived image with numbered marks (Set-of-Mark).
* `diff(frame_a, frame_b, region?, channel?, direction?, align=true, flatten?, highpass_nm?)`:
  both frames flattened (default `highpass` with a 3 nm median background, so steps and creep
  bows do not dominate the correlation), sub-pixel cross-correlation registration within a
  ±40 % search window (shift in nm; confidence = the main NCC peak divided by the best NCC
  farther away than twice the image's autocorrelation length, so a lattice ridge is not read
  as infinite confidence), the aligned difference
  image (b − shifted a) as a derived image, and changed regions (centroid nm, sign, area
  nm², peak change). Frames with different geometry are resampled onto the overlap in
  scan nm first.
* `stats(frame, region?, channel?, direction?)`: mean, std, RMS roughness after plane
  subtraction, min/max, percentiles, forward/backward correlation (when both exist), a
  line-noise index (median |row-to-row offset| / std), fraction of rows acquired.

### 6.4 Memory and notes (WP-B `notes.py`, lead `host.py`)

* `history(view="events"|"attempts", start_turn?, end_turn?, limit=64)` (lead): exact
  action/result records by turn; `attempts` summarises the episode so far (acts per
  action type, frames, spectra, reports, budget).
* `read_guide()`, `write_guide(content)` (≤ 64 KiB), `read_working()`,
  `write_working(content)` (≤ 16 KiB): GUIDE.md persists for the whole episode (and is the
  place for durable lessons); WORKING.md is the scratch state, persists across
  compactions. Writes replace the whole file. GUIDE.md starts as `No reliable model yet.\n`.

## 7. Claude Code runner (WP-B)

Port VISTA's `backends/claude/{runner,mcp_bridge,mcp_host,compact_hook,lifecycle}.py`
(MIT; keep a short attribution comment at the top of each derived file) to Windows,
without Docker:

* Command: `claude -p --input-format stream-json --output-format stream-json --verbose
  --include-hook-events --model <id> --effort <e> --system-prompt <text> --setting-sources ""
  --settings <segment settings.json> --mcp-config <segment mcp.json> --strict-mcp-config
  --tools "" --allowedTools mcp__stm__<each tool> --permission-mode default
  (--session-id <uuid4> | --resume <id>)`, cwd = an empty directory under the run dir.
* Child environment: the parent environment minus every variable that ties a process to
  the calling Claude Code session (`CLAUDECODE`, every `CLAUDE_CODE_*`, `CLAUDE_PID`,
  `CLAUDE_EFFORT`), plus `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, `CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`,
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`,
  `MAX_MCP_OUTPUT_TOKENS=100000`, `CLAUDE_CODE_AUTO_COMPACT_WINDOW=<cfg, default 200000>`,
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=<cfg, default 90>`. Authentication is the user's normal
  Claude Code login (do not copy credentials anywhere).
* MCP config: server `stm`, command = the venv python with `-I` and the absolute path of
  `mcp_bridge.py`, env `VISTA_RPC_PORT`, `VISTA_RPC_TOKEN`.
* Input: the first user message (text + base64 PNG image blocks) from
  `host.initial_message()` or `host.recovery_message()`; stdin stays open so the runner can
  send continuation messages; close stdin to end a segment.
* Output parsing: `system/init` (check the tool list is exactly the expected `mcp__stm__*`
  set and the MCP server is connected; record the resolved model id), `assistant`
  (tool_use ids of `act`), `user` (a `tool_result` for a pending `act` id ⇒
  `host.acknowledge_act()`), `system/compact_boundary`, `rate_limit_event`, `result`
  (usage, cost, num_turns, final text).
* Turn end: final text ending with `[DONE]` or `[ABORT]` (`stmbench.harness.ic_driver.marker_outcome`)
  ⇒ `host.finish`. Otherwise, if not terminal, send the continuation text (once per turn;
  N consecutive turns without any `act` or report ⇒ outcome `stalled`).
* Compaction (VISTA's checkpoint protocol): PreCompact and Stop hooks run
  `compact_hook.py`; on PreCompact without a ready checkpoint the hook blocks compaction
  with "Review and update WORKING.md so it preserves relevant existing information and
  contains the complete continuation state needed after compaction." and the host is put
  in checkpoint mode (only notes tools execute); `write_working` completes the checkpoint;
  the runner ends the segment and starts a fresh session with `host.recovery_message()`.
  The same checkpoint is requested when the session's delivered images reach the image
  limit (default 500). A native `compact_boundary` before the handoff is an error.
* Preflight (`ClaudeCodeRunner.preflight()`): a throw-away segment, "This is a runtime
  preflight, not a task. Do not call tools. Reply exactly READY.", same flags; checks the
  init envelope and the reply.
* Robustness: decision lease (no tool call for `turn_timeout_s` ⇒ end segment, recover),
  bounded retries with backoff on API overload, no zombie processes on Windows
  (`terminate()` then `kill()` after a grace period), UTF-8 everywhere.
* Logs under `<run>/vista/claude_io/`: `segment_NNN.{stdout.jsonl,stderr.txt,input.json
  (image data replaced by sha256 + size),command.json,settings.json,mcp.json,bridge.jsonl}`
  and `manifest.json` (per segment: session id, model, exit code, boundary, usage,
  cost_usd, num_turns; totals).

## 8. Episode host and ledger (lead)

`run_vista_episode(scenario, seed, arm, model, effort, out, ...)` builds the world and
`RuntimeHost` exactly as `stmbench.harness.episode.run_episode` does, pauses the sim
clock except inside `act`, enforces the scenario budgets after every act, runs the Claude
Code runner, then judges and writes a ledger with the same files and fields as an LLM
episode (`episode.json`, `driver.json`, `events.json`, `tip_timeline.json`,
`host_facts.json`, `session/`), mode `V-<arm>`, plus `vista/` (archive, notes, claude_io,
manifest). Run directory: `<out>/<scenario>/seed<N>/V-<arm>_<model>_<effort>/<run_id>/`.

## 9. Static perception benchmark (WP-D)

`vista_spm/staticbench/`: frames with known truth generated on the simulator,
questions in the BabyVision style, answered through the same `inspect`/`read_values`/
measurement tools with no instrument actions. Item types: tip condition (good / blunt /
double / unstable / dirty), Au(111) herringbone period and orientation, adatom count and
positions, did an adatom move between two frames and by how much, step height, drift
between frames, mid-scan tip change row, atomic resolution (FFT Bragg-peak metric as
truth). Generator, item schema (JSONL), scorer with per-type tolerances, a QA host that
implements `protocol.RunnerHost` without `act`, and a report table.

## 10. Arms (lead, `arms.py`)

| arm | observation | notes | inspect | read_values + measure | advisor |
|---|---|---|---|---|---|
| `full` | image | yes | yes | yes | no |
| `mem` | image | yes | yes | no | no |
| `notes` | image | yes | no | no | no |
| `img` | image | no | no | no | no |
| `text` | 64×64 numeric grid | yes | no | read_values only | no |
| `vigil-text` | MAST vision summaries as text, no image | yes | no | no | MAST text only |
| `advisor` | image | yes | yes | yes | MAST detectors on request |

## 11. Definition of done for each work package

Code + tests passing with the commands in §4; a short `README` section at the top of each
module docstring; no edits outside the owned paths; a final report listing files, public
API, test results (command + counts), what was verified against real data, and open
issues. Prefer small, explicit, well-named functions; no silent fallbacks (a failure is an
error reply that says what happened).
