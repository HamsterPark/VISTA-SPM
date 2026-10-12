# Results

These files are records of past runs. They are aggregates only: episode ledgers, transcripts and frames stay outside the repository. Each file is kept as it was produced and is not regenerated. The analysis is in [docs/notes/2026-10-10-first-results.zh-CN.md](../docs/notes/2026-10-10-first-results.zh-CN.md) (Chinese).

## m1: the first ablation matrix (2026-10-09/10)

The run covered four arms (full, mem, img, vigil-text) × three scenarios (P1, P4, P5) × seeds 0–2, with claude-opus-5-5 at effort high. All 36 episodes are valid. The images had no graph paper (added 2026-10-12); `--grid off` reproduces them.

| File | Contents |
|---|---|
| `m1/summary.md` | Arm × scenario means: claims verified, reproduced, tip diagnosis, acts, instrument time, tool calls, images, estimated USD. |
| `m1/rows.csv` | One row per episode. |
| `m1/p4_relaxed.json` | P4 re-judged offline. Each episode carries the official rule and a relaxed one: a verification frame of at most 0.2 nm/px instead of 0.1 nm/px. This is a secondary metric. |
| `m1/code_manifest.json` | Digest and file hashes of the frozen code, and the five mid-run patches with their reasons and the episodes they apply to. |

Points to keep in mind:

- **Billing.** 23 of the 36 final episodes billed an API key before the subscription guard existed, and 13 ran on the subscription. The model was the same throughout.
- **Cost.** USD figures are Claude Code's estimates.
- **Each step changes more than one thing.** mem adds notes, the frame archive and `inspect` over img; full adds the exact readout and five measurement tools over mem. Neither comparison isolates one tool, and the static benchmark's images-only arm has no tools at all.
- **Chance floor.** The claim fractions are not chance-corrected. Reporting fixed constants passes about 0.26 of P1 claims, about 0.41 of P5 claims and none of P4 (audit of 2026-10-12).
- **P4 evidence rule.** The official rule requires a frame of at most 0.1 nm/px covering the reported site after the last move. The task text does not state this rule.
- **Tip diagnosis.** The simulator blurs only along the fast-scan axis, so blunt tips look sharp across it. Until that is fixed, tip-diagnosis scores are unreliable.

## staticbench: STM-BabyVision

| File | Contents |
|---|---|
| `staticbench/v1_baseline.md`, `.csv` | The 230-item v1 set answered by the classical baseline (92.6 %), with the oracle and abstain rows. |
| `staticbench/v1_manifest.json` | Generation parameters, item counts, class balance and difficulty mix of the v1 set. The frames themselves are regenerated with `generate --types all --seeds 0-9`. |
| `staticbench/pilot1.md`, `.csv` | Claude pilot: 12 items, one per type, with tools and with images only. |
| `staticbench/hard1.md`, `.csv` | Claude on the 37-item hard set, with tools (31) and with images only (23); the classical baseline scored 30. |

The Claude runs saw the frames without graph paper (added 2026-10-12); `qa --grid off` reproduces them.
