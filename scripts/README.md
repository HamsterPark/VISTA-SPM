# Matrix tooling

These scripts ran the first ablation matrix (`m1`, 2026-10-09/10). Run them with the environment described in the [README](../README.md#setup); they run closed-loop episodes, so they need MAST.

| Script | Purpose |
|---|---|
| `snapshot.py <dest>` | Copies `vista_spm/`, `stmbench/` and `stmsim/` to `<dest>` and writes a hash manifest with each checkout's commit. |
| `run_matrix.py` | Runs seeds × arms × scenarios, one subprocess per episode. Details below. |
| `aggregate.py <out>` | Writes one row per episode (`--csv`) and the arm × scenario means (`--md`). |
| `find_failures.py <out>` | Lists episodes to rerun: missing, without a ledger, or where the simulator connection was lost. `--move` renames them aside so `run_matrix.py` reruns exactly those. |
| `rejudge.py <out>` | Re-judges finished episodes offline with an overridden evidence rule. This gives a secondary metric; the ledger's verdict never changes. |

`run_matrix.py` behaves as follows:

- **Order:** seed-major, so a stopped matrix still holds complete, balanced blocks.
- **Resuming:** it skips episodes that already have a valid ledger.
- **Session limit:** if an episode hits the subscription's usage limit, the script moves it aside and requeues it. It pauses until the reset time reported by Claude Code.
- **Weekly guard:** it stops starting episodes when the seven-day usage reaches `--weekly-guard`.

## The m1 run

```bash
python scripts/snapshot.py tmp/code/m1
python scripts/run_matrix.py --cwd tmp/code/m1 --out <out> --run-id m1 \
    --arms full,vigil-text,img,mem \
    --scenarios P1_barth1990_au111,P4_atom_positioning_cu111,P5_sader_jarvis_qplus \
    --seeds 0,1,2 --parallel 2
python scripts/find_failures.py <out> --run-id m1 --arms full,vigil-text,img,mem \
    --scenarios P1_barth1990_au111,P4_atom_positioning_cu111,P5_sader_jarvis_qplus --move
python scripts/aggregate.py <out> --run-id m1 --csv results/m1/rows.csv --md results/m1/summary.md
python scripts/rejudge.py <out> --scenario P4_atom_positioning_cu111 --run-id m1 \
    --evidence target_site:max_nm_per_px=0.2 --json results/m1/p4_relaxed.json
```

`run_matrix.py` and `find_failures.py` assume the default model (`claude-opus-5-5`) and effort (`high`); pass `--model` / `--effort` for others.

m1 ran when the harness was still part of STM-Bench, so its manifest ([results/m1/code_manifest.json](../results/m1/code_manifest.json)) lists the old paths. It also lists the five patches made during the run, each with its reason and the episodes it applies to.
