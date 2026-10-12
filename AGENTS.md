# AGENTS.md: repository guide

VISTA-SPM is an image-first visual harness for a multimodal model operating the STM-Bench simulator through MAST. [README.md](README.md) gives the overview; [docs/DESIGN.md](docs/DESIGN.md) is the specification. Read `vista_spm/protocol.py` before changing any module.

## What this repository depends on

- **STM-Bench** (public; packages `stmbench` and `stmsim`), installed editable. It provides the scenarios, the simulator, `RuntimeHost` and the judge.
  - The harness imports a few private helpers from it: `stmbench.harness.episode`, `ic_driver` and `cli`.
  - It needs `RuntimeHost(controller_timeout_s=...)`, from HamsterPark/STM-Bench#1.
  - Never edit STM-Bench from here. Propose the change there.
- **MAST** (private), reached through `MASTv2` on `PYTHONPATH` and `MAST_ROOT`. It provides the skills behind `act`, the `.sxm`/`.dat` readers and VIGIL. Tests that import it carry `requires_mast`.
- **dsh-spm** (public) ports the visual memory to TypeScript. Its golden file `spec/golden/visual_memory.json` is exported from `vista_spm/frames.py`, `archive.py` and `viewtools.py`. After a change to their behaviour or model-facing vocabulary, re-export the golden there.

## Invariants

1. **Truth stays out of observations.**
   - Tool replies, images, file names and act descriptions never carry hidden scenario values, tolerances or judge-only data.
   - The ledger keeps truth for the judge and for replay after the episode.
2. **The model-facing surface changes only on purpose.**
   - Treat tool names, descriptions, reply keys, units and decimals as experiment parameters.
   - Before and after a change, compare `python -m vista_spm tools --arm <arm> --json` for every arm.
   - Record any difference with the results it affects.
3. **Billing stays on the subscription.**
   - `RunnerConfig.auth = "subscription"` strips `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` from the Claude Code child process. It also aborts a session whose init reports an API key source.
   - STM-Bench's RuntimeHost exports MAST's provider keys into the environment, which is how roughly $52 of API usage happened on 2026-10-09.
   - Do not switch to `auth="env"` without the owner's explicit approval.
4. **One act, one observation.** An `act` runs only after the previous act's result reached the model. The instrument clock runs only during actions.
5. **Tools measure; they do not interpret.** No tool may say "double tip" or "atomic resolution". That is the model's judgement.

## Running experiments

- **Freeze the code first.** Run `scripts/snapshot.py` and run the matrix from the snapshot (`--cwd`), so edits during a run do not change running episodes. If a patch must go in mid-run, record it in the snapshot's `MANIFEST.json` (what, why, which episodes it applies to) and rerun the affected episodes.
- **Keep heavy jobs off the machine.** Episodes share it with the simulator, whose scan render is synchronous. Heavy parallel jobs (full test suites, builds) can push a render past the controller timeout. `scripts/find_failures.py` lists episodes that lost the instrument.
- **Keep run output out of git.** Ledgers, transcripts and frames go under `STM_BENCH_DATA` or `tmp/`. Only aggregated tables go into `results/`, which is a record and is not regenerated.
- **Report claim fractions with their chance floor.** Report full-reproduction counts, paired comparisons and the guessing floor of each scenario next to the fractions.

## Code rules

- Write code, comments and docstrings in English. Chinese is fine in `docs/notes/` and `README.zh-CN.md`.
- **No machine-specific paths** in source or scripts. Reach outside the checkout through `stmbench.paths` or a command-line argument. Machine notes live in `tmp/local_notes.md`, which is git-ignored.
- **The controller vendor's name** appears only inside MAST's identifiers (`nanonis_files`, `NanonisConfig`, ...) and the `.sxm` header token, as in STM-Bench.
- `tests/test_repo_hygiene.py` enforces both rules above.
- **Tests must not read local files.** Synthesize arrays or write a small `.sxm` in `tmp_path`.
- Run tests with `python -m pytest -q -m "not isolated"`, then `python -m pytest -q -m isolated`. Report pass, skip and deselect counts together.

## Commits

- Commit as `HamsterPark <121485008+HamsterPark@users.noreply.github.com>`, the GitHub noreply address.
- Make one logical change per commit, with a present-tense subject and a body that says why.
- The repository is private for now. Before making it public:
  - choose a license;
  - review `docs/notes/` and `results/`, which are verbatim records written for internal use.
