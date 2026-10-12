"""List matrix episodes that must be rerun, and optionally move them aside.

An episode is rerun when it is missing, has no episode.json, or its instrument failed under it
(simulator connection timeouts: "Connection lost for role" in the episode log, or act errors
with TimeoutError), or when it is named with --also (e.g. episodes run before a harness fix).
Moving aside renames <run_id> to <run_id>.<tag>.<ts>; run_matrix.py with the same arguments then
reruns exactly those.

usage: python scripts/find_failures.py <out_root> --run-id m1 --arms a,b --scenarios A,B --seeds 0,1,2
                               [--model claude-opus-5-5] [--effort high]
                               [--also seed0:vigil-text:P1_barth1990_au111,...] [--move]
"""
import argparse
import json
import time
from pathlib import Path


def infra_failed(run: Path, log: Path) -> str | None:
    if log.is_file():
        text = log.read_text(encoding="utf-8", errors="replace")
        n = text.count("Connection lost for role")
        if n:
            return f"{n} simulator connection losses in the episode log"
    try:
        ep = json.loads((run / "episode.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no episode.json"
    bad = [e for e in (ep.get("skill_result") or {}).get("results") or []
           if isinstance(e, dict) and "TimeoutError" in json.dumps(e)]
    return f"{len(bad)} act results with TimeoutError" if bad else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--also", default="")
    ap.add_argument("--move", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    also = {tuple(x.split(":")) for x in a.also.split(",") if x}
    todo = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in a.arms.split(","):
            for sc in a.scenarios.split(","):
                run = root / sc / f"seed{seed}" / f"V-{arm}_{a.model}_{a.effort}" / a.run_id
                log = root / "logs" / f"seed{seed}_{arm}_{sc}_{a.run_id}.log"
                why = None
                if (f"seed{seed}", arm, sc) in also:
                    why = "named with --also"
                elif not run.exists():
                    why = "missing"
                else:
                    why = infra_failed(run, log)
                if why:
                    todo.append((seed, arm, sc, why, run, log))
    for seed, arm, sc, why, run, log in todo:
        print(f"seed{seed} {arm} {sc}: {why}")
        if a.move and run.exists():
            tag = "infra" if why not in ("named with --also", "missing", "no episode.json") else "redo"
            ts = int(time.time())
            run.rename(run.with_name(f"{a.run_id}.{tag}.{ts}"))
            if log.is_file():
                log.rename(log.with_name(log.stem + f".{tag}.{ts}.log"))
    print(f"{len(todo)} episode(s) to rerun" + (" (moved aside)" if a.move else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
