"""Run a VISTA-SPM ablation matrix: seeds x arms x scenarios, one subprocess per episode.

- Order: seed-major (all arms and scenarios for seed 0, then seed 1, ...), so a stopped matrix
  still holds complete, balanced blocks.
- Resumable: an episode whose run dir holds a valid episode.json is skipped; a run dir without
  one (the child died) is moved aside to <run_id>.stale.<ts> and rerun.
- Usage limits: an episode that met the account's usage limit is moved aside to
  <run_id>.usage_limit.<ts> and requeued; nothing new starts until the window's reset time
  (from Claude Code's rate_limit_event), plus a margin.
- Weekly guard: when any rate_limit_event reports seven_day utilization >= --weekly-guard,
  no new episode starts; running ones finish; the script exits and says so.
- --cwd: run the children from a frozen code snapshot (python -m puts the cwd first on sys.path).
- --grid off: scan images without the metric grid, as in m1; the runs are filed as <arm>-nogrid.

usage: python scripts/run_matrix.py --out DIR --run-id ID --arms full,img --scenarios A,B --seeds 0,1,2
                            [--cwd SNAPSHOT] [--parallel 2] [--model ..] [--effort high]
                            [--max-usd 40] [--weekly-guard 0.7] [--grid off]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

LIMIT_MARKERS = ("hit your session limit", "hit your weekly limit", "hit your usage limit", "usage_limit_reached")


def run_dir(out: Path, scenario: str, seed: int, arm: str, model: str, effort: str, run_id: str) -> Path:
    return out / scenario / f"seed{seed}" / f"V-{arm}_{model}_{effort}" / run_id


def limit_info(d: Path) -> dict:
    """{'limited': bool, 'resets_at': epoch|None, 'seven_day': float|None, 'five_hour': float|None}"""
    out = {"limited": False, "resets_at": None, "seven_day": None, "five_hour": None}
    io = d / "vista" / "claude_io"
    files = sorted(io.glob("*.stdout.jsonl")) if io.is_dir() else []
    for f in files:
        for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
            low = raw.lower()
            if any(m in low for m in LIMIT_MARKERS):
                out["limited"] = True
            if '"rate_limit_event"' not in raw:
                continue
            try:
                info = json.loads(raw).get("rate_limit_info") or {}
            except ValueError:
                continue
            wins = info.get("unifiedWindows") or {}
            for k in ("seven_day", "five_hour"):
                u = (wins.get(k) or {}).get("utilization")
                if isinstance(u, (int, float)):
                    out[k] = max(out[k] or 0.0, float(u))
            if info.get("status") == "rejected":
                out["limited"] = True
                r = info.get("resetsAt")
                if isinstance(r, (int, float)):
                    out["resets_at"] = max(out["resets_at"] or 0, int(r))
    return out


def summary(d: Path) -> str:
    try:
        ep = json.loads((d / "episode.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "no ledger"
    sr = ep.get("skill_result") or {}
    return (f"verified {ep.get('claims_verified')}/{ep.get('claims_total')} diag={ep.get('diag_correct')} "
            f"outcome={sr.get('outcome')} acts={sr.get('acts')} "
            f"sim%={100 * (ep.get('sim_consumed_fraction') or 0):.0f} "
            f"usd={(sr.get('billing') or {}).get('usd_est')} err={str(sr.get('error'))[:120]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--max-usd", type=float, default=40.0)
    ap.add_argument("--weekly-guard", type=float, default=0.7)
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--grid", choices=["on", "off"], default=None)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    filed = (lambda arm: f"{arm}-nogrid") if a.grid == "off" else (lambda arm: arm)
    queue = []
    for seed in seeds:
        for arm in a.arms.split(","):
            for scenario in a.scenarios.split(","):
                d = run_dir(out, scenario, seed, filed(arm), a.model, a.effort, a.run_id)
                if (d / "episode.json").is_file() and not limit_info(d)["limited"]:
                    print(f"skip (done) seed{seed} {arm} {scenario}", flush=True)
                    continue
                if d.exists():
                    tag = "usage_limit" if limit_info(d)["limited"] else "stale"
                    d.rename(d.with_name(f"{a.run_id}.{tag}.{int(time.time())}"))
                    print(f"moved aside ({tag}): {d}", flush=True)
                queue.append((scenario, seed, arm))
    print(f"plan: {len(queue)} episodes, parallel {a.parallel}, cwd {a.cwd or os.getcwd()}", flush=True)
    running = []
    pause_until = 0.0
    stop_new = False
    while queue or running:
        now = time.time()
        while queue and len(running) < a.parallel and not stop_new and now >= pause_until:
            scenario, seed, arm = queue.pop(0)
            log = logs / f"seed{seed}_{arm}_{scenario}_{a.run_id}.log"
            cmd = [a.python, "-m", "vista_spm", "run", "--scenario", scenario, "--seeds", str(seed),
                   "--arm", arm, "--model", a.model, "--effort", a.effort, "--out", str(out), "--run", a.run_id,
                   "--runner-opt", f"max_budget_usd={a.max_usd}"] + (["--grid", a.grid] if a.grid else [])
            fh = open(log, "a", encoding="utf-8")
            p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=a.cwd or None, env=os.environ.copy())
            running.append((p, scenario, seed, arm, fh, time.time()))
            print(f"{time.strftime('%H:%M:%S')} start seed{seed} {arm} {scenario} pid={p.pid}", flush=True)
            time.sleep(20)          # stagger MAST runtime start-ups
            now = time.time()
        for item in list(running):
            p, scenario, seed, arm, fh, t0 = item
            if p.poll() is None:
                continue
            fh.close()
            running.remove(item)
            d = run_dir(out, scenario, seed, filed(arm), a.model, a.effort, a.run_id)
            li = limit_info(d) if d.exists() else {"limited": False, "resets_at": None, "seven_day": None}
            if li["limited"]:
                if d.exists():
                    d.rename(d.with_name(f"{a.run_id}.usage_limit.{int(time.time())}"))
                queue.insert(0, (scenario, seed, arm))
                resume = (li["resets_at"] or (time.time() + 1800)) + 180
                pause_until = max(pause_until, resume)
                print(f"{time.strftime('%H:%M:%S')} USAGE LIMIT seed{seed} {arm} {scenario}: requeued; "
                      f"pausing until {time.strftime('%H:%M', time.localtime(pause_until))}", flush=True)
                continue
            print(f"{time.strftime('%H:%M:%S')} end seed{seed} {arm} {scenario} exit={p.returncode} "
                  f"wall={time.time() - t0:.0f}s :: {summary(d)}", flush=True)
            if li.get("seven_day") is not None and li["seven_day"] >= a.weekly_guard and not stop_new:
                stop_new = True
                print(f"WEEKLY GUARD: seven_day utilization {li['seven_day']:.2f} >= {a.weekly_guard}; "
                      f"no new episodes ({len(queue)} left in queue)", flush=True)
        if stop_new and not running:
            break
        time.sleep(5)
    left = [f"seed{s} {ar} {sc}" for sc, s, ar in queue]
    print(f"matrix {'stopped by weekly guard' if stop_new else 'done'}; left: {left}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
