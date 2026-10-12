"""VISTA-SPM command line.

    python -m vista_spm run --scenario P4_atom --seeds 0 --arm full [--model claude-opus-5-5]
                                 [--effort high] [--out DIR] [--run ID] [--display-max 768]
    python -m vista_spm tools --arm full [--qplus] [--json]      # what the model would see
    python -m vista_spm preflight [--model ..] [--effort low]   # Claude Code + MCP sanity check

One episode per ``run`` invocation and seed, in this process (MAST's runtime holds
process-level singletons: run seeds sequentially, or parallel episodes in separate processes).
Run directories: ``<out>/<scenario>/seed<N>/V-<arm>_<model>_<effort>/<run_id>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from stmbench.cli import _csv, _resolve
from stmbench.paths import data_path

from .arms import ARMS, get_arm


def cmd_run(a) -> int:
    from stmsim.scenario import Scenario

    from .episode import run_vista_episode

    options: dict = {"preflight": not a.no_preflight}
    for kv in a.runner_opt or []:
        k, _, v = kv.partition("=")
        options[k.strip()] = _parse_value(v.strip())
    rows = []
    for path in _resolve(a.scenario):
        for seed in [int(s) for s in _csv(a.seeds)]:
            sc = Scenario.load(path)
            res = run_vista_episode(sc, seed=seed, arm=a.arm, model=a.model, effort=a.effort, out=a.out,
                                    run_id=a.run, time_scale=a.time_scale, max_acts=a.max_acts,
                                    display_max=a.display_max, runner_options=options)
            sr = res.skill_result or {}
            rows.append(res)
            print(f"{sc.id:28s} seed={seed} arm={a.arm} success={res.success} "
                  f"verified={res.claims_verified}/{res.claims_total} outcome={sr.get('outcome')} "
                  f"acts={sr.get('acts')} sim={res.sim_consumed_s:.0f}s wall={res.wall_s:.0f}s "
                  f"usd≈{(sr.get('billing') or {}).get('usd_est')} → {res.out_dir}", flush=True)
    if rows:
        print(f"== {sum(r.success for r in rows)}/{len(rows)} reproduced")
    return 0


def _offline_host(arm_name: str, qplus: bool, claim_ids: list[str]):
    """A VistaHost on fakes, only to list what the model would be offered."""
    from .host import VistaHost

    class _NoCtx:
        def run(self, name, params):
            raise RuntimeError("offline")

    class _Clock:
        def pause(self): ...
        def resume(self): ...
        def sim(self): return 0.0

    tmp = Path(tempfile.mkdtemp(prefix="vista_tools_"))
    (tmp / "session").mkdir()
    return VistaHost(arm=get_arm(arm_name), vista_dir=tmp / "vista", task="(offline listing)",
                     claim_ids=claim_ids, ctx=_NoCtx(), clock=_Clock(), session_dir=tmp / "session",
                     time_scale=1.0, budget_probe=lambda: (0.0, 0), is_qplus=qplus)


def cmd_tools(a) -> int:
    from .prompts import system_prompt

    host = _offline_host(a.arm, a.qplus, _csv(a.claims) or ["example_claim"])
    tools = host.list_tools()
    if a.json:
        print(json.dumps({"system_prompt": system_prompt(get_arm(a.arm)), "tools": tools},
                         ensure_ascii=False, indent=1))
        return 0
    print(system_prompt(get_arm(a.arm)))
    total = 0
    for t in tools:
        n = len(t["description"]) + len(json.dumps(t["inputSchema"]))
        total += n
        print(f"- {t['name']:18s} {n:6d} chars  {t['description'][:90]}…")
    print(f"== {len(tools)} tools, {total} characters of descriptions + schemas (≈{total // 4} tokens)")
    return 0


def cmd_preflight(a) -> int:
    from .claude_runner import ClaudeCodeRunner, RunnerConfig
    from .prompts import system_prompt
    from .rpc import RpcServer

    host = _offline_host(a.arm, False, ["example_claim"])
    io = Path(tempfile.mkdtemp(prefix="vista_preflight_"))
    rpc = RpcServer(host, log_path=io / "rpc.jsonl")
    rpc.start()
    try:
        runner = ClaudeCodeRunner(host, rpc, RunnerConfig(model=a.model, effort=a.effort,
                                                          system_prompt=system_prompt(get_arm(a.arm)), io_dir=io))
        out = runner.preflight()
    finally:
        rpc.stop()
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    return 0 if (not isinstance(out, dict) or out.get("ok", True)) else 1


def cmd_replay(a) -> int:
    """Build the stmbench replay page for a VISTA run (on a converted copy; the ledger is not touched)."""
    from stmbench.replay.build import build_replay
    from .replay_compat import replay_copy

    tmp = Path(tempfile.mkdtemp(prefix="vista_replay_")) / Path(a.run_dir).name
    src = replay_copy(a.run_dir, tmp)
    page = build_replay(str(src), a.out, single_file=a.single_file, bare=False, truth=not a.no_truth,
                        max_px=a.max_px)
    print(f"replay → {page}")
    return 0


def _parse_value(v: str):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vista_spm", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run VISTA-SPM episodes")
    r.add_argument("--scenario", required=True, help="YAML path or id prefix (e.g. P4_atom)")
    r.add_argument("--seeds", default="0")
    r.add_argument("--arm", default="full", choices=sorted(ARMS))
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--out", default=data_path("vista_runs"))
    r.add_argument("--run", default=None, help="run id (default: UTC timestamp)")
    r.add_argument("--time-scale", type=float, default=None)
    r.add_argument("--max-acts", type=int, default=400)
    r.add_argument("--display-max", type=int, default=None)
    r.add_argument("--no-preflight", action="store_true")
    r.add_argument("--runner-opt", action="append", help="RunnerConfig field=value (JSON value), repeatable")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("tools", help="print the tool surface of an arm")
    t.add_argument("--arm", default="full", choices=sorted(ARMS))
    t.add_argument("--qplus", action="store_true")
    t.add_argument("--claims", default="")
    t.add_argument("--json", action="store_true")
    t.set_defaults(fn=cmd_tools)
    p = sub.add_parser("preflight", help="check Claude Code + the MCP bridge end to end (one cheap call)")
    p.add_argument("--arm", default="full", choices=sorted(ARMS))
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="low")
    p.set_defaults(fn=cmd_preflight)
    rp = sub.add_parser("replay", help="build the stmbench replay page for a finished VISTA run")
    rp.add_argument("run_dir")
    rp.add_argument("--out", default=None)
    rp.add_argument("--single-file", action="store_true")
    rp.add_argument("--no-truth", action="store_true")
    rp.add_argument("--max-px", type=int, default=400)
    rp.set_defaults(fn=cmd_replay)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
