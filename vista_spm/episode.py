"""One VISTA-SPM episode: scenario → world + MAST runtime → VistaHost + Claude Code → verdict.

The world, runtime host, fault scheduler, judge and ledger are exactly those of
``stmbench.harness.episode.run_episode`` (its helpers are reused, not copied), so
``stmbench.report`` and ``stmbench.replay`` read a VISTA episode like any other. What differs
is the driver: instead of MAST's agent loop, Claude Code drives ``VistaHost`` through MCP.

Run directory: ``<out>/<scenario>/seed<N>/V-<arm>_<model>_<effort>/<run_id>/`` with the usual
``episode.json``, ``driver.json``, ``events.json``, ``tip_timeline.json``, ``host_facts.json``,
``session/`` and ``mast_root/``, plus ``vista/`` (archive, notes, claude_io, manifest).

``driver``: tests and scripted checks pass a callable ``driver(host) -> dict`` that plays the
episode through ``host.call(...)`` instead of launching Claude Code.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from stmbench.harness.episode import (BILLING_UNKNOWN, EpisodeResult, _budget_caps, _json_default, _safe,
                                      _verdict_extras, apply_budget_overrun, claim_counts, claim_run_dir,
                                      count_atomic_frames, default_run_id, diag_correct, provenance,
                                      tip_timeline)
from stmbench.trackB.truth_criteria import judge
from stmsim.scenario import Scenario

from .arms import Arm, with_options
from .host import VistaHost
from .replay_compat import replay_events

#: MAST's controller reply timeout for simulator episodes: stmsim renders a whole frame inside
#: Scan_Action (2-15 s for large Au(111) frames); MAST's 5 s default turned that into lost connections.
SIM_REPLY_TIMEOUT_S = 60.0


def vista_run_dir(out: str | Path, scenario_id: str, seed: int, arm: str, model: str, effort: str,
                  run_id: str) -> Path:
    leaf = f"V-{_safe(arm)}_{_safe(model)}_{_safe(effort)}"
    return Path(out) / _safe(scenario_id) / f"seed{int(seed)}" / leaf / _safe(run_id)


def run_vista_episode(scenario: Scenario, *, seed: int, arm: str | Arm, model: str = "claude-opus-5-5",
                      effort: str = "high", out: str | Path, run_id: str | None = None,
                      time_scale: float | None = None, max_acts: int = 400, display_max: int | None = None,
                      grid: bool | None = None, vision_backend: str | None = None,
                      runner_options: dict | None = None,
                      driver: Callable[[VistaHost], dict] | None = None,
                      on_host: Callable[[VistaHost], None] | None = None) -> EpisodeResult:
    from stmbench.harness.runtime_host import RuntimeHost

    arm_obj = with_options(arm, display_max=display_max, grid=grid)
    mode = f"V-{arm_obj.name}"
    rid = run_id or default_run_id()
    out_dir = claim_run_dir(vista_run_dir(out, scenario.id, seed, arm_obj.name, model, effort, rid))
    if time_scale is not None:
        scenario.time_scale = time_scale
    max_sim_s, max_cmds = _budget_caps(scenario.budget)
    caps = {"max_acts": max_acts, "max_sim_s": max_sim_s, "max_wire_cmds": max_cmds}
    world = scenario.build_world(seed, session_dir=out_dir / "session")
    host = RuntimeHost(world, out_dir, vision_backend=vision_backend,
                       controller_timeout_s=SIM_REPLY_TIMEOUT_S)
    host.start()
    disp = host.dispatcher
    sched = scenario.scheduler(world, server=host.server)
    disp.fault_hook = lambda cmd, args: sched.tick(cmd)
    sched.tick()
    truth_before = world.truth()
    t0 = time.perf_counter()
    vhost: VistaHost | None = None
    run_info: dict = {}
    error: str | None = None
    world.clock.pause()                   # sim time runs only inside `act`
    try:
        vhost = VistaHost.from_runtime(host, scenario, arm_obj, out_dir / "vista", max_sim_s=max_sim_s,
                                       max_cmds=max_cmds, max_acts=max_acts, run_id=f"vista-{scenario.id}")
        if on_host is not None:
            on_host(vhost)
        if driver is not None:
            run_info = {"driver": "scripted", **(driver(vhost) or {})}
            if vhost.outcome is None:
                vhost.finish("done", run_info.get("final_text", ""))
        else:
            run_info = _run_claude(vhost, out_dir, arm_obj, model, effort, runner_options or {})
    except Exception as exc:  # noqa: BLE001 — the ledger is written whatever happened
        import traceback
        error = f"{type(exc).__name__}: {exc}"
        run_info = {**run_info, "error": error, "traceback": traceback.format_exc()}
        if vhost is not None and vhost.outcome is None:
            vhost.finish("error", "")
    finally:
        try:
            world.clock.resume()
        except Exception:  # noqa: BLE001
            pass
        wall = time.perf_counter() - t0
        truth_after = world.truth()
        calls = disp.call_log
        wire_errors = sum(1 for c in calls if not c[3].startswith("ok") and "NeedModule" not in c[3])
        cmd_counts = Counter(c[1] for c in calls)
        host.stop()

    skill_result = _skill_result(vhost, run_info, mode, model, effort, max_sim_s, max_cmds)
    if scenario.success.get("kind") == "atomic_resolution":
        skill_result["frames_atomic"] = count_atomic_frames(out_dir / "session", scenario.material)
    consumed = float(truth_after.get("sim_s", 0.0)) - float(truth_before.get("sim_s", 0.0))
    fraction = (consumed / max_sim_s) if max_sim_s else None
    faults = [{"kind": f.kind, "at_sim_s": f.at_sim_s, "when": f.when, "fired": bool(f.fired),
               "fired_at_sim_s": f.fired_at_sim_s} for f in sched.faults]
    extra = _verdict_extras(scenario, world, cmd_counts, skill_result, truth_after,
                            consumed_fraction=fraction, faults=faults)
    v = judge(scenario.success.get("kind", "tip_repaired"), {**truth_after, "events_tail": world.events[-50:]},
              scenario.success.get("thresholds"), **extra)
    v = apply_budget_overrun(v, consumed, max_sim_s)
    tip_after = truth_after.get("tip") if isinstance(truth_after.get("tip"), dict) else {}
    tip_dead = bool(tip_after["dead"]) if "dead" in tip_after else None
    damage = None
    surface = getattr(world, "surface", None)
    if callable(getattr(surface, "damage_area_nm2", None)):
        try:
            damage = float(surface.damage_area_nm2())
        except Exception:  # noqa: BLE001
            damage = None
    diag = diag_correct(skill_result.get("diagnosis"), truth_after, scenario.success.get("thresholds"))
    paper = claim_counts(scenario, v) if scenario.paper else {}
    prov = provenance(model, getattr(host, "mast_root", out_dir / "mast_root") / "config" / "ui_settings.json")
    res = EpisodeResult(
        scenario_id=scenario.id, seed=seed, mode=mode, success=v.success, partial=v.partial,
        verdict=v.as_dict(), truth_before=truth_before, truth_after=truth_after,
        sim_time_s=world.clock.sim(), wall_s=wall, wire_cmds=len(calls), wire_errors=wire_errors,
        events=dict(Counter(e["kind"] for e in world.events)), skill_result=skill_result, out_dir=str(out_dir),
        run_id=rid, model_id=model, policy=None, time_scale=float(scenario.time_scale), caps=caps,
        sim_consumed_s=consumed, sim_consumed_fraction=fraction, faults=faults,
        **prov, sampling_params_json={"model_id": model, "effort": effort, "source": "claude code"},
        envelope_violations=_envelope_violations(vhost), tip_dead=tip_dead, damage_area_nm2=damage,
        wrong_action_count=None, diag_correct=diag, billing=skill_result.get("billing"), **paper)
    _write(out_dir / "episode.json", res.as_dict())
    # ``events`` is the replay stream stmbench.replay reads; ``host`` keeps the full host record
    host_ledger = vhost.ledger() if vhost is not None else {}
    replay_stream = replay_events(host_ledger.get("driver_events") or [], host_ledger.get("events") or [],
                                  out_dir / "vista" / "claude_io") if vhost is not None else []
    _write(out_dir / "driver.json", {"run": run_info, "events": replay_stream, "host": host_ledger})
    (out_dir / "events.json").write_text(json.dumps(world.events, default=str, ensure_ascii=False, indent=0),
                                         encoding="utf-8")
    (out_dir / "tip_timeline.json").write_text(
        json.dumps(tip_timeline(world, truth_before.get("tip")), default=_json_default, ensure_ascii=False,
                   indent=0), encoding="utf-8")
    _write(out_dir / "host_facts.json", getattr(host, "facts", {}))
    return res


def _run_claude(vhost: VistaHost, out_dir: Path, arm: Arm, model: str, effort: str, options: dict) -> dict:
    from .claude_runner import ClaudeCodeRunner, RunnerConfig
    from .prompts import CONTINUE_TEXT, system_prompt
    from .rpc import RpcServer

    io_dir = out_dir / "vista" / "claude_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    rpc = RpcServer(vhost, log_path=io_dir / "rpc.jsonl")
    rpc.start()
    try:
        cfg_fields = {f.name for f in dataclasses.fields(RunnerConfig)}
        cfg_kw: dict[str, Any] = {"model": model, "effort": effort, "system_prompt": system_prompt(arm),
                                  "io_dir": io_dir, "continue_text": CONTINUE_TEXT}
        if "compaction" in cfg_fields:
            cfg_kw["compaction"] = arm.compaction
        cfg_kw.update({k: v for k, v in options.items() if k in cfg_fields})
        # RunnerConfig.preflight (default on) makes run() start with the throw-away READY segment
        runner = ClaudeCodeRunner(vhost, rpc, RunnerConfig(**{k: v for k, v in cfg_kw.items() if k in cfg_fields}))
        result = runner.run()
        return {"driver": "claude_code", **_asdict(result)}
    finally:
        rpc.stop()


def _skill_result(vhost: VistaHost | None, run_info: dict, mode: str, model: str, effort: str,
                  max_sim_s: float | None, max_cmds: int | None) -> dict:
    if vhost is None:
        return {"driver": True, "mode": mode, "model_id": model, "error": run_info.get("error"),
                "results": {}, "diagnosis": {}, "billing": dict(BILLING_UNKNOWN)}
    usage = run_info.get("usage") or run_info.get("usage_totals") or {}
    billing = dict(BILLING_UNKNOWN)
    if usage:
        billing.update({"tokens_in": usage.get("input_tokens"), "tokens_out": usage.get("output_tokens"),
                        "usd_est": usage.get("cost_usd"), "cost_known": usage.get("cost_usd") is not None,
                        "n_calls": usage.get("num_turns"), "ledger": "claude code result events",
                        "error": None, "cache_read_tokens": usage.get("cache_read_input_tokens"),
                        "cache_creation_tokens": usage.get("cache_creation_input_tokens")})
    b = vhost._budget()
    return {"driver": True, "mode": mode, "model_id": model, "effort": effort, "arm": vhost.arm.name,
            "arm_config": dataclasses.asdict(vhost.arm),
            "outcome": vhost.outcome, "stop_reason": vhost.stop_reason, "turns": vhost.turn,
            "acts": vhost.acts, "tool_calls": vhost.tool_calls, "gate_fallbacks": vhost.gate_fallbacks,
            "model_calls": usage.get("num_turns") if usage else None,
            "segments": len(vhost.segments), "error": run_info.get("error"),
            "final_text": (vhost.final_text or "")[:2000],
            "tool_sequence": [e.get("name") for e in vhost.driver_events if e.get("kind") == "tool_start"],
            "action_sequence": [e.get("action") for e in vhost.events if e.get("kind") == "act"],
            "budget": {"max_sim_s": max_sim_s, "max_wire_cmds": max_cmds},
            "sim_s": b.get("instrument_s_used"), "wire_cmds": b.get("commands_used"), "clock_paused": True,
            "diagnosis": vhost.diagnosis(), "results": vhost.results_folded(), "billing": billing}


def _envelope_violations(vhost: VistaHost | None) -> int | None:
    if vhost is None:
        return None
    from stmbench.harness.episode import count_envelope_violations
    return count_envelope_violations(vhost.driver_events)


def _asdict(x: Any) -> dict:
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, dict):
        return dict(x)
    return {"result": str(x)}


def _write(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, default=_json_default, ensure_ascii=False, indent=1), encoding="utf-8")
