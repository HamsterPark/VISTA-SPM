"""Make a VISTA-SPM ledger readable by ``stmbench.replay``.

The replay builder reads ``driver.json["events"]``: ``tool_start`` / ``tool_end`` pairs (name,
args, wall time ``t``, instrument time ``sim_s``, ``preview``, ``ok``, ``tool_call_id``) and
``message`` events (assistant narration). Its captions know MAST's skill names and SI
arguments (``ScanAt``, ``MoveAtomTo``, ``MoveToXY``).

:func:`replay_events` builds that stream from the host's tool-call record and the Claude Code
transcripts: ``act`` calls are renamed to the MAST skill they ran, with the resolved positions
in metres, and the model's text between tool calls becomes ``message`` events placed at the
instrument time of the surrounding calls. The original ledger files are never rewritten;
:func:`replay_copy` prepares a converted copy of a finished run directory for the builder.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _pairs(driver_events: list[dict]) -> list[tuple[dict, dict]]:
    """tool_start / tool_end in call order (the host serialises calls, so they alternate)."""
    out: list[tuple[dict, dict]] = []
    start: dict | None = None
    for e in driver_events or []:
        k = e.get("kind")
        if k == "tool_start":
            start = e
        elif k == "tool_end" and start is not None:
            out.append((start, e))
            start = None
    if start is not None:
        out.append((start, dict(start, kind="tool_end")))
    return out


def _act_index(host_events: list[dict]) -> dict[int, dict]:
    return {int(e["turn"]): e for e in host_events or [] if e.get("kind") == "act" and "turn" in e}


def _m(v: Any) -> float | None:
    return float(v) * 1e-9 if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _caption_call(name: str, args: dict, act: dict | None) -> tuple[str, dict]:
    """(name, args) as the replay's captions understand them."""
    if name != "act":
        return name, {k: v for k, v in (args or {}).items() if isinstance(v, (str, int, float, bool))}
    action = str((args or {}).get("action") or "act")
    params = (act or {}).get("params") or (args or {}).get("params") or {}
    resolved = (act or {}).get("resolved") or {}
    if action == "scan":
        cx, cy = (resolved.get("center_nm") or [None, None])[:2]
        out = {"center_x_m": _m(cx), "center_y_m": _m(cy), "size_m": _m(params.get("size_nm")),
               "pixels": params.get("pixels")}
        return "ScanAt", {k: v for k, v in out.items() if v is not None}
    if action == "move_atom":
        ax, ay = (resolved.get("atom_nm") or [None, None])[:2]
        tx, ty = (resolved.get("target_nm") or [None, None])[:2]
        out = {"atom_x_m": _m(ax), "atom_y_m": _m(ay), "target_x_m": _m(tx), "target_y_m": _m(ty),
               "manip_bias_v": params.get("bias_v"),
               "manip_setpoint_a": (float(params["current_pa"]) * 1e-12) if "current_pa" in params else None}
        return "MoveAtomTo", {k: v for k, v in out.items() if v is not None}
    if action == "move_tip":
        x, y = (resolved.get("to_nm") or [None, None])[:2]
        return "MoveToXY", {k: v for k, v in {"x_m": _m(x), "y_m": _m(y)}.items() if v is not None}
    flat = {k: v for k, v in params.items() if isinstance(v, (str, int, float, bool))}
    return action, flat


def _transcript_blocks(io_dir: Path) -> list[dict]:
    """Assistant text and tool_use blocks, in order, across every segment's transcript."""
    blocks: list[dict] = []
    if not io_dir.is_dir():
        return blocks
    for f in sorted(io_dir.glob("segment_*.stdout.jsonl")):
        for raw in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            if ev.get("type") != "assistant":
                continue
            for b in (ev.get("message") or {}).get("content") or []:
                if b.get("type") == "text" and str(b.get("text") or "").strip():
                    blocks.append({"type": "text", "text": str(b["text"])})
                elif b.get("type") == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.get("id"),
                                   "name": str(b.get("name") or "").split("__")[-1]})
    return blocks


def replay_events(driver_events: list[dict], host_events: list[dict], io_dir: str | Path) -> list[dict]:
    pairs = _pairs(driver_events)
    acts = _act_index(host_events)
    blocks = _transcript_blocks(Path(io_dir))
    out: list[dict] = []
    t_last = float(pairs[0][0].get("t", 0.0)) if pairs else 0.0
    sim_last = pairs[0][0].get("sim_s") if pairs else None
    i = 0

    def emit_call(start: dict, end: dict, call_id: str) -> None:
        nonlocal t_last, sim_last
        act = acts.get(int(end.get("turn", -1))) if start.get("name") == "act" else None
        name, args = _caption_call(str(start.get("name") or ""), start.get("args") or {}, act)
        out.append({"kind": "tool_start", "name": name, "args": args, "tool_call_id": call_id,
                    "t": start.get("t"), "sim_s": start.get("sim_s")})
        out.append({"kind": "tool_end", "name": name, "tool_call_id": call_id, "ok": not end.get("is_error"),
                    "preview": end.get("preview", ""), "t": end.get("t"), "sim_s": end.get("sim_s")})
        t_last, sim_last = float(end.get("t", t_last) or t_last), end.get("sim_s", sim_last)

    for b in blocks:
        if b["type"] == "text":
            out.append({"kind": "message", "role": "assistant", "text": b["text"], "t": t_last, "sim_s": sim_last})
        elif i < len(pairs):
            emit_call(pairs[i][0], pairs[i][1], str(b.get("id") or f"call-{i}"))
            i += 1
    for k in range(i, len(pairs)):                      # calls the transcript did not show
        emit_call(pairs[k][0], pairs[k][1], f"call-{k}")
    return out


def replay_copy(run_dir: str | Path, dest: str | Path) -> Path:
    """Copy a finished run directory to ``dest`` with a replay-compatible ``driver.json``."""
    src, dst = Path(run_dir), Path(dest)
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("mast_root"))
    drv = json.loads((src / "driver.json").read_text(encoding="utf-8"))
    host = drv.get("host") or drv
    if not any(e.get("kind") == "tool_start" for e in drv.get("events") or []):
        drv["events"] = replay_events(host.get("driver_events") or [], host.get("events") or [],
                                      src / "vista" / "claude_io")
    (dst / "driver.json").write_text(json.dumps(drv, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    return dst
