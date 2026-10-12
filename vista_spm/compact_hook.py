#!/usr/bin/env python3
# Derived from VISTA (MIT), github.com/joshhhhhan/VISTA, backends/claude/compact_hook.py
"""Claude Code hook (PreCompact, PostCompact, Stop) for VISTA's checkpoint protocol.

README
------
Each segment's ``settings.json`` runs this file (exec form: ``command`` = the venv python,
``args`` = ``["-I", "-X", "utf8", "<this file>"]``) on PreCompact, PostCompact and Stop.
Standard library only; importable as ``vista_spm.compact_hook`` so the runner uses
the same :func:`decide`.

Decision logic (VISTA's, unchanged), from the episode's checkpoint state
``none | requested | ready``:

* PreCompact, state ``none`` or ``requested`` → block compaction with
  :data:`CHECKPOINT_INSTRUCTION`; from ``none`` the host is asked for a checkpoint
  (``request_checkpoint``), which puts it in notes-only mode until WORKING.md is written.
* PreCompact, state ``ready`` → block with :data:`HANDOFF_REASON` (the runner is about to
  replace the session).
* Stop, state ``requested`` → block the stop with :data:`CHECKPOINT_INSTRUCTION` (Claude
  Code feeds the reason back to the model as "Stop hook feedback").
* Everything else (Stop otherwise, PostCompact) → no decision (``{}``). PostCompact means
  a native compaction happened; the runner treats that as an error.
* When the episode is already terminal: PreCompact is blocked with the handoff reason and
  Stop is allowed.
* ``native`` compaction (arms without notes tools, env ``VISTA_COMPACTION=native``): never
  block; Claude Code compacts on its own and the runner only records the events.

Transport: the event goes to the RpcServer as ``episode/hook`` (env ``VISTA_RPC_PORT``,
``VISTA_RPC_TOKEN``, optional ``VISTA_RPC_HOST``); the runner's hook callback applies
:func:`decide` to the live host state and returns the output, which this script prints.
If the RpcServer cannot be reached, the fallback uses marker files in ``VISTA_HOOK_DIR``
(``checkpoint.requested`` / ``checkpoint.ready``, which the runner mirrors from the host
state and picks up again). Every invocation is appended to
``VISTA_HOOK_DIR/hook_events.jsonl``. The script always exits 0 and prints one JSON
object (``{}`` = no decision); a malformed event is allowed through, never blocked.

Verified with Claude Code 2.1.295 on Windows (2026-10-09):

* PreCompact input: ``session_id, transcript_path, cwd, prompt_id, hook_event_name,
  trigger ("auto"|"manual"), custom_instructions``; it fires before an API request once the
  context passes ``CLAUDE_CODE_AUTO_COMPACT_WINDOW`` x ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`` %.
  Both exit code 2 and stdout ``{"decision": "block", "reason": ...}`` block it: the
  stream shows ``system/status compacting`` then ``status null`` with no
  ``compact_boundary``, and the turn continues uncompacted (the model is not told).
* Stop input: ``... permission_mode, effort, stop_hook_active, last_assistant_message,
  background_tasks, session_crons``. ``{"decision": "block", "reason": R}`` makes Claude
  Code inject a user message ``"Stop hook feedback:\\nR"`` and continue the turn.
* PostCompact input carries ``trigger`` and ``compact_summary``.
* ``--include-hook-events`` streams ``hook_started``/``hook_response`` for Stop only.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

CHECKPOINT_INSTRUCTION = (
    "Review and update WORKING.md so it preserves relevant existing information and "
    "contains the complete continuation state needed after compaction."
)
HANDOFF_REASON = "The checkpoint is saved; end this response for handoff."
REQUEST_MARKER = "checkpoint.requested"
READY_MARKER = "checkpoint.ready"
EVENTS_FILE = "hook_events.jsonl"
CHECKPOINT_STATES = ("none", "requested", "ready")
_KEEP_KEYS = ("session_id", "hook_event_name", "trigger", "custom_instructions",
              "stop_hook_active", "prompt_id", "permission_mode")


def decide(event_name: str | None, checkpoint: str, *, terminal: bool = False,
           native: bool = False) -> tuple[dict, str | None]:
    """``(hook output, side effect)``; the side effect is ``"request_checkpoint"`` or ``None``."""
    if native:
        return {}, None
    if checkpoint not in CHECKPOINT_STATES:
        checkpoint = "none"
    if event_name == "PreCompact":
        if terminal or checkpoint == "ready":
            return {"decision": "block", "reason": HANDOFF_REASON}, None
        effect = "request_checkpoint" if checkpoint == "none" else None
        return {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}, effect
    if event_name == "Stop" and not terminal and checkpoint == "requested":
        return {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}, None
    return {}, None


def sanitize_event(event: Any) -> dict:
    """The fields worth logging and forwarding (summaries and messages are truncated)."""
    if not isinstance(event, dict):
        return {"hook_event_name": None, "malformed": True}
    record = {key: event[key] for key in _KEEP_KEYS if key in event}
    for key in ("last_assistant_message", "compact_summary"):
        value = event.get(key)
        if isinstance(value, str):
            record[key] = value[:2000]
            record[key + "_chars"] = len(value)
    return record


def marker_state(hook_dir: Path | None) -> str:
    if hook_dir is None:
        return "none"
    if (hook_dir / READY_MARKER).is_file():
        return "ready"
    if (hook_dir / REQUEST_MARKER).is_file():
        return "requested"
    return "none"


def _rpc_hook(record: dict, timeout: float, env: Any) -> dict:
    port = env.get("VISTA_RPC_PORT", "")
    token = env.get("VISTA_RPC_TOKEN", "")
    host = env.get("VISTA_RPC_HOST", "127.0.0.1") or "127.0.0.1"
    if not port.isdigit() or not token:
        raise RuntimeError("VISTA_RPC_PORT/VISTA_RPC_TOKEN not set")
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        sock.settimeout(timeout)
        reader = sock.makefile("rb")
        try:
            for rid, method, params in ((1, "auth", {"token": token}),
                                        (2, "episode/hook", {"event": record})):
                line = json.dumps({"id": rid, "method": method, "params": params},
                                  ensure_ascii=False, separators=(",", ":")) + "\n"
                sock.sendall(line.encode("utf-8"))
                raw = reader.readline()
                if not raw:
                    raise ConnectionError("RpcServer closed the connection")
                response = json.loads(raw.decode("utf-8"))
                if response.get("error") is not None:
                    raise RuntimeError(f"RpcServer error: {response['error']}")
            result = response.get("result") or {}
        finally:
            reader.close()
    output = result.get("output") if isinstance(result, dict) else None
    if not isinstance(output, dict):
        raise RuntimeError("RpcServer returned no hook output")
    return output


def _append(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError:
        pass


def run(raw: bytes, environ: dict | None = None) -> dict:
    """Process one hook invocation; returns the output to print. Never raises."""
    env = os.environ if environ is None else environ
    hook_dir_value = env.get("VISTA_HOOK_DIR", "")
    hook_dir = Path(hook_dir_value) if hook_dir_value else None
    try:
        event = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        event = None
    record = sanitize_event(event)
    log = {"t": round(time.time(), 3), "event": record}
    output: dict = {}
    if record.get("malformed"):
        log["path"] = "malformed"
    else:
        try:
            timeout = float(env.get("VISTA_HOOK_RPC_TIMEOUT", "20"))
        except ValueError:
            timeout = 20.0
        try:
            output = _rpc_hook(record, timeout, env)
            log["path"] = "rpc"
        except Exception as exc:  # noqa: BLE001 — fall back to marker files
            log["path"] = "markers"
            log["rpc_error"] = f"{type(exc).__name__}: {exc}"
            state = marker_state(hook_dir)
            output, effect = decide(record.get("hook_event_name"), state,
                                    native=env.get("VISTA_COMPACTION") == "native")
            log["checkpoint"] = state
            if effect == "request_checkpoint" and hook_dir is not None:
                try:
                    hook_dir.mkdir(parents=True, exist_ok=True)
                    (hook_dir / REQUEST_MARKER).touch()
                    log["effect"] = effect
                except OSError as mexc:
                    log["marker_error"] = f"{type(mexc).__name__}: {mexc}"
    log["output"] = output
    if hook_dir is not None:
        _append(hook_dir / EVENTS_FILE, log)
    return output


def main() -> int:
    try:
        raw = sys.stdin.buffer.read()
    except (OSError, ValueError):
        raw = b""
    output = run(raw)
    sys.stdout.buffer.write((json.dumps(output, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
