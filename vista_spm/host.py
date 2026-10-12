"""VistaHost: the episode host behind the MCP tools (implements ``protocol.RunnerHost``).

It owns, for one episode: the MAST ``ExecutionContext`` the actions run through, the
simulator clock (paused except while an action executes), the frame archive, the notes,
the judge's report channels, the action history, the one-action gate, the budgets and the
checkpoint state. Tools are assembled per ablation arm (``arms.py``).

What the host never does: read the simulator's hidden state. Its inputs are the controller's
replies (through MAST skills) and the files the instrument saved into the session directory.

Dependencies are injected so the host can be tested without MAST or the simulator; use
:meth:`VistaHost.from_runtime` to wire it to a running ``stmbench.harness.runtime_host.RuntimeHost``.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from . import prompts
from .actions import ACTIONS, ActionError, ActionRunner, act_description, act_schema, validate_params
from .arms import Arm
from .protocol import (ImageBlock, ToolDef, ToolReply, compact_json, error_reply, json_reply)

RESULT_TOOL_NAME = "report_result"
TIP_TOOL_NAME = "report_tip_state"
#: record names the judge folds (``harness.results.fold_results`` / ``ic_driver._diagnosis_from``)
JUDGE_RESULT = "ReportResult"
JUDGE_TIP = "ReportTipState"

NOTES_TOOL_NAMES = ("read_guide", "write_guide", "read_working", "write_working")

GATE_REFUSAL = ("Observe the result of the previous act before another action. No action was executed.")
CHECKPOINT_REFUSAL = prompts.CHECKPOINT_INSTRUCTION + " No other tool executes until WORKING.md is saved."


def _terminal_text(reason: str) -> str:
    return (f"The instrument is no longer available ({reason}). No further actions can be executed. "
            "Report any remaining results with report_result, then end your reply with [DONE] "
            "(or [ABORT] if the task could not be completed).")


class VistaHost:
    def __init__(self, *, arm: Arm, vista_dir: str | Path, task: str, claim_ids: list[str],
                 ctx: Any, clock: Any, session_dir: str | Path, time_scale: float,
                 budget_probe: Callable[[], tuple[float | None, int | None]],
                 max_sim_s: float | None = None, max_cmds: int | None = None, max_acts: int = 400,
                 is_qplus: bool = False, archive: Any = None, notes: Any = None,
                 gate_fallback_s: float = 10.0, vigil: Any = None,
                 state_skills: tuple[str, ...] = ("GetBias", "GetSetpoint", "GetCurrent", "GetZPosition",
                                                  "GetScanXYPosition", "GetDriftCompensation")):
        self.arm = arm
        self.vista_dir = Path(vista_dir)
        self.vista_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.claim_ids = [str(c) for c in claim_ids]
        self.ctx = ctx
        self.clock = clock
        self.session_dir = Path(session_dir)
        self.time_scale = float(time_scale or 1.0)
        self._budget_probe = budget_probe
        self.max_sim_s = max_sim_s
        self.max_cmds = max_cmds
        self.max_acts = int(max_acts)
        self.is_qplus = bool(is_qplus)
        self.state_skills = state_skills
        self.gate_fallback_s = float(gate_fallback_s)
        self.vigil = vigil

        if archive is None:
            from .archive import Archive
            archive = Archive(self.vista_dir / "archive", display_max=arm.display_max,
                              grid=arm.grid and arm.observation == "image")
        self.archive = archive
        if notes is None and arm.notes:
            from .notes import Notes
            notes = Notes(self.vista_dir / "notes")
        self.notes = notes

        self.actions = [n for n, s in ACTIONS.items() if self.is_qplus or not s.qplus_only]
        self.runner = ActionRunner(ctx, resolve_point=self._resolve_point, time_scale=self.time_scale,
                                   clock=clock, is_qplus=self.is_qplus, list_files=self._session_files)

        self._lock = threading.RLock()          # tool calls
        self._gate_lock = threading.Lock()      # act gate (acknowledged from the runner thread)
        self._act_pending = False
        self._act_done_t = 0.0
        self.gate_fallbacks = 0

        self.turn = 0
        self.acts = 0
        self.tool_calls = 0
        self.events: list[dict] = []            # per act / report, for `history` and the ledger
        self.driver_events: list[dict] = []     # tool_start / tool_end stream (driver.json)
        self.result_reports: list[dict] = []
        self.tip_reports: list[dict] = []
        self.terminal = False                   # the episode is over (finish() was called)
        self.instrument_closed = False          # no more actions (budget); reports still accepted
        self.stop_reason: str | None = None
        self.checkpoint = "none"
        self.checkpoint_reason: str | None = None
        self.images_delivered = 0
        self.segment = -1
        self.segments: list[dict] = []
        self.outcome: str | None = None
        self.final_text = ""
        self._known_files: set[str] = set()
        self._t0 = time.time()

        self.sim0, self.cmds0 = self._probe()
        self._scan_existing_files()
        self.initial_state = self._state_echo()
        self._tools: dict[str, tuple[ToolDef, Callable[[dict], ToolReply]]] = self._build_tools()

    # ── construction from a running RuntimeHost ─────────────────────────────
    @classmethod
    def from_runtime(cls, runtime_host, scenario, arm: Arm, vista_dir: str | Path, *,
                     max_sim_s: float | None, max_cmds: int | None, max_acts: int = 400,
                     run_id: str = "vista") -> "VistaHost":
        from stmbench.harness.ic_driver import _budget_probe

        world = runtime_host.world
        facts = getattr(runtime_host, "facts", {}) or {}
        is_qplus = facts.get("is_qplus") is True
        # ``constraint`` claims (e.g. "the neighbours did not move") are checked against the
        # truth directly and count as reported (truth_criteria.verify_claims): not offered here
        return cls(arm=arm, vista_dir=vista_dir, task=scenario.task,
                   claim_ids=[str(c.get("id")) for c in (scenario.claims or [])
                              if str(c.get("kind")) != "constraint"],
                   ctx=runtime_host.context(run_id=run_id), clock=world.clock,
                   session_dir=world.session_dir, time_scale=float(scenario.time_scale or 1.0),
                   budget_probe=lambda: _budget_probe(runtime_host), max_sim_s=max_sim_s,
                   max_cmds=max_cmds, max_acts=max_acts, is_qplus=is_qplus)

    # ── tools ───────────────────────────────────────────────────────────────
    def _build_tools(self) -> dict[str, tuple[ToolDef, Callable[[dict], ToolReply]]]:
        t: dict[str, tuple[ToolDef, Callable[[dict], ToolReply]]] = {}
        grid = bool(getattr(self.archive, "grid", False))
        t["act"] = (ToolDef("act", act_description(self.actions, self.arm.observation, grid), act_schema(self.actions),
                            title="Instrument action", read_only=False, destructive=True), self._act)
        if self.claim_ids:
            t[RESULT_TOOL_NAME] = (ToolDef(RESULT_TOOL_NAME, (
                "Record one result of the task for scoring. This is the only channel the results are read "
                "from: one call per result, repeat to update (the last call counts); a result never reported "
                "counts as not reproduced. Position results need x_nm and y_nm in scan-frame nm. Does not "
                "touch the instrument."), _result_schema(self.claim_ids), title="Report a result"),
                self._report_result)
        t[TIP_TOOL_NAME] = (ToolDef(TIP_TOOL_NAME, (
            "Record your current judgement of the tunnelling junction and the tip. The last call counts; "
            "not reporting means no judgement. Does not touch the instrument."), _TIP_SCHEMA,
            title="Report tip state"), self._report_tip)
        if self.arm.history:
            t["history"] = (ToolDef("history", (
                "Read objective records of this episode without changing anything. `events` returns the exact "
                "action records (parameters as resolved, controller result, instrument state afterwards, new "
                "frame ids, instrument time) by turn; use start_turn, end_turn and limit to select a range. "
                "`attempts` summarises the episode so far: actions by type, frames and spectra, reports, "
                "instrument time used."), _HISTORY_SCHEMA, title="Episode history"), self._history)
        if self.arm.notes:
            from .notes import NOTES_TOOLS
            for name in NOTES_TOOL_NAMES:
                t[name] = (NOTES_TOOLS[name], self._notes_handler(name))
        if self.arm.inspect or self.arm.read_values:
            from .viewtools import view_tools
            for name, (tdef, fn) in view_tools(self.arm.display_max, grid=grid).items():
                if (name == "inspect" and self.arm.inspect) or (name == "read_values" and self.arm.read_values):
                    t[name] = (tdef, self._archive_handler(fn))
        if self.arm.measure:
            from .measure import MEASURE_TOOLS
            for name, (tdef, fn) in MEASURE_TOOLS.items():
                t[name] = (tdef, self._archive_handler(fn))
        if self.arm.advisor:
            t["advisor"] = (ToolDef("advisor", (
                "Run MAST's own image detectors on one archived scan frame and return their outputs as text "
                "(tip-quality and double-tip scores, mid-scan tip-change rows, atomic-lattice check, and the "
                "learned vision model's heads when available). These are another program's readings, not "
                "ground truth. Does not change the instrument."), _ADVISOR_SCHEMA, title="MAST detectors"),
                self._advisor)
        return t

    def list_tools(self) -> list[dict]:
        return [tdef.to_mcp() for tdef, _ in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> ToolReply:
        with self._lock:
            self.tool_calls += 1
            args = dict(args or {}) if isinstance(args, dict) else {}
            rec_start = {"kind": "tool_start", "name": name, "args": _clip_args(args), "turn": self.turn,
                         "t": time.time(), "sim_s": self._sim_now()}
            self.driver_events.append(rec_start)
            entry = self._tools.get(name)
            if entry is None:
                reply = error_reply(f"unknown tool {name!r}", available=list(self._tools))
            elif self.checkpoint == "requested" and name not in NOTES_TOOL_NAMES:
                reply = error_reply(CHECKPOINT_REFUSAL)
            else:
                try:
                    reply = entry[1](args)
                except Exception as exc:  # noqa: BLE001 — the host never raises into the RPC layer
                    reply = error_reply(f"internal error in {name}: {type(exc).__name__}: {exc}")
            self.images_delivered += len(reply.images)
            self.driver_events.append({"kind": "tool_end", "name": name, "turn": self.turn,
                                       "is_error": reply.is_error, "effect": reply.effect,
                                       "n_images": len(reply.images), "preview": reply.text[:600],
                                       "t": time.time(), "sim_s": self._sim_now()})
            return reply

    # ── act ─────────────────────────────────────────────────────────────────
    def _act(self, args: dict) -> ToolReply:
        if self.terminal or self.instrument_closed:
            return error_reply(_terminal_text(self.stop_reason or "episode over"), budget=self._budget())
        with self._gate_lock:
            if self._act_pending:
                waited = time.monotonic() - self._act_done_t
                if waited < self.gate_fallback_s:
                    return error_reply(GATE_REFUSAL)
                self.gate_fallbacks += 1          # the runner never acknowledged; do not deadlock
                self._act_pending = False
        name = args.get("action")
        spec = ACTIONS.get(name) if name in self.actions else None
        if spec is None:
            return error_reply(f"unknown action {name!r}", actions=self.actions)
        try:
            params = validate_params(spec, args.get("params"))
        except ActionError as exc:
            return error_reply(str(exc), action=name)
        turn = self.turn + 1
        sim_before = self._sim_now()
        t_wall = time.time()
        self.clock.resume()
        try:
            outcome = self.runner.run(spec, params)
        finally:
            self.clock.pause()
        self.turn = turn
        self.acts += 1
        new_entries = self._ingest_new_files(turn)
        state = self._state_echo()
        budget = self._budget()
        sim_after = self._sim_now()
        record = {"turn": turn, "action": name, "params": params, "note": str(args.get("note") or "")[:1000],
                  "ok": outcome.ok, "error": outcome.error, "facts": outcome.facts, "resolved": outcome.resolved,
                  "steps": outcome.steps, "new_frames": [e.fid for e in new_entries], "state": state,
                  "instrument_s": (None if sim_before is None or sim_after is None else sim_after - sim_before),
                  "budget": budget, "t_wall": t_wall}
        self.events.append({"kind": "act", **record})
        self._check_budget(budget)

        payload: dict = {"turn": turn, "action": name, "ok": outcome.ok}
        if outcome.error:
            payload["error"] = outcome.error
        if outcome.resolved:
            payload["resolved"] = outcome.resolved
        if outcome.facts:
            payload["result"] = outcome.facts
        payload["instrument_s"] = record["instrument_s"]
        payload["new_frames"] = [e.summary() for e in new_entries]
        if not new_entries:
            latest = self.archive.latest("s")
            payload["latest_frame"] = latest.fid if latest is not None else None
        payload["instrument"] = state
        payload["budget"] = budget
        images, extra = self._observation_for(new_entries)
        payload.update(extra)
        if self.instrument_closed:
            payload["episode"] = _terminal_text(self.stop_reason or "budget")
        reply = json_reply(payload, images=images, effect="environment",
                           meta={"turn": turn, "action": name})
        reply.is_error = not outcome.ok
        with self._gate_lock:
            self._act_pending = True
            self._act_done_t = time.monotonic()
        return reply

    def _observation_for(self, entries: list) -> tuple[list[ImageBlock], dict]:
        """The observation pushed into context after an act: the newest scan (or spectrum)."""
        if not entries:
            return [], {}
        scans = [e for e in entries if e.kind == "s"]
        spectra = [e for e in entries if e.kind == "d"]
        shown = ([scans[-1]] if scans else []) + spectra[-2:]
        if self.arm.observation == "image":
            images: list[ImageBlock] = []
            for e in shown:
                images += list(self.archive.observation(e.fid))
            return images, {}
        if self.arm.observation == "text":
            return [], {"observation_text": {e.fid: self.archive.text_grid(e.fid) for e in shown}}
        if self.arm.observation == "vigil":
            # scans: MAST's detectors and vision model as text; spectra: the numbers (MAST's skills
            # returned curve data as numbers; its vision model never saw curves)
            return [], {"observation_text": {e.fid: (self.archive.text_grid(e.fid) if e.kind == "d"
                                                     else self._vigil_text(e.fid)) for e in shown}}
        return [], {}

    def _vigil_text(self, fid: str) -> str:
        if self.vigil is None:
            from .advisor import vigil_summary
            return vigil_summary(self.archive, fid)
        return self.vigil(self.archive, fid)

    # ── reports ─────────────────────────────────────────────────────────────
    def _report_result(self, args: dict) -> ToolReply:
        from stmbench.harness.results import fold_results

        cid = args.get("claim_id")
        if cid not in self.claim_ids:
            return error_reply(f"unknown claim_id {cid!r}", claim_ids=self.claim_ids)
        val = args.get("value")
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            return error_reply("value must be a number")
        clean = {k: v for k, v in args.items() if k in _RESULT_FIELDS}
        rec = {"name": JUDGE_RESULT, "turn": self.turn, "args": clean}
        self.result_reports.append(rec)
        self.events.append({"kind": "report", "turn": self.turn, "report": "result", "args": clean,
                            "t_wall": time.time()})
        folded = fold_results(self.result_reports)
        left = [c for c in self.claim_ids if c not in folded["claims"]]
        return json_reply({"recorded": clean, "not_yet_reported": left}, effect="report")

    def _report_tip(self, args: dict) -> ToolReply:
        clean = {k: v for k, v in args.items() if k in ("junction", "atomic_resolution", "tip_state", "reason")}
        bad = [k for k in ("junction", "tip_state") if k in clean and clean[k] not in _TIP_ENUMS[k]]
        if bad:
            return error_reply(f"invalid value for {bad}", allowed={k: _TIP_ENUMS[k] for k in bad})
        rec = {"name": JUDGE_TIP, "turn": self.turn, "args": clean}
        self.tip_reports.append(rec)
        self.events.append({"kind": "report", "turn": self.turn, "report": "tip_state", "args": clean,
                            "t_wall": time.time()})
        return json_reply({"recorded": clean}, effect="report")

    def results_folded(self) -> dict:
        from stmbench.harness.results import fold_results
        return fold_results(self.result_reports)

    def diagnosis(self) -> dict:
        from stmbench.harness.ic_driver import _diagnosis_from
        return _diagnosis_from(list(self.tip_reports), self.final_text)

    # ── history ─────────────────────────────────────────────────────────────
    def _history(self, args: dict) -> ToolReply:
        view = args.get("view", "events")
        if view == "attempts":
            return json_reply(self._attempts())
        if view != "events":
            return error_reply("view must be 'events' or 'attempts'")
        lo = int(args.get("start_turn", 0) or 0)
        hi = args.get("end_turn")
        hi = int(hi) if isinstance(hi, (int, float)) else None
        limit = max(1, min(128, int(args.get("limit", 64) or 64)))
        rows = [_public_event(e) for e in self.events
                if e.get("turn", 0) >= lo and (hi is None or e.get("turn", 0) <= hi)]
        return json_reply({"view": "events", "n": len(rows), "events": rows[-limit:],
                           "truncated": len(rows) > limit})

    def _attempts(self) -> dict:
        by_action: dict[str, dict] = {}
        for e in self.events:
            if e.get("kind") != "act":
                continue
            d = by_action.setdefault(e["action"], {"n": 0, "failed": 0})
            d["n"] += 1
            d["failed"] += 0 if e.get("ok") else 1
        # entry attributes (fid, kind letter) are the archive's stable Python API; the summary
        # dicts are model-facing and may spell things differently
        frames = [e.fid for e in self.archive.frames(kind="s")]
        spectra = [e.fid for e in self.archive.frames(kind="d")]
        return {"view": "attempts", "turn": self.turn, "actions": by_action,
                "frames": frames[-40:], "n_frames": len(frames),
                "spectra": spectra[-40:], "n_spectra": len(spectra),
                "results_reported": sorted(self.results_folded()["claims"]),
                "results_missing": [c for c in self.claim_ids if c not in self.results_folded()["claims"]],
                "tip_reports": len(self.tip_reports), "budget": self._budget()}

    # ── notes, archive tools, advisor ───────────────────────────────────────
    def _notes_handler(self, name: str) -> Callable[[dict], ToolReply]:
        def handler(args: dict) -> ToolReply:
            handle = getattr(self.notes, "handle", None)
            if callable(handle):
                reply = handle(name, args)
            else:
                fn = getattr(self.notes, name)
                try:
                    out = fn(**args) if name.startswith("write_") else fn()
                except (TypeError, ValueError) as exc:
                    return error_reply(f"{name}: {exc}")
                reply = (json_reply({"content": out}) if name.startswith("read_")
                         else json_reply({"ok": True, "chars": len(str(args.get("content", "")))}, effect="notes"))
            if name == "write_working" and self.checkpoint == "requested" and not reply.is_error:
                self.checkpoint = "ready"
                reply = ToolReply(text=reply.text + "\nWORKING.md updated; compact checkpoint ready.",
                                  images=reply.images, is_error=False, effect="notes", meta=reply.meta)
            return reply
        return handler

    def _archive_handler(self, fn: Callable[[Any, dict], ToolReply]) -> Callable[[dict], ToolReply]:
        def handler(args: dict) -> ToolReply:
            return fn(self.archive, args)
        return handler

    def _advisor(self, args: dict) -> ToolReply:
        from .advisor import advisor_reply
        return advisor_reply(self.archive, args)

    # ── RunnerHost ──────────────────────────────────────────────────────────
    def initial_message(self) -> tuple[str, list[ImageBlock]]:
        text = prompts.task_block(self.task, claim_ids=self.claim_ids, budget_s=self.max_sim_s)
        text += ("\n\nInstrument state now: " + compact_json(self.initial_state)
                 + "\nNo frame has been acquired yet in this session.")
        return text, []

    def recovery_message(self) -> tuple[str, list[ImageBlock]]:
        latest = self.archive.latest("s")
        last_act = next((_public_event(e) for e in reversed(self.events) if e.get("kind") == "act"), None)
        record = {"current_observation": latest.summary() if latest is not None else None,
                  "instrument": self._state_echo(), "last_action_result": last_act,
                  "episode_so_far": self._attempts()}
        notes = {}
        if self.notes is not None:
            notes = {"guide": _safe_read(self.notes, "read_guide"),
                     "working_memory": _safe_read(self.notes, "read_working")}
        text = ("Environment record:\n" + compact_json(record))
        if notes:
            text += "\n\nAgent-authored notes:\n" + compact_json(notes)
        text += "\n\n" + prompts.task_block(self.task, claim_ids=self.claim_ids, budget_s=self.max_sim_s)
        text += "\n\nContinue the task with as little instrument time and as few instrument actions as possible."
        images: list[ImageBlock] = []
        if latest is not None and self.arm.observation == "image":
            images = list(self.archive.observation(latest.fid))
        return text, images

    def acknowledge_act(self) -> None:
        with self._gate_lock:
            self._act_pending = False

    def request_checkpoint(self, reason: str) -> None:
        with self._lock:
            if self.checkpoint == "none" and self.arm.notes:
                self.checkpoint = "requested"
                self.checkpoint_reason = str(reason)
                self.driver_events.append({"kind": "checkpoint_requested", "reason": str(reason),
                                           "turn": self.turn, "t": time.time()})

    def status(self) -> dict:
        return {"terminal": self.terminal, "instrument_closed": self.instrument_closed,
                "stop_reason": self.stop_reason, "checkpoint": self.checkpoint,
                "images_delivered": self.images_delivered, "acts": self.acts, "turn": self.turn,
                "tool_calls": self.tool_calls, "budget": self._budget(), "segment": self.segment,
                "outcome": self.outcome}

    def begin_segment(self, index: int, session_id: str) -> None:
        with self._lock:
            self.segment = int(index)
            self.images_delivered = 0
            self.checkpoint = "none"            # every segment is a fresh context
            self.checkpoint_reason = None
            self.segments.append({"index": int(index), "session_id": session_id, "start_turn": self.turn,
                                  "t": time.time()})
            self.driver_events.append({"kind": "segment_start", "index": int(index), "session_id": session_id,
                                       "turn": self.turn, "t": time.time()})

    def end_segment(self, index: int, info: dict) -> None:
        with self._lock:
            for s in self.segments:
                if s["index"] == int(index):
                    s.update({"end_turn": self.turn, "info": _jsonable(info), "t_end": time.time()})
            self.driver_events.append({"kind": "segment_end", "index": int(index), "info": _jsonable(info),
                                       "turn": self.turn, "t": time.time()})
        with self._gate_lock:
            self._act_pending = False         # a fresh session sees the current state in its first message

    def finish(self, outcome: str, final_text: str) -> None:
        with self._lock:
            self.outcome = str(outcome)
            self.final_text = str(final_text or "")
            if not self.terminal:
                self.terminal = True
                self.stop_reason = self.stop_reason or str(outcome)
            self.driver_events.append({"kind": "finish", "outcome": self.outcome, "turn": self.turn,
                                       "final_text": self.final_text[:2000], "t": time.time()})

    # ── internals ───────────────────────────────────────────────────────────
    def _resolve_point(self, spec: dict) -> tuple[float, float, dict]:
        if not isinstance(spec, dict):
            raise ActionError("a position must be {x_nm, y_nm} or {frame, x, y}")
        if "x_nm" in spec or "y_nm" in spec:
            x, y = spec.get("x_nm"), spec.get("y_nm")
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (x, y)):
                raise ActionError("x_nm and y_nm must both be numbers")
            return float(x), float(y), {"x_nm": float(x), "y_nm": float(y)}
        fid, x, y = spec.get("frame"), spec.get("x"), spec.get("y")
        if not isinstance(fid, str) or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                               for v in (x, y)):
            raise ActionError("a frame position needs frame (id), x and y (display pixels)")
        try:
            entry = self.archive.get(fid)
        except KeyError as exc:
            raise ActionError(str(exc).strip("'\"")) from None
        if entry.kind not in ("s", "p"):
            raise ActionError(f"{fid} is not a scan frame")
        from .frames import display_to_scan_nm
        X, Y = display_to_scan_nm(entry, float(x), float(y))
        return float(X), float(Y), {"frame": fid, "x": float(x), "y": float(y),
                                    "x_nm": round(float(X), 4), "y_nm": round(float(Y), 4)}

    def _session_files(self) -> set[str]:
        if not self.session_dir.is_dir():
            return set()
        return {p.name for p in self.session_dir.iterdir() if p.suffix.lower() in (".sxm", ".dat")}

    def _scan_existing_files(self) -> None:
        if self.session_dir.is_dir():
            for p in self.session_dir.iterdir():
                if p.suffix.lower() in (".sxm", ".dat"):
                    self._known_files.add(p.name)

    def _ingest_new_files(self, turn: int) -> list:
        if not self.session_dir.is_dir():
            return []
        new = [p for p in self.session_dir.iterdir()
               if p.suffix.lower() in (".sxm", ".dat") and p.name not in self._known_files]
        new.sort(key=lambda p: (p.stat().st_mtime, p.name))
        out = []
        sim = self._sim_now()
        for p in new:
            self._known_files.add(p.name)
            try:
                if p.suffix.lower() == ".sxm":
                    out.append(self.archive.add_scan(turn, p, sim_s=sim))
                else:
                    out.append(self.archive.add_spectrum(turn, p, sim_s=sim))
            except Exception as exc:  # noqa: BLE001 — a broken file is recorded, not fatal
                self.events.append({"kind": "archive_error", "turn": turn, "file": p.name,
                                    "error": f"{type(exc).__name__}: {exc}"})
        return out

    def _state_echo(self) -> dict:
        """Controller readbacks after an action, read while the clock is paused (free).

        Explicit per-skill mapping into model units; an unreadable value is left out, never
        guessed."""
        out: dict = {}
        for name in self.state_skills:
            try:
                res = self.ctx.run(name, {})
            except Exception:  # noqa: BLE001
                continue
            d = getattr(res, "data", None)
            if not getattr(res, "success", False) or not isinstance(d, dict):
                continue
            mapper = _STATE_MAP.get(name)
            if mapper is not None:
                out.update(mapper(d))
            else:
                for k, v in d.items():
                    nk, nv = _normalise_reading(k, v)
                    if nk is not None:
                        out[nk] = nv
        return out

    def _probe(self) -> tuple[float | None, int | None]:
        try:
            return self._budget_probe()
        except Exception:  # noqa: BLE001
            return None, None

    def _sim_now(self) -> float | None:
        try:
            return float(self.clock.sim())
        except Exception:  # noqa: BLE001
            return None

    def _budget(self) -> dict:
        sim, cmds = self._probe()
        used_s = (sim - self.sim0) if (sim is not None and self.sim0 is not None) else None
        used_c = (cmds - self.cmds0) if (cmds is not None and self.cmds0 is not None) else None
        return {"instrument_s_used": None if used_s is None else round(used_s, 1),
                "instrument_s_max": self.max_sim_s, "commands_used": used_c, "commands_max": self.max_cmds,
                "actions": self.acts, "actions_max": self.max_acts}

    def _check_budget(self, b: dict) -> None:
        """Close the instrument when a budget is spent. The episode is NOT over: the model
        still reports results and ends its reply with [DONE]; the runner ends the episode
        on the marker (or on its stall guard)."""
        if self.terminal or self.instrument_closed:
            return
        used = b.get("instrument_s_used")
        if self.max_sim_s and used is not None and used >= self.max_sim_s:
            self.instrument_closed, self.stop_reason = True, "budget_sim"
        elif self.max_cmds and b.get("commands_used") is not None and b["commands_used"] >= self.max_cmds:
            self.instrument_closed, self.stop_reason = True, "budget_cmds"
        elif self.acts >= self.max_acts:
            self.instrument_closed, self.stop_reason = True, "max_acts"

    # ── ledger views ────────────────────────────────────────────────────────
    def ledger(self) -> dict:
        return {"arm": self.arm.name, "outcome": self.outcome, "stop_reason": self.stop_reason,
                "turns": self.turn, "acts": self.acts, "tool_calls": self.tool_calls,
                "instrument_closed": self.instrument_closed,
                "gate_fallbacks": self.gate_fallbacks, "segments": self.segments,
                "events": self.events, "driver_events": self.driver_events,
                "result_reports": self.result_reports, "tip_reports": self.tip_reports,
                "final_text": self.final_text, "budget": self._budget(), "tools": self.tool_names()}


# ── helpers ─────────────────────────────────────────────────────────────────
_RESULT_FIELDS = ("claim_id", "value", "unit", "sigma", "x_nm", "y_nm", "note")
_TIP_ENUMS = {"junction": ["clean", "dirty", "unknown"],
              "tip_state": ["sharp", "blunt", "double", "unstable", "unknown"]}


def _result_schema(ids: list[str]) -> dict:
    return {"type": "object", "properties": {
        "claim_id": {"type": "string", "enum": list(ids), "description": "which result (see the task)"},
        "value": {"type": "number", "description": "the value, in the unit the task asks for"},
        "unit": {"type": "string", "description": "the unit you used (checked by a person, not scored)"},
        "sigma": {"type": "number", "description": "uncertainty (optional)"},
        "x_nm": {"type": "number", "description": "position results: x in scan-frame nm"},
        "y_nm": {"type": "number", "description": "position results: y in scan-frame nm"},
        "note": {"type": "string", "maxLength": 1000, "description": "basis, one or two sentences (optional)"}},
        "required": ["claim_id", "value"], "additionalProperties": False}


_TIP_SCHEMA = {"type": "object", "properties": {
    "junction": {"type": "string", "enum": _TIP_ENUMS["junction"],
                 "description": "is the tunnelling junction clean (barrier of several eV) or dirty; unknown if not measured"},
    "atomic_resolution": {"type": "boolean", "description": "you currently have an atomically resolved image"},
    "tip_state": {"type": "string", "enum": _TIP_ENUMS["tip_state"],
                  "description": "sharp / blunt / double (duplicated features) / unstable / unknown"},
    "reason": {"type": "string", "maxLength": 1000, "description": "the evidence, one or two sentences"}},
    "required": ["junction", "atomic_resolution", "tip_state", "reason"], "additionalProperties": False}

_HISTORY_SCHEMA = {"type": "object", "properties": {
    "view": {"type": "string", "enum": ["events", "attempts"]},
    "start_turn": {"type": "integer", "minimum": 0, "maximum": 999999},
    "end_turn": {"type": "integer", "minimum": 0, "maximum": 999999},
    "limit": {"type": "integer", "minimum": 1, "maximum": 128, "default": 64}},
    "required": ["view"], "additionalProperties": False}

_ADVISOR_SCHEMA = {"type": "object", "properties": {
    "frame": {"type": "string", "description": "an archived scan frame id, e.g. t0007.s0"}},
    "required": ["frame"], "additionalProperties": False}


def _num(d: dict, *keys: str) -> float | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def _scaled(v: float | None, factor: float, nd: int) -> float | None:
    return None if v is None else round(v * factor, nd)


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


_STATE_MAP: dict[str, Callable[[dict], dict]] = {
    "GetBias": lambda d: _drop_none({"bias_v": _scaled(_num(d, "bias_v", "bias"), 1.0, 6)}),
    "GetSetpoint": lambda d: _drop_none({"setpoint_pa": _scaled(_num(d, "setpoint_a", "setpoint"), 1e12, 3)}),
    "GetCurrent": lambda d: _drop_none({"current_pa": _scaled(_num(d, "current_a", "current"), 1e12, 3)}),
    "GetZPosition": lambda d: _drop_none({"z_nm": _scaled(_num(d, "z_pos_m", "z_m", "z"), 1e9, 4)}),
    "GetScanXYPosition": lambda d: _drop_none({"tip_x_nm": _scaled(_num(d, "x_m", "x"), 1e9, 4),
                                               "tip_y_nm": _scaled(_num(d, "y_m", "y"), 1e9, 4)}),
    "GetDriftCompensation": lambda d: {"drift_compensation": _drop_none({
        "on": d.get("enabled") if isinstance(d.get("enabled"), bool) else None,
        "vx_nm_s": _scaled(_num(d, "vx"), 1e9, 5), "vy_nm_s": _scaled(_num(d, "vy"), 1e9, 5),
        "vz_nm_s": _scaled(_num(d, "vz"), 1e9, 5),
        "saturated": (any(bool(d.get(k)) for k in ("x_saturated", "y_saturated", "z_saturated"))
                      if any(k in d for k in ("x_saturated", "y_saturated", "z_saturated")) else None)})},
}


def _normalise_reading(key: str, value: Any) -> tuple[str | None, Any]:
    """Controller readback → model units. Unknown or non-scalar fields are dropped."""
    if isinstance(value, bool):
        return key, value
    if not isinstance(value, (int, float)):
        return None, None
    v = float(value)
    if key.endswith("_a"):
        return key[:-2] + "_pa", round(v * 1e12, 3)
    if key.endswith("_m"):
        return key[:-2] + "_nm", round(v * 1e9, 4)
    if key.endswith("_m_s"):
        return key[:-4] + "_nm_s", round(v * 1e9, 5)
    return key, (round(v, 6) if abs(v) < 1e6 else v)


def _public_event(e: dict) -> dict:
    """An event as the model may read it (no step-level MAST internals)."""
    keep = ("kind", "turn", "action", "params", "note", "ok", "error", "facts", "resolved", "new_frames",
            "state", "instrument_s", "budget", "report", "args")
    return {k: e[k] for k in keep if k in e}


def _clip_args(args: dict) -> dict:
    s = json.dumps(args, ensure_ascii=False, default=str)
    return args if len(s) <= 4000 else {"_clipped": s[:4000]}


def _safe_read(notes: Any, method: str) -> str:
    try:
        out = getattr(notes, method)()
    except Exception:  # noqa: BLE001
        return ""
    if out is None:
        return ""
    return out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, default=str)


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, ensure_ascii=False, default=str))
