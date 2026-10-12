"""QAHost: one STM-BabyVision item as a ``protocol.RunnerHost`` — look and measure, no act.

README
------
The Claude Code runner (WP-B, ``vista_spm.claude_runner.ClaudeCodeRunner``) drives this
host exactly as it drives an episode host: ``initial_message()`` → tool calls over RPC →
``finish``. Differences from ``host.VistaHost``:

* the item's frames are archived at construction (turn 0; frame k of the question is
  ``frame_ids[k]``) and shown in the first message, each as its default observation image;
* tools are the arm's visual-memory and measurement tools (``viewtools.VIEW_TOOLS`` /
  ``measure.MEASURE_TOOLS``, bound to this item's archive) plus ``answer``; there is **no
  act**, no notes and no history (one question, one short session);
* ``answer(value | abstain, confidence?, note?)`` validates the value against the item's
  ``answer_format`` (an invalid value is refused with the reason, nothing recorded); a valid
  call records the answer and makes the host terminal — calling it again replaces it;
* a tool-call budget (default 60): past it only ``answer`` executes;
* ``request_checkpoint`` needs nothing saved: the checkpoint is ready at once and
  ``recovery_message()`` repeats the question, the data and any answer already recorded.

``result()`` is the line the answers file gets: ``{"id", "answer": {...} | null, "outcome",
"tool_calls", "tools_used", "images_delivered", "segments", "wall_s", "log"}``.

Never shows the model anything from ``truth`` or ``provenance``: the question text and the
frame files are the whole input.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..protocol import (DISPLAY_MAX_DEFAULT, ImageBlock, ToolDef, ToolReply, compact_json, error_reply,
                        json_reply)
from . import items as I

ANSWER_TOOL = "answer"
DEFAULT_MAX_TOOL_CALLS = 60

_VALUE_HELP = {
    "choice": "one of: {choices}",
    "number": "a number in {unit}",
    "vector": "[a, b] in {unit}",
    "position_list": "a list [[x, y], ...] of scan-frame coordinates in nm (empty list if none)",
    "count": "a non-negative integer",
    "bool": "true or false",
}

SYSTEM_HINT = (
    "You are answering one question about scanning tunnelling microscope (STM) data from a simulated "
    "instrument. There is no instrument to operate: the data below were recorded earlier and are archived "
    "losslessly. Look at them, re-inspect and measure them with the tools where that helps, then call "
    "`answer` and end your reply with [DONE].")
SYSTEM_HINT_IMAGES_ONLY = (
    "You are answering one question about scanning tunnelling microscope (STM) data from a simulated "
    "instrument. There is no instrument to operate: the data below were recorded earlier. You have the "
    "images below and no tool except `answer`; work from what you see, then call `answer` and end your "
    "reply with [DONE].")

#: system prompt for a QA session (the episode prompt talks about acting on an instrument)
QA_SYSTEM_PROMPT = (
    "You answer questions about scanning tunnelling microscope data. Each task gives you archived "
    "frames (topographs, spectra) and one question. Use the tools to look at the data again and to "
    "measure it; they never change anything. Tools measure, you interpret. Call `answer` once with "
    "your answer in the requested format (or abstain=true if you cannot decide), then end your "
    "reply with [DONE].")
QA_SYSTEM_PROMPT_IMAGES_ONLY = (
    "You answer questions about scanning tunnelling microscope data. Each task gives you images of "
    "archived frames (topographs, spectra) and one question. Your only tool is `answer`: judge and "
    "estimate from the images and the numbers given with them. Call `answer` once with your answer in "
    "the requested format (or abstain=true if you cannot decide), then end your reply with [DONE].")

#: how display pixels map to scan-frame nm (protocol.DISPLAY_CONVENTION, spelled out for the model)
COORD_HELP = (
    "Coordinates: scan-frame nm as in each frame's header (centre = scan offset, x right, y up). A frame "
    "with field w x h nm is shown as an image of W x H display pixels (display_size_px; the integer scale is "
    "in each image's label). A display pixel position (x, y), measured from the image's top-left corner with "
    "y down, is at u = -w/2 + x*w/W, v = h/2 - y*h/H relative to the centre; rotated by the scan angle a and "
    "shifted by the centre (cx, cy): X = cx + cos(a)*u - sin(a)*v, Y = cy + sin(a)*u + cos(a)*v.")


def runner_settings(*, images_only: bool = False, **overrides: Any) -> dict:
    """``ClaudeCodeRunner`` ``RunnerConfig`` fields for a QA session (pass them through, with
    ``io_dir`` / ``model`` / ``effort`` as usual): no act to acknowledge, ``answer`` is the only
    progress, native compaction (there are no notes to checkpoint into), a QA system prompt
    (the images-only variant names no measuring tools)."""
    s = {"act_tools": (), "progress_tools": (ANSWER_TOOL,), "compaction": "native",
         "system_prompt": QA_SYSTEM_PROMPT_IMAGES_ONLY if images_only else QA_SYSTEM_PROMPT,
         "stall_turns": 2, "max_segments": 3,
         # an item is cheap to retry later; a usage limit lasts hours, so give up on API errors early
         "max_recoveries": 1, "api_retry_backoff_s": (10.0, 30.0),
         "continue_text": f"Continue: answer the question with the `{ANSWER_TOOL}` tool, then end your reply with [DONE]."}
    s.update(overrides)
    return s


def _arm(arm: Any):
    if arm is None or isinstance(arm, str):
        from ..arms import get_arm

        return get_arm(arm or "full")
    return arm


class QAHost:
    def __init__(self, item: dict, dataset_dir: str | Path, work_dir: str | Path, *, arm: Any = "full",
                 display_max: int | None = None, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
                 archive: Any = None, view_tools: dict | None = None, measure_tools: dict | None = None):
        errs = I.validate_item(item)
        if errs:
            raise ValueError(f"invalid item {item.get('id')}: {errs}")
        self.item = item
        self.dataset_dir = Path(dataset_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.arm = _arm(arm)
        if self.arm.observation not in ("image", "text"):
            raise ValueError(f"arm {self.arm.name!r}: the QA host serves image or text observations only")
        self.display_max = int(display_max or getattr(self.arm, "display_max", DISPLAY_MAX_DEFAULT))
        self.max_tool_calls = int(max_tool_calls)
        if archive is None:
            from ..archive import Archive

            root = self.work_dir / "archive"
            if root.is_dir() and any(root.iterdir()):
                # an Archive re-opens what is there (crash recovery) and would number this
                # item's frames after the old ones: frame k would no longer be frame_ids[k]
                raise FileExistsError(f"{root} already holds an archive; give each QA session a fresh work_dir")
            archive = Archive(root, display_max=self.display_max)
        self.archive = archive
        self.frame_ids: list[str] = []
        for rel in item["frames"]:
            path = self.dataset_dir / rel
            if not path.is_file():
                raise FileNotFoundError(f"item {item['id']}: frame file missing: {path}")
            e = archive.add_spectrum(0, path) if path.suffix.lower() == ".dat" else archive.add_scan(0, path)
            self.frame_ids.append(e.fid)
        self._lock = threading.RLock()
        self._tools = self._build_tools(view_tools, measure_tools)
        self.answer: dict | None = None
        self.answer_calls = 0
        self.tool_calls = 0
        self.tools_used: dict[str, int] = {}
        self.log: list[dict] = []
        self.terminal = False
        self.stop_reason: str | None = None
        self.outcome: str | None = None
        self.final_text = ""
        self.checkpoint = "none"
        self.images_delivered = 0
        self.images_total = 0
        self.segment = -1
        self.segments: list[dict] = []
        self._t0 = time.time()

    # ── tools ───────────────────────────────────────────────────────────────
    def _build_tools(self, view_tools: dict | None, measure_tools: dict | None) -> dict[str, tuple[ToolDef, Callable]]:
        t: dict[str, tuple[ToolDef, Callable]] = {}
        if self.arm.inspect or self.arm.read_values:
            if view_tools is None:
                from .. import viewtools as V

                view_tools = V.VIEW_TOOLS if self.display_max == DISPLAY_MAX_DEFAULT else V.view_tools(self.display_max)
            for name, (tdef, fn) in view_tools.items():
                if (name == "inspect" and self.arm.inspect) or (name == "read_values" and self.arm.read_values):
                    t[name] = (tdef, fn)
        if self.arm.measure:
            if measure_tools is None:
                from .. import measure as M

                measure_tools = M.MEASURE_TOOLS if self.display_max == DISPLAY_MAX_DEFAULT else M.measure_tools(self.display_max)
            for name, (tdef, fn) in measure_tools.items():
                t[name] = (tdef, fn)
        t[ANSWER_TOOL] = (self.answer_tooldef(), None)
        return t

    def answer_tooldef(self) -> ToolDef:
        af = self.item["answer_format"]
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"value": I.answer_value_schema(self.item),
                                 "abstain": {"type": "boolean", "description": "true = decline to answer "
                                             "(scored separately from a wrong answer); omit value then."},
                                 "confidence": {"type": "number", "minimum": 0, "maximum": 1,
                                                "description": "Your probability that the value is right."},
                                 "note": {"type": "string", "maxLength": 1000,
                                          "description": "One or two sentences on how you decided."}}}
        how = _VALUE_HELP[af["kind"]].format(choices=", ".join(af.get("choices") or []), unit=af.get("unit", ""))
        return ToolDef(ANSWER_TOOL, (
            f"Record your answer to the question: value = {how}. A valid call ends the task (call it again only "
            f"to replace the answer). Use abstain=true instead of a value if you cannot decide. Does not touch "
            f"the data."), schema, title="Answer", read_only=False)

    def list_tools(self) -> list[dict]:
        return [tdef.to_mcp() for tdef, _ in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, args: dict) -> ToolReply:
        with self._lock:
            t0 = time.perf_counter()
            self.tool_calls += 1
            self.tools_used[name] = self.tools_used.get(name, 0) + 1
            args = dict(args or {}) if isinstance(args, dict) else {}
            entry = self._tools.get(name)
            if entry is None:
                reply = error_reply(f"unknown tool {name!r}", available=list(self._tools))
            elif name == ANSWER_TOOL:
                reply = self._answer(args)
            elif self.tool_calls > self.max_tool_calls:
                reply = error_reply(f"tool budget of {self.max_tool_calls} calls used up; call answer now")
            else:
                try:
                    reply = entry[1](self.archive, args)
                except Exception as exc:  # noqa: BLE001 — the host never raises into the RPC layer
                    reply = error_reply(f"internal error in {name}: {type(exc).__name__}: {exc}")
            self.images_delivered += len(reply.images)
            self.images_total += len(reply.images)
            self.log.append({"n": self.tool_calls, "tool": name, "args": _clip(args), "is_error": reply.is_error,
                             "n_images": len(reply.images), "wall_s": round(time.perf_counter() - t0, 3),
                             "t": round(time.time() - self._t0, 3)})
            return reply

    def _answer(self, args: dict) -> ToolReply:
        unknown = sorted(set(args) - {"value", "abstain", "confidence", "note"})
        if unknown:
            return error_reply(f"unknown argument(s) {unknown}; use value, abstain, confidence, note")
        abstain = args.get("abstain") is True
        if not abstain and "value" not in args:
            return error_reply("give a value (or abstain=true)")
        value, abstained, err = I.normalize_answer(self.item, {"value": args.get("value"), "abstain": abstain})
        if err is not None:
            return error_reply(f"answer not recorded: {err}",
                               expected=_VALUE_HELP[self.item["answer_format"]["kind"]].format(
                                   choices=", ".join(self.item["answer_format"].get("choices") or []),
                                   unit=self.item["answer_format"].get("unit", "")))
        conf = args.get("confidence")
        if conf is not None:
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                return error_reply("confidence must be a number in [0, 1]")
            if not 0.0 <= conf <= 1.0:
                return error_reply("confidence must be a number in [0, 1]")
        self.answer_calls += 1
        self.answer = {"abstain": True} if abstained else {"value": value, "abstain": False}
        if conf is not None:
            self.answer["confidence"] = conf
        if args.get("note"):
            self.answer["note"] = str(args["note"])[:1000]
        self.terminal = True
        # "done" is one of the runner's outcomes: a host that turns terminal mid-turn is reported
        # by its stop_reason, and anything else would read as "budget"
        self.stop_reason = self.stop_reason or "done"
        return json_reply({"recorded": self.answer, "replaced_previous": self.answer_calls > 1,
                           "next": "End your reply with [DONE]."}, effect="report")

    # ── RunnerHost ──────────────────────────────────────────────────────────
    def _data_block(self) -> str:
        lines = []
        for k, fid in enumerate(self.frame_ids):
            e = self.archive.get(fid)
            lines.append(f"- {I.frame_label(self.item, k)} = {fid}: " + compact_json(e.summary()))
        return "\n".join(lines)

    @property
    def images_only(self) -> bool:
        """True when ``answer`` is the only tool (the images-only arm)."""
        return list(self._tools) == [ANSWER_TOOL]

    def _answer_help(self) -> str:
        af = self.item["answer_format"]
        how = _VALUE_HELP[af["kind"]].format(choices=", ".join(af.get("choices") or []), unit=af.get("unit", ""))
        budget = "" if self.images_only else f" You may use at most {self.max_tool_calls} tool calls."
        return (f"How to answer: call `{ANSWER_TOOL}` with value = {how}. If you cannot decide, call it with "
                f"abstain=true instead (scored separately from a wrong answer).{budget}")

    def _message(self, preface: str) -> tuple[str, list[ImageBlock]]:
        data_head = ("Data (the question's frame/spectrum numbers and their ids):" if self.images_only else
                     "Data (the question's frame/spectrum numbers; the tools address them by id):")
        coord = COORD_HELP + ("" if self.images_only else
                              " Tools that take display pixels use that image's pixels and also report scan-frame nm.")
        text = [preface, "", "Question:", self.item["question"], "",
                data_head, self._data_block(), "", coord, "", self._answer_help()]
        images: list[ImageBlock] = []
        if self.arm.observation == "image":
            for fid in self.frame_ids:
                images += self.archive.observation(fid)
        else:
            for fid in self.frame_ids:
                text += ["", self.archive.text_grid(fid)]
        self.images_delivered += len(images)
        self.images_total += len(images)
        return "\n".join(text), images

    def _hint(self) -> str:
        return SYSTEM_HINT_IMAGES_ONLY if self.images_only else SYSTEM_HINT

    def initial_message(self) -> tuple[str, list[ImageBlock]]:
        return self._message(self._hint())

    def recovery_message(self) -> tuple[str, list[ImageBlock]]:
        pre = self._hint() + " (This is a fresh session continuing the same question.)"
        if self.answer is not None:
            pre += " Answer already recorded: " + compact_json(self.answer) + "."
        pre += f" Tool calls used so far: {self.tool_calls}."
        return self._message(pre)

    def acknowledge_act(self) -> None:
        """No act tool here: nothing to gate."""

    def request_checkpoint(self, reason: str) -> None:
        with self._lock:
            self.checkpoint = "ready"          # nothing to save; recovery repeats the question
            self.log.append({"event": "checkpoint", "reason": str(reason), "t": round(time.time() - self._t0, 3)})

    def status(self) -> dict:
        return {"terminal": self.terminal, "stop_reason": self.stop_reason, "checkpoint": self.checkpoint,
                "images_delivered": self.images_delivered, "acts": 0, "turn": 0, "tool_calls": self.tool_calls,
                "budget": {"max_tool_calls": self.max_tool_calls,
                           "tool_calls_left": max(0, self.max_tool_calls - self.tool_calls)},
                "answered": self.answer is not None, "segment": self.segment, "outcome": self.outcome}

    def begin_segment(self, index: int, session_id: str) -> None:
        with self._lock:
            self.segment = int(index)
            self.images_delivered = 0
            if self.checkpoint == "ready":
                self.checkpoint = "none"
            self.segments.append({"index": int(index), "session_id": str(session_id),
                                  "t": round(time.time() - self._t0, 3)})

    def end_segment(self, index: int, info: dict) -> None:
        with self._lock:
            for s in self.segments:
                if s["index"] == int(index):
                    s["info"] = json.loads(json.dumps(info, default=str))

    def finish(self, outcome: str, final_text: str) -> None:
        with self._lock:
            self.outcome = str(outcome)
            self.final_text = str(final_text or "")
            self.terminal = True
            self.stop_reason = self.stop_reason or str(outcome)

    # ── result ──────────────────────────────────────────────────────────────
    def result(self) -> dict:
        return {"id": self.item["id"], "answer": self.answer, "outcome": self.outcome or (
                    "answered" if self.answer is not None else None),
                "tool_calls": self.tool_calls, "tools_used": dict(self.tools_used),
                "images_delivered": self.images_total, "segments": self.segments,
                "wall_s": round(time.time() - self._t0, 2), "final_text": self.final_text[-2000:],
                "frame_ids": list(self.frame_ids), "arm": self.arm.name, "log": self.log}


def run_items(dataset: str | Path, out_path: str | Path, drive: Callable[["QAHost"], Any], *,
              work_root: str | Path, arm: Any = "full", types: list[str] | None = None,
              ids: list[str] | None = None, max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
              log: Callable[[str], None] | None = None) -> list[dict]:
    """Answer items one QA session at a time. ``drive(host)`` runs the model against the host
    (e.g. a ``ClaudeCodeRunner`` built around it) and returns when the session is over. Each
    ``host.result()`` is appended to ``out_path`` as soon as it exists (a crash keeps what was
    answered); the file is a valid answers file for ``score``. Items already in ``out_path`` are
    skipped, so an interrupted run resumes."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set(I.load_answers(out)) if out.exists() else set()
    rows: list[dict] = []
    for it in I.load_items(dataset):
        if (types and it["type"] not in types) or (ids and it["id"] not in ids) or it["id"] in done:
            continue
        base = Path(work_root) / it["id"]
        n_prev = sum(1 for p in base.iterdir() if p.is_dir()) if base.is_dir() else 0
        host = QAHost(it, dataset, base / f"session{n_prev}", arm=arm, max_tool_calls=max_tool_calls)
        try:
            drive(host)
        except Exception as exc:  # noqa: BLE001 — one failed session must not end the run
            host.finish("error", f"{type(exc).__name__}: {exc}")
        res = host.result()
        with open(out, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
        rows.append(res)
        if log is not None:
            log(f"{it['id']}: {res['answer']} ({res['tool_calls']} tool calls, {res['wall_s']} s)")
    return rows


def _clip(args: dict, n: int = 400) -> dict:
    s = json.dumps(args, default=str, ensure_ascii=False)
    return args if len(s) <= n else {"_clipped": s[:n]}
