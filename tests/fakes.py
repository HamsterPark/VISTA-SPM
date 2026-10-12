"""Test doubles for the VISTA-SPM runner (WP-B): a toy ``RunnerHost`` and a PNG helper.

``ToyHost`` implements :class:`vista_spm.protocol.RunnerHost` without any instrument:

* ``act(action, note?)`` — ``action`` in ``look | count``; every executed act increments a
  counter and returns one PNG with ``counter`` white squares, so a model can be asked to
  report what it sees. The next act is refused until :meth:`acknowledge_act` (VISTA's
  one-action-one-observation gate).
* ``report_result(claim_id, value)`` — recorded, never touches the "instrument".
* the four notes tools from :mod:`vista_spm.notes`.
* While a checkpoint is requested only the notes tools execute; ``write_working`` makes the
  checkpoint ``ready``. ``begin_segment`` resets it to ``none``.
* ``max_acts`` acts make the host terminal with stop reason ``budget``.

Everything is guarded by one re-entrant lock (the runner calls the host from several
threads). ``events`` records every interaction for assertions.
"""
from __future__ import annotations

import struct
import threading
import zlib
from pathlib import Path

from vista_spm.compact_hook import CHECKPOINT_INSTRUCTION
from vista_spm.notes import NOTES_TOOL_NAMES, NOTES_TOOLS, Notes
from vista_spm.protocol import ImageBlock, ToolDef, ToolReply, error_reply, json_reply

TASK_TEXT = (
    "You operate a toy instrument through tools. Call `act` with action \"count\" to take one "
    "observation; each observation is an image with some white squares on black. "
)


def png_squares(n: int, size: int = 96, cell: int = 16) -> bytes:
    """Greyscale PNG (``size`` x ``size``) with ``n`` white ``cell``-px squares on black."""
    per_row = max(1, size // (2 * cell))
    pixels = bytearray(size * size)
    for k in range(max(0, int(n))):
        r, c = divmod(k, per_row)
        y0, x0 = cell // 2 + r * 2 * cell, cell // 2 + c * 2 * cell
        for y in range(y0, min(size, y0 + cell)):
            for x in range(x0, min(size, x0 + cell)):
                pixels[y * size + x] = 255
    raw = b"".join(b"\x00" + bytes(pixels[y * size:(y + 1) * size]) for y in range(size))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


ACT_TOOL = ToolDef(
    name="act",
    title="Act on the toy instrument",
    description=("Execute exactly one action on the toy instrument and return the new observation "
                 "image. The result must be observed before another action can execute."),
    input_schema={"type": "object", "additionalProperties": False, "required": ["action"],
                  "properties": {"action": {"type": "string", "enum": ["look", "count"]},
                                 "note": {"type": "string", "maxLength": 500}}},
    read_only=False, destructive=False, idempotent=False)

REPORT_TOOL = ToolDef(
    name="report_result",
    title="Report a result",
    description="Record a measured result for the judge. Never changes the instrument.",
    input_schema={"type": "object", "additionalProperties": False, "required": ["claim_id", "value"],
                  "properties": {"claim_id": {"type": "string"}, "value": {"type": "number"}}},
    read_only=False, destructive=False, idempotent=False)


class ToyHost:
    """A toy :class:`~vista_spm.protocol.RunnerHost` (see the module docstring)."""

    def __init__(self, root: str | Path, *, max_acts: int | None = None, task_text: str = TASK_TEXT,
                 act_delay_s: float = 0.0, image_size: int = 96) -> None:
        self.root = Path(root)
        self.lock = threading.RLock()
        self.notes = Notes(self.root / "notes", provenance=self._provenance)
        self.max_acts = max_acts
        self.task_text = task_text
        self.act_delay_s = act_delay_s
        self.image_size = int(image_size)
        self.counter = 0
        self.acts = 0
        self.ack_pending = False
        self.checkpoint = "none"
        self.checkpoint_reasons: list[str] = []
        self.terminal = False
        self.stop_reason: str | None = None
        self.images_delivered = 0
        self.reports: list[dict] = []
        self.events: list[tuple] = []
        self.finished: tuple[str, str] | None = None
        self.segments: list[tuple] = []

    # ── ToolHandler ──────────────────────────────────────────────────────────
    def tool_defs(self) -> list[ToolDef]:
        return [ACT_TOOL, REPORT_TOOL, *NOTES_TOOLS.values()]

    def list_tools(self) -> list[dict]:
        return [tool.to_mcp() for tool in self.tool_defs()]

    def call(self, name: str, args: dict) -> ToolReply:
        with self.lock:
            self.events.append(("call", name, dict(args)))
            if self.terminal and name not in NOTES_TOOL_NAMES:
                return error_reply("The episode has ended; nothing was executed.")
            if self.checkpoint == "requested" and name not in NOTES_TOOL_NAMES:
                return error_reply(f"{CHECKPOINT_INSTRUCTION} Nothing else was executed.")
            if self.checkpoint == "ready" and name not in NOTES_TOOL_NAMES:
                return error_reply("The checkpoint is saved; end this response for handoff.")
            if name in NOTES_TOOL_NAMES:
                reply = self.notes.call(name, args)
                if name == "write_working" and not reply.is_error and self.checkpoint == "requested":
                    self.checkpoint = "ready"
                    self.events.append(("checkpoint_ready",))
                    reply = ToolReply(text=reply.text + "\nWORKING.md saved; checkpoint ready. "
                                      "End this response now; the episode continues in a fresh context.",
                                      effect=reply.effect, meta=reply.meta)
                return reply
            if name == "act":
                return self._act(args)
            if name == "report_result":
                self.reports.append(dict(args))
                return json_reply({"recorded": dict(args)}, effect="report")
            return error_reply(f"unknown tool {name!r}")

    def _act(self, args: dict) -> ToolReply:
        if self.ack_pending:
            self.events.append(("act_refused",))
            return error_reply("Observe the previous act result before another action; nothing was executed.")
        if args.get("action") not in ("look", "count"):
            return error_reply("action must be 'look' or 'count'")
        if self.act_delay_s:
            import time
            time.sleep(self.act_delay_s)
        self.counter += 1
        self.acts += 1
        self.ack_pending = True
        if self.max_acts is not None and self.acts >= self.max_acts:
            self.terminal = True
            self.stop_reason = "budget"
        image = self.current_image()
        self.images_delivered += 1
        self.events.append(("act", self.counter))
        return json_reply({"turn": self.acts, "action": args.get("action"), "frame": f"t{self.acts:04d}.s0",
                           "budget": {"acts_left": None if self.max_acts is None else self.max_acts - self.acts}},
                          images=[image], effect="environment")

    def current_image(self) -> ImageBlock:
        size = self.image_size
        return ImageBlock(png=png_squares(self.counter, size=size, cell=max(4, size // 6)),
                          label=f'<visual kind="current" frame="t{self.acts:04d}.s0" size="{size}x{size}"/>')

    # ── RunnerHost ───────────────────────────────────────────────────────────
    def initial_message(self) -> tuple[str, list[ImageBlock]]:
        with self.lock:
            self.events.append(("initial_message",))
            self.images_delivered += 1
            return self.task_text, [self.current_image()]

    def recovery_message(self) -> tuple[str, list[ImageBlock]]:
        with self.lock:
            self.events.append(("recovery_message",))
            snap = self.notes.snapshot()
            text = (f"{self.task_text}\n\nEnvironment record: acts so far {self.acts}; the current "
                    f"observation follows.\n\nGUIDE.md:\n{snap['guide']}\nWORKING.md:\n"
                    f"{snap['working'] or '(none)'}\nContinue the task.")
            self.images_delivered += 1
            return text, [self.current_image()]

    def acknowledge_act(self) -> None:
        with self.lock:
            self.events.append(("ack", self.ack_pending))
            self.ack_pending = False

    def request_checkpoint(self, reason: str) -> None:
        with self.lock:
            self.events.append(("request_checkpoint", reason))
            self.checkpoint_reasons.append(reason)
            if self.checkpoint == "none" and not self.terminal:
                self.checkpoint = "requested"

    def status(self) -> dict:
        with self.lock:
            return {"terminal": self.terminal, "stop_reason": self.stop_reason, "checkpoint": self.checkpoint,
                    "images_delivered": self.images_delivered, "acts": self.acts, "turn": self.acts,
                    "budget": {"max_acts": self.max_acts}}

    def begin_segment(self, index: int, session_id: str) -> None:
        with self.lock:
            self.events.append(("begin_segment", index, session_id))
            self.segments.append(("begin", index, session_id))
            self.checkpoint = "none"

    def end_segment(self, index: int, info: dict) -> None:
        with self.lock:
            self.events.append(("end_segment", index, info.get("boundary")))
            self.segments.append(("end", index, dict(info)))

    def finish(self, outcome: str, final_text: str) -> None:
        with self.lock:
            self.events.append(("finish", outcome))
            self.finished = (outcome, final_text)

    def _provenance(self) -> dict:
        return {"acts": self.acts, "counter": self.counter}

    # ── helpers for assertions ───────────────────────────────────────────────
    def names(self, kind: str) -> list[tuple]:
        with self.lock:
            return [e for e in self.events if e[0] == kind]
