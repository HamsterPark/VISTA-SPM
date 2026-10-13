"""Shared types of the VISTA-SPM harness. Every module codes against this file.

Nothing here imports MAST, the simulator, numpy or PIL, so the MCP bridge (which runs as a
separate, standard-library-only process launched by Claude Code) can import it too.

Contents
--------
* :class:`ImageBlock` / :class:`ToolReply` — what a tool returns, and its MCP rendering.
* :class:`ToolDef` — a tool's name, description and JSON schema, and its MCP rendering.
* :class:`ToolHandler` / :class:`RunnerHost` — the two interfaces between the pieces:
  the RPC server calls a ``ToolHandler``; the Claude Code runner drives a ``RunnerHost``.
* Frame ids (:func:`frame_id`, :func:`parse_frame_id`) and the display-coordinate
  convention (see the docstring of :data:`DISPLAY_CONVENTION`).
"""
from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: MCP server name; Claude Code exposes our tools as ``mcp__<SERVER_NAME>__<tool>``.
SERVER_NAME = "stm"
#: MCP protocol revision the bridge announces (the one VISTA's bridge uses).
MCP_PROTOCOL = "2025-06-18"

#: Longest side, in pixels, of a rendered observation unless an arm overrides it.
DISPLAY_MAX_DEFAULT = 768

#: Hard caps shared by the tool schemas and their implementations (VISTA's numbers where
#: they exist).
MAX_INSPECT_VIEWS = 16
MAX_READ_VIEWS = 64
MAX_READ_SAMPLES = 4096
MAX_QUESTION_CHARS = 1024
MAX_LABEL_CHARS = 128
MAX_GUIDE_CHARS = 64 * 1024
MAX_WORKING_CHARS = 16 * 1024

DISPLAY_CONVENTION = """
Display coordinates (what the model sees and points at)
--------------------------------------------------------
A frame with ``nx`` x ``ny`` pixels is shown as an image of ``nx*k`` x ``ny*k`` pixels,
``k = max(1, DISPLAY_MAX // max(nx, ny))`` (integer nearest-neighbour upscaling, never
smoothing). A frame larger than ``DISPLAY_MAX`` is shown at ``k = 1/ceil(max(nx,ny)/DISPLAY_MAX)``
(integer block-mean downsampling); ``k`` is always reported next to the image.

Display coordinates ``(x, y)`` are continuous, origin at the top-left CORNER of the image,
``x`` to the right, ``y`` down. Row 0 of the image is the frame's top edge (MAST's
``sxm_oriented_frames`` convention: ``scan_dir=up`` already flipped, backward already
un-mirrored). The native continuous coordinate is ``(x/k, y/k)``; native pixel ``i`` covers
``[i, i+1)`` and its centre is ``i + 0.5``.

Scan-frame nm (what the instrument, the actions and ``report_result`` use):
``u = -w/2 + (x/k) * w/nx``, ``v = +h/2 - (y/k) * h/ny`` (w, h in nm), then rotate by the
frame's scan angle about its centre and add the centre::

    X = cx + cos(a) u - sin(a) v ,   Y = cy + sin(a) u + cos(a) v

This is ``stmbench.human.render.pixel_to_scan_nm(geom, x/k - 0.5, y/k - 0.5)``, the
function the mode-H page uses and that a test pins to the judge's footprint rule.
"""

#: Frame-id kinds: ``s`` saved scan (.sxm), ``p`` partial snapshot of a scan in progress,
#: ``d`` saved spectrum (.dat), ``m`` derived image a measurement tool produced.
FRAME_KINDS = ("s", "p", "d", "m")
_FRAME_RE = re.compile(r"^t(\d{4,})\.([spdm])(\d+)$")


def frame_id(turn: int, kind: str, index: int) -> str:
    """``frame_id(7, "s", 0) == "t0007.s0"``. ``turn`` is the action turn that produced it
    (turn 0 = the observation the episode started from)."""
    if kind not in FRAME_KINDS:
        raise ValueError(f"frame kind {kind!r} not in {FRAME_KINDS}")
    if turn < 0 or index < 0:
        raise ValueError("turn and index must be >= 0")
    return f"t{int(turn):04d}.{kind}{int(index)}"


def parse_frame_id(fid: str) -> tuple[int, str, int]:
    m = _FRAME_RE.match(str(fid or "").strip())
    if not m:
        raise ValueError(f"not a frame id: {fid!r} (expected like 't0007.s0')")
    return int(m.group(1)), m.group(2), int(m.group(3))


def compact_json(obj: Any) -> str:
    """The text block of a tool reply: compact, UTF-8, numbers as numbers. NaN and +-Infinity are
    not JSON, so they are written as ``null`` (a reply carrying them would not parse)."""
    return json.dumps(_finite(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False,
                      default=_json_default)


def _finite(o: Any) -> Any:
    """``o`` with every non-finite float replaced by ``None``. numpy scalars and arrays become the
    Python values :func:`_json_default` would give them, so finite values serialize as before."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    converted = _json_default(o)                 # str(o) stays a string; numpy values are checked
    return converted if isinstance(converted, str) else _finite(converted)


def _json_default(o: Any) -> Any:
    item = getattr(o, "item", None)              # numpy scalars
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(o, "tolist", None)          # numpy arrays
    if callable(tolist):
        return tolist()
    return str(o)


@dataclass(frozen=True)
class ImageBlock:
    """One image of a tool reply. ``label`` is the text block placed right before it, e.g.
    ``<visual kind="current" frame="t0007.s0" channel="Z" direction="forward" scale="2" size="768x768"/>``.
    Images carry no text inside them; everything the model needs to read is in the label
    or in the reply text (VISTA's rule)."""

    png: bytes
    label: str

    def b64(self) -> str:
        return base64.b64encode(self.png).decode("ascii")


@dataclass
class ToolReply:
    """What every tool returns.

    ``text`` is the first content block (compact JSON from :func:`compact_json`, or a short
    sentence for errors). ``images`` follow, each as a label text block plus an image block.
    ``effect``: ``"none"`` (read-only), ``"notes"`` (GUIDE/WORKING changed), ``"report"``
    (a result/diagnosis was recorded), ``"environment"`` (the instrument acted — only
    ``act`` and the episode-control tools). ``meta`` is for the host's own ledger and is
    never sent to the model.
    """

    text: str
    images: list[ImageBlock] = field(default_factory=list)
    is_error: bool = False
    effect: str = "none"
    meta: dict = field(default_factory=dict)

    def to_mcp(self) -> dict:
        content: list[dict] = [{"type": "text", "text": self.text}]
        for im in self.images:
            content.append({"type": "text", "text": im.label})
            content.append({"type": "image", "data": im.b64(), "mimeType": "image/png"})
        return {"content": content, "isError": bool(self.is_error)}


def json_reply(obj: Any, *, images: list[ImageBlock] | None = None, effect: str = "none",
               meta: dict | None = None) -> ToolReply:
    return ToolReply(text=compact_json(obj), images=list(images or []), effect=effect, meta=dict(meta or {}))


def error_reply(message: str, **details: Any) -> ToolReply:
    """A refused or failed call. The message says what to do differently; nothing executed
    unless the message says otherwise."""
    body = {"error": str(message)}
    if details:
        body.update(details)
    return ToolReply(text=compact_json(body), is_error=True)


@dataclass(frozen=True)
class ToolDef:
    """A tool as the model sees it. ``input_schema`` is JSON Schema (draft 2020-12) with
    ``additionalProperties: false`` at the top level."""

    name: str
    description: str
    input_schema: dict
    title: str = ""
    read_only: bool = True
    destructive: bool = False
    idempotent: bool = False

    def to_mcp(self) -> dict:
        return {"name": self.name, "title": self.title or self.name, "description": self.description,
                "inputSchema": self.input_schema,
                "annotations": {"readOnlyHint": self.read_only, "destructiveHint": self.destructive,
                                "idempotentHint": self.idempotent, "openWorldHint": False}}


@runtime_checkable
class ToolHandler(Protocol):
    """What the RPC server (and through it the MCP bridge) talks to."""

    def list_tools(self) -> list[dict]:
        """MCP tool dicts (:meth:`ToolDef.to_mcp`), in the order the model should see them."""
        ...

    def call(self, name: str, args: dict) -> ToolReply:
        """Execute one tool call. Must never raise: failures come back as ``is_error`` replies."""
        ...


@runtime_checkable
class RunnerHost(ToolHandler, Protocol):
    """What ``claude_runner`` needs from the episode host (``host.VistaHost``; tests use a fake).

    Threads: the RPC thread calls ``list_tools``, ``call``, ``status`` and ``request_checkpoint``
    (image-limit observer, hook callback); the runner's stdout reader calls ``acknowledge_act``;
    the runner thread calls the rest. Implementations must be thread-safe.

    Contracts the runner relies on: ``begin_segment`` resets the checkpoint to ``"none"`` (every
    segment is a fresh context); ``recovery_message`` is self-contained, task text included (it
    can follow a failed initial segment); ``acknowledge_act`` is also called after every recovery
    message and for act results that are errors; ``status()["acts"]`` feeds the runner's
    checkpoint-loop guard; ``status()["terminal"]`` ends the episode at once, whereas
    ``status()["instrument_closed"]`` (budget spent) only refuses further actions, so results
    can still be reported before the reply ends with [DONE].

    Life cycle, driven by the runner::

        initial_message() -> segment 0 starts (begin_segment) -> tool calls arrive over RPC ...
        -> the runner sees an ``act`` tool_result reach the model -> acknowledge_act()
        -> a turn ends without [DONE]/[ABORT] -> continuation text -> ...
        -> compaction checkpoint (request_checkpoint ... status()["checkpoint"] == "ready")
        -> end_segment -> recovery_message() -> next segment -> ... -> finish(outcome, text)
    """

    def initial_message(self) -> tuple[str, list[ImageBlock]]:
        """Task text + the starting observation, for the first user turn of segment 0."""
        ...

    def recovery_message(self) -> tuple[str, list[ImageBlock]]:
        """First user turn of a fresh session after a checkpoint or a crash: environment
        record (current observation metadata, last action result, recent history), the
        agent's notes (GUIDE.md, WORKING.md), and the current frame image."""
        ...

    def acknowledge_act(self) -> None:
        """The result of the last ``act`` is now in the model's context: the next ``act``
        may execute (VISTA's one-action-one-observation gate)."""
        ...

    def request_checkpoint(self, reason: str) -> None:
        """Ask the model to save WORKING.md before the context is replaced. While a
        checkpoint is requested only the notes tools execute."""
        ...

    def status(self) -> dict:
        """``{"terminal": bool, "instrument_closed": bool, "stop_reason": str | None,
        "checkpoint": "none"|"requested"|"ready", "images_delivered": int, "acts": int, "turn": int,
        "budget": {...}}``."""
        ...

    def begin_segment(self, index: int, session_id: str) -> None:
        ...

    def end_segment(self, index: int, info: dict) -> None:
        """``info``: exit code, context boundary reason, usage of the segment."""
        ...

    def finish(self, outcome: str, final_text: str) -> None:
        """``outcome`` in {"done", "abort", "budget", "error", "stalled", "max_segments"}."""
        ...
