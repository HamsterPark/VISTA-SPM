"""STM-BabyVision item schema: one JSON object per line of ``items.jsonl``.

README
------
An *item* is one question about one or more frames the simulator produced, with the
simulator's truth and the tolerance it is scored against::

    {"id": "move.detect.3fa9c1d2",            # neutral: never encodes the answer
     "type": "move", "subtype": "detect",
     "question": "...",                       # English, self-contained, says unit and format
     "frames": ["frames/9d0c2e1f7a3b.sxm", "frames/51aa0c3e9d10.sxm"],   # relative to the dataset dir
     "answer_format": {"kind": "bool"},       # choice | number | vector | position_list | count | bool
     "truth": {"value": true, ...},
     "tolerance": {...},
     "provenance": {"scenario": ..., "seed": ..., "tip_condition": ..., "hidden_draws": {...},
                    "sim_s": [...], "frames": [{geometry...}], ...},
     "difficulty": "normal"}

``frames`` are referred to in the question as *frame 1, frame 2, ...* in list order (a
``.dat`` spectrum is *spectrum k*). Coordinates are **scan-frame nm**: the instrument's own
frame recorded in each file's header (``SCAN_OFFSET`` is the frame centre, x to the right,
y up, rotated by ``SCAN_ANGLE``) - the frame ``report_result`` and the measurement tools use.

Answer shapes (the ``value`` of an answer, see :func:`normalize_answer`):

==============  ============================================================
kind            value
==============  ============================================================
choice          one of ``answer_format.choices`` (case-insensitive)
number          a finite number in ``answer_format.unit``
vector          ``[a, b]`` (two finite numbers) in ``answer_format.unit``
position_list   ``[[x, y], ...]`` scan-frame nm (may be empty)
count           a non-negative integer
bool            ``true`` / ``false``
==============  ============================================================

An answer may instead abstain (``{"abstain": true}``); abstentions are scored separately.
Nothing here imports numpy, MAST or the simulator.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA_VERSION = "staticbench/1"

#: item types in the order the report lists them
ITEM_TYPES = ("tip", "adatoms", "move", "drift", "herringbone", "step", "tipchange", "atomic")
SUBTYPES: dict[str, tuple[str, ...]] = {
    "tip": ("condition",),
    "adatoms": ("count", "positions"),
    "move": ("detect", "which", "distance"),
    "drift": ("vector",),
    "herringbone": ("period", "orientation"),
    "step": ("height",),
    "tipchange": ("row",),
    "atomic": ("resolved",),
}
ANSWER_KINDS = ("choice", "number", "vector", "position_list", "count", "bool")
TIP_CLASSES = ("good", "blunt", "double", "unstable", "dirty")
DIFFICULTIES = ("easy", "normal", "hard")
#: more points than this in a position_list answer is refused (a guard, not a physics limit)
MAX_POINTS = 400

_REQUIRED = ("id", "type", "subtype", "question", "frames", "answer_format", "truth",
             "tolerance", "provenance")


@dataclass
class Item:
    id: str
    type: str
    subtype: str
    question: str
    frames: list[str]
    answer_format: dict
    truth: dict
    tolerance: dict
    provenance: dict = field(default_factory=dict)
    difficulty: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("difficulty") is None:
            d.pop("difficulty")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(**{k: d[k] for k in _REQUIRED}, difficulty=d.get("difficulty"))


# -- ids and names ------------------------------------------------------------
def short_hash(*parts: Any, n: int = 8) -> str:
    """Deterministic neutral token: the first ``n`` hex digits of sha1 over the parts."""
    h = hashlib.sha1("/".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return h[:n]


def item_id(type_: str, subtype: str, *parts: Any) -> str:
    """``<type>.<subtype>.<hash>`` - the hash never depends on the truth."""
    return f"{type_}.{subtype}.{short_hash(type_, subtype, *parts)}"


def frame_label(item: dict, k: int) -> str:
    """How the question names ``frames[k]``: ``frame 1`` / ``spectrum 2`` (1-based, list order)."""
    p = str(item["frames"][k]).lower()
    return f"{'spectrum' if p.endswith('.dat') else 'frame'} {k + 1}"


# -- validation ---------------------------------------------------------------
def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_pair(p: Any) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_finite(v) for v in p)


def validate_item(d: dict, *, root: str | Path | None = None) -> list[str]:
    """Every problem with one item dict (empty list = valid). With ``root`` the frame files
    must exist under it."""
    errs: list[str] = []
    if not isinstance(d, dict):
        return ["item is not an object"]
    for k in _REQUIRED:
        if k not in d:
            errs.append(f"missing {k!r}")
    if errs:
        return errs
    iid = str(d["id"])
    if not re.fullmatch(r"[a-z]+\.[a-z_]+\.[0-9a-f]{6,}", iid):
        errs.append(f"id {iid!r} is not '<type>.<subtype>.<hex>'")
    t, st = d["type"], d["subtype"]
    if t not in ITEM_TYPES:
        errs.append(f"type {t!r} not in {ITEM_TYPES}")
    elif st not in SUBTYPES[t]:
        errs.append(f"subtype {st!r} not in {SUBTYPES[t]} for type {t!r}")
    q = d["question"]
    if not isinstance(q, str) or len(q.strip()) < 20:
        errs.append("question missing or too short")
    frames = d["frames"]
    if not isinstance(frames, list) or not frames:
        errs.append("frames must be a non-empty list")
    else:
        for f in frames:
            s = str(f)
            pp = PurePosixPath(s)
            if pp.is_absolute() or ":" in s or ".." in pp.parts or "\\" in s:
                errs.append(f"frame path {f!r} must be relative POSIX without '..'")
            elif not s.lower().endswith((".sxm", ".dat")):
                errs.append(f"frame {f!r} is neither .sxm nor .dat")
            elif root is not None and not (Path(root) / s).is_file():
                errs.append(f"frame file missing: {f}")
    if d.get("difficulty") is not None and d["difficulty"] not in DIFFICULTIES:
        errs.append(f"difficulty {d['difficulty']!r} not in {DIFFICULTIES}")
    af, truth, tol = d["answer_format"], d["truth"], d["tolerance"]
    if not isinstance(af, dict) or af.get("kind") not in ANSWER_KINDS:
        errs.append(f"answer_format.kind must be one of {ANSWER_KINDS}")
        return errs
    if not isinstance(truth, dict) or "value" not in truth:
        errs.append("truth must be an object with a 'value'")
        return errs
    if not isinstance(tol, dict):
        errs.append("tolerance must be an object")
        return errs
    errs += _validate_kind(af, truth, tol)
    try:
        json.dumps(d, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errs.append(f"item is not strict JSON: {exc}")
    errs += leak_problems(d)
    return errs


def _validate_kind(af: dict, truth: dict, tol: dict) -> list[str]:
    kind = af["kind"]
    v = truth["value"]
    errs: list[str] = []
    if kind == "choice":
        ch = af.get("choices")
        if not isinstance(ch, list) or len(ch) < 2 or not all(isinstance(c, str) for c in ch):
            errs.append("choice needs answer_format.choices (>= 2 strings)")
        elif v not in ch:
            errs.append(f"truth {v!r} not among the choices")
    elif kind == "number":
        if not _finite(v):
            errs.append("number truth must be finite")
        if not af.get("unit"):
            errs.append("number needs answer_format.unit")
        for a in truth.get("accept", []) or []:
            if not _finite(a):
                errs.append("truth.accept must hold finite numbers")
        if not any(k in tol for k in ("abs", "rel")):
            errs.append("number tolerance needs 'abs' and/or 'rel'")
    elif kind == "vector":
        if not _is_pair(v):
            errs.append("vector truth must be [a, b]")
        if not af.get("unit"):
            errs.append("vector needs answer_format.unit")
        if not any(k in tol for k in ("abs", "rel")):
            errs.append("vector tolerance needs 'abs' and/or 'rel'")
    elif kind == "position_list":
        if not isinstance(v, list) or not all(_is_pair(p) for p in v):
            errs.append("position_list truth must be a list of [x, y]")
        opt = truth.get("optional", [])
        if not isinstance(opt, list) or not all(_is_pair(p) for p in opt):
            errs.append("truth.optional must be a list of [x, y]")
        if not _finite(tol.get("match_nm")):
            errs.append("position_list tolerance needs 'match_nm'")
    elif kind == "count":
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
            errs.append("count truth must be a non-negative integer")
        lo, hi = truth.get("min", v), truth.get("max", v)
        if not (isinstance(lo, int) and isinstance(hi, int) and lo <= v <= hi):
            errs.append("count truth needs min <= value <= max")
    elif kind == "bool":
        if not isinstance(v, bool):
            errs.append("bool truth must be true/false")
    return errs


def leak_problems(d: dict) -> list[str]:
    """Hidden facts that must never appear in the question text.

    * the tip condition, in any item that is not itself about the tip;
    * the numeric truth itself (two or more decimals, which no question needs to quote).
    """
    errs: list[str] = []
    q = str(d.get("question", "")).lower()
    prov = d.get("provenance") or {}
    cond = str(prov.get("tip_condition") or "").lower()
    if d.get("type") != "tip" and cond and cond != "good":
        if re.search(rf"\b{re.escape(cond)}\b", q):
            errs.append(f"question mentions the hidden tip condition {cond!r}")
    vals: list[float] = []
    v = (d.get("truth") or {}).get("value")
    if _finite(v):
        vals.append(float(v))
    elif _is_pair(v):
        vals += [float(x) for x in v]
    for x in vals:
        if x != round(x, 1):              # e.g. 6.29 - an integer row or count may appear legitimately
            for s in (f"{x:.2f}", f"{x:.3f}"):
                if s in q:
                    errs.append(f"question quotes the truth value {s}")
                    break
    return errs


# -- answers ------------------------------------------------------------------
_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0"}


def normalize_answer(item: dict, answer: Any) -> tuple[Any, bool, str | None]:
    """``(value, abstained, error)`` for one raw answer.

    ``answer`` is the ``answer`` object of an answers-file line (``{"value": ..., "abstain": ...}``)
    or the bare value. A malformed value is an error (scored as wrong, not as abstained)."""
    if isinstance(answer, dict):
        if answer.get("abstain") is True:
            return None, True, None
        if "value" not in answer:
            return None, False, "answer has no 'value'"
        raw = answer["value"]
    else:
        raw = answer
    if raw is None:
        return None, True, None
    af = item["answer_format"]
    kind = af["kind"]
    try:
        if kind == "choice":
            s = str(raw).strip().lower()
            choices = [str(c) for c in af.get("choices", [])]
            for c in choices:
                if s == c.lower():
                    return c, False, None
            return None, False, f"{raw!r} is not one of {choices}"
        if kind == "number":
            x = float(raw)
            if not math.isfinite(x):
                return None, False, "not a finite number"
            return x, False, None
        if kind == "vector":
            if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
                return None, False, "vector must be [a, b]"
            a, b = float(raw[0]), float(raw[1])
            if not (math.isfinite(a) and math.isfinite(b)):
                return None, False, "vector components must be finite"
            return [a, b], False, None
        if kind == "position_list":
            if not isinstance(raw, (list, tuple)):
                return None, False, "position_list must be a list of [x, y]"
            if len(raw) > MAX_POINTS:
                return None, False, f"more than {MAX_POINTS} points"
            pts = []
            for p in raw:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    p = [p["x"], p["y"]]
                if not (isinstance(p, (list, tuple)) and len(p) == 2):
                    return None, False, f"point {p!r} is not [x, y]"
                x, y = float(p[0]), float(p[1])
                if not (math.isfinite(x) and math.isfinite(y)):
                    return None, False, "point coordinates must be finite"
                pts.append([x, y])
            return pts, False, None
        if kind == "count":
            if isinstance(raw, bool):
                return None, False, "count must be a non-negative integer"
            x = float(raw)
            if not math.isfinite(x) or x < 0 or abs(x - round(x)) > 1e-9:
                return None, False, "count must be a non-negative integer"
            return int(round(x)), False, None
        if kind == "bool":
            if isinstance(raw, bool):
                return raw, False, None
            s = str(raw).strip().lower()
            if s in _TRUE:
                return True, False, None
            if s in _FALSE:
                return False, False, None
            return None, False, f"{raw!r} is not true/false"
    except (TypeError, ValueError) as exc:
        return None, False, f"malformed {kind} answer: {exc}"
    return None, False, f"unknown answer kind {kind!r}"


def answer_value_schema(item: dict) -> dict:
    """JSON schema of the ``value`` argument the QA host's ``answer`` tool takes."""
    af = item["answer_format"]
    kind = af["kind"]
    unit = af.get("unit")
    if kind == "choice":
        return {"type": "string", "enum": list(af["choices"]), "description": "One of the choices."}
    if kind == "number":
        return {"type": "number", "description": f"The value in {unit}."}
    if kind == "vector":
        return {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
                "description": f"[a, b] in {unit}."}
    if kind == "position_list":
        return {"type": "array", "maxItems": MAX_POINTS,
                "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                "description": "[[x, y], ...] in scan-frame nm (an empty list is allowed)."}
    if kind == "count":
        return {"type": "integer", "minimum": 0, "description": "A non-negative integer."}
    if kind == "bool":
        return {"type": "boolean"}
    raise ValueError(f"unknown answer kind {kind!r}")


# -- files --------------------------------------------------------------------
def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: not JSON ({exc})") from None
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
            n += 1
    return n


def load_items(dataset: str | Path) -> list[dict]:
    """Items of a dataset directory (``<dir>/items.jsonl``) or of a JSONL file."""
    p = Path(dataset)
    return read_jsonl(p / "items.jsonl" if p.is_dir() else p)


def load_answers(path: str | Path) -> dict[str, dict | None]:
    """``{item_id: answer_object}`` from a JSONL of ``{"id", "answer": {...}}`` (later lines win).
    A line ``{"id": ..., "value": ...}`` (no ``answer`` wrapper) is accepted too; ``"answer": null``
    (a QA session that ended without answering) maps to ``None`` = not answered."""
    out: dict[str, dict | None] = {}
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"{path}: answer line without 'id': {row!r}")
        if "answer" in row:
            ans = row["answer"]
            if ans is not None and not isinstance(ans, dict):
                ans = {"value": ans}
        else:
            ans = {k: v for k, v in row.items() if k in ("value", "abstain", "confidence", "note")}
        out[str(row["id"])] = ans
    return out
