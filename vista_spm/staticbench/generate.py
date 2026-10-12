"""Generate an STM-BabyVision dataset: frames with known truth, and questions about them.

README
------
::

    python -m vista_spm.staticbench generate --out tmp/staticbench/v1 \
        --types tip,adatoms,move,drift,herringbone --seeds 0-9 [--jobs 8] [--overlays]

writes ``<out>/items.jsonl`` (one item per line, schema in :mod:`.items`), the frames the
items refer to under ``<out>/frames/`` (``.sxm`` / ``.dat`` with neutral names), and
``<out>/manifest.json`` (counts, class balance, failures, wall time, versions). With
``--overlays`` it also renders every frame with its truth drawn on top into
``<out>/_overlays/`` for a person to check (never referenced by an item).

Every item comes from one *recipe* ``(type, seed, index, params)``; a recipe builds its own
world (``_sim.make_world``) with a frozen clock and a seed derived from the recipe, so the
dataset is a pure function of ``(types, seeds)`` and recipes can run in parallel processes.

Item types and what is varied (per dataset seed):

* ``tip`` — one item per class (good / blunt / double / unstable / dirty), Cu(111)+Fe or
  Au(111)+adatoms alternating; a topograph plus a dI/dV spectrum on bare terrace.
* ``adatoms`` — two Cu(111)+Fe frames: one good tip, one double or blunt tip; count and
  positions.
* ``move`` — two frame pairs of one window: one with an atom dragged in between (the
  ``MoveAtomTo`` sequence, see ``_sim.Bench.drag_atom``), one without; detect / which / distance.
* ``drift`` — two frame pairs of one window, minutes apart (one with partial drift
  compensation on even seeds).
* ``herringbone`` — two Au(111) frames (80-100 nm and 120-150 nm) inside one rotational
  domain; period and stripe orientation.
* ``step`` — one frame across monatomic steps; terrace height difference between two points.
* ``tipchange`` — one frame with exactly one spontaneous tip change; the row.
* ``atomic`` — two small frames (sharp vs smeared apex); atomic resolution yes/no, truth by
  MAST's ``assess_atomic_phase`` (a reference metric, documented as such).
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import items as I
from ._sim import Bench, FrameRec, _plain, apparent_arms, make_world

GENERATOR_VERSION = "staticbench-gen/1"
DEFAULT_TYPES = ("tip", "adatoms", "move", "drift", "herringbone")
TYPE_CODE = {t: k + 1 for k, t in enumerate(I.ITEM_TYPES)}

COORD_NOTE = ("Coordinates are scan-frame nanometres as recorded in the frame header: the frame "
              "centre is at the scan offset, x points right and y points up in the image.")
ARTEFACT_NOTE = "The tip may be imperfect: report real surface atoms, not imaging artefacts."


class GenerationError(RuntimeError):
    """A recipe could not produce a valid item within its attempts."""


@dataclass
class Recipe:
    type: str
    seed: int
    index: int
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.type}/{self.seed}/{self.index}"

    def as_dict(self) -> dict:
        return {"type": self.type, "seed": self.seed, "index": self.index, "params": dict(self.params)}


@dataclass
class Ctx:
    out: Path
    keep_work: bool = False
    overlays: bool = False

    @property
    def frames_dir(self) -> Path:
        return self.out / "frames"

    def session(self, r: Recipe, attempt: int) -> Path:
        return self.out / "_work" / I.short_hash("work", r.key, attempt, n=10) / "session"

    def export(self, src: str | Path, r: Recipe, label: str) -> str:
        """Copy a saved frame/spectrum into ``frames/`` under a neutral name; returns the
        dataset-relative POSIX path. An ``.sxm`` header's ``SCAN_FILE`` (the session path it was
        saved under) is rewritten to that neutral path, so no file says where or how it was made."""
        src = Path(src)
        name = I.short_hash("frame", r.key, label, n=12) + src.suffix.lower()
        dest = self.frames_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        rel = f"frames/{name}"
        if src.suffix.lower() == ".sxm":
            dest.write_bytes(neutral_sxm_bytes(src.read_bytes(), rel))
        else:
            shutil.copyfile(src, dest)
        return rel


_DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")


def neutral_sxm_bytes(raw: bytes, scan_file: str) -> bytes:
    """The ``.sxm`` bytes with the header's ``SCAN_FILE`` value replaced by ``scan_file``; the
    data block after the ``\\x1a\\x04`` marker is untouched (readers locate it by the marker)."""
    end = raw.find(b":SCANIT_END:")
    mark = raw.find(b"\x1a\x04", end if end >= 0 else 0)
    if end < 0 or mark < 0:
        raise GenerationError("not a simulator .sxm: no ':SCANIT_END:' header end")
    lines = raw[:mark].decode("gbk").split("\n")
    for k in range(len(lines) - 1):
        if lines[k].strip() == ":SCAN_FILE:":
            lines[k + 1] = scan_file
            break
    else:
        raise GenerationError("not a simulator .sxm: no ':SCAN_FILE:' entry")
    head = "\n".join(lines)
    if _DRIVE_PATH.search(head):
        raise GenerationError("an .sxm header still names a drive path after the rewrite")
    return head.encode("gbk") + raw[mark:]


# ── planning ─────────────────────────────────────────────────────────────────
def parse_seeds(spec: str) -> list[int]:
    """``"0-9"`` / ``"0,2,5"`` / ``"0-3,7"`` → sorted unique seeds."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError(f"no seeds in {spec!r}")
    return sorted(out)


def parse_types(spec: str) -> list[str]:
    if str(spec).strip() in ("all", "*"):
        return list(I.ITEM_TYPES)
    types = [t.strip() for t in str(spec).split(",") if t.strip()]
    bad = [t for t in types if t not in I.ITEM_TYPES]
    if bad:
        raise ValueError(f"unknown item type(s) {bad}; known: {I.ITEM_TYPES}")
    return types


def plan(types: list[str], seeds: list[int]) -> list[Recipe]:
    """The recipes for these types and seeds (deterministic, class-balanced per seed)."""
    rs: list[Recipe] = []
    for s in seeds:
        if "tip" in types:
            for c, cond in enumerate(I.TIP_CLASSES):
                rs.append(Recipe("tip", s, c, {"condition": cond,
                                               "substrate": "cu" if (s + c) % 2 == 0 else "au"}))
        if "adatoms" in types:
            rs.append(Recipe("adatoms", s, 0, {"tip": "good"}))
            rs.append(Recipe("adatoms", s, 1, {"tip": "double" if s % 2 == 0 else "blunt"}))
        if "move" in types:
            rs.append(Recipe("move", s, 0, {"moved": True}))
            rs.append(Recipe("move", s, 1, {"moved": False}))
        if "drift" in types:
            rs.append(Recipe("drift", s, 0, {"substrate": "cu", "compensate": False}))
            rs.append(Recipe("drift", s, 1, {"substrate": "au", "compensate": s % 2 == 0}))
        if "herringbone" in types:
            rs.append(Recipe("herringbone", s, 0, {"size": "small"}))
            rs.append(Recipe("herringbone", s, 1, {"size": "large"}))
        if "step" in types:
            rs.append(Recipe("step", s, 0, {"substrate": "au" if s % 2 == 0 else "cu"}))
        if "tipchange" in types:
            rs.append(Recipe("tipchange", s, 0, {"substrate": "cu" if s % 2 == 0 else "au",
                                                 "scan_dir": "up" if s % 3 == 2 else "down"}))
        if "atomic" in types:
            sub = "au" if s % 2 == 0 else "cu"
            rs.append(Recipe("atomic", s, 0, {"want": True, "substrate": sub}))
            rs.append(Recipe("atomic", s, 1, {"want": False, "substrate": sub}))
    return rs


def _rng(r: Recipe, salt: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(r.seed), TYPE_CODE[r.type], int(r.index), 0x5B, int(salt)])


def _world_seed(r: Recipe, attempt: int) -> int:
    return int(r.seed) * 1000 + TYPE_CODE[r.type] * 100 + int(r.index) * 10 + int(attempt)


# ── shared pieces ────────────────────────────────────────────────────────────
def _predict_scan_xy(b: Bench, ax: float, ay: float, t: float) -> tuple[float, float]:
    ox, oy = b.offset_nm(t)
    return ax - ox, ay - oy


def _window_with_atoms(b: Bench, rng: np.random.Generator, size: float, *, min_atoms: int,
                       max_atoms: int = 999, frame_time_s: float = 160.0, reach_nm: float = 50.0,
                       margin_nm: float = 1.5, tries: int = 80, step_free: bool = False) -> tuple[float, float]:
    """A scan-frame centre whose frame (predicted at mid-frame time) holds ``min_atoms`` to
    ``max_atoms`` adatoms at least ``margin_nm`` inside the edges (and, with ``step_free``, no
    step edge: a step parallel to the fast axis is indistinguishable from a tip change)."""
    atoms = [a for a in b.atoms() if abs(a["x_nm"]) < reach_nm and abs(a["y_nm"]) < reach_nm]
    t_mid = b.now + 2.0 + frame_time_s / 2
    best, best_n = None, -1
    for _ in range(tries):
        if atoms:
            a = atoms[int(rng.integers(len(atoms)))]
            sx = a["x_nm"] + float(rng.uniform(-size / 3, size / 3))
            sy = a["y_nm"] + float(rng.uniform(-size / 3, size / 3))
        else:
            sx, sy = (float(v) for v in rng.uniform(-reach_nm, reach_nm, 2))
        if step_free and not b.w.surface.step_free_window(sx * 1e-9, sy * 1e-9, size * 1.25e-9, n=25):
            continue
        cx, cy = _predict_scan_xy(b, sx, sy, t_mid)
        n = 0
        for q in b.atoms():
            px, py = _predict_scan_xy(b, q["x_nm"], q["y_nm"], t_mid)
            if abs(px - cx) <= size / 2 - margin_nm and abs(py - cy) <= size / 2 - margin_nm:
                n += 1
        if min_atoms <= n <= max_atoms:
            return round(cx, 3), round(cy, 3)
        if n >= min_atoms and (best is None or n < best_n):
            best, best_n = (cx, cy), n             # too crowded, but the least crowded seen
    if best is None:
        raise GenerationError(f"no window of {size} nm with >= {min_atoms} atoms")
    return round(best[0], 3), round(best[1], 3)


def _atoms_in_frame(rec: FrameRec, margin_nm: float = 0.4) -> tuple[list, list, list]:
    """``(core, optional, records)``: apparent scan positions of the adatoms whose centre lies
    inside the frame by >= margin (core) or within +-margin of an edge (optional)."""
    core, opt, recs = [], [], []
    for a in rec.atoms:
        x, y, row = rec.sample_to_scan(a["x_nm"], a["y_nm"])
        if rec.inside(x, y, margin_nm):
            core.append([round(x, 4), round(y, 4)])
            where = "core"
        elif rec.inside(x, y, -margin_nm):
            opt.append([round(x, 4), round(y, 4)])
            where = "edge"
        else:
            continue
        col, prow = rec.scan_to_pixel(x, y)
        recs.append({"id": a["id"], "x_nm": round(x, 4), "y_nm": round(y, 4), "col": round(col, 2),
                     "row": round(prow, 2), "sample_nm": [round(a["x_nm"], 4), round(a["y_nm"], 4)],
                     "where": where})
    return core, opt, recs


def _min_pair_distance(points: list) -> float:
    best = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            best = min(best, math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1]))
    return best


def _frame_prov(rec: FrameRec) -> dict:
    return _plain(rec.summary())


def _base_prov(r: Recipe, b: Bench, sc, template: str, attempt: int, *, tip: dict,
               overrides: dict | None = None) -> dict:
    return {"generator": GENERATOR_VERSION, "recipe": r.as_dict(), "scenario": sc.id,
            "template": template, "seed": r.seed, "world_seed": _world_seed(r, attempt),
            "attempt": attempt, "tip_condition": tip.get("condition", "good"),
            "tip_params": _plain(tip), "overrides": _plain(overrides or {}),
            **b.provenance()}


def _no_tip_change(b: Bench) -> bool:
    return not b.w.tip.events


def _ghost_visibility(tip: dict, phi_ev: float) -> tuple[float, float]:
    """(separation nm, relative ghost weight) of a double tip's second apex."""
    from stmsim.physics.junction import kappa_per_m

    apexes = tip.get("apexes") or []
    if len(apexes) < 2:
        return 0.0, 0.0
    a0, a1 = apexes[0], apexes[1]
    d = math.hypot(a1[0] - a0[0], a1[1] - a0[1]) * 1e9
    kappa = kappa_per_m(phi_ev)
    vis = (a1[3] / a0[3]) * math.exp(2 * kappa * min(a1[2] - a0[2], 0.0))
    return d, vis


def _bare_spot(b: Bench, rec: FrameRec, rng: np.random.Generator, *, clear_nm: float = 2.5,
               step_clear_nm: float = 3.0, margin_nm: float = 2.0) -> tuple[float, float]:
    """A scan point inside the frame whose sample point NOW is bare terrace."""
    atoms = b.atoms()
    ox, oy = b.offset_nm()
    for _ in range(200):
        u = float(rng.uniform(-rec.w_nm / 2 + margin_nm, rec.w_nm / 2 - margin_nm))
        v = float(rng.uniform(-rec.h_nm / 2 + margin_nm, rec.h_nm / 2 - margin_nm))
        x, y = rec.uv_to_scan(u, v)
        sx, sy = x + ox, y + oy
        if any(math.hypot(a["x_nm"] - sx, a["y_nm"] - sy) < clear_nm for a in atoms):
            continue
        if b.step_distance_nm(sx, sy, radius_nm=step_clear_nm, n=8) < step_clear_nm:
            continue
        return round(x, 3), round(y, 3)
    raise GenerationError("no bare spot for a spectrum inside the frame")


def _material(b: Bench) -> str:
    return str(b.w.surface.material.name)


def _a(material: str) -> str:
    """``a Cu(111)`` / ``an Au(111)``."""
    return ("an " if material[:1].upper() in "AEIOU" else "a ") + material


def _item(type_: str, subtype: str, r: Recipe, *, question: str, frames: list[str], answer_format: dict,
          truth: dict, tolerance: dict, provenance: dict, difficulty: str = "normal") -> dict:
    it = I.Item(id=I.item_id(type_, subtype, r.seed, r.index, "sb1"), type=type_, subtype=subtype,
                question=" ".join(question.split()), frames=frames, answer_format=answer_format,
                truth=_plain(truth), tolerance=tolerance, provenance=_plain(provenance),
                difficulty=difficulty).to_dict()
    errs = I.validate_item(it)
    if errs:
        raise GenerationError(f"{it['id']}: invalid item: {errs}")
    return it


# ── tip condition ────────────────────────────────────────────────────────────
TIP_DEFINITIONS = ("'good' (sharp, single, stable and clean apex), 'blunt' (features broadened or "
                   "smeared, no fine detail), 'double' (features duplicated by a second apex), 'unstable' "
                   "(abrupt tip changes during acquisition: jumps, streaks or a discontinuity in the image "
                   "or in the spectrum), 'dirty' (an adsorbate on the apex: the spectrum shows a feature "
                   "that does not belong to the clean sample)")


def build_tip(r: Recipe, ctx: Ctx) -> list[dict]:
    cond, sub = r.params["condition"], r.params["substrate"]
    rng = _rng(r)
    scale = float(rng.uniform(0.0, 1.0))
    angle = float(rng.uniform(0.0, 360.0))
    template = "cu_fe" if sub == "cu" else "au_fe"
    size = 20.0 if sub == "cu" else 30.0
    density = float(rng.uniform(1.0, 1.6)) if sub == "cu" else float(rng.uniform(0.5, 0.8))
    tip = {"condition": cond, "condition_scale": scale, "condition_angle_deg": angle}
    got = None
    for attempt in range(8):
        w, sc = make_world(template, _world_seed(r, attempt), ctx.session(r, attempt), tip=tip,
                           adatoms={"field_per_100nm2": density})
        b = Bench(w, sc)
        try:
            cx, cy = _window_with_atoms(b, rng, size, min_atoms=3, max_atoms=12, step_free=True)
        except GenerationError:
            continue
        # an unstable tip must show it in the topograph (a jump in a spectrum alone can sit where
        # the current crosses zero and look like a real spectral step); every other class must
        # show no tip change at all
        for k in range(10 if cond == "unstable" else 1):
            rec = b.scan(cx, cy, size, 256, line_time_s=0.3)
            if cond == "unstable" and not rec.tip_change_rows:
                b.wait(5.0)
                continue
            if cond != "unstable" and not _no_tip_change(b):
                break
            try:
                sx, sy = _bare_spot(b, rec, rng)
            except GenerationError:
                break
            spec = b.spectrum(sx, sy)
            if cond != "unstable" and spec.jump:
                break
            got = (rec, spec)
            break
        if got is not None:
            break
    if got is None:
        raise GenerationError(f"tip/{cond}: no frame with consistent evidence in 8 worlds")
    rec, spec = got
    tip0 = rec.tip_start
    evidence: dict[str, Any] = {"topograph_tip_change_rows": [rec.row_of_acq(k) for k in rec.tip_change_rows],
                                "spectrum_jump": bool(spec.jump), "phi_junction_ev": spec.phi_ev,
                                "tip_at_frame_start": tip0}
    difficulty = "normal"
    if cond == "blunt":
        evidence["radius_nm"] = tip0["radius_nm"]
        difficulty = "hard" if tip0["radius_nm"] < 3.8 else "normal"
    elif cond == "double":
        d, vis = _ghost_visibility(tip0, spec.phi_ev)
        evidence.update({"apex_separation_nm": d, "ghost_relative_weight": vis})
        difficulty = "hard" if (d < 1.2 or vis < 0.15) else "normal"
    elif cond == "unstable":
        dzs = [abs(float(e["detail"].get("dz_pm", 0.0))) for e in rec.tip_events if e["kind"] == "spontaneous_change"]
        evidence["tip_change_dz_pm"] = dzs
        difficulty = "hard" if (not dzs or max(dzs) < 40.0) else "normal"
    elif cond == "dirty":
        evidence["tip_state_peak_bias_v"] = round(0.45 - 0.7 * scale, 4)
    mat = _material(b)
    frames = [ctx.export(rec.path, r, "topo"), ctx.export(spec.path, r, "spec")]
    q = (f"Frame 1 is a constant-current STM topograph ({size:g} nm x {size:g} nm, 256 x 256 px) of "
         f"{_a(mat)} surface with a few adsorbed atoms, and spectrum 2 is a dI/dV point spectrum (lock-in "
         f"channel 'LI Demod 1 X' against 'Bias calc (V)', current in 'Current (A)') taken with the same tip on "
         f"bare terrace inside that area. Judge the condition of the tip from these data. Choose exactly one "
         f"label: {TIP_DEFINITIONS}. Answer with one of: good, blunt, double, unstable, dirty.")
    prov = _base_prov(r, b, sc, template, attempt, tip=tip, overrides={"field_per_100nm2": density})
    prov.update({"frames": [_frame_prov(rec), _plain(spec.summary())], "sim_s": [rec.t_start, spec.sim_s]})
    if ctx.overlays:
        _overlay_frame(ctx, r, "topo", rec, marks=[], text=f"tip={cond}")
        _overlay_spectrum(ctx, r, "spec", spec)
    return [_item("tip", "condition", r, question=q, frames=frames,
                  answer_format={"kind": "choice", "choices": list(I.TIP_CLASSES)},
                  truth={"value": cond, "evidence": evidence}, tolerance={}, provenance=prov,
                  difficulty=difficulty)]


# ── adatoms: count and positions ─────────────────────────────────────────────
def build_adatoms(r: Recipe, ctx: Ctx) -> list[dict]:
    variant = r.params["tip"]
    rng = _rng(r)
    size = float(rng.choice([15.0, 20.0, 25.0, 30.0]))
    density = float(rng.uniform(0.6, 1.8))
    tip = {"condition": variant, "condition_scale": float(rng.uniform(0.3, 1.0)),
           "condition_angle_deg": float(rng.uniform(0.0, 360.0))}
    for attempt in range(6):
        w, sc = make_world("cu_fe", _world_seed(r, attempt), ctx.session(r, attempt), tip=tip,
                           adatoms={"field_per_100nm2": density})
        b = Bench(w, sc)
        try:
            cx, cy = _window_with_atoms(b, rng, size, min_atoms=2, max_atoms=25)
        except GenerationError:
            continue
        rec = b.scan(cx, cy, size, 256)
        if _no_tip_change(b) and not rec.hops:
            break
    else:
        raise GenerationError("adatoms: no window, or the tip changed / atoms moved, in every attempt")
    core, opt, recs = _atoms_in_frame(rec)
    min_sep = _min_pair_distance(core + opt)
    difficulty = "hard" if (variant != "good" or min_sep < 1.0) else "normal"
    frames = [ctx.export(rec.path, r, "frame")]
    head = (f"Frame 1 is a constant-current STM topograph ({size:g} nm x {size:g} nm, 256 x 256 px) of "
            f"Cu(111) with isolated adsorbed Fe atoms, which image as compact protrusions.")
    prov = _base_prov(r, b, sc, "cu_fe", attempt, tip=tip, overrides={"field_per_100nm2": density})
    prov.update({"frames": [_frame_prov(rec)], "sim_s": [rec.t_start], "atoms": recs,
                 "min_pair_distance_nm": min_sep if math.isfinite(min_sep) else None})
    if ctx.overlays:
        _overlay_frame(ctx, r, "frame", rec, marks=[("o", p) for p in core] + [("x", p) for p in opt],
                       text=f"tip={variant} n={len(core)}+{len(opt)}")
    out = [
        _item("adatoms", "count", r, frames=frames, question=(
            head + " How many adsorbed atoms have their centre inside the frame? Atoms cut by the frame "
            "edge count if their centre is inside. " + ARTEFACT_NOTE + " Answer with a non-negative integer."),
            answer_format={"kind": "count"},
            truth={"value": len(core), "min": len(core), "max": len(core) + len(opt), "n_edge": len(opt)},
            tolerance={"near": 1}, provenance=prov, difficulty=difficulty),
        _item("adatoms", "positions", r, frames=frames, question=(
            head + " List the centre of every adsorbed atom whose centre lies inside the frame, as "
            "scan-frame coordinates [x, y] in nm. " + COORD_NOTE + " " + ARTEFACT_NOTE +
            " Answer with a list of [x, y] pairs (an empty list if there are none)."),
            answer_format={"kind": "position_list", "unit": "nm", "frame": "scan"},
            truth={"value": core, "optional": opt}, tolerance={"match_nm": 0.3}, provenance=prov,
            difficulty=difficulty),
    ]
    return out


# ── move: before / after a manipulation ──────────────────────────────────────
def _isolated(atoms: list[dict], a: dict, r_nm: float) -> bool:
    return all(math.hypot(q["x_nm"] - a["x_nm"], q["y_nm"] - a["y_nm"]) >= r_nm for q in atoms if q["id"] != a["id"])


def build_move(r: Recipe, ctx: Ctx) -> list[dict]:
    moved = bool(r.params["moved"])
    rng = _rng(r)
    size = float(rng.choice([15.0, 18.0, 20.0]))
    density = float(rng.uniform(1.5, 2.5))
    tip = {"condition": "good"}
    frame_t = 256 * 0.6
    result = None
    for attempt in range(10):
        w, sc = make_world("cu_fe", _world_seed(r, attempt), ctx.session(r, attempt), tip=tip,
                           adatoms={"field_per_100nm2": density})
        b = Bench(w, sc)
        atoms = b.atoms()
        cand = [a for a in atoms if max(abs(a["x_nm"]), abs(a["y_nm"])) < 45 and _isolated(atoms, a, 3.0)]
        if not cand:
            continue
        A = cand[int(rng.integers(len(cand)))]
        if moved:
            target = None
            for _ in range(40):
                d = float(rng.uniform(1.5, 3.5))
                th = float(rng.uniform(0, 2 * math.pi))
                tx, ty = A["x_nm"] + d * math.cos(th), A["y_nm"] + d * math.sin(th)
                if all(math.hypot(q["x_nm"] - tx, q["y_nm"] - ty) >= 1.5 for q in atoms if q["id"] != A["id"]):
                    target = (tx, ty)
                    break
            if target is None:
                continue
            mid = ((A["x_nm"] + target[0]) / 2, (A["y_nm"] + target[1]) / 2)
            half_span = math.hypot(target[0] - A["x_nm"], target[1] - A["y_nm"]) / 2
        else:
            target, mid, half_span = None, (A["x_nm"], A["y_nm"]), 0.0
        # the window: both positions of the atom inside both frames, with room for the drift
        t1 = b.now + 2.0 + frame_t / 2
        t2 = t1 + frame_t + 40.0
        jitter = max(0.0, size / 2 - half_span - 4.5)
        sx = mid[0] + float(rng.uniform(-jitter, jitter))
        sy = mid[1] + float(rng.uniform(-jitter, jitter))
        o1, o2 = b.offset_nm(t1), b.offset_nm(t2)
        cx, cy = sx - (o1[0] + o2[0]) / 2, sy - (o1[1] + o2[1]) / 2
        rec1 = b.scan(round(cx, 3), round(cy, 3), size, 256)
        if not _no_tip_change(b) or rec1.hops:
            continue
        drag = None
        if moved:
            ok = False
            drag = []
            for k in range(3):
                r_th, r_pick = b.effective_thresholds(A["id"])
                R = max(0.5 * r_th, 2.0 * r_pick)
                if R > 0.8 * r_th:
                    R = math.sqrt(r_th * r_pick)
                res = b.drag_atom(A["id"], target, r_kohm=R)
                drag.append({k2: v for k2, v in res.items() if k2 != "events"} | {"n_events": len(res["events"])})
                if b.atom_lost_or_carried():
                    break
                now_a = next(q for q in b.atoms() if q["id"] == A["id"])
                if math.hypot(now_a["x_nm"] - A["x_nm"], now_a["y_nm"] - A["y_nm"]) >= 0.7:
                    ok = True
                    break
            if not ok or b.atom_lost_or_carried():
                continue
        else:
            b.wait(float(rng.uniform(20.0, 60.0)))
        rec2 = b.scan(rec1.cx_nm, rec1.cy_nm, size, 256)
        if not _no_tip_change(b) or rec2.hops or b.atom_lost_or_carried():
            continue
        before = {a["id"]: a for a in rec1.atoms}
        after = {a["id"]: a for a in rec2.atoms}
        movers = [i for i in before if i in after and (before[i]["i"], before[i]["j"]) != (after[i]["i"], after[i]["j"])]
        if moved and movers != [A["id"]]:
            continue
        if not moved and movers:
            continue
        p1 = rec1.sample_to_scan(before[A["id"]]["x_nm"], before[A["id"]]["y_nm"])
        p2 = rec2.sample_to_scan(after[A["id"]]["x_nm"], after[A["id"]]["y_nm"])
        if not (rec1.inside(p1[0], p1[1], 1.5) and rec2.inside(p2[0], p2[1], 1.5)):
            continue
        if moved:
            q1 = rec1.sample_to_scan(after[A["id"]]["x_nm"], after[A["id"]]["y_nm"])
            q0 = rec2.sample_to_scan(before[A["id"]]["x_nm"], before[A["id"]]["y_nm"])
            if not (rec1.inside(q1[0], q1[1], 1.0) and rec2.inside(q0[0], q0[1], 1.0)):
                continue
        # a movement is only defined relative to something that stayed: >= 2 other atoms in both frames
        n_ref = sum(1 for i in before if i != A["id"] and i in after
                    and rec1.inside(*rec1.sample_to_scan(before[i]["x_nm"], before[i]["y_nm"])[:2], 1.0)
                    and rec2.inside(*rec2.sample_to_scan(after[i]["x_nm"], after[i]["y_nm"])[:2], 1.0))
        if n_ref < 2:
            continue
        result = (attempt, w, sc, b, A, rec1, rec2, before, after, drag, p1, p2, n_ref)
        break
    if result is None:
        raise GenerationError(f"move(moved={moved}): no valid frame pair in 10 worlds")
    attempt, w, sc, b, A, rec1, rec2, before, after, drag, p1, p2, n_ref = result
    bA, aA = before[A["id"]], after[A["id"]]
    dist = math.hypot(aA["x_nm"] - bA["x_nm"], aA["y_nm"] - bA["y_nm"])          # on the surface
    # as the images show it: drift during a frame stretches it (up to ~15 % along y for a slow,
    # small frame), so the separation between where the atom is and where it would have been is
    # measured in each frame's own coordinates; any of the three is accepted
    q2a, q2b = rec2.sample_to_scan(aA["x_nm"], aA["y_nm"]), rec2.sample_to_scan(bA["x_nm"], bA["y_nm"])
    q1a, q1b = rec1.sample_to_scan(aA["x_nm"], aA["y_nm"]), rec1.sample_to_scan(bA["x_nm"], bA["y_nm"])
    dist_f2 = math.hypot(q2a[0] - q2b[0], q2a[1] - q2b[1])
    dist_f1 = math.hypot(q1a[0] - q1b[0], q1a[1] - q1b[1])
    shift, _spread = apparent_shift(rec1, rec2)
    shift = [float(shift[0]), float(shift[1])]
    dt = rec2.t_start - rec1.t_start
    frames = [ctx.export(rec1.path, r, "before"), ctx.export(rec2.path, r, "after")]
    head = (f"Frames 1 and 2 are constant-current STM topographs ({size:g} nm x {size:g} nm, 256 x 256 px) "
            f"of the same scan window on Cu(111) with adsorbed Fe atoms, recorded one after the other with "
            f"identical settings ({dt:.0f} s between their starts). Thermal drift may shift the whole image "
            f"between them; that alone is not a movement.")
    prov = _base_prov(r, b, sc, "cu_fe", attempt, tip=tip, overrides={"field_per_100nm2": density})
    _, _, recs1 = _atoms_in_frame(rec1)
    prov.update({"frames": [_frame_prov(rec1), _frame_prov(rec2)], "sim_s": [rec1.t_start, rec2.t_start],
                 "atom_id": A["id"], "drag": drag, "atoms_frame1": recs1, "n_reference_atoms": n_ref,
                 "image_shift_nm": [round(v, 4) for v in shift],
                 "moved_atom": {"frame1_scan_nm": [p1[0], p1[1]], "frame2_scan_nm": [p2[0], p2[1]],
                                "sample_before_nm": [before[A["id"]]["x_nm"], before[A["id"]]["y_nm"]],
                                "sample_after_nm": [after[A["id"]]["x_nm"], after[A["id"]]["y_nm"]]}
                 if moved else None})
    if ctx.overlays:
        _overlay_frame(ctx, r, "before", rec1, marks=[("o", [p1[0], p1[1]])], text=f"moved={moved}")
        _overlay_frame(ctx, r, "after", rec2, marks=[("o", [p2[0], p2[1]])], text=f"d={dist:.2f} nm")
    out = [_item("move", "detect", r, frames=frames, question=(
        head + " Did any adsorbed atom change its position on the surface (relative to the other atoms and the "
        "substrate) between frame 1 and frame 2? Answer true or false."),
        answer_format={"kind": "bool"}, truth={"value": moved}, tolerance={}, provenance=prov,
        difficulty="hard" if (moved and dist < 1.0) else "normal")]
    if moved:
        out.append(_item("move", "which", r, frames=frames, question=(
            head + " Exactly one adsorbed atom changed its position on the surface between the frames. Give "
            "the position of that atom as it appears in frame 1, as scan-frame coordinates [x, y] in nm. "
            + COORD_NOTE + " Answer with [x, y]."),
            answer_format={"kind": "vector", "unit": "nm", "frame": "scan"},
            truth={"value": [round(p1[0], 4), round(p1[1], 4)], "frame2_nm": [round(p2[0], 4), round(p2[1], 4)]},
            tolerance={"abs": 0.3}, provenance=prov, difficulty="normal"))
        out.append(_item("move", "distance", r, frames=frames, question=(
            head + " Exactly one adsorbed atom changed its position on the surface between the frames. How far "
            "did it move on the surface, in nm? Measure it relative to the surface, i.e. after removing any "
            "rigid shift of the whole image between the frames. Answer with a number in nm."),
            answer_format={"kind": "number", "unit": "nm"},
            truth={"value": round(dist_f2, 4), "accept": [round(dist_f1, 4), round(dist, 4)],
                   "surface_nm": round(dist, 4), "frame1_nm": round(dist_f1, 4), "frame2_nm": round(dist_f2, 4),
                   "drag_attempts": len(drag or [])},
            tolerance={"abs": 0.25, "rel": 0.15}, provenance=prov,
            difficulty="hard" if dist < 1.0 else "normal"))
    return out


# ── drift between two frames ─────────────────────────────────────────────────
def apparent_shift(rec1: FrameRec, rec2: FrameRec, n: int = 9) -> tuple[np.ndarray, float]:
    """How far a fixed point of the sample moves in the image from frame 1 to frame 2
    (scan-frame nm), as the median over a grid of points of frame 1, and the largest deviation
    from it across the grid.

    Not ``-(o(t2) - o(t1))`` at one row: a feature that drifts down the frame is imaged a few
    rows later in frame 2, and the drift during those rows adds to its displacement (0.3 nm on
    a 6 nm shift at 1 nm/min) - what the images show is the per-row solution for each point."""
    g = np.linspace(-0.4, 0.4, n)
    ds = []
    for fu in g:
        for fv in g:
            x, y = rec1.uv_to_scan(fu * rec1.w_nm, fv * rec1.h_nm)
            sx, sy = rec1.scan_to_sample(x, y)
            p1 = rec1.sample_to_scan(sx, sy)
            p2 = rec2.sample_to_scan(sx, sy)
            ds.append((p2[0] - p1[0], p2[1] - p1[1]))
    d = np.asarray(ds)
    med = np.median(d, axis=0)
    spread = float(np.max(np.hypot(d[:, 0] - med[0], d[:, 1] - med[1])))
    return med, spread


def build_drift(r: Recipe, ctx: Ctx) -> list[dict]:
    sub, comp = r.params["substrate"], bool(r.params.get("compensate"))
    rng = _rng(r)
    template = "cu_fe" if sub == "cu" else "au_fe"
    size = float(rng.choice([20.0, 30.0, 40.0]))
    density = float(rng.uniform(0.8, 1.4)) if sub == "cu" else float(rng.uniform(0.4, 0.8))
    gap = float(rng.uniform(0.0, 600.0))
    comp_frac = float(rng.uniform(0.6, 1.0)) if comp else 0.0
    tip = {"condition": "good"}
    for attempt in range(6):
        w, sc = make_world(template, _world_seed(r, attempt), ctx.session(r, attempt), tip=tip,
                           adatoms={"field_per_100nm2": density})
        b = Bench(w, sc)
        try:
            cx, cy = _window_with_atoms(b, rng, size, min_atoms=3, max_atoms=40, frame_time_s=2 * 154 + gap)
        except GenerationError:
            continue
        if comp:
            v = w.drift_v_m_per_s
            w.set_drift_comp(True, float(-comp_frac * v[0]), float(-comp_frac * v[1]), 0.0)
        rec1 = b.scan(cx, cy, size, 256)
        b.wait(gap)
        rec2 = b.scan(cx, cy, size, 256)
        if not _no_tip_change(b) or rec1.hops or rec2.hops:
            continue
        n_common = sum(1 for a in rec2.atoms if rec1.inside(*rec1.sample_to_scan(a["x_nm"], a["y_nm"])[:2], 1.0)
                       and rec2.inside(*rec2.sample_to_scan(a["x_nm"], a["y_nm"])[:2], 1.0))
        if n_common >= 2:
            break
    else:
        raise GenerationError("drift: no pair with >= 2 common features")
    shift, spread = apparent_shift(rec1, rec2)
    dt = rec2.t_start - rec1.t_start
    frames = [ctx.export(rec1.path, r, "f1"), ctx.export(rec2.path, r, "f2")]
    mat = _material(b)
    q = (f"Frames 1 and 2 are constant-current STM topographs ({size:g} nm x {size:g} nm, 256 x 256 px) of "
         f"{mat} recorded on the same scan window with identical settings; frame 2 started {dt:.0f} s after "
         f"frame 1 started. Between them the sample drifted. By how much did surface features move in the "
         f"image from frame 1 to frame 2? Give the displacement [dx, dy] in nm along the scan-frame axes "
         f"(x to the right, y up in the image). Answer with [dx, dy].")
    prov = _base_prov(r, b, sc, template, attempt, tip=tip,
                      overrides={"field_per_100nm2": density, "gap_s": gap, "drift_compensation_frac": comp_frac})
    prov.update({"frames": [_frame_prov(rec1), _frame_prov(rec2)], "sim_s": [rec1.t_start, rec2.t_start],
                 "shift_row_spread_nm": spread, "n_common_atoms": n_common,
                 "drift_compensation": bool(comp)})
    if ctx.overlays:
        _overlay_frame(ctx, r, "f1", rec1, marks=[], text="drift f1")
        _overlay_frame(ctx, r, "f2", rec2, marks=[], text=f"shift=({shift[0]:.2f},{shift[1]:.2f})")
    mag = float(np.hypot(*shift))
    return [_item("drift", "vector", r, question=q, frames=frames,
                  answer_format={"kind": "vector", "unit": "nm", "frame": "scan"},
                  truth={"value": [round(float(shift[0]), 4), round(float(shift[1]), 4)], "magnitude_nm": mag},
                  tolerance={"abs": 0.3, "rel": 0.2}, provenance=prov,
                  difficulty="hard" if spread > 0.3 else "normal")]


# ── herringbone ──────────────────────────────────────────────────────────────
def _single_domain_window(b: Bench, rng: np.random.Generator, size: float, frame_time_s: float,
                          tries: int = 800) -> tuple[float, float, dict]:
    """A scan centre whose frame (with margin for the drift during it) is step-free and lies in
    one rotational-domain cell, at least 4 nm from its walls."""
    t_mid = b.now + 2.0 + frame_time_s / 2
    o = b.offset_nm(t_mid)
    half = size * 0.58
    g = np.linspace(-half, half, 7)
    for _ in range(tries):
        sx, sy = (float(v) for v in rng.uniform(-500.0, 500.0, 2))
        if not b.w.surface.step_free_window(sx * 1e-9, sy * 1e-9, size * 1.12e-9, n=31):
            continue
        o0 = b.orientation_at(sx, sy)
        if o0 is None:
            raise GenerationError("surface has no herringbone")
        seed_i = o0["seed_index"]
        ok = True
        for gx in g:
            for gy in g:
                oo = b.orientation_at(sx + gx, sy + gy)
                if oo["seed_index"] != seed_i or (oo.get("boundary_distance_nm") or 1e9) < 4.0:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return round(sx - o[0], 3), round(sy - o[1], 3), o0
    raise GenerationError(f"no step-free single-domain window of {size:.0f} nm")


def build_herringbone(r: Recipe, ctx: Ctx) -> list[dict]:
    big = r.params["size"] == "large"
    rng = _rng(r)
    size = float(rng.uniform(120.0, 150.0)) if big else float(rng.uniform(80.0, 100.0))
    size = round(size, 1)
    px = 384 if big else 320
    line_t = max(0.3, size / 400.0)
    tip = {"condition": "good"}
    for attempt in range(5):
        w, sc = make_world("au_hb", _world_seed(r, attempt), ctx.session(r, attempt), tip=tip)
        b = Bench(w, sc)
        try:
            cx, cy, _ = _single_domain_window(b, rng, size, px * 2 * line_t)
        except GenerationError:
            continue
        rec = b.scan(cx, cy, size, px, line_time_s=line_t)
        if _no_tip_change(b):              # a 5-10 min frame: the good tip changes in ~10 % of them
            break
    else:
        raise GenerationError("herringbone: no clean single-domain frame in 5 worlds")
    mx, my = rec.scan_to_sample(cx, cy)
    orient = b.orientation_at(mx, my)
    hb = b.herringbone_params()
    app = apparent_arms(rec, orient, hb)
    p1, p2 = app["arms"][0]["period_nm"], app["arms"][1]["period_nm"]
    a1, a2 = app["arms"][0]["angle_deg"], app["arms"][1]["angle_deg"]
    frames = [ctx.export(rec.path, r, "frame")]
    head = (f"Frame 1 is a constant-current STM topograph ({size:g} nm x {size:g} nm, {px} x {px} px) of "
            f"reconstructed Au(111) showing the herringbone (22 x sqrt(3)) reconstruction.")
    prov = _base_prov(r, b, sc, "au_hb", attempt, tip=tip)
    prov.update({"frames": [_frame_prov(rec)], "sim_s": [rec.t_start], "orientation_at_centre": orient,
                 "herringbone": hb, "apparent": app})
    if ctx.overlays:
        _overlay_frame(ctx, r, "frame", rec, marks=[("line", a1), ("line", a2)],
                       text=f"P={p1:.2f}/{p2:.2f} nm  ang={a1:.1f}/{a2:.1f}")
    distorted = abs(p1 - p2) > 0.1
    return [
        _item("herringbone", "period", r, frames=frames, question=(
            head + " What is the period of the reconstruction stripes in this image: the spacing between "
            "neighbouring parallel stripes, measured perpendicular to them where they run straight (within one "
            "arm of the zigzag)? Answer with a number in nm."),
            answer_format={"kind": "number", "unit": "nm"},
            truth={"value": round((p1 + p2) / 2, 4), "accept": [round(p1, 4), round(p2, 4)],
                   "undistorted_nm": round(app["arms"][0]["period_undistorted_nm"], 4)},
            tolerance={"abs": 0.15}, provenance=prov, difficulty="hard" if distorted else "normal"),
        _item("herringbone", "orientation", r, frames=frames, question=(
            head + " In which direction do the reconstruction stripes run in this image? Give the angle in "
            "degrees in [0, 180), measured counter-clockwise from the image's +x axis (x pointing right, y "
            "pointing up). The stripes zigzag; the direction of either arm of the zigzag or their mean "
            "direction is accepted. Answer with a number in degrees."),
            answer_format={"kind": "number", "unit": "deg", "mod": 180},
            truth={"value": round(app["mean_angle_deg"], 3), "accept": [round(a1, 3), round(a2, 3)]},
            tolerance={"abs": 6.0, "mod": 180}, provenance=prov, difficulty="normal"),
    ]


# ── steps ────────────────────────────────────────────────────────────────────
def build_step(r: Recipe, ctx: Ctx) -> list[dict]:
    sub = r.params["substrate"]
    rng = _rng(r)
    template = "cu_fe" if sub == "cu" else "au_hb"
    size = float(rng.choice([50.0, 60.0, 80.0]))
    terrace = float(rng.uniform(12.0, 25.0))
    tip = {"condition": "good"}
    over: dict = {"surface": {"terrace_median_nm": terrace}}
    if template == "cu_fe":
        over["adatoms"] = {"field_per_100nm2": 0.05}
    for attempt in range(6):
        w, sc = make_world(template, _world_seed(r, attempt), ctx.session(r, attempt), tip=tip, **over)
        b = Bench(w, sc)
        t_mid = b.now + 2.0 + 256 * 2 * max(0.3, size / 400) / 2
        o = b.offset_nm(t_mid)
        chosen = None
        for _ in range(200):
            sx, sy = (float(v) for v in rng.uniform(-200.0, 200.0, 2))
            if min(abs(sx), abs(sy)) < 30 and template == "cu_fe":
                continue                       # keep away from the placed target + bystanders
            g = np.linspace(-size / 2, size / 2, 9)
            levels = {b.terrace_level(sx + gx, sy + gy) for gx in g for gy in g}
            if 2 <= len(levels) <= 4:
                chosen = (sx - o[0], sy - o[1])
                break
        if chosen is None:
            continue
        rec = b.scan(round(chosen[0], 3), round(chosen[1], 3), size, 256)
        if not _no_tip_change(b):
            continue
        pts = None
        for _ in range(400):
            ua, va, ub, vb = (float(v) for v in rng.uniform(-size / 2 + 4, size / 2 - 4, 4))
            A = rec.uv_to_scan(ua, va)
            B = rec.uv_to_scan(ub, vb)
            sA, sB = rec.scan_to_sample(*A), rec.scan_to_sample(*B)
            if b.step_distance_nm(*sA, radius_nm=3.0, n=8) < 3.0 or b.step_distance_nm(*sB, radius_nm=3.0, n=8) < 3.0:
                continue
            lA, lB = b.terrace_level(*sA), b.terrace_level(*sB)
            if lA == lB or math.hypot(B[0] - A[0], B[1] - A[1]) < size / 4:
                continue
            pts = (A, B, lA, lB)
            break
        if pts is not None:
            break
    else:
        raise GenerationError("step: no frame with two points on different terraces")
    A, B, lA, lB = pts
    step_pm = float(b.w.surface.material.step_m) * 1e12
    dh = (lB - lA) * step_pm
    mat = _material(b)
    frames = [ctx.export(rec.path, r, "frame")]
    q = (f"Frame 1 is a constant-current STM topograph ({size:g} nm x {size:g} nm, 256 x 256 px) of {_a(mat)} "
         f"surface with monatomic steps. Point A = ({A[0]:.1f}, {A[1]:.1f}) and point B = ({B[0]:.1f}, {B[1]:.1f}) "
         f"are given in scan-frame nm. {COORD_NOTE} How much higher is the terrace under B than the terrace "
         f"under A, in pm? Give a negative number if B lies lower. Report only the terrace-level difference "
         f"(the sum of the step heights between the two terraces), not any overall slope of the image. "
         f"Answer with a number in pm.")
    prov = _base_prov(r, b, sc, template, attempt, tip=tip, overrides=over)
    prov.update({"frames": [_frame_prov(rec)], "sim_s": [rec.t_start], "points_scan_nm": [list(A), list(B)],
                 "levels": [lA, lB], "step_pm": step_pm})
    if ctx.overlays:
        _overlay_frame(ctx, r, "frame", rec, marks=[("o", list(A)), ("x", list(B))], text=f"dh={dh:.0f} pm")
    return [_item("step", "height", r, question=q, frames=frames, answer_format={"kind": "number", "unit": "pm"},
                  truth={"value": round(dh, 2), "n_steps": int(lB - lA), "step_pm": step_pm},
                  tolerance={"abs": 30.0, "rel": 0.1}, provenance=prov,
                  difficulty="hard" if abs(lB - lA) >= 3 else "normal")]


# ── mid-scan tip change ──────────────────────────────────────────────────────
def build_tipchange(r: Recipe, ctx: Ctx) -> list[dict]:
    sub, scan_dir = r.params["substrate"], r.params["scan_dir"]
    rng = _rng(r)
    template = "cu_fe" if sub == "cu" else "au_fe"
    size = 20.0 if sub == "cu" else 30.0
    tip = {"condition": "unstable", "condition_scale": float(rng.uniform(0.0, 0.3))}
    got = None
    for attempt in range(8):
        w, sc = make_world(template, _world_seed(r, attempt), ctx.session(r, attempt), tip=tip,
                           adatoms={"field_per_100nm2": 0.8 if sub == "cu" else 0.5})
        b = Bench(w, sc)
        try:
            cx, cy = _window_with_atoms(b, rng, size, min_atoms=1, max_atoms=20, step_free=True)
        except GenerationError:
            continue
        for k in range(6):
            rec = b.scan(cx, cy, size, 256, line_time_s=0.3, scan_dir=scan_dir)
            evs = [e for e in rec.tip_events if e["kind"] == "spontaneous_change"]
            if len(rec.tip_change_rows) == 1 and len(evs) == 1:
                kk = rec.tip_change_rows[0]
                dz = abs(float(evs[0]["detail"].get("dz_pm", 0.0)))
                if 12 <= kk <= rec.ny - 12 and dz >= 30.0:
                    got = (attempt, w, sc, b, rec, kk, evs[0])
                    break
            b.wait(5.0)
        if got:
            break
    if got is None:
        raise GenerationError("tipchange: no frame with exactly one clear tip change")
    attempt, w, sc, b, rec, kk, ev = got
    row = kk if rec.scan_dir == "down" else rec.ny - kk
    frames = [ctx.export(rec.path, r, "frame")]
    q = (f"Frame 1 (256 x 256 px) is a constant-current STM topograph during which the tip changed abruptly "
         f"once, so the image changes discontinuously from one scan line to the next. At which image row does "
         f"the discontinuity occur? Rows are numbered from 0 (top row of the image) to 255 (bottom row); give "
         f"the index of the first row below the discontinuity. Answer with an integer row index.")
    prov = _base_prov(r, b, sc, template, attempt, tip=tip)
    prov.update({"frames": [_frame_prov(rec)], "sim_s": [rec.t_start], "tip_event": ev,
                 "acq_row": kk, "scan_dir": rec.scan_dir})
    if ctx.overlays:
        _overlay_frame(ctx, r, "frame", rec, marks=[("row", row)], text=f"row={row} dz={ev['detail'].get('dz_pm', 0):.0f}pm")
    return [_item("tipchange", "row", r, question=q, frames=frames,
                  answer_format={"kind": "number", "unit": "row"},
                  truth={"value": int(row), "dz_pm": float(ev["detail"].get("dz_pm", 0.0)),
                         "scan_dir": rec.scan_dir},
                  tolerance={"abs": 2}, provenance=prov,
                  difficulty="hard" if abs(float(ev["detail"].get("dz_pm", 0.0))) < 50 else "normal")]


# ── atomic resolution ────────────────────────────────────────────────────────
def atomic_reference(path: str | Path, material: str) -> dict:
    """MAST's ``assess_atomic_phase`` on both Z directions of a saved frame (the B4 judge's
    rule) — the *reference* answer for an atomic-resolution item."""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames
    from mast.vision.atomic_phase import assess_atomic_phase
    from mast.vision.lattice_calibration import first_order_period_nm

    expected = first_order_period_nm(material)
    fr = sxm_oriented_frames(read_sxm(str(path)), "Z")
    out = {"expected_a_nm": expected, "nm_per_px": fr.get("nm_per_px")}
    for side in ("forward", "backward"):
        res = assess_atomic_phase(np.asarray(fr[side], float), nm_per_px=fr.get("nm_per_px"),
                                  expected_a_nm=expected)
        out[side] = {"passed": bool(res.passed), "reasons": list(res.reasons), "snr": float(res.snr),
                     "angular_concentration": float(res.angular_concentration),
                     "period_nm": res.period_nm, "period_fast_axis_nm": res.period_fast_axis_nm}
    return out


def build_atomic(r: Recipe, ctx: Ctx) -> list[dict]:
    want, sub = bool(r.params["want"]), r.params["substrate"]
    rng = _rng(r)
    template = "au_hb" if sub == "au" else "cu_fe"
    size = 5.0
    # MAST's judge flips between apex sigma 0.08 and 0.10 nm (Cu: 1.9 pm imaged corrugation passes,
    # 0.9 pm fails); the classes stay clear of that band so the picture and the metric agree
    sigma = float(rng.uniform(0.035, 0.065)) if want else float(rng.uniform(0.12, 0.25))
    tip = {"condition": "good", "apex_sigma_nm": sigma, "radius_nm": 1.0}
    over: dict = {}
    if template == "cu_fe":
        over["adatoms"] = {"field_per_100nm2": 0.3}
    for attempt in range(6):
        w, sc = make_world(template, _world_seed(r, attempt), ctx.session(r, attempt), tip=tip, **over)
        b = Bench(w, sc)
        t_mid = b.now + 2.0 + 256 * 0.3 / 2
        o = b.offset_nm(t_mid)
        spot = None
        for _ in range(300):
            sx, sy = (float(v) for v in rng.uniform(-60.0, 60.0, 2))
            if b.step_distance_nm(sx, sy, radius_nm=6.0, n=12) < 6.0:
                continue
            if any(math.hypot(a["x_nm"] - sx, a["y_nm"] - sy) < 6.0 for a in b.atoms()):
                continue
            spot = (sx - o[0], sy - o[1])
            break
        if spot is None:
            continue
        rec = b.scan(round(spot[0], 3), round(spot[1], 3), size, 256, line_time_s=0.15)
        if _no_tip_change(b):
            break
    else:
        raise GenerationError("atomic: no clean frame")
    mat = _material(b)
    ref = atomic_reference(rec.path, mat)
    truth_v = bool(ref["forward"]["passed"])
    tsnap = b.tip_initial
    period_m = float(b.w.surface.material.first_order_period_m)
    transfer = float(b.w.tip.atomic_transfer(period_m))
    physics = {"apex_sigma_nm": tsnap["apex_sigma_nm"], "transfer": transfer,
               "imaged_corrugation_pm": transfer * float(b.w.surface.material.corrugation_m) * 1e12,
               "z_noise_floor_pm": float(b.w.noise.z_floor_m) * 1e12}
    frames = [ctx.export(rec.path, r, "frame")]
    q = (f"Frame 1 is a small constant-current STM topograph ({size:g} nm x {size:g} nm, 256 x 256 px) of "
         f"{mat}. Does it resolve the atomic lattice of the surface, i.e. a periodic arrangement of individual "
         f"atoms (close-packed rows with the substrate's atomic spacing) clearly visible across the frame? "
         f"Answer true or false.")
    prov = _base_prov(r, b, sc, template, attempt, tip=tip, overrides=over)
    prov.update({"frames": [_frame_prov(rec)], "sim_s": [rec.t_start], "physics": physics})
    if ctx.overlays:
        _overlay_frame(ctx, r, "frame", rec, marks=[], text=f"atomic={truth_v} sigma={sigma:.3f}")
    agree = truth_v == want
    return [_item("atomic", "resolved", r, question=q, frames=frames, answer_format={"kind": "bool"},
                  truth={"value": truth_v, "reference": "mast.vision.atomic_phase.assess_atomic_phase on the "
                         "forward Z channel (expected period = first-order lattice period); a reference metric, "
                         "not ground truth", "judge": ref, "physics": physics, "recipe_intent": want},
                  tolerance={}, provenance=prov, difficulty="normal" if agree else "hard")]


BUILDERS: dict[str, Callable[[Recipe, Ctx], list[dict]]] = {
    "tip": build_tip, "adatoms": build_adatoms, "move": build_move, "drift": build_drift,
    "herringbone": build_herringbone, "step": build_step, "tipchange": build_tipchange,
    "atomic": build_atomic,
}


# ── overlays (for people, never for the model) ───────────────────────────────
def _overlay_frame(ctx: Ctx, r: Recipe, label: str, rec: FrameRec, *, marks: list, text: str = "") -> None:
    """Render a frame's Z (plane-flattened) with truth marks: ``("o", [x, y])`` circle,
    ``("x", [x, y])`` cross (optional point), ``("row", r)`` horizontal line, ``("line", angle_deg)``
    a direction through the centre."""
    from PIL import Image, ImageDraw

    from stmbench.human.render import array_png, frame_array

    z, _ = frame_array(rec.path, channel="Z", direction="forward", flatten_mode="plane")
    png, _, _ = array_png(z, clip_pct=0.5)
    import io

    im = Image.open(io.BytesIO(png)).convert("RGB")
    k = max(1, 768 // max(rec.nx, rec.ny))
    im = im.resize((rec.nx * k, rec.ny * k), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    for kind, val in marks:
        if kind in ("o", "x"):
            col, row = rec.scan_to_pixel(val[0], val[1])
            X, Y = (col + 0.5) * k, (row + 0.5) * k
            rr = max(6, int(0.45 / rec.nm_per_px * k))
            if kind == "o":
                dr.ellipse([X - rr, Y - rr, X + rr, Y + rr], outline=(255, 60, 60), width=2)
            else:
                dr.line([X - rr, Y - rr, X + rr, Y + rr], fill=(60, 160, 255), width=2)
                dr.line([X - rr, Y + rr, X + rr, Y - rr], fill=(60, 160, 255), width=2)
        elif kind == "row":
            Y = float(val) * k
            dr.line([0, Y, im.width, Y], fill=(255, 60, 60), width=1)
        elif kind == "line":
            a = math.radians(float(val))
            cxp, cyp = im.width / 2, im.height / 2
            L = im.width * 0.45
            dr.line([cxp - L * math.cos(a), cyp + L * math.sin(a), cxp + L * math.cos(a), cyp - L * math.sin(a)],
                    fill=(255, 60, 60), width=2)
    if text:
        dr.rectangle([0, 0, 8 * len(text) + 8, 16], fill=(0, 0, 0))
        dr.text((4, 2), text, fill=(255, 255, 0))
    out = ctx.out / "_overlays" / f"{r.type}_s{r.seed}_i{r.index}_{label}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out)


def _overlay_spectrum(ctx: Ctx, r: Recipe, label: str, spec) -> None:
    from stmbench.human.render import render_dat

    png = render_dat(spec.path, y="LI Demod 1 X (A)")
    out = ctx.out / "_overlays" / f"{r.type}_s{r.seed}_i{r.index}_{label}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(png)


# ── running ──────────────────────────────────────────────────────────────────
def run_recipe(recipe: dict, out: str, keep_work: bool = False, overlays: bool = False) -> dict:
    """Process-pool entry point: one recipe → ``{"recipe", "items", "error", "wall_s"}``."""
    r = Recipe(**recipe)
    ctx = Ctx(out=Path(out), keep_work=keep_work, overlays=overlays)
    t0 = time.perf_counter()
    try:
        its = BUILDERS[r.type](r, ctx)
        err = None
    except Exception as exc:  # noqa: BLE001 — reported in the manifest, never silently dropped
        its = []
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    finally:
        if not keep_work:
            _clean_recipe_work(ctx, r)
    return {"recipe": r.as_dict(), "items": its, "error": err, "wall_s": time.perf_counter() - t0}


def _clean_recipe_work(ctx: Ctx, r: Recipe) -> None:
    for attempt in range(12):
        d = ctx.session(r, attempt).parent
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def generate(out: str | Path, types: list[str], seeds: list[int], *, jobs: int = 1,
             keep_work: bool = False, overlays: bool = False, log: Callable[[str], None] | None = None) -> dict:
    """Build the dataset; returns the manifest (also written to ``<out>/manifest.json``)."""
    out = Path(out)
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    if (out / "items.jsonl").exists():
        raise FileExistsError(f"{out / 'items.jsonl'} exists; choose a new --out (datasets are never overwritten)")
    out.mkdir(parents=True, exist_ok=True)
    recipes = plan(types, seeds)
    t0 = time.perf_counter()
    results: list[dict] = []
    if jobs <= 1:
        for k, r in enumerate(recipes, 1):
            res = run_recipe(r.as_dict(), str(out), keep_work, overlays)
            results.append(res)
            log(f"[{k}/{len(recipes)}] {r.key} -> {len(res['items'])} item(s){' ERROR' if res['error'] else ''}"
                f" ({res['wall_s']:.1f} s)")
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=int(jobs)) as ex:
            futs = {ex.submit(run_recipe, r.as_dict(), str(out), keep_work, overlays): r for r in recipes}
            for k, fut in enumerate(as_completed(futs), 1):
                r = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:  # noqa: BLE001 — a crashed worker is a failure, not a hang
                    res = {"recipe": r.as_dict(), "items": [], "error": f"worker crashed: {exc!r}", "wall_s": 0.0}
                results.append(res)
                log(f"[{k}/{len(recipes)}] {r.key} -> {len(res['items'])} item(s)"
                    f"{' ERROR' if res['error'] else ''} ({res['wall_s']:.1f} s)")
    if not keep_work:
        shutil.rmtree(out / "_work", ignore_errors=True)
    order = {t: k for k, t in enumerate(I.ITEM_TYPES)}
    results.sort(key=lambda x: (order[x["recipe"]["type"]], x["recipe"]["seed"], x["recipe"]["index"]))
    # file order: by type, then by the neutral id - the recipe order (e.g. the tip classes in a
    # fixed sequence per seed) must not be readable from an item's position
    all_items = sorted((it for res in results for it in res["items"]),
                       key=lambda it: (order[it["type"]], it["id"]))
    ids = Counter(it["id"] for it in all_items)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        raise RuntimeError(f"duplicate item ids: {dup[:5]}")
    n = I.write_jsonl(out / "items.jsonl", all_items)
    wall = time.perf_counter() - t0
    manifest = build_manifest(all_items, results, types=types, seeds=seeds, wall_s=wall, jobs=jobs)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {n} items to {out / 'items.jsonl'} in {wall:.1f} s; failures: {len(manifest['failures'])}")
    return manifest


def build_manifest(all_items: list[dict], results: list[dict], *, types, seeds, wall_s: float, jobs: int) -> dict:
    by_sub = Counter(f"{it['type']}.{it['subtype']}" for it in all_items)
    balance: dict[str, dict] = {}
    for it in all_items:
        key = f"{it['type']}.{it['subtype']}"
        kind = it["answer_format"]["kind"]
        if kind in ("choice", "bool"):
            balance.setdefault(key, Counter())[str(it["truth"]["value"])] += 1
        elif kind == "count":
            balance.setdefault(key, Counter())[f"n={it['truth']['value']}"] += 1
    diff = Counter(f"{it['type']}.{it.get('difficulty', 'normal')}" for it in all_items)
    try:
        import stmsim
        sim_v = getattr(stmsim, "__version__", None)
    except Exception:  # noqa: BLE001
        sim_v = None
    return {"schema": I.SCHEMA_VERSION, "generator": GENERATOR_VERSION, "types": list(types),
            "seeds": list(seeds), "n_items": len(all_items), "items_per_subtype": dict(sorted(by_sub.items())),
            "class_balance": {k: dict(sorted(v.items())) for k, v in sorted(balance.items())},
            "difficulty": dict(sorted(diff.items())),
            "n_frames": len({f for it in all_items for f in it["frames"]}),
            "failures": [{"recipe": res["recipe"], "error": res["error"]} for res in results if res["error"]],
            "recipe_wall_s": {res["recipe"]["type"] + f"/{res['recipe']['seed']}/{res['recipe']['index']}":
                              round(res["wall_s"], 2) for res in results},
            "wall_s": round(wall_s, 1), "jobs": int(jobs), "sim_version": sim_v,
            "python": sys.version.split()[0], "platform": sys.platform}
