"""Driving the simulator for static frames: deterministic worlds, acquisitions, truth geometry.

README
------
The generator does not go through MAST. It builds the same ``stmsim`` world a scenario
builds (``Scenario.build_world``), freezes the world clock, and calls the world's own verbs —
the ones the controller modules call when MAST sends ``Scan.FrameSet`` / ``Scan.Action`` /
``Scan.Save`` / ``BiasSpectr.Start`` / ``FolMe.XYPosSet`` — advancing instrument time
explicitly. The frames and spectra are therefore written by the same renderer and the same
``.sxm`` / ``.dat`` writers an episode uses, and generation is **deterministic** (a MAST-driven
run is not: sim time there follows the wall clock of the polling loop).

Truth geometry. The renderer images sample point ``scan_xy(pixel) + drift_offset(t_row)``
with one offset per acquired row, so a feature at sample position ``a`` appears in the scan
frame at ``p`` with ``p + o(t(p)) = a``. :meth:`FrameRec.sample_to_scan` solves that with the
per-row offsets recorded right after the frame was rendered (sub-pixel; validated against
blob centroids to < 0.06 nm). ``stmbench.trackB.claims.to_sample_frame`` uses one offset
per frame (the start); across a 150 s frame the drift moves 1-3 nm, so the per-row solution
is what an item's truth must use.

Nothing here reads anything the generated question exposes; the truth is the world's.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import stmbench

_SCENARIO_DIR = Path(stmbench.__file__).resolve().parent / "trackB" / "scenarios"

#: base scenario files the templates start from (read, never modified)
_BASE = {"cu": "P4_atom_positioning_cu111.yaml", "au": "P1_barth1990_au111.yaml"}

#: templates: (base, has adatoms, description)
TEMPLATES = {
    "cu_fe": ("cu", True, "Cu(111) with a sparse field of Fe adatoms (P4 physics)"),
    "au_hb": ("au", False, "Au(111) herringbone with rotational domains (P1 physics)"),
    "au_fe": ("au", True, "Au(111) herringbone plus a sparse field of adatoms"),
}

SPECIES = {"cu_fe": "Fe", "au_fe": "Fe"}


def _yaml(name: str) -> dict:
    import yaml

    return yaml.safe_load((_SCENARIO_DIR / name).read_text(encoding="utf-8"))


def template_dict(template: str) -> dict:
    """The scenario dict a template starts from (paper blocks and the hidden tip draw removed)."""
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template!r} (known: {sorted(TEMPLATES)})")
    base, with_adatoms, _ = TEMPLATES[template]
    d = copy.deepcopy(_yaml(_BASE[base]))
    for k in ("paper", "claims", "notes"):
        d.pop(k, None)
    d["id"] = f"SB_{template}"
    d["family"] = "SB"
    d["task"] = "staticbench frame generation (no task)"
    d["success"] = {"kind": "tip_repaired"}
    hidden = dict(d.get("hidden") or {})
    hidden.pop("tip", None)                  # the tip condition is forced per recipe
    ini = d.setdefault("initial", {})
    if with_adatoms and "adatoms" not in ini:
        p4 = _yaml(_BASE["cu"])
        ini["adatoms"] = copy.deepcopy(p4["initial"]["adatoms"])
        hidden["adatoms"] = copy.deepcopy(p4["hidden"]["adatoms"])
    if not with_adatoms:
        ini.pop("adatoms", None)
        hidden.pop("adatoms", None)
    if "adatoms" in ini:
        ini["adatoms"]["field_half_nm"] = 90.0     # frames stay within +-60 nm of the origin
    d["hidden"] = hidden
    return d


def make_scenario(template: str, *, tip: dict | None = None, surface: dict | None = None,
                  adatoms: dict | None = None, initial: dict | None = None):
    from stmsim.scenario import Scenario

    d = template_dict(template)
    ini = d["initial"]
    ini.setdefault("tip", {}).update(dict(tip or {}))
    ini.setdefault("surface", {}).update(dict(surface or {}))
    if adatoms:
        if "adatoms" not in ini:
            raise ValueError(f"template {template!r} has no adatoms block")
        ini["adatoms"].update(dict(adatoms))
    ini.update(dict(initial or {}))
    known = set(Scenario.__dataclass_fields__)
    sc = Scenario(**{k: v for k, v in d.items() if k in known})
    sc.validate()
    return sc


def freeze_clock(world, t0_sim: float) -> None:
    """Stop the world's clock and put instrument time at exactly ``t0_sim``.

    ``wall()`` is pinned to 0 (the private ``_t0`` is moved onto the pause instant), so the
    sim and slow clocks read exactly their offsets from now on and only
    :meth:`Bench.wait` moves them — the frames do not depend on how fast this machine is."""
    c = world.clock
    c.pause()
    c._t0 = c._pause_started          # noqa: SLF001 — wall() == 0.0 exactly while paused
    c._paused_total = 0.0             # noqa: SLF001
    c._sim_offset = float(t0_sim)     # noqa: SLF001
    c._slow_offset = float(t0_sim) * float(c.slow_scale) / float(c.time_scale)   # noqa: SLF001
    world.transients.clear()


#: what ``time.monotonic`` reads while a world is built (any constant: only differences matter)
_BUILD_MONOTONIC = 1000.0


def make_world(template: str, world_seed: int, session_dir: str | Path, *, t0_sim: float = 600.0,
               **overrides):
    """``(world, scenario)`` with the clock frozen at ``t0_sim``.

    The world's clock starts running inside ``World.__init__`` and ``build_world`` reads it
    (the coarse gap is set from the surface height under the drift offset *now*), so the build
    itself runs with ``time.monotonic`` held constant: the world is born at sim time 0 whatever
    this machine's speed, and frames come out bit-identical run to run."""
    import time as _time
    from unittest import mock

    sc = make_scenario(template, **overrides)
    with mock.patch.object(_time, "monotonic", lambda: _BUILD_MONOTONIC):
        w = sc.build_world(int(world_seed), session_dir=Path(session_dir))
        freeze_clock(w, t0_sim)
    return w, sc


# ── records ──────────────────────────────────────────────────────────────────
@dataclass
class FrameRec:
    """One saved frame and what the truth needs about it."""
    path: Path
    cx_nm: float
    cy_nm: float
    w_nm: float
    h_nm: float
    angle_deg: float
    nx: int
    ny: int
    line_time_s: float
    scan_dir: str
    t_start: float
    per_row_s: float
    offsets_nm: np.ndarray                    # (ny, 2): drift offset at each ACQUISITION row
    tip_change_rows: list[int]                # acquisition rows whose render drew a tip change
    tip_events: list[dict]
    hops: list[dict]                          # adatom events while the frame was rendered
    bias_v: float
    setpoint_a: float
    atoms: list[dict] = field(default_factory=list)   # on-surface adatoms right after the frame
    tip_start: dict = field(default_factory=dict)     # the tip as the frame's first row met it

    # geometry ----------------------------------------------------------------
    @property
    def nm_per_px(self) -> float:
        return self.w_nm / self.nx

    @property
    def t_end(self) -> float:
        return self.t_start + self.per_row_s * self.ny

    def t_mid(self) -> float:
        return self.t_start + self.per_row_s * (self.ny - 1) / 2.0

    def geometry(self) -> dict:
        return {"cx_nm": self.cx_nm, "cy_nm": self.cy_nm, "w_nm": self.w_nm, "h_nm": self.h_nm,
                "angle_deg": self.angle_deg, "nx": self.nx, "ny": self.ny,
                "nm_per_px": self.nm_per_px, "line_time_s": self.line_time_s,
                "scan_dir": self.scan_dir, "t_start_sim_s": self.t_start,
                "frame_time_s": self.per_row_s * self.ny, "bias_v": self.bias_v,
                "setpoint_a": self.setpoint_a}

    def _rot(self) -> tuple[float, float]:
        a = math.radians(self.angle_deg)
        return math.cos(a), math.sin(a)

    def scan_to_uv(self, x: float, y: float) -> tuple[float, float]:
        c, s = self._rot()
        dx, dy = x - self.cx_nm, y - self.cy_nm
        return c * dx + s * dy, -s * dx + c * dy

    def uv_to_scan(self, u: float, v: float) -> tuple[float, float]:
        c, s = self._rot()
        return self.cx_nm + c * u - s * v, self.cy_nm + s * u + c * v

    def scan_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Scan nm → (col, row) of the oriented image, pixel centres at integers, row 0 = top."""
        u, v = self.scan_to_uv(x, y)
        return ((u + self.w_nm / 2) * self.nx / self.w_nm - 0.5,
                (self.h_nm / 2 - v) * self.ny / self.h_nm - 0.5)

    def pixel_to_scan(self, col: float, row: float) -> tuple[float, float]:
        u = -self.w_nm / 2 + (col + 0.5) * self.w_nm / self.nx
        v = self.h_nm / 2 - (row + 0.5) * self.h_nm / self.ny
        return self.uv_to_scan(u, v)

    def acq_of_row(self, row: float) -> float:
        """Geometric row (0 = top) → acquisition index (continuous)."""
        return row if self.scan_dir == "down" else (self.ny - 1) - row

    def row_of_acq(self, k: float) -> float:
        return k if self.scan_dir == "down" else (self.ny - 1) - k

    def offset_at_row(self, row: float) -> tuple[float, float]:
        k = min(max(self.acq_of_row(row), 0.0), self.ny - 1.0)
        ks = np.arange(self.ny, dtype=float)
        return (float(np.interp(k, ks, self.offsets_nm[:, 0])),
                float(np.interp(k, ks, self.offsets_nm[:, 1])))

    def time_at_row(self, row: float) -> float:
        k = min(max(self.acq_of_row(row), 0.0), self.ny - 1.0)
        return self.t_start + self.per_row_s * k

    def sample_to_scan(self, ax: float, ay: float) -> tuple[float, float, float]:
        """Sample-frame nm → (x, y, row): where the feature appears in this frame's scan
        coordinates, and the (continuous, 0 = top) row that imaged it."""
        ox, oy = self.offsets_nm[self.ny // 2]
        row = 0.0
        for _ in range(12):
            x, y = ax - ox, ay - oy
            _, row = self.scan_to_pixel(x, y)
            nox, noy = self.offset_at_row(row)
            if abs(nox - ox) < 1e-7 and abs(noy - oy) < 1e-7:
                ox, oy = nox, noy
                break
            ox, oy = nox, noy
        x, y = ax - ox, ay - oy
        _, row = self.scan_to_pixel(x, y)
        return float(x), float(y), float(row)

    def scan_to_sample(self, x: float, y: float) -> tuple[float, float]:
        _, row = self.scan_to_pixel(x, y)
        ox, oy = self.offset_at_row(row)
        return float(x + ox), float(y + oy)

    def inside(self, x: float, y: float, margin_nm: float = 0.0) -> bool:
        """True when the scan point lies inside the frame by at least ``margin_nm``
        (negative margin = allowed outside)."""
        u, v = self.scan_to_uv(x, y)
        return (abs(u) <= self.w_nm / 2 - margin_nm) and (abs(v) <= self.h_nm / 2 - margin_nm)

    def drift_gradient(self) -> tuple[float, float, float]:
        """``(gx, gy, nonlinearity_nm)``: the slope of the stage-frame drift offset against the
        image v coordinate (nm/nm) from a straight-line fit over the rows, and the largest
        residual of that fit (creep makes it curve)."""
        rows = np.arange(self.ny, dtype=float)
        v = self.h_nm / 2 - (rows + 0.5) * self.h_nm / self.ny
        k = np.array([self.acq_of_row(r) for r in rows]).astype(int)
        ox, oy = self.offsets_nm[k, 0], self.offsets_nm[k, 1]
        px = np.polyfit(v, ox, 1)
        py = np.polyfit(v, oy, 1)
        res = max(float(np.max(np.abs(np.polyval(px, v) - ox))), float(np.max(np.abs(np.polyval(py, v) - oy))))
        return float(px[0]), float(py[0]), res

    def summary(self) -> dict:
        return {**self.geometry(), "tip_change_rows": list(self.tip_change_rows),
                "n_tip_events": len(self.tip_events), "n_adatom_events": len(self.hops),
                "drift_first_row_nm": [float(v) for v in self.offsets_nm[0]],
                "drift_last_row_nm": [float(v) for v in self.offsets_nm[-1]]}


@dataclass
class SpecRec:
    path: Path
    x_nm: float
    y_nm: float
    sample_xy_nm: tuple[float, float]
    v0: float
    v1: float
    n: int
    jump: bool
    phi_ev: float
    sim_s: float
    stab_bias_v: float
    stab_setpoint_a: float
    near_kind: str = ""
    near_nm: float = float("nan")

    def summary(self) -> dict:
        return {"x_nm": self.x_nm, "y_nm": self.y_nm, "v0": self.v0, "v1": self.v1, "n": self.n,
                "jump": bool(self.jump), "phi_ev": self.phi_ev, "sim_s": self.sim_s,
                "stab_bias_v": self.stab_bias_v, "stab_setpoint_a": self.stab_setpoint_a,
                "near_kind": self.near_kind, "near_nm": self.near_nm}


# ── the bench ────────────────────────────────────────────────────────────────
class Bench:
    """One world and the acquisitions made on it, in instrument time."""

    #: fast-axis speed above which the simulator may scratch the surface with a blunt tip
    MAX_SPEED_NM_S = 400.0

    def __init__(self, world, scenario=None):
        self.w = world
        self.scenario = scenario
        self.frames: list[FrameRec] = []
        self.specs: list[SpecRec] = []
        self.tip_initial = tip_summary(world)

    @property
    def now(self) -> float:
        return float(self.w.clock.sim())

    def wait(self, seconds: float) -> None:
        if seconds > 0:
            self.w.clock.advance_sim(float(seconds))

    def offset_nm(self, t_sim: float | None = None) -> tuple[float, float]:
        ox, oy, _ = self.w.drift_offset(self.now if t_sim is None else float(t_sim))
        return ox * 1e9, oy * 1e9

    # adatoms ------------------------------------------------------------------
    def registry(self):
        return self.w.surface.site.adatoms

    def atoms(self) -> list[dict]:
        reg = self.registry()
        if reg is None:
            return []
        out = []
        for a in reg.atoms.values():
            if a.status != "on_surface":
                continue
            x, y = reg.site_xy(a.i, a.j, a.sub)
            out.append({"id": int(a.id), "x_nm": x * 1e9, "y_nm": y * 1e9, "i": int(a.i), "j": int(a.j),
                        "role": a.role})
        return out

    def atom_lost_or_carried(self) -> bool:
        reg = self.registry()
        return reg is not None and any(a.status != "on_surface" for a in reg.atoms.values())

    # acquisitions -------------------------------------------------------------
    def scan(self, cx_nm: float, cy_nm: float, size_nm: float, px: int, *, angle_deg: float = 0.0,
             line_time_s: float | None = None, scan_dir: str = "down", settle_s: float = 2.0) -> FrameRec:
        """Frame-set, settle, scan, save — what ``ScanAt`` + ``SaveScan`` make the controller do."""
        w = self.w
        st = w.scan
        if line_time_s is None:
            line_time_s = max(0.3, size_nm / self.MAX_SPEED_NM_S)
        st.cx, st.cy = cx_nm * 1e-9, cy_nm * 1e-9
        st.w = st.h = size_nm * 1e-9
        st.angle_deg = float(angle_deg)
        st.nx = st.ny = int(px)
        st.line_time_fwd_s = st.line_time_bwd_s = float(line_time_s)
        st.series_name = "sim"
        st.comment = ""
        st.continuous = st.autosave = False
        w.move_xy(st.cx, st.cy)
        self.wait(settle_s)
        n_tip0 = len(w.tip.events)
        n_ev0 = len(w.events)
        tip_start = tip_summary(w)
        w.scan_start(direction_up=(scan_dir == "up"))
        fr = w.frame
        offs = np.array([w.drift_offset(float(t))[:2] for t in fr.row_times_sim], dtype=float) * 1e9
        tip_events = []
        per_row = st.line_time_fwd_s + st.line_time_bwd_s
        for k, e in enumerate(w.tip.events[n_tip0:]):
            tip_events.append({"kind": e.kind, "sim_s": float(e.sim_s), "detail": _plain(e.detail),
                               "acq_row": int(round((float(e.sim_s) - fr.t_start_sim) / per_row))})
        hops = [_plain(e) for e in w.events[n_ev0:] if e.get("kind") in ("adatom_hop", "adatom_picked")]
        self.wait(st.frame_time_s + 0.5)
        path = w.save_frame()
        rec = FrameRec(path=Path(path), cx_nm=cx_nm, cy_nm=cy_nm, w_nm=size_nm, h_nm=size_nm,
                       angle_deg=float(angle_deg), nx=int(px), ny=int(px), line_time_s=float(line_time_s),
                       scan_dir=scan_dir, t_start=float(fr.t_start_sim), per_row_s=float(per_row),
                       offsets_nm=offs, tip_change_rows=[int(r) for r in fr.tip_change_rows],
                       tip_events=tip_events, hops=hops, bias_v=float(w.bias_v),
                       setpoint_a=float(w.zctrl.setpoint_a), atoms=self.atoms(), tip_start=tip_start)
        self.frames.append(rec)
        return rec

    def spectrum(self, x_nm: float, y_nm: float, *, v0: float = -1.0, v1: float = 1.0, n: int = 201,
                 stab_bias_v: float = 0.5, stab_setpoint_a: float = 0.5e-9,
                 lockin_amp_v: float = 0.01) -> SpecRec:
        """A dI/dV point spectrum at a scan-frame point (``BiasSpectr.Start`` with autosave)."""
        from stmsim.io.dat_writer import save_bias_spectrum

        w = self.w
        bias0, sp0 = w.bias_v, w.zctrl.setpoint_a
        w.move_xy(x_nm * 1e-9, y_nm * 1e-9)
        w.set_bias(stab_bias_v)
        w.set_setpoint(stab_setpoint_a)
        w.transients.clear()
        w.lockin_on = True
        w.lockin_amp_v = float(lockin_amp_v)
        self.wait(1.0)
        cur = w.sts_curve(v0, v1, int(n), settle_s=0.005, integ_s=0.005)
        path = save_bias_spectrum(w, "spec", v=cur["v"], i=cur["i"], didv=cur["didv"])
        w.dats_saved.append(path)
        sim_s = self.now
        self.wait(int(n) * 0.02 + 0.5)
        w.set_setpoint(sp0)
        w.set_bias(bias0)
        w.transients.clear()
        w.lockin_on = False
        rec = SpecRec(path=Path(path), x_nm=float(x_nm), y_nm=float(y_nm),
                      sample_xy_nm=(float(cur["sx_nm"]), float(cur["sy_nm"])), v0=float(v0), v1=float(v1),
                      n=int(n), jump=bool(cur["jump"]), phi_ev=float(cur["phi_ev"]), sim_s=sim_s,
                      stab_bias_v=float(stab_bias_v), stab_setpoint_a=float(stab_setpoint_a),
                      near_kind=str(cur.get("near_kind", "")), near_nm=float(cur.get("near_nm", float("nan"))))
        self.specs.append(rec)
        return rec

    def drag_atom(self, atom_id: int, target_sample_nm: tuple[float, float], *, r_kohm: float,
                  speed_nm_s: float = 0.2, approach_offset_nm: float = 0.3,
                  manip_bias_v: float = 0.01, waypoint_nm: float = 0.1) -> dict:
        """``MoveAtomTo`` without the verify scan: start behind the atom, lower the junction
        resistance, drag along waypoints to the target at ``speed_nm_s``, raise the setpoint
        back before leaving. Coordinates are fixed in the scan frame at the start (as MAST does)."""
        w = self.w
        reg = self.registry()
        atom = reg.atoms[int(atom_id)]
        ax, ay = (v * 1e9 for v in reg.site_xy(atom.i, atom.j, atom.sub))
        ox, oy = self.offset_nm()
        a_scan = np.array([ax - ox, ay - oy])
        t_scan = np.array([target_sample_nm[0] - ox, target_sample_nm[1] - oy])
        d = t_scan - a_scan
        L = float(np.hypot(*d))
        u = d / L if L > 0 else np.array([1.0, 0.0])
        start = a_scan - u * approach_offset_nm
        bias0, sp0 = w.bias_v, w.zctrl.setpoint_a
        w.move_xy_path(start[0] * 1e-9, start[1] * 1e-9)          # imaging resistance: no drag
        self.wait(1.0)
        sign = 1.0 if bias0 >= 0 else -1.0
        w.set_bias(sign * manip_bias_v)
        w.set_setpoint(manip_bias_v / (r_kohm * 1e3))
        w.transients.clear()
        events: list[dict] = []
        path_len = float(np.hypot(*(t_scan - start)))
        n = max(1, int(math.ceil(path_len / waypoint_nm)))
        prev = start
        for k in range(1, n + 1):
            p = start + (t_scan - start) * (k / n)
            evs = w.move_xy_path(p[0] * 1e-9, p[1] * 1e-9, speed_m_s=speed_nm_s * 1e-9)
            events += [_plain(e) for e in evs]
            self.wait(float(np.hypot(*(p - prev))) / speed_nm_s)
            prev = p
        w.set_setpoint(sp0)
        self.wait(0.5)
        w.set_bias(bias0)
        w.transients.clear()
        return {"atom_id": int(atom_id), "r_kohm": float(r_kohm), "speed_nm_s": float(speed_nm_s),
                "start_scan_nm": [float(v) for v in start], "target_scan_nm": [float(v) for v in t_scan],
                "events": events}

    # truth helpers ------------------------------------------------------------
    def effective_thresholds(self, atom_id: int) -> tuple[float, float]:
        """(pull, pick-up) junction resistance thresholds in kOhm for this atom and this tip
        as the simulator's drag model will apply them (generator privilege)."""
        reg = self.registry()
        atom = reg.atoms[int(atom_id)]
        grip = self.w.grip()
        hold = reg.hold_factor(atom, self.w.surface.terrace_level)
        r_th = reg.params.r_threshold_ohm * grip.threshold_scale * hold
        r_pick = reg.params.r_pick_ohm * grip.threshold_scale * atom.r_scale
        return r_th / 1e3, r_pick / 1e3

    def terrace_level(self, x_nm: float, y_nm: float) -> int:
        lv = self.w.surface.terrace_level(np.array([x_nm * 1e-9]), np.array([y_nm * 1e-9]))
        return int(lv[0])

    def step_distance_nm(self, x_nm: float, y_nm: float, radius_nm: float = 6.0, n: int = 25) -> float:
        """Distance (nm, sample frame) to the nearest point on another terrace, by a polar
        search out to ``radius_nm`` (inf when none)."""
        lv0 = self.terrace_level(x_nm, y_nm)
        best = float("inf")
        rs = np.linspace(radius_nm / n, radius_nm, n)
        th = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        R, T = np.meshgrid(rs, th)
        xs = (x_nm + R * np.cos(T)).ravel() * 1e-9
        ys = (y_nm + R * np.sin(T)).ravel() * 1e-9
        lv = self.w.surface.terrace_level(xs, ys)
        other = np.flatnonzero(lv != lv0)
        if other.size:
            best = float(np.min(R.ravel()[other]))
        return best

    def orientation_at(self, x_nm: float, y_nm: float) -> dict | None:
        return self.w.surface.orientation_at(x_nm * 1e-9, y_nm * 1e-9)

    def herringbone_params(self) -> dict:
        p = self.w.surface.herringbone
        return {"period_nm": p.period_m * 1e9, "shear": p.shear, "arm_half_angle_deg": p.arm_half_angle_deg,
                "chevron_period_nm": p.chevron_period_m * 1e9, "amp_pm": p.amp_m * 1e12,
                "period_along_arm_nm": p.period_m * 1e9 * math.cos(math.atan(p.shear)),
                "domain_spacing_nm": (p.domain_spacing_m * 1e9) if p.domain_spacing_m else None}

    def provenance(self) -> dict:
        w = self.w
        return {"hidden_draws": _plain(dict((w.hidden or {}).get("draws", {}))),
                "material": w.surface.material.name,
                "drift_nm_per_min": float(np.hypot(*w.drift_v_m_per_s[:2])) * 6e10,
                "tip_initial": self.tip_initial}


# ── truth computations ───────────────────────────────────────────────────────
def apparent_arms(rec: FrameRec, orient: dict, hb: dict) -> dict:
    """The herringbone arms as they appear in this frame's image (period nm, stripe angle in
    image degrees CCW from +x, mod 180), with the drift during the frame folded in.

    In the stage frame an arm is the plane wave with wavevector ``(2π/T)(e_u ± tanα e_v)``
    (``e_u`` the compression direction ``stripe - 90°``). The renderer images sample point
    ``c + R(θ)(u, v) + o(v)``; with ``o`` linear in ``v`` (slope ``g``) the image sees
    ``k' = R(θ)ᵀk + (0, k·g)``."""
    T = float(hb["period_nm"])
    tan_a = float(hb["shear"])
    th = math.radians(float(orient["stripe_deg"]) - 90.0)
    eu = np.array([math.cos(th), math.sin(th)])
    ev = np.array([-math.sin(th), math.cos(th)])
    gx, gy, nonlin = rec.drift_gradient()
    g = np.array([gx, gy])
    a = math.radians(rec.angle_deg)
    rt = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])     # R(θ)ᵀ
    arms = []
    for sgn in (+1.0, -1.0):
        k = (2 * math.pi / T) * (eu + sgn * tan_a * ev)
        kp = rt @ k + np.array([0.0, float(k @ g)])
        k0 = rt @ k
        arms.append({"sign": sgn,
                     "period_nm": float(2 * math.pi / np.linalg.norm(kp)),
                     "angle_deg": float((math.degrees(math.atan2(kp[1], kp[0])) + 90.0) % 180.0),
                     "period_undistorted_nm": float(2 * math.pi / np.linalg.norm(k0)),
                     "angle_undistorted_deg": float((math.degrees(math.atan2(k0[1], k0[0])) + 90.0) % 180.0)})
    a1, a2 = arms[0]["angle_deg"], arms[1]["angle_deg"]
    return {"arms": arms, "mean_angle_deg": circ_mean_180(a1, a2),
            "drift_gradient": [gx, gy], "drift_nonlinearity_nm": nonlin}


def circ_mean_180(a: float, b: float) -> float:
    """Mean of two axial angles (mod 180)."""
    d = ((b - a + 90.0) % 180.0) - 90.0
    return (a + d / 2.0) % 180.0


def angle_diff_180(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def tip_summary(world) -> dict:
    s = world.tip.snapshot()
    keep = ("apexes", "n_apex", "multi", "radius_nm", "apex_sigma_nm", "phi_ev", "lambda_per_s",
            "metastable", "flicker_dz_pm", "ldos", "carried", "length_nm")
    return _plain({k: s[k] for k in keep if k in s})


def _plain(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays → Python, tuples → lists, NaN → None."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
