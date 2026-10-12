"""MAST's own detectors as a tool (arm ``advisor``) and as a text observation (arm ``vigil-text``).

``advisor`` gives the model, on request, what MAST's perception stack says about an archived
scan frame: the classical tip-quality fusion, the double-tip detector, the mid-scan tip-change
detector, the atomic-lattice check, and the learned VIGIL heads when the model weights are
available. ``vigil-text`` replaces the image observation with a deterministic text rendering of
the same readings — the representation MAST fed its agents before this harness, run through the
same harness and model so the two representations can be compared directly.

These are another program's readings, labelled as such; nothing here consults the simulator.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from .protocol import ToolReply, compact_json, error_reply, json_reply


def _jsonish(x: Any, depth: int = 0) -> Any:
    if depth > 3:
        return None
    if isinstance(x, (bool, str)) or x is None:
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return None if not math.isfinite(v) else (int(v) if float(v).is_integer() and abs(v) < 1e12 else round(v, 6))
    if isinstance(x, dict):
        out = {str(k): _jsonish(v, depth + 1) for k, v in list(x.items())[:40]}
        return {k: v for k, v in out.items() if v is not None}
    if isinstance(x, (list, tuple)) and len(x) <= 12:
        return [_jsonish(v, depth + 1) for v in x]
    if hasattr(x, "model_dump"):
        return _jsonish(x.model_dump(), depth + 1)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonish(dataclasses.asdict(x), depth + 1)
    if hasattr(x, "__dict__"):
        return _jsonish({k: v for k, v in vars(x).items() if not k.startswith("_")}, depth + 1)
    return None


def _acquired(a: np.ndarray | None) -> np.ndarray | None:
    if a is None:
        return None
    a = np.asarray(a, dtype=float)
    rows = np.isfinite(a).all(axis=1)
    return a[rows] if rows.sum() >= 8 else None


def mast_readings(archive, fid: str) -> dict:
    """Every MAST detector that runs on this frame, keyed by detector; errors are reported."""
    entry = archive.get(fid)
    if getattr(entry, "kind", "s") not in ("s", "p"):
        raise ValueError(f"{fid} is not a scan frame")
    fwd = _acquired(archive.array(fid, "Z", "forward"))
    try:
        bwd = _acquired(archive.array(fid, "Z", "backward"))
    except Exception:  # noqa: BLE001
        bwd = None
    if fwd is None:
        raise ValueError(f"{fid}: fewer than 8 acquired rows")
    if bwd is not None and bwd.shape != fwd.shape:
        bwd = None
    nm = getattr(entry, "nm_per_px", None)
    out: dict = {}

    def run(key, fn):
        try:
            out[key] = _jsonish(fn())
        except Exception as exc:  # noqa: BLE001
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}

    from mast.vision.tip_quality import assess_tip_quality_classical
    from mast.vision.double_tip import detect_double_tip
    from mast.vision.tip_change import detect_tip_change
    from mast.vision.atomic_phase import assess_atomic_phase

    run("tip_quality_classical", lambda: assess_tip_quality_classical(fwd, bwd=bwd, nm_per_px=nm))
    run("double_tip", lambda: detect_double_tip(fwd, nm_per_px=nm))
    run("tip_change", lambda: detect_tip_change(np.stack([fwd, bwd]) if bwd is not None else fwd, nm_per_px=nm))
    run("atomic_lattice", lambda: assess_atomic_phase(fwd, nm_per_px=nm))
    try:
        from mast.vision.module import VisionModule

        vm = VisionModule.get()
        run("vigil_coarse", lambda: vm.assess_tip_coarse(fwd))
        run("vigil_fine", lambda: vm.assess_tip_fine(fwd))
        out["vision_backend"] = str(getattr(vm, "backend_name", None) or type(getattr(vm, "_backend", None)).__name__)
    except Exception as exc:  # noqa: BLE001
        out["vigil"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


ADVISOR_NOTE = ("Readings of MAST's detectors and learned vision model on this frame. They are another "
                "program's outputs, with known failure modes (e.g. no reliable double-tip detector), not "
                "ground truth.")


def advisor_reply(archive, args: dict) -> ToolReply:
    fid = args.get("frame")
    if not isinstance(fid, str):
        return error_reply("frame must be an archived scan frame id, e.g. t0007.s0")
    try:
        readings = mast_readings(archive, fid)
    except KeyError as exc:
        return error_reply(str(exc).strip("'\""))
    except Exception as exc:  # noqa: BLE001
        return error_reply(f"advisor could not run on {fid}: {type(exc).__name__}: {exc}")
    return json_reply({"frame": fid, "note": ADVISOR_NOTE, "readings": readings})


def vigil_summary(archive, fid: str) -> str:
    """The text observation of arm ``vigil-text``: frame facts + MAST's readings, as text.

    Deterministic (no model call), in the spirit of MAST's ``buffer_summarizer`` templates:
    labels, confidences and numbers, no adjectives the detectors did not produce."""
    entry = archive.get(fid)
    head = entry.summary() if hasattr(entry, "summary") else {"frame": fid}
    try:
        r = mast_readings(archive, fid)
    except Exception as exc:  # noqa: BLE001
        return f"frame {fid}: {compact_json(head)}; MAST readings unavailable ({type(exc).__name__}: {exc})"
    parts = [f"frame {fid}: {compact_json(head)}"]
    tq = r.get("tip_quality_classical") or {}
    if "error" not in tq:
        parts.append("classical tip quality: " + _kv(tq, ("label", "verdict", "good", "quality", "score",
                                                          "reasons", "sharpness", "instability")))
    vc = r.get("vigil_coarse") or {}
    if vc and "error" not in vc:
        parts.append("learned model (coarse): " + _kv(vc, ("label", "confidence", "tip_radius_nm")))
    vf = r.get("vigil_fine") or {}
    if vf and "error" not in vf:
        parts.append("learned model (fine): " + _kv(vf, ("label", "morph", "is_usable", "switching", "drift")))
    dt = r.get("double_tip") or {}
    if "error" not in dt:
        parts.append("double-tip detector (unreliable by design): " + _kv(dt, ("detected", "is_double", "score",
                                                                               "separation_nm", "vector_px")))
    tc = r.get("tip_change") or {}
    if "error" not in tc:
        parts.append("mid-scan tip change: " + _kv(tc, ("changed", "change_row", "score", "threshold")))
    al = r.get("atomic_lattice") or {}
    if "error" not in al:
        parts.append("atomic lattice check: " + _kv(al, ("passed", "period_fast_axis_nm", "snr",
                                                         "angular_concentration", "reasons")))
    return "\n".join(parts)


def _kv(d: dict, keys: tuple[str, ...]) -> str:
    items = [f"{k}={d[k]}" for k in keys if k in d and d[k] is not None]
    return ", ".join(items) if items else "(no fields)"
