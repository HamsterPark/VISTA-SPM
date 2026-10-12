"""Classical reference answers for STM-BabyVision items: a non-LLM row for the leaderboard.

README
------
``python -m vista_spm.staticbench answer <dataset> --mode baseline --out answers.jsonl``
answers every item from the frame files alone (never from ``truth`` or ``provenance``), with
plain image processing:

* adatoms — median-background subtraction, local maxima above a noise threshold, intensity
  centroids (sub-pixel), a compactness test that rejects step edges, and suppression of
  fainter copies that repeat at one common offset (a double tip's ghosts);
* tip — row-to-row jumps measured column-wise (a tip change shifts every column, a slanted
  step does not), a jump in the spectrum's current, a peak in dI/dV above a running-median
  baseline (a state on the tip), repeated ghost offsets (double), and the fast-axis/slow-axis
  width ratio of compact features (blunt: the simulator smears along the fast axis only);
* move / drift — phase correlation between the frames refined by matching detected atoms;
* herringbone — Hann-windowed FFT, the strongest peaks in the 4.5–9 nm band (one per arm);
* step — a line profile from A to B with the background slope taken as the median derivative;
* tipchange — the row with the largest column-median jump;
* atomic — FFT peak-to-background ratio in the lattice band (not MAST's judge, which is the truth).

Every answer is wrapped as ``{"value": ..., "abstain": false}``; a failure abstains with the
error in ``note`` (it is not hidden).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from . import items as I

# ── frames ───────────────────────────────────────────────────────────────────


def load_scan(path: str | Path) -> dict:
    """Oriented Z (and Current) forward/backward arrays in SI plus the frame geometry."""
    from mast.io.nanonis_files import read_sxm, sxm_oriented_frames

    from stmbench.human.render import frame_geometry

    scan = read_sxm(str(path))
    z = sxm_oriented_frames(scan, "Z")
    geom = frame_geometry(scan.get("header") or {})
    if geom is None or z.get("forward") is None:
        raise ValueError(f"{path}: no Z frame or no geometry")
    nmpp = float(geom["w_nm"]) / int(geom["nx"])
    return {"z": np.asarray(z["forward"], float), "z_bwd": (np.asarray(z["backward"], float)
                                                             if z.get("backward") is not None else None),
            "geom": geom, "nm_per_px": nmpp}


def load_spectrum(path: str | Path) -> dict:
    from mast.io.nanonis_files import read_dat

    d = read_dat(str(path))
    cols = {k: np.asarray(v, float) for k, v in (d.get("columns") or {}).items()}
    names = list(cols)
    if not names:
        raise ValueError(f"{path}: no data columns")
    v = cols[names[0]]
    cur = next((cols[n] for n in names if n.lower().startswith("current") and "bwd" not in n.lower()), None)
    li = next((cols[n] for n in names if "demod" in n.lower() or "lix" in n.lower().replace(" ", "")), None)
    return {"v": v, "i": cur, "didv": li, "columns": names}


def pix_to_scan(geom: dict, col: float, row: float) -> tuple[float, float]:
    """Oriented-array pixel (centres at integers, row 0 = top) → scan-frame nm."""
    w, h = float(geom["w_nm"]), float(geom["h_nm"])
    nx, ny = int(geom["nx"]), int(geom["ny"])
    u = -w / 2 + (col + 0.5) * w / nx
    v = h / 2 - (row + 0.5) * h / ny
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    return float(geom["cx_nm"]) + math.cos(a) * u - math.sin(a) * v, float(geom["cy_nm"]) + math.sin(a) * u + math.cos(a) * v


def scan_to_pix(geom: dict, x: float, y: float) -> tuple[float, float]:
    w, h = float(geom["w_nm"]), float(geom["h_nm"])
    nx, ny = int(geom["nx"]), int(geom["ny"])
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    dx, dy = x - float(geom["cx_nm"]), y - float(geom["cy_nm"])
    u = math.cos(a) * dx + math.sin(a) * dy
    v = -math.sin(a) * dx + math.cos(a) * dy
    return (u + w / 2) * nx / w - 0.5, (h / 2 - v) * ny / h - 0.5


def img_vec_to_scan(geom: dict, du_nm: float, dv_nm: float) -> tuple[float, float]:
    """A displacement along the image axes (u right, v up) → scan-frame components."""
    a = math.radians(float(geom.get("angle_deg", 0.0) or 0.0))
    return math.cos(a) * du_nm - math.sin(a) * dv_nm, math.sin(a) * du_nm + math.cos(a) * dv_nm


def _fill(z: np.ndarray) -> np.ndarray:
    z = np.array(z, dtype=float)
    ok = np.isfinite(z)
    if not ok.all():
        z[~ok] = float(np.nanmedian(z)) if ok.any() else 0.0
    return z


def plane(z: np.ndarray) -> np.ndarray:
    z = _fill(z)
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = np.column_stack([xx.ravel(), yy.ravel(), np.ones(z.size)])
    coef, *_ = np.linalg.lstsq(a, z.ravel(), rcond=None)
    return z - (coef[0] * xx + coef[1] * yy + coef[2])


def _odd(n: float) -> int:
    k = max(3, int(round(n)))
    return k if k % 2 else k + 1


def background(z: np.ndarray, nmpp: float, scale_nm: float = 2.5) -> np.ndarray:
    """Median background over ``scale_nm`` (edge preserving: steps stay steps), computed on a
    2x-downsampled copy when the kernel is large."""
    from scipy import ndimage

    k = scale_nm / nmpp
    if k > 24 and min(z.shape) >= 64:
        small = z[::2, ::2]
        bg = ndimage.median_filter(small, size=_odd(k / 2), mode="nearest")
        bg = ndimage.zoom(bg, (z.shape[0] / bg.shape[0], z.shape[1] / bg.shape[1]), order=1)
        return bg[: z.shape[0], : z.shape[1]]
    return ndimage.median_filter(z, size=_odd(k), mode="nearest")


# ── compact features ─────────────────────────────────────────────────────────
def detect_blobs(z: np.ndarray, nmpp: float, *, min_h_m: float = 20e-12, k_sigma: float = 6.0,
                 sep_nm: float = 0.6, r_nm: float = 0.35, compact_frac: float = 0.55,
                 bg_nm: float = 1.6) -> list[dict]:
    """Compact protrusions: ``[{col, row, h_m, wx_nm, wy_nm}]`` brightest first."""
    from scipy import ndimage

    z = plane(z)
    hp = z - background(z, nmpp, bg_nm)
    sm = ndimage.gaussian_filter(hp, max(0.5, 0.08 / nmpp))
    med = float(np.median(sm))
    noise = 1.4826 * float(np.median(np.abs(sm - med)))
    thr = med + max(min_h_m, k_sigma * noise)
    mx = ndimage.maximum_filter(sm, size=_odd(sep_nm / nmpp))
    peaks = np.argwhere((sm == mx) & (sm > thr))
    ny, nx = z.shape
    r = max(2, int(round(r_nm / nmpp)))
    rc = max(2, int(round(0.8 / nmpp)))
    out = []
    for row, col in peaks:
        h = float(sm[row, col] - med)
        ring = [sm[row + dr, col + dc] - med for dr, dc in ((rc, 0), (-rc, 0), (0, rc), (0, -rc))
                if 0 <= row + dr < ny and 0 <= col + dc < nx]
        if len(ring) >= 3 and max(ring) > compact_frac * h:
            continue                                # a ridge (step edge), not a compact bump
        r0, r1, c0, c1 = max(0, row - r), min(ny, row + r + 1), max(0, col - r), min(nx, col + r + 1)
        win = np.clip(hp[r0:r1, c0:c1] - 0.3 * h, 0.0, None)
        s = float(win.sum())
        if s <= 0:
            continue
        yy, xx = np.mgrid[r0:r1, c0:c1]
        cx, cy = float((win * xx).sum() / s), float((win * yy).sum() / s)
        r2 = max(3, int(round(0.7 / nmpp)))
        q0, q1, p0, p1 = max(0, row - r2), min(ny, row + r2 + 1), max(0, col - r2), min(nx, col + r2 + 1)
        w2 = np.clip(hp[q0:q1, p0:p1] - 0.5 * h, 0.0, None)
        s2 = float(w2.sum()) or 1.0
        y2, x2 = np.mgrid[q0:q1, p0:p1]
        wx = math.sqrt(float((w2 * (x2 - cx) ** 2).sum()) / s2) * nmpp
        wy = math.sqrt(float((w2 * (y2 - cy) ** 2).sum()) / s2) * nmpp
        out.append({"col": cx, "row": cy, "h_m": h, "wx_nm": wx, "wy_nm": wy})
    out.sort(key=lambda b: -b["h_m"])
    return out


def ghost_offset(strong: list[dict], weak: list[dict], nmpp: float, shape: tuple[int, int], *,
                 tol_nm: float = 0.3) -> tuple[list[float] | None, list[int]]:
    """The offset at which fainter copies (``weak`` candidates, 10-80 % of the height) repeat the
    ``strong`` features — a double tip — and the indices into ``weak`` of those copies.
    ``(None, [])`` unless at least two strong features, and half of those whose copy would fall
    inside the frame, share one offset."""
    ny, nx = shape
    pairs = []
    for i, a in enumerate(strong):
        for j, b in enumerate(weak):
            if not (0.1 * a["h_m"] <= b["h_m"] <= 0.8 * a["h_m"]):
                continue
            dx = (b["col"] - a["col"]) * nmpp
            dy = (b["row"] - a["row"]) * nmpp
            if 0.6 <= math.hypot(dx, dy) <= 4.5:
                pairs.append((dx, dy, i, j))
    best, best_members = None, []
    for dx, dy, _, _ in pairs:
        members = [(i, j) for ex, ey, i, j in pairs if math.hypot(ex - dx, ey - dy) <= tol_nm]
        if len({i for i, _ in members}) > len({i for i, _ in best_members}):
            best, best_members = [dx, dy], members
    if best is None:
        return None, []
    n_src = len({i for i, _ in best_members})
    could = sum(1 for a in strong if 2 <= a["col"] + best[0] / nmpp <= nx - 3 and 2 <= a["row"] + best[1] / nmpp <= ny - 3)
    if n_src < 2 or n_src < 0.5 * max(1, could):
        return None, []
    return best, sorted({j for _, j in best_members})


def find_atoms(fr: dict) -> tuple[list[dict], list[float] | None]:
    """Compact features minus the copies a second apex makes; and that copy offset (or None)."""
    nmpp = fr["nm_per_px"]
    strong = detect_blobs(fr["z"], nmpp)
    weak = detect_blobs(fr["z"], nmpp, min_h_m=6e-12, k_sigma=4.0)
    off, ghost_idx = ghost_offset(strong, weak, nmpp, fr["z"].shape)
    if off is None:
        return strong, None
    ghosts = [weak[j] for j in ghost_idx]
    keep = [b for b in strong if all(math.hypot(b["col"] - g["col"], b["row"] - g["row"]) * nmpp > 0.3 for g in ghosts)]
    return keep, off


def atoms_in_frame(fr: dict) -> list[list[float]]:
    atoms, _ = find_atoms(fr)
    return [list(pix_to_scan(fr["geom"], b["col"], b["row"])) for b in atoms]


# ── rows and spectra ─────────────────────────────────────────────────────────
def row_jumps(z: np.ndarray) -> np.ndarray:
    """``D[k] = median over columns of (z[k] - z[k-1])`` for k >= 1 (D[0] = 0)."""
    z = _fill(z)
    d = np.median(np.diff(z, axis=0), axis=1)
    return np.concatenate([[0.0], d])


def strongest_row_jump(z: np.ndarray) -> tuple[int, float, float]:
    """(row, |jump| m, robust noise m) of the largest column-median row-to-row jump."""
    d = row_jumps(z)
    med = float(np.median(d[1:]))
    noise = 1.4826 * float(np.median(np.abs(d[1:] - med))) + 1e-15
    k = int(np.argmax(np.abs(d - med)))
    return k, float(abs(d[k] - med)), noise


def spectrum_features(sp: dict) -> dict:
    from scipy import ndimage, signal

    out = {"jump": False, "peak": False, "peak_v": None, "peak_rel": 0.0}
    v = np.asarray(sp["v"], float)
    if sp.get("i") is not None and len(v) > 10:
        i = np.asarray(sp["i"], float)
        d = np.diff(i)
        dd = d - ndimage.median_filter(d, size=9, mode="nearest")
        noise = 1.4826 * float(np.median(np.abs(dd - np.median(dd)))) + 1e-30
        out["jump_z"] = float(np.max(np.abs(dd)) / noise)
        out["jump"] = bool(out["jump_z"] > 25.0)
    if sp.get("didv") is not None and len(v) > 20:
        y = np.asarray(sp["didv"], float)
        ys = signal.savgol_filter(y, 9, 2) if len(y) > 9 else y
        base = ndimage.median_filter(ys, size=_odd(0.45 / max(abs(v[1] - v[0]), 1e-6)), mode="nearest")
        res = ys - base
        level = float(np.median(np.abs(base))) + 1e-30
        noise = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-30
        k = int(np.argmax(res))
        out["peak_rel"] = float(res[k] / level)
        out["peak_snr"] = float(res[k] / noise)
        out["peak"] = bool(res[k] > 0.25 * level and res[k] > 6.0 * noise)
        out["peak_v"] = float(v[k])
    return out


# ── per type ─────────────────────────────────────────────────────────────────
def _frame_paths(item: dict, root: Path) -> list[Path]:
    return [root / f for f in item["frames"]]


def answer_tip(item: dict, root: Path) -> tuple[Any, dict]:
    paths = _frame_paths(item, root)
    fr = load_scan(paths[0])
    sp = load_spectrum(paths[1]) if len(paths) > 1 else None
    k, jump, noise = strongest_row_jump(fr["z"])
    topo_jump = jump > max(15e-12, 10.0 * noise)
    spec = spectrum_features(sp) if sp is not None else {}
    main, off = find_atoms(fr)
    ratio = float(np.median([b["wx_nm"] / max(b["wy_nm"], 1e-3) for b in main])) if main else 1.0
    meta = {"row_jump_pm": jump * 1e12, "row_jump_noise_pm": noise * 1e12, "spectrum": spec,
            "n_atoms": len(main), "ghost_offset_nm": off, "width_ratio_x_y": ratio}
    if topo_jump or spec.get("jump"):
        return "unstable", meta
    if spec.get("peak"):
        return "dirty", meta
    if off is not None:
        return "double", meta
    # the simulator smears every feature along the fast axis by the tip radius (x/y ~1.3-1.5 for a
    # 1 nm tip); a 3-8 nm tip doubles that
    if ratio > BLUNT_RATIO:
        return "blunt", meta
    return "good", meta


#: fast-axis/slow-axis width ratio of compact features above which the baseline calls a tip blunt
BLUNT_RATIO = 1.75


def answer_adatoms(item: dict, root: Path) -> tuple[Any, dict]:
    fr = load_scan(_frame_paths(item, root)[0])
    pts = atoms_in_frame(fr)
    meta = {"n": len(pts)}
    if item["subtype"] == "count":
        return len(pts), meta
    return [[round(x, 3), round(y, 3)] for x, y in pts], meta


def register(f1: dict, f2: dict) -> tuple[float, float, dict]:
    """Displacement (scan-frame nm) of features from frame 1 to frame 2: phase correlation on
    the background-subtracted images, refined with matched compact features when there are any."""
    from skimage.registration import phase_cross_correlation

    nmpp = f1["nm_per_px"]
    a = plane(f1["z"])
    b = plane(f2["z"])
    a = a - background(a, nmpp, 3.0)
    b = b - background(b, nmpp, 3.0)
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=20, normalization=None)
    # skimage: the shift that registers b onto a, so features moved by -shift (rows, cols)
    d_row, d_col = -float(shift[0]), -float(shift[1])
    du, dv = d_col * nmpp, -d_row * nmpp
    dx, dy = img_vec_to_scan(f1["geom"], du, dv)
    meta = {"phase_corr_nm": [dx, dy]}
    p1 = atoms_in_frame(f1)
    p2 = atoms_in_frame(f2)
    if p1 and p2:
        dd = []
        for x1, y1 in p1:
            best = min(p2, key=lambda q: math.hypot(q[0] - x1 - dx, q[1] - y1 - dy))
            e = math.hypot(best[0] - x1 - dx, best[1] - y1 - dy)
            if e < 0.6:
                dd.append((best[0] - x1, best[1] - y1))
        if len(dd) >= 2:
            dx, dy = float(np.median([d[0] for d in dd])), float(np.median([d[1] for d in dd]))
            meta["n_matched"] = len(dd)
    return dx, dy, meta


def answer_drift(item: dict, root: Path) -> tuple[Any, dict]:
    p = _frame_paths(item, root)
    dx, dy, meta = register(load_scan(p[0]), load_scan(p[1]))
    return [round(dx, 4), round(dy, 4)], meta


def answer_move(item: dict, root: Path) -> tuple[Any, dict]:
    p = _frame_paths(item, root)
    f1, f2 = load_scan(p[0]), load_scan(p[1])
    dx, dy, meta = register(f1, f2)
    a1, a2 = atoms_in_frame(f1), atoms_in_frame(f2)
    g2 = f2["geom"]
    lost, found = [], list(a2)
    for x, y in a1:
        px, py = x + dx, y + dy
        col, row = scan_to_pix(g2, px, py)
        inside = (0.8 / f2["nm_per_px"]) <= col <= g2["nx"] - 1 - 0.8 / f2["nm_per_px"] and \
                 (0.8 / f2["nm_per_px"]) <= row <= g2["ny"] - 1 - 0.8 / f2["nm_per_px"]
        hit = [q for q in found if math.hypot(q[0] - px, q[1] - py) < 0.4]
        if hit:
            found.remove(min(hit, key=lambda q: math.hypot(q[0] - px, q[1] - py)))
        elif inside:
            lost.append((x, y))
    new = [q for q in found if (lambda c: 0.8 / f1["nm_per_px"] <= c[0] <= f1["geom"]["nx"] - 1 - 0.8 / f1["nm_per_px"]
                                and 0.8 / f1["nm_per_px"] <= c[1] <= f1["geom"]["ny"] - 1 - 0.8 / f1["nm_per_px"])(
        scan_to_pix(f1["geom"], q[0] - dx, q[1] - dy))]
    meta.update({"shift_nm": [dx, dy], "n_frame1": len(a1), "n_frame2": len(a2), "lost": lost, "new": new})
    st = item["subtype"]
    if st == "detect":
        return bool(lost and new), meta
    if not lost:
        raise ValueError("no atom of frame 1 is missing from frame 2")
    if st == "which":
        x, y = lost[0]
        return [round(x, 3), round(y, 3)], meta
    if not new:
        raise ValueError("no new atom in frame 2 to pair with the missing one")
    best = min(((math.hypot(q[0] - dx - x, q[1] - dy - y), (x, y), q) for x, y in lost for q in new), key=lambda t: t[0])
    return round(best[0], 4), meta


def fft_peaks(z: np.ndarray, nmpp: float, band_nm: tuple[float, float], n: int = 6) -> list[dict]:
    """Strongest local maxima of the Hann-windowed spectrum whose period lies in ``band_nm``;
    one per +-k pair, sub-bin by a 3x3 centroid. ``angle_deg`` is the stripe direction (k + 90
    degrees, image axes with v up, mod 180)."""
    from scipy import ndimage

    z = plane(z)
    z = z - z.mean()
    ny, nx = z.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.abs(np.fft.fftshift(np.fft.fft2(z * win)))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=nmpp))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=nmpp))
    FX, FY = np.meshgrid(fx, fy)
    KR = np.hypot(FX, FY)
    band = (KR >= 1.0 / band_nm[1]) & (KR <= 1.0 / band_nm[0])
    Fb = np.where(band, F, 0.0)
    mx = ndimage.maximum_filter(Fb, size=3)
    pk = np.argwhere((Fb == mx) & (Fb > 0))
    cand = sorted(((float(Fb[r, c]), int(r), int(c)) for r, c in pk), reverse=True)
    out: list[dict] = []
    for val, r, c in cand:
        r0, r1, c0, c1 = max(0, r - 1), min(ny, r + 2), max(0, c - 1), min(nx, c + 2)
        wgt = F[r0:r1, c0:c1] ** 2
        kx = float((wgt * FX[r0:r1, c0:c1]).sum() / wgt.sum())
        ky = float((wgt * FY[r0:r1, c0:c1]).sum() / wgt.sum())
        ku, kv = kx, -ky
        if kv < 0 or (kv == 0 and ku < 0):
            ku, kv = -ku, -kv
        if any(math.hypot(ku - p["ku"], kv - p["kv"]) < 1.5 / (nmpp * max(nx, ny)) for p in out):
            continue
        k = math.hypot(ku, kv)
        out.append({"ku": ku, "kv": kv, "period_nm": 1.0 / k, "strength": val,
                    "angle_deg": (math.degrees(math.atan2(kv, ku)) + 90.0) % 180.0})
        if len(out) >= n:
            break
    return out


def answer_herringbone(item: dict, root: Path) -> tuple[Any, dict]:
    fr = load_scan(_frame_paths(item, root)[0])
    peaks = fft_peaks(fr["z"], fr["nm_per_px"], (4.5, 9.0))
    if not peaks:
        raise ValueError("no spectral peak in the 4.5-9 nm band")
    top = peaks[0]
    second = next((p for p in peaks[1:] if _adiff(p["angle_deg"], top["angle_deg"]) > 5.0
                   and p["strength"] > 0.3 * top["strength"]), None)
    meta = {"peaks": peaks[:4]}
    if item["subtype"] == "period":
        ps = [top["period_nm"]] + ([second["period_nm"]] if second else [])
        return round(float(np.mean(ps)), 4), meta
    return round(top["angle_deg"], 3), meta


def _adiff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def answer_step(item: dict, root: Path) -> tuple[Any, dict]:
    from scipy import ndimage

    fr = load_scan(_frame_paths(item, root)[0])
    (xa, ya), (xb, yb) = _points_from_question(item["question"])     # the question, never the truth
    ca, ra = scan_to_pix(fr["geom"], xa, ya)
    cb, rb = scan_to_pix(fr["geom"], xb, yb)
    n = int(max(abs(cb - ca), abs(rb - ra)) * 2) + 2
    cols = np.linspace(ca, cb, n)
    rows = np.linspace(ra, rb, n)
    z = _fill(fr["z"])
    zs = ndimage.gaussian_filter(z, 1.0)
    prof = ndimage.map_coordinates(zs, [rows, cols], order=1, mode="nearest")
    d = np.diff(prof)
    slope = float(np.median(d))
    total = float(prof[-1] - prof[0]) - slope * (n - 1)
    return round(total * 1e12, 2), {"n_samples": n, "slope_pm_per_sample": slope * 1e12}


def _points_from_question(q: str) -> list[list[float]]:
    import re

    m = re.findall(r"point [AB] = \((-?[\d.]+), (-?[\d.]+)\)", q, flags=re.IGNORECASE)
    if len(m) != 2:
        raise ValueError("could not read points A and B from the question")
    return [[float(a), float(b)] for a, b in m]


def answer_tipchange(item: dict, root: Path) -> tuple[Any, dict]:
    fr = load_scan(_frame_paths(item, root)[0])
    k, jump, noise = strongest_row_jump(fr["z"])
    return int(k), {"jump_pm": jump * 1e12, "noise_pm": noise * 1e12}


def answer_atomic(item: dict, root: Path) -> tuple[Any, dict]:
    from scipy import ndimage

    fr = load_scan(_frame_paths(item, root)[0])
    nmpp = fr["nm_per_px"]
    z = plane(fr["z"])
    z = z - ndimage.gaussian_filter(z, 0.6 / nmpp)
    ny, nx = z.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    F = np.abs(np.fft.fftshift(np.fft.fft2(z * win)))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=nmpp))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=nmpp))
    KR = np.hypot(*np.meshgrid(fx, fy))
    band = (KR >= 1 / 0.33) & (KR <= 1 / 0.18)
    ratio = float(F[band].max() / (np.median(F[band]) + 1e-30))
    return bool(ratio > 40.0), {"peak_to_median": ratio}


ANSWERERS = {"tip": answer_tip, "adatoms": answer_adatoms, "move": answer_move, "drift": answer_drift,
             "herringbone": answer_herringbone, "step": answer_step, "tipchange": answer_tipchange,
             "atomic": answer_atomic}


def answer_item(item: dict, root: str | Path) -> dict:
    """``{"id", "answer": {"value"|"abstain", ...}, "meta"}`` for one item; never raises."""
    root = Path(root)
    try:
        value, meta = ANSWERERS[item["type"]](item, root)
        return {"id": item["id"], "answer": {"value": value, "abstain": False}, "meta": _jsonable(meta)}
    except Exception as exc:  # noqa: BLE001 — an unanswerable item abstains, with the reason
        return {"id": item["id"], "answer": {"abstain": True, "note": f"{type(exc).__name__}: {exc}"}, "meta": {}}


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_baseline(dataset: str | Path, out: str | Path, *, types: list[str] | None = None,
                 log=None) -> list[dict]:
    rows = []
    items = I.load_items(dataset)
    for it in items:
        if types and it["type"] not in types:
            continue
        rows.append(answer_item(it, dataset))
        if log is not None:
            log(f"{it['id']}: {rows[-1]['answer']}")
    I.write_jsonl(out, rows)
    return rows
