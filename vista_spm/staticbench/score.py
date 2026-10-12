"""Scoring STM-BabyVision answers against the simulator's truth, per item type.

README
------
``score_item(item, answer)`` → one record; ``score_all(items, answers)`` → all of them (an item
without an answer counts as *not answered*, which is reported apart from an explicit abstention);
``aggregate(records)`` → per-subtype tables. Rules (tolerances live in each item, so a
dataset can tighten them without code changes):

==============  =====================================================================
kind            correct when
==============  =====================================================================
choice / bool   the value equals the truth
number          ``min |a - t|`` over the truth and ``truth.accept`` (mod ``tolerance.mod`` for
                angles) ``<= max(abs, rel * |t|)``; herringbone: 0.15 nm, 6 deg mod 180;
                move distance 0.25 nm or 15 %; step 30 pm or 10 %; tip-change row +-2
vector          ``|a - t| <= max(abs, rel * |t|)``; drift 0.3 nm or 20 %, a position 0.3 nm
count           ``truth.min <= a <= truth.max`` (atoms on the edge are optional);
                ``within_near``: inside the range widened by ``tolerance.near`` (1)
position_list   Hungarian matching within ``match_nm`` (0.3 nm): no miss and no spurious
                point; ``score`` = F1 (matches to edge points neither help nor hurt)
==============  =====================================================================

An abstention (``{"abstain": true}``) is never correct; it is counted separately so a model
that declines is distinguishable from one that is wrong. A malformed answer is ``invalid``
(wrong, and counted).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

from . import items as I


def _angle_diff(a: float, b: float, mod: float) -> float:
    d = abs(float(a) - float(b)) % mod
    return min(d, mod - d)


def _wrap(d: float, mod: float) -> float:
    """Signed difference wrapped into (-mod/2, mod/2]."""
    d = (d + mod / 2.0) % mod - mod / 2.0
    return d if d != -mod / 2.0 else mod / 2.0


def match_points(pred: list, core: list, optional: list | None = None, gate_nm: float = 0.3) -> dict:
    """One-to-one matching (Hungarian, distances above ``gate_nm`` never match) of predicted
    points to the truth's core and optional points."""
    optional = list(optional or [])
    truth = [(float(x), float(y), False) for x, y in core] + [(float(x), float(y), True) for x, y in optional]
    pred = [(float(x), float(y)) for x, y in pred]
    tp, matched_err, opt_hits, pairs = 0, [], 0, []
    if pred and truth:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        big = 1e6
        cost = np.full((len(pred), len(truth)), big)
        for i, (px, py) in enumerate(pred):
            for j, (tx, ty, opt) in enumerate(truth):
                d = math.hypot(px - tx, py - ty)
                if d <= gate_nm:
                    cost[i, j] = d + (1e-6 if opt else 0.0)       # prefer a core point on a tie
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i, j] >= big:
                continue
            pairs.append((int(i), int(j)))
            if truth[j][2]:
                opt_hits += 1
            else:
                tp += 1
                matched_err.append(math.hypot(pred[i][0] - truth[j][0], pred[i][1] - truth[j][1]))
    n_core = len(core)
    fp = len(pred) - tp - opt_hits
    fn = n_core - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0          # no prediction: vacuously precise
    recall = tp / n_core if n_core else 1.0
    if n_core == 0 and fp == 0:
        f1 = 1.0
    elif tp == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "optional_matched": opt_hits, "precision": precision,
            "recall": recall, "f1": f1,
            "mean_err_nm": (sum(matched_err) / len(matched_err)) if matched_err else None}


def score_item(item: dict, answer: Any) -> dict:
    """One scored record. ``answer`` is the answer object (``{"value"}`` / ``{"abstain": true}``),
    a bare value, or ``None`` (no answer at all)."""
    af = item["answer_format"]
    kind = af["kind"]
    truth = item["truth"]
    tol = item.get("tolerance") or {}
    rec: dict[str, Any] = {"id": item["id"], "type": item["type"], "subtype": item["subtype"], "kind": kind,
                           "difficulty": item.get("difficulty", "normal"), "answered": False, "abstained": False,
                           "missing": answer is None, "invalid": False, "correct": False, "score": 0.0,
                           "error": None, "note": None, "pred": None, "truth": truth.get("value")}
    if answer is None:
        rec["note"] = "no answer"
        return rec
    value, abstained, err = I.normalize_answer(item, answer)
    if abstained:
        rec["abstained"] = True
        return rec
    if err is not None:
        rec["invalid"] = True
        rec["note"] = err
        return rec
    rec["answered"] = True
    rec["pred"] = value
    t = truth["value"]
    if kind in ("choice", "bool"):
        rec["correct"] = value == t
    elif kind == "number":
        mod = tol.get("mod") or af.get("mod")
        cands = [t] + list(truth.get("accept") or [])
        if mod:
            errs = [_angle_diff(value, c, float(mod)) for c in cands]
            rec["signed_error"] = _wrap(float(value) - float(t), float(mod))
        else:
            errs = [abs(float(value) - float(c)) for c in cands]
            rec["signed_error"] = float(value) - float(t)
        best = min(errs)
        lim = max(float(tol.get("abs", 0.0) or 0.0), float(tol.get("rel", 0.0) or 0.0) * abs(float(t)))
        rec["error"] = best
        rec["limit"] = lim
        rec["correct"] = best <= lim + 1e-9
    elif kind == "vector":
        dx, dy = float(value[0]) - float(t[0]), float(value[1]) - float(t[1])
        e = math.hypot(dx, dy)
        lim = max(float(tol.get("abs", 0.0) or 0.0), float(tol.get("rel", 0.0) or 0.0) * math.hypot(*t))
        rec.update({"error": e, "limit": lim, "error_xy": [dx, dy], "correct": e <= lim + 1e-9})
    elif kind == "count":
        lo, hi = int(truth.get("min", t)), int(truth.get("max", t))
        near = int(tol.get("near", 1))
        v = int(value)
        rec["error"] = 0 if lo <= v <= hi else (lo - v if v < lo else v - hi)
        rec["signed_error"] = v - int(t)
        rec["correct"] = lo <= v <= hi
        rec["within_near"] = lo - near <= v <= hi + near
    elif kind == "position_list":
        m = match_points(value, t, truth.get("optional"), float(tol.get("match_nm", 0.3)))
        rec.update({"match": m, "error": m["mean_err_nm"], "correct": m["fn"] == 0 and m["fp"] == 0})
        rec["score"] = float(m["f1"])
    if kind != "position_list":
        rec["score"] = 1.0 if rec["correct"] else 0.0
    return rec


def score_all(items: list[dict], answers: dict[str, Any]) -> list[dict]:
    """Score every item; answers for unknown ids are ignored (and reported by the caller)."""
    return [score_item(it, answers.get(it["id"])) for it in items]


def _stats(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "median": None, "p90": None, "rmse": None}
    s = sorted(xs)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p90 = s[min(n - 1, int(math.ceil(0.9 * n)) - 1)]
    return {"mean": sum(s) / n, "median": med, "p90": p90, "rmse": math.sqrt(sum(x * x for x in s) / n)}


def aggregate(records: list[dict]) -> dict:
    """Per-subtype and overall tables (plain dicts, JSON-safe)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[f"{r['type']}.{r['subtype']}"].append(r)
    order = {t: k for k, t in enumerate(I.ITEM_TYPES)}
    rows = []
    for key in sorted(groups, key=lambda k: (order.get(k.split(".")[0], 99), k)):
        rs = groups[key]
        n = len(rs)
        answered = [r for r in rs if r["answered"]]
        row: dict[str, Any] = {
            "subtype": key, "kind": rs[0]["kind"], "n": n, "answered": len(answered),
            "abstained": sum(r["abstained"] for r in rs), "missing": sum(r["missing"] for r in rs),
            "invalid": sum(r["invalid"] for r in rs), "correct": sum(r["correct"] for r in rs),
            "accuracy": sum(r["correct"] for r in rs) / n if n else None,
            "accuracy_answered": (sum(r["correct"] for r in answered) / len(answered)) if answered else None,
            "mean_score": sum(r["score"] for r in rs) / n if n else None}
        errs = [float(r["error"]) for r in answered if isinstance(r.get("error"), (int, float))]
        if errs and rs[0]["kind"] in ("number", "vector", "count", "position_list"):
            row["error"] = _stats(errs)
        if rs[0]["kind"] == "count":
            row["within_near"] = sum(bool(r.get("within_near")) for r in rs) / n
        if rs[0]["kind"] == "position_list":
            ms = [r["match"] for r in answered]
            row["precision"] = sum(m["precision"] for m in ms) / len(ms) if ms else None
            row["recall"] = sum(m["recall"] for m in ms) / len(ms) if ms else None
            row["f1"] = sum(m["f1"] for m in ms) / len(ms) if ms else None
        if rs[0]["kind"] in ("choice", "bool"):
            conf: dict[str, Counter] = defaultdict(Counter)
            for r in rs:
                pred = "abstain" if r["abstained"] else ("invalid" if r["invalid"] or r["missing"] else str(r["pred"]))
                conf[str(r["truth"])][pred] += 1
            row["confusion"] = {k: dict(v) for k, v in sorted(conf.items())}
            row["per_class_recall"] = {k: v.get(k, 0) / sum(v.values()) for k, v in sorted(conf.items())}
        diff: dict[str, list] = defaultdict(list)
        for r in rs:
            diff[r["difficulty"] or "normal"].append(r["correct"])
        row["by_difficulty"] = {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(diff.items())}
        rows.append(row)
    n = len(records)
    accs = [r["accuracy"] for r in rows if r["accuracy"] is not None]
    overall = {"n": n, "correct": sum(r["correct"] for r in records),
               "accuracy": (sum(r["correct"] for r in records) / n) if n else None,
               "macro_accuracy": (sum(accs) / len(accs)) if accs else None,
               "mean_score": (sum(r["score"] for r in records) / n) if n else None,
               "abstained": sum(r["abstained"] for r in records), "missing": sum(r["missing"] for r in records),
               "invalid": sum(r["invalid"] for r in records)}
    return {"overall": overall, "subtypes": rows}
