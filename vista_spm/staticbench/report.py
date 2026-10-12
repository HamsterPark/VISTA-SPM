"""Tables of STM-BabyVision results: one row per (run, item subtype), as CSV and Markdown.

README
------
``write_report({"baseline": records, "claude-x": records2}, out_dir)`` writes
``report.csv`` (machine-readable, one row per run and subtype) and ``report.md`` (an overall
table per run, a per-subtype table with every run side by side, the error statistics in the
item's unit, and the confusion matrices of the choice / bool subtypes). ``records`` are the
output of :func:`.score.score_all`.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from . import items as I
from .score import aggregate

UNITS = {"herringbone.period": "nm", "herringbone.orientation": "deg", "move.distance": "nm",
         "move.which": "nm", "drift.vector": "nm", "adatoms.positions": "nm", "adatoms.count": "atoms",
         "step.height": "pm", "tipchange.row": "rows"}

CSV_FIELDS = ("run", "subtype", "kind", "n", "answered", "abstained", "missing", "invalid", "correct",
              "accuracy", "accuracy_answered", "mean_score", "err_unit", "err_mean", "err_median", "err_p90",
              "within_near", "precision", "recall", "f1")


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: Any) -> str:
    return "-" if x is None else f"{100.0 * float(x):.1f}%"


def csv_rows(runs: dict[str, list[dict]]) -> list[dict]:
    out = []
    for name, recs in runs.items():
        agg = aggregate(recs)
        for row in agg["subtypes"]:
            e = row.get("error") or {}
            out.append({"run": name, "subtype": row["subtype"], "kind": row["kind"], "n": row["n"],
                        "answered": row["answered"], "abstained": row["abstained"], "missing": row["missing"],
                        "invalid": row["invalid"], "correct": row["correct"], "accuracy": row["accuracy"],
                        "accuracy_answered": row["accuracy_answered"], "mean_score": row["mean_score"],
                        "err_unit": UNITS.get(row["subtype"], ""), "err_mean": e.get("mean"),
                        "err_median": e.get("median"), "err_p90": e.get("p90"),
                        "within_near": row.get("within_near"), "precision": row.get("precision"),
                        "recall": row.get("recall"), "f1": row.get("f1")})
        o = agg["overall"]
        out.append({"run": name, "subtype": "ALL", "kind": "", "n": o["n"], "answered": None,
                    "abstained": o["abstained"], "missing": o["missing"], "invalid": o["invalid"],
                    "correct": o["correct"], "accuracy": o["accuracy"], "accuracy_answered": None,
                    "mean_score": o["mean_score"]})
    return out


def markdown(runs: dict[str, list[dict]], *, title: str = "STM-BabyVision results",
             dataset: str | None = None, notes: list[str] | None = None) -> str:
    aggs = {name: aggregate(recs) for name, recs in runs.items()}
    lines = [f"# {title}", ""]
    if dataset:
        lines += [f"Dataset: `{dataset}`", ""]
    for n in notes or []:
        lines += [n, ""]
    lines += ["## Overall", "", "| run | items | correct | accuracy | macro accuracy (subtypes) | mean score | abstained | missing | invalid |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, agg in aggs.items():
        o = agg["overall"]
        lines.append(f"| {name} | {o['n']} | {o['correct']} | {_pct(o['accuracy'])} | {_pct(o['macro_accuracy'])} | "
                     f"{_f(o['mean_score'])} | {o['abstained']} | {o['missing']} | {o['invalid']} |")
    lines += ["", "## Per subtype", "",
              "Accuracy counts an abstention or a missing answer as wrong; *answered* is accuracy over the "
              "items that got a valid answer. Errors are over answered items, in the unit shown "
              "(position lists: mean distance of matched points).", "",
              "| subtype | run | n | accuracy | answered acc. | abstain | error median | error p90 | extra |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    keys: list[str] = []
    for agg in aggs.values():
        for row in agg["subtypes"]:
            if row["subtype"] not in keys:
                keys.append(row["subtype"])
    order = {t: k for k, t in enumerate(I.ITEM_TYPES)}
    keys.sort(key=lambda k: (order.get(k.split(".")[0], 99), k))
    for key in keys:
        for name, agg in aggs.items():
            row = next((r for r in agg["subtypes"] if r["subtype"] == key), None)
            if row is None:
                continue
            e = row.get("error") or {}
            unit = UNITS.get(key, "")
            extra = []
            if row.get("within_near") is not None:
                extra.append(f"within +-1: {_pct(row['within_near'])}")
            if row.get("f1") is not None:
                extra.append(f"P {_f(row['precision'], 2)} / R {_f(row['recall'], 2)} / F1 {_f(row['f1'], 2)}")
            if row.get("per_class_recall"):
                extra.append("recall " + ", ".join(f"{k}: {_pct(v)}" for k, v in row["per_class_recall"].items()))
            lines.append(f"| {key} | {name} | {row['n']} | {_pct(row['accuracy'])} | {_pct(row['accuracy_answered'])} | "
                         f"{row['abstained']} | {_f(e.get('median'))} {unit if e else ''} | {_f(e.get('p90'))} | "
                         f"{'; '.join(extra)} |")
    conf_keys = [k for k in keys if any(next((r for r in a["subtypes"] if r["subtype"] == k and r.get("confusion")), None)
                                        for a in aggs.values())]
    if conf_keys:
        lines += ["", "## Confusion matrices (rows = truth, columns = answer)", ""]
        for key in conf_keys:
            for name, agg in aggs.items():
                row = next((r for r in agg["subtypes"] if r["subtype"] == key), None)
                if row is None or not row.get("confusion"):
                    continue
                conf = row["confusion"]
                cols = sorted({c for v in conf.values() for c in v} | set(conf))
                lines += [f"**{key} — {name}**", "", "| truth \\ answer | " + " | ".join(cols) + " |",
                          "|---|" + "---:|" * len(cols)]
                for t in sorted(conf):
                    lines.append(f"| {t} | " + " | ".join(str(conf[t].get(c, 0)) for c in cols) + " |")
                lines.append("")
    diff_lines = []
    for name, agg in aggs.items():
        for row in agg["subtypes"]:
            bd = row.get("by_difficulty") or {}
            if len(bd) > 1:
                diff_lines.append(f"| {row['subtype']} | {name} | " + ", ".join(
                    f"{k}: {_pct(v['accuracy'])} (n={v['n']})" for k, v in bd.items()) + " |")
    if diff_lines:
        lines += ["## By difficulty", "", "| subtype | run | accuracy by difficulty |", "|---|---|---|"] + diff_lines + [""]
    return "\n".join(lines) + "\n"


def write_report(runs: dict[str, list[dict]], out_dir: str | Path, *, title: str = "STM-BabyVision results",
                 dataset: str | None = None, notes: list[str] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = csv_rows(runs)
    with open(out / "report.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CSV_FIELDS})
    md = markdown(runs, title=title, dataset=dataset, notes=notes)
    (out / "report.md").write_text(md, encoding="utf-8")
    return {"csv": str(out / "report.csv"), "md": str(out / "report.md"), "rows": len(rows)}
