"""Aggregate VISTA-SPM episodes: one row per episode, then arm x scenario means.

usage: python scripts/aggregate.py <out_root> [--run-id ID] [--csv rows.csv] [--md summary.md]
Reads <out_root>/<scenario>/seed<N>/V-<arm>_<model>_<effort>/<run_id>/episode.json (+ driver.json).
"""
import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RUN_RE = re.compile(r"V-(?P<arm>[a-z\-]+)_(?P<model>.+)_(?P<effort>low|medium|high|xhigh|max)$")


def rows(root: Path, run_id: str | None):
    for ep_path in sorted(root.glob("*/seed*/V-*/*/episode.json")):
        run_dir = ep_path.parent
        if run_id and run_dir.name != run_id:
            continue
        if ".stale." in run_dir.name:
            continue
        m = RUN_RE.match(run_dir.parent.name)
        if not m:
            continue
        ep = json.loads(ep_path.read_text(encoding="utf-8"))
        sr = ep.get("skill_result") or {}
        drv = {}
        try:
            drv = json.loads((run_dir / "driver.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        host = drv.get("host") or drv
        tools = Counter(e.get("name") for e in host.get("driver_events") or [] if e.get("kind") == "tool_start")
        n_img = sum(int(e.get("n_images") or 0) for e in host.get("driver_events") or [] if e.get("kind") == "tool_end")
        usage = (sr.get("billing") or {})
        yield {
            "scenario": run_dir.parent.parent.parent.name, "seed": run_dir.parent.parent.name.replace("seed", ""),
            "arm": m["arm"], "model": m["model"], "effort": m["effort"], "run_id": run_dir.name,
            "claims_total": ep.get("claims_total"), "claims_verified": ep.get("claims_verified"),
            "claims_reported": ep.get("claims_reported"),
            "frac_verified": (ep.get("claims_verified") or 0) / max(1, ep.get("claims_total") or 1),
            "reproduced": bool(ep.get("reproduced")), "diag_correct": ep.get("diag_correct"),
            "outcome": sr.get("outcome"), "acts": sr.get("acts"),
            "sim_frac": ep.get("sim_consumed_fraction"), "wall_s": ep.get("wall_s"),
            "usd_est": usage.get("usd_est"), "model_calls": sr.get("model_calls"),
            "images_returned": n_img, "tool_calls": sum(tools.values()),
            "inspect": tools.get("inspect", 0), "read_values": tools.get("read_values", 0),
            "measure": sum(tools.get(k, 0) for k in ("line_profile", "fft_peaks", "detect_blobs", "diff", "stats")),
            "notes_writes": tools.get("write_guide", 0) + tools.get("write_working", 0),
            "advisor": tools.get("advisor", 0), "history": tools.get("history", 0),
            "segments": len(list((run_dir / "vista" / "claude_io").glob("segment_*.stdout.jsonl"))),
        }


def mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return statistics.fmean(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--run-id")
    ap.add_argument("--csv")
    ap.add_argument("--md")
    a = ap.parse_args()
    rs = list(rows(Path(a.root), a.run_id))
    if a.csv and rs:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    groups = defaultdict(list)
    for r in rs:
        groups[(r["arm"], r["scenario"])].append(r)
    arms = sorted({r["arm"] for r in rs}, key=lambda x: ["full", "read", "mem", "notes", "img", "text", "vigil-text", "advisor"].index(x)
                  if x in ["full", "read", "mem", "notes", "img", "text", "vigil-text", "advisor"] else 99)
    scen = sorted({r["scenario"] for r in rs})
    lines = ["| arm | scenario | n | claims verified | reproduced | tip diag correct | acts | sim used | tool calls | images | usd est |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        for sc in scen:
            g = groups.get((arm, sc)) or []
            if not g:
                continue
            lines.append(f"| {arm} | {sc} | {len(g)} | {mean(r['frac_verified'] for r in g):.2f} | "
                         f"{sum(r['reproduced'] for r in g)}/{len(g)} | {mean(r['diag_correct'] for r in g):.2f} | "
                         f"{mean(r['acts'] for r in g):.1f} | {mean(r['sim_frac'] for r in g):.2f} | "
                         f"{mean(r['tool_calls'] for r in g):.0f} | {mean(r['images_returned'] for r in g):.0f} | "
                         f"{mean(r['usd_est'] for r in g):.2f} |")
        g = [r for r in rs if r["arm"] == arm]
        lines.append(f"| **{arm}** | **all** | {len(g)} | **{mean(r['frac_verified'] for r in g):.2f}** | "
                     f"{sum(r['reproduced'] for r in g)}/{len(g)} | {mean(r['diag_correct'] for r in g):.2f} | "
                     f"{mean(r['acts'] for r in g):.1f} | {mean(r['sim_frac'] for r in g):.2f} | "
                     f"{mean(r['tool_calls'] for r in g):.0f} | {mean(r['images_returned'] for r in g):.0f} | "
                     f"{mean(r['usd_est'] for r in g):.2f} |")
    out = "\n".join(lines)
    print(out)
    print(f"\n{len(rs)} episodes; total usd est {sum(r['usd_est'] or 0 for r in rs):.2f}")
    if a.md:
        Path(a.md).write_text(out + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
