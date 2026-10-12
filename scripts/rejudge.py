"""Re-judge finished episodes offline with overridden evidence rules (a secondary, clearly
labelled metric; the official verdict in episode.json is never changed).

Uses the same judge call as stmbench.harness.episode: judge(kind, {**truth_after, events_tail},
thresholds, claims=..., report=..., events=..., orientation_at=None). P1 orientation claims need
the live surface (orientation_at), so this script is meant for P4/P5-style claims.

usage: python scripts/rejudge.py <out_root> --scenario P4_atom_positioning_cu111 --run-id m1
                         [--evidence target_site:max_nm_per_px=0.2] [--json out.json]
"""
import argparse
import json
from pathlib import Path

from stmsim.scenario import Scenario

from stmbench.cli import _resolve
from stmbench.trackB.truth_criteria import judge


def verdict(scenario, ep: dict, events: list, thr_extra: dict) -> dict:
    thr = {**(scenario.success.get("thresholds") or {}), **thr_extra}
    truth_after = ep.get("truth_after") or {}
    v = judge(scenario.success.get("kind"), {**truth_after, "events_tail": events[-50:]}, thr,
              claims=scenario.claims, report=(ep.get("skill_result") or {}).get("results") or {},
              events=events, orientation_at=None)
    return v.as_dict() if hasattr(v, "as_dict") else dict(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--run-id", default="m1")
    ap.add_argument("--evidence", action="append", default=[],
                    help="claim:key=value override of that claim's evidence rule (repeatable)")
    ap.add_argument("--json")
    a = ap.parse_args()
    path = _resolve(a.scenario)[0]
    sc = Scenario.load(path)
    over: dict = {}
    for item in a.evidence:
        cid, _, kv = item.partition(":")
        k, _, val = kv.partition("=")
        over.setdefault(cid, {})[k] = json.loads(val)
    rows = []
    for run in sorted(Path(a.root).glob(f"{sc.id}/seed*/V-*/{a.run_id}")):
        try:
            ep = json.loads((run / "episode.json").read_text(encoding="utf-8"))
            events = json.loads((run / "events.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        off = verdict(sc, ep, events, {})
        rel = verdict(sc, ep, events, {"claim_evidence": over}) if over else None
        def short(v):
            det = v.get("details") or {}
            cl = det.get("claims") or {}
            items = cl.items() if isinstance(cl, dict) else [(c[0], c[1]) for c in cl]
            out = {"n_verified": det.get("n_verified")}
            for cid, c in items:
                t = c.get("truth")
                out[cid] = {"value_ok": c.get("value_ok"), "evidence_ok": c.get("evidence_ok"),
                            "atom_id": c.get("atom_id"),
                            "truth": t if not isinstance(t, (list, dict)) or len(json.dumps(t)) < 300 else "…"}
            return out
        row = {"seed": run.parts[-3], "arm": run.parts[-2], "official_recomputed": short(off),
               "official_in_ledger": ep.get("claims_verified"),
               "relaxed": short(rel) if rel else None}
        rows.append(row)
        print(f"{row['seed']} {row['arm']:<36} official={row['official_in_ledger']} "
              f"recomputed={row['official_recomputed']['n_verified']} "
              f"relaxed={(row['relaxed'] or {}).get('n_verified')}")
        if rel:
            for cid in (row["relaxed"] or {}):
                if cid != "n_verified":
                    print(f"      {cid}: {json.dumps(row['relaxed'][cid], ensure_ascii=False, default=str)[:300]}")
    if a.json:
        Path(a.json).write_text(json.dumps({"scenario": sc.id, "override": over, "rows": rows}, indent=1,
                                           ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
