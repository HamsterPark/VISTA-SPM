"""A VISTA-SPM episode end to end on the simulator with a scripted driver (no model).

Needs MAST (``requires_mast``) and must be the only RuntimeHost in its process (``isolated``):
    python -m pytest tests/test_episode.py -m isolated -q
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import stmbench

from tests.conftest import requires_mast

pytestmark = [requires_mast, pytest.mark.isolated]

SCENARIO = Path(stmbench.__file__).resolve().parent / "trackB" / "scenarios" / "P4_atom_positioning_cu111.yaml"


def test_scripted_episode_writes_a_judged_ledger(tmp_path):
    from stmsim.scenario import Scenario

    from vista_spm.episode import run_vista_episode

    seen: dict = {}

    def driver(h):
        r = h.call("act", {"action": "scan", "params": {"center": {"x_nm": 0, "y_nm": 0}, "size_nm": 30,
                                                          "pixels": 128}})
        body = json.loads(r.text)
        seen["scan"] = body
        seen["images"] = len(r.images)
        h.acknowledge_act()
        seen["refused"] = h.call("act", {"action": "wait", "params": {"seconds": 5}}).is_error
        h.call("report_result", {"claim_id": "target_site", "value": 0.0, "x_nm": 0.0, "y_nm": 0.0})
        h.call("report_tip_state", {"junction": "unknown", "atomic_resolution": False, "tip_state": "unknown",
                                    "reason": "test"})
        h.finish("done", "test [DONE]")
        return {"final_text": "test [DONE]"}

    res = run_vista_episode(Scenario.load(SCENARIO), seed=0, arm="img", model="scripted", effort="none",
                            out=tmp_path, run_id="t", driver=driver)
    assert seen["scan"]["ok"] is True and len(seen["scan"]["new_frames"]) == 1
    assert seen["images"] == 1
    assert seen["refused"] is False                     # acknowledged → the next act runs
    out = Path(res.out_dir)
    for name in ("episode.json", "driver.json", "events.json", "tip_timeline.json", "host_facts.json"):
        assert (out / name).is_file(), name
    ep = json.loads((out / "episode.json").read_text(encoding="utf-8"))
    assert ep["mode"] == "V-img" and ep["claims_total"] == 2 and ep["claims_reported"] >= 1
    assert ep["skill_result"]["outcome"] == "done"
    assert 0 < ep["sim_consumed_s"] < 3600                # the clock ran only inside the two acts
    assert len(list((out / "session").glob("*.sxm"))) == 1  # one scan, one file (no duplicate save)
