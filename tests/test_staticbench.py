"""STM-BabyVision (vista_spm.staticbench): schema, scorer, report, truth geometry, QA host,
and a tiny end-to-end generation + baseline run (requires_mast)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from vista_spm.staticbench import items as I
from vista_spm.staticbench.score import aggregate, match_points, score_all, score_item
from tests.conftest import requires_mast


# ── helpers ──────────────────────────────────────────────────────────────────
def _item(kind: str, **kw) -> dict:
    base = {"id": "drift.vector.0123abcd", "type": "drift", "subtype": "vector",
            "question": "How far did the surface features move between frame 1 and frame 2? Answer [dx, dy] in nm.",
            "frames": ["frames/a.sxm", "frames/b.sxm"], "answer_format": {"kind": kind},
            "truth": {"value": None}, "tolerance": {}, "provenance": {"seed": 0}, "difficulty": "normal"}
    if kind == "choice":
        base.update(id="tip.condition.0123abcd", type="tip", subtype="condition",
                    answer_format={"kind": "choice", "choices": list(I.TIP_CLASSES)}, truth={"value": "double"},
                    question="Judge the condition of the tip from frame 1 and spectrum 2. Answer with one label.",
                    frames=["frames/a.sxm", "frames/s.dat"])
    elif kind == "number":
        base.update(id="herringbone.orientation.0123abcd", type="herringbone", subtype="orientation",
                    answer_format={"kind": "number", "unit": "deg", "mod": 180},
                    truth={"value": 3.0, "accept": [175.0, 10.0]}, tolerance={"abs": 6.0, "mod": 180},
                    question="In which direction do the stripes run in frame 1? Answer in degrees.",
                    frames=["frames/a.sxm"])
    elif kind == "vector":
        base.update(truth={"value": [1.0, -5.0]}, tolerance={"abs": 0.3, "rel": 0.2},
                    answer_format={"kind": "vector", "unit": "nm"})
    elif kind == "position_list":
        base.update(id="adatoms.positions.0123abcd", type="adatoms", subtype="positions",
                    answer_format={"kind": "position_list", "unit": "nm"},
                    truth={"value": [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], "optional": [[9.9, 0.0]]},
                    tolerance={"match_nm": 0.3}, frames=["frames/a.sxm"],
                    question="List the centre of every adsorbed atom in frame 1 as [x, y] in nm.")
    elif kind == "count":
        base.update(id="adatoms.count.0123abcd", type="adatoms", subtype="count",
                    answer_format={"kind": "count"}, truth={"value": 5, "min": 5, "max": 6},
                    tolerance={"near": 1}, frames=["frames/a.sxm"],
                    question="How many adsorbed atoms have their centre inside frame 1? Answer an integer.")
    elif kind == "bool":
        base.update(id="move.detect.0123abcd", type="move", subtype="detect", answer_format={"kind": "bool"},
                    truth={"value": True}, question="Did any adsorbed atom move between frame 1 and frame 2? Answer true or false.")
    base.update(kw)
    return base


# ── schema ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", I.ANSWER_KINDS)
def test_valid_items_validate(kind):
    assert I.validate_item(_item(kind)) == []


def test_validation_catches_problems():
    assert any("missing" in e for e in I.validate_item({"id": "x"}))
    assert any("not among the choices" in e for e in I.validate_item(_item("choice", truth={"value": "fuzzy"})))
    assert any("relative POSIX" in e for e in I.validate_item(_item("bool", frames=["/abs/a.sxm"])))
    assert any("relative POSIX" in e for e in I.validate_item(_item("bool", frames=["x:y.sxm"])))
    assert any("relative POSIX" in e for e in I.validate_item(_item("bool", frames=["../a.sxm"])))
    assert any("neither .sxm nor .dat" in e for e in I.validate_item(_item("bool", frames=["a.png"])))
    assert any("subtype" in e for e in I.validate_item(_item("bool", subtype="period")))
    assert any("match_nm" in e for e in I.validate_item(_item("position_list", tolerance={})))
    assert any("min <= value <= max" in e for e in I.validate_item(_item("count", truth={"value": 5, "min": 6})))
    assert any("'abs' and/or 'rel'" in e for e in I.validate_item(_item("vector", tolerance={})))
    assert any("id" in e for e in I.validate_item(_item("bool", id="move.detect.double-tip")))


def test_frame_files_checked_against_root(tmp_path):
    it = _item("bool")
    assert any("missing" in e for e in I.validate_item(it, root=tmp_path))
    for f in it["frames"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(b"x")
    assert I.validate_item(it, root=tmp_path) == []


def test_leak_checks():
    it = _item("count", provenance={"tip_condition": "double"},
               question="The tip is double. How many adsorbed atoms have their centre inside frame 1?")
    assert any("hidden tip condition" in e for e in I.validate_item(it))
    it = _item("vector", truth={"value": [1.234, -5.0]},
               question="Frames 1 and 2 drifted by about 1.23 nm. Give [dx, dy] in nm for frame 2 vs frame 1.")
    assert any("quotes the truth" in e for e in I.validate_item(it))
    # the tip item names every class as a choice: not a leak
    assert I.validate_item(_item("choice")) == []


def test_ids_are_neutral_and_deterministic():
    a = I.item_id("tip", "condition", 3, 1, "sb1")
    assert a == I.item_id("tip", "condition", 3, 1, "sb1")
    assert a != I.item_id("tip", "condition", 3, 2, "sb1")
    assert a.startswith("tip.condition.") and not any(c in a for c in I.TIP_CLASSES)


def test_normalize_answer_shapes():
    ch = _item("choice")
    assert I.normalize_answer(ch, {"value": " Double "}) == ("double", False, None)
    assert I.normalize_answer(ch, {"value": "sharp"})[2] is not None
    assert I.normalize_answer(ch, {"abstain": True}) == (None, True, None)
    assert I.normalize_answer(ch, None) == (None, True, None)
    b = _item("bool")
    assert I.normalize_answer(b, {"value": "yes"})[0] is True
    assert I.normalize_answer(b, {"value": "False"})[0] is False
    assert I.normalize_answer(b, {"value": "maybe"})[2] is not None
    c = _item("count")
    assert I.normalize_answer(c, {"value": 4.0})[0] == 4
    assert I.normalize_answer(c, {"value": 4.5})[2] is not None
    assert I.normalize_answer(c, {"value": True})[2] is not None
    assert I.normalize_answer(c, {"value": -1})[2] is not None
    v = _item("vector")
    assert I.normalize_answer(v, {"value": [1, 2]})[0] == [1.0, 2.0]
    assert I.normalize_answer(v, {"value": [1, 2, 3]})[2] is not None
    assert I.normalize_answer(v, {"value": [1, float("nan")]})[2] is not None
    p = _item("position_list")
    assert I.normalize_answer(p, {"value": [{"x": 1, "y": 2}, [3, 4]]})[0] == [[1.0, 2.0], [3.0, 4.0]]
    assert I.normalize_answer(p, {"value": []})[0] == []
    assert I.normalize_answer(p, {"value": [[1]]})[2] is not None
    assert I.normalize_answer(p, {"novalue": 1})[2] is not None


def test_answer_value_schema_per_kind():
    assert I.answer_value_schema(_item("choice"))["enum"] == list(I.TIP_CLASSES)
    assert I.answer_value_schema(_item("count"))["type"] == "integer"
    assert I.answer_value_schema(_item("vector"))["minItems"] == 2
    assert I.answer_value_schema(_item("bool"))["type"] == "boolean"


def test_jsonl_roundtrip(tmp_path):
    rows = [_item("bool"), _item("count")]
    I.write_jsonl(tmp_path / "items.jsonl", rows)
    assert I.load_items(tmp_path) == rows
    I.write_jsonl(tmp_path / "ans.jsonl", [{"id": "a", "answer": {"value": 1}}, {"id": "b", "value": 2},
                                           {"id": "a", "answer": {"abstain": True}}, {"id": "c", "answer": None},
                                           {"id": "d", "answer": 3}])
    ans = I.load_answers(tmp_path / "ans.jsonl")
    assert ans == {"a": {"abstain": True}, "b": {"value": 2}, "c": None, "d": {"value": 3}}
    assert score_item(_item("count"), ans["c"])["missing"]


# ── scorer ───────────────────────────────────────────────────────────────────
def test_choice_and_bool_scoring():
    ch = _item("choice")
    assert score_item(ch, {"value": "double"})["correct"]
    r = score_item(ch, {"value": "blunt"})
    assert r["answered"] and not r["correct"] and r["score"] == 0.0
    r = score_item(ch, {"abstain": True})
    assert r["abstained"] and not r["correct"] and not r["answered"]
    r = score_item(ch, None)
    assert r["missing"] and not r["abstained"]
    r = score_item(ch, {"value": "fuzzy"})
    assert r["invalid"] and not r["correct"]
    assert score_item(_item("bool"), {"value": True})["correct"]
    assert not score_item(_item("bool"), {"value": False})["correct"]


def test_number_scoring_with_accept_and_angle_wrap():
    it = _item("number")                                  # truth 3 deg, accept 175 and 10, tol 6 mod 180
    assert score_item(it, {"value": 179.0})["correct"]    # 4 deg from 3 across the wrap
    r = score_item(it, {"value": 168.0})                  # 7 from the accepted 175: outside 6
    assert not r["correct"] and math.isclose(r["error"], 7.0)
    assert score_item(it, {"value": 170.0})["correct"]    # 5 from 175
    assert not score_item(it, {"value": 90.0})["correct"]
    assert score_item(it, {"value": 14.0})["correct"]     # within 6 of accepted 10
    r = score_item(it, {"value": 178.0})
    assert math.isclose(r["signed_error"], -5.0)
    step = _item("number", truth={"value": -470.8}, tolerance={"abs": 30.0, "rel": 0.1},
                 answer_format={"kind": "number", "unit": "pm"})
    assert score_item(step, {"value": -430.0})["correct"]     # 40.8 < 47.08
    assert not score_item(step, {"value": -235.4})["correct"]
    assert not score_item(step, {"value": 470.8})["correct"]


def test_number_without_mod_uses_absolute_difference():
    it = _item("number", answer_format={"kind": "number", "unit": "nm"}, truth={"value": 6.3, "accept": [6.2, 6.4]},
               tolerance={"abs": 0.15})
    assert score_item(it, {"value": 6.52})["correct"]          # 0.12 from accepted 6.4
    assert not score_item(it, {"value": 6.6})["correct"]


def test_vector_scoring_abs_or_rel():
    it = _item("vector")                                 # truth [1, -5]: |t| = 5.10 -> limit max(0.3, 1.02)
    assert score_item(it, {"value": [1.5, -5.8]})["correct"]    # error 0.94
    assert not score_item(it, {"value": [2.2, -5.0]})["correct"]
    small = _item("vector", truth={"value": [0.1, 0.0]})
    assert score_item(small, {"value": [0.35, 0.1]})["correct"]  # 0.27 < 0.3
    assert not score_item(small, {"value": [0.5, 0.0]})["correct"]


def test_count_scoring_with_edge_range():
    it = _item("count")                                  # value 5, accepted 5..6
    assert score_item(it, {"value": 5})["correct"]
    assert score_item(it, {"value": 6})["correct"]
    r = score_item(it, {"value": 7})
    assert not r["correct"] and r["within_near"] and r["error"] == 1
    r = score_item(it, {"value": 3})
    assert not r["correct"] and not r["within_near"] and r["error"] == 2


def test_position_matching_hungarian():
    core = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]
    m = match_points([[0.05, 0.0], [2.0, 0.1], [0.0, 1.9]], core, [], 0.3)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 0, 0) and m["f1"] == 1.0
    m = match_points([[0.0, 0.0], [2.0, 0.0]], core, [], 0.3)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 1)
    m = match_points([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [5.0, 5.0]], core, [], 0.3)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 0)
    m = match_points([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [9.8, 0.0]], core, [[9.9, 0.0]], 0.3)
    assert (m["tp"], m["fp"], m["fn"], m["optional_matched"]) == (3, 0, 0, 1)
    m = match_points([[0.0, 0.5]], [[0.0, 0.0]], [], 0.3)          # outside the gate: a miss and a spurious point
    assert (m["tp"], m["fp"], m["fn"]) == (0, 1, 1) and m["f1"] == 0.0
    # optimal, not greedy: p0 is nearest to t0, but only the assignment p0->t1, p1->t0 matches both
    m = match_points([[0.0, 0.0], [-0.28, 0.0]], [[-0.05, 0.0], [0.25, 0.0]], [], 0.3)
    assert m["tp"] == 2
    assert match_points([], [], [], 0.3)["f1"] == 1.0
    assert match_points([], core, [], 0.3)["f1"] == 0.0


def test_position_list_item_score_is_f1():
    it = _item("position_list")
    r = score_item(it, {"value": [[0.0, 0.0], [2.0, 0.0]]})
    assert not r["correct"] and math.isclose(r["score"], 0.8)
    assert score_item(it, {"value": [[0, 0], [2, 0], [0, 2], [9.9, 0.1]]})["correct"]


def test_aggregate_tables():
    its = [_item("choice", id="tip.condition.00000001"), _item("choice", id="tip.condition.00000002", truth={"value": "good"}),
           _item("count", id="adatoms.count.00000003"), _item("vector", id="drift.vector.00000004")]
    ans = {"tip.condition.00000001": {"value": "double"}, "tip.condition.00000002": {"abstain": True},
           "adatoms.count.00000003": {"value": 7}}
    recs = score_all(its, ans)
    agg = aggregate(recs)
    o = agg["overall"]
    assert o["n"] == 4 and o["correct"] == 1 and o["abstained"] == 1 and o["missing"] == 1
    rows = {r["subtype"]: r for r in agg["subtypes"]}
    assert rows["tip.condition"]["accuracy"] == 0.5 and rows["tip.condition"]["accuracy_answered"] == 1.0
    assert rows["tip.condition"]["confusion"] == {"double": {"double": 1}, "good": {"abstain": 1}}
    assert rows["adatoms.count"]["within_near"] == 1.0
    assert rows["drift.vector"]["answered"] == 0


def test_report_files(tmp_path):
    from vista_spm.staticbench.report import write_report

    its = [_item("choice", id="tip.condition.00000001"), _item("number", id="herringbone.orientation.00000002")]
    recs_a = score_all(its, {"tip.condition.00000001": {"value": "double"}, "herringbone.orientation.00000002": {"value": 2.0}})
    recs_b = score_all(its, {})
    res = write_report({"baseline": recs_a, "nobody": recs_b}, tmp_path / "rep", dataset="x")
    csv_text = Path(res["csv"]).read_text(encoding="utf-8")
    assert "baseline,tip.condition" in csv_text and "nobody,ALL" in csv_text
    md = Path(res["md"]).read_text(encoding="utf-8")
    assert "| baseline | 2 | 2 | 100.0%" in md and "Confusion matrices" in md


def test_neutral_sxm_header_rewrite():
    from vista_spm.staticbench.generate import GenerationError, neutral_sxm_bytes

    session = "Q" + ":" + "\\work\\ab12\\session\\sim001.sxm"          # a drive-rooted path, built
    head = f":SCAN_PIXELS:\n       256       256\n:SCAN_FILE:\n{session}\n:COMMENT:\n\n:SCANIT_END:\n\n"
    raw = head.encode("gbk") + b"\x1a\x04" + bytes(range(256))
    out = neutral_sxm_bytes(raw, "frames/0123456789ab.sxm")
    h, data = out.split(b"\x1a\x04", 1)
    assert data == bytes(range(256))
    assert b":SCAN_FILE:\nframes/0123456789ab.sxm\n" in h and session.encode() not in h
    with pytest.raises(GenerationError):
        neutral_sxm_bytes(b"no header here", "frames/x.sxm")
    leaky = (head.replace(":COMMENT:\n\n", f":COMMENT:\n{session}\n")).encode("gbk") + b"\x1a\x04"
    with pytest.raises(GenerationError):                                # another field still names a drive path
        neutral_sxm_bytes(leaky, "frames/x.sxm")


# ── truth geometry (no simulator needed) ─────────────────────────────────────
def _rec(offsets: np.ndarray, *, angle: float = 0.0, scan_dir: str = "down", n: int = 64, size: float = 20.0):
    from vista_spm.staticbench._sim import FrameRec

    return FrameRec(path=Path("x.sxm"), cx_nm=3.0, cy_nm=-2.0, w_nm=size, h_nm=size, angle_deg=angle, nx=n, ny=n,
                    line_time_s=0.3, scan_dir=scan_dir, t_start=100.0, per_row_s=0.6, offsets_nm=offsets,
                    tip_change_rows=[], tip_events=[], hops=[], bias_v=0.1, setpoint_a=1e-10)


@pytest.mark.parametrize("angle,scan_dir", [(0.0, "down"), (30.0, "down"), (-15.0, "up")])
def test_sample_to_scan_inverts_per_row_drift(angle, scan_dir):
    n = 64
    k = np.arange(n, dtype=float)
    offs = np.column_stack([5.0 + 0.02 * k, -3.0 + 0.035 * k + 0.0002 * k ** 2])     # drift + creep curvature
    rec = _rec(offs, angle=angle, scan_dir=scan_dir, n=n)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y = rec.uv_to_scan(*(rng.uniform(-9.0, 9.0, 2)))
        sx, sy = rec.scan_to_sample(x, y)
        bx, by, row = rec.sample_to_scan(sx, sy)
        assert math.hypot(bx - x, by - y) < 1e-5
        assert abs(row - rec.scan_to_pixel(x, y)[1]) < 1e-6
    col, row = rec.scan_to_pixel(rec.cx_nm, rec.cy_nm)
    assert abs(col - (n / 2 - 0.5)) < 1e-9 and abs(row - (n / 2 - 0.5)) < 1e-9


def test_pixel_convention_matches_human_render():
    from stmbench.human.render import pixel_to_scan_nm

    rec = _rec(np.zeros((64, 2)), angle=25.0)
    geom = {"cx_nm": rec.cx_nm, "cy_nm": rec.cy_nm, "w_nm": rec.w_nm, "h_nm": rec.h_nm, "angle_deg": 25.0,
            "nx": 64, "ny": 64}
    for col, row in ((0, 0), (10.5, 40.25), (63, 63)):
        assert np.allclose(rec.pixel_to_scan(col, row), pixel_to_scan_nm(geom, col, row), atol=1e-9)


def test_apparent_arms_without_and_with_drift():
    from vista_spm.staticbench._sim import angle_diff_180, apparent_arms

    hb = {"period_nm": 6.4, "shear": math.tan(math.radians(12.0))}
    orient = {"stripe_deg": 120.0}
    still = apparent_arms(_rec(np.zeros((64, 2))), orient, hb)
    for arm in still["arms"]:
        assert math.isclose(arm["period_nm"], 6.4 * math.cos(math.radians(12.0)), rel_tol=1e-9)
    angles = sorted(a["angle_deg"] for a in still["arms"])
    assert np.allclose(angles, [108.0, 132.0])
    assert angle_diff_180(still["mean_angle_deg"], 120.0) < 1e-9
    rot = apparent_arms(_rec(np.zeros((64, 2)), angle=30.0), orient, hb)        # the image turns with the scan
    assert angle_diff_180(rot["mean_angle_deg"], 90.0) < 1e-9
    # a drift that moves the sample by +2 nm in y over the frame stretches the image along y
    k = np.arange(64, dtype=float)
    drifted = apparent_arms(_rec(np.column_stack([np.zeros(64), -2.0 * k / 63])), orient, hb)
    assert all(abs(a["period_nm"] - b["period_nm"]) > 1e-3 for a, b in zip(drifted["arms"], still["arms"]))


# ── QA host with stubs (no MAST, no archive code) ────────────────────────────
class _Entry:
    def __init__(self, fid):
        self.fid = fid

    def summary(self):
        return {"frame": self.fid, "kind": "scan"}


class _StubArchive:
    def __init__(self):
        self.fids = []

    def add_scan(self, turn, path):
        e = _Entry(f"t0000.s{sum(f.startswith('t0000.s') for f in self.fids)}")
        self.fids.append(e.fid)
        return e

    def add_spectrum(self, turn, path):
        e = _Entry(f"t0000.d{sum(f.startswith('t0000.d') for f in self.fids)}")
        self.fids.append(e.fid)
        return e

    def get(self, fid):
        return _Entry(fid)

    def observation(self, fid):
        from vista_spm.protocol import ImageBlock

        return [ImageBlock(png=b"\x89PNG", label=f'<visual kind="current" frame="{fid}"/>')]

    def text_grid(self, fid, n=64):
        return f"grid {fid}"


def _stub_host(tmp_path, kind="choice", **kw):
    from vista_spm.protocol import ToolDef, json_reply
    from vista_spm.staticbench.qa_host import QAHost

    it = _item(kind)
    for f in it["frames"]:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_bytes(b"stub")
    calls = []

    def inspect(archive, args):
        calls.append(args)
        return json_reply({"ok": True})

    view = {"inspect": (ToolDef("inspect", "look", {"type": "object"}), inspect),
            "read_values": (ToolDef("read_values", "read", {"type": "object"}), inspect)}
    host = QAHost(it, tmp_path, tmp_path / "work", archive=_StubArchive(), view_tools=view, measure_tools={}, **kw)
    return host, it, calls


def test_qa_host_protocol_and_answer(tmp_path):
    from vista_spm.protocol import RunnerHost

    host, it, calls = _stub_host(tmp_path)
    assert isinstance(host, RunnerHost)
    names = host.tool_names()
    assert names[-1] == "answer" and "inspect" in names and "act" not in names
    text, images = host.initial_message()
    assert it["question"] in text and "frame 1 = t0000.s0" in text and "spectrum 2 = t0000.d0" in text
    assert len(images) == 2
    assert "truth" not in text and "double" in text            # the choices are listed, the truth is not marked
    r = host.call("inspect", {"views": []})
    assert not r.is_error and calls
    r = host.call("answer", {"value": "sharp"})
    assert r.is_error and host.answer is None and not host.status()["terminal"]
    r = host.call("answer", {"value": "Double", "confidence": 0.8, "note": "ghost copies"})
    assert not r.is_error and host.answer == {"value": "double", "abstain": False, "confidence": 0.8, "note": "ghost copies"}
    st = host.status()
    assert st["terminal"] and st["acts"] == 0 and st["answered"]
    assert "Answer already recorded" in host.recovery_message()[0]
    host.finish("done", "ok [DONE]")
    res = host.result()
    assert res["answer"]["value"] == "double" and res["outcome"] == "done" and res["tools_used"]["answer"] == 2
    json.dumps(res)


def test_qa_host_abstain_budget_and_checkpoint(tmp_path):
    host, _, _ = _stub_host(tmp_path, kind="count", max_tool_calls=2)
    assert host.call("inspect", {}).is_error is False
    assert host.call("inspect", {}).is_error is False
    r = host.call("inspect", {})
    assert r.is_error and "budget" in r.text
    assert host.call("nonsense", {}).is_error
    assert host.call("answer", {}).is_error                     # neither value nor abstain
    assert not host.call("answer", {"abstain": True}).is_error
    assert host.answer == {"abstain": True}
    host.request_checkpoint("image limit")
    assert host.status()["checkpoint"] == "ready"
    host.begin_segment(1, "s1")
    assert host.status()["checkpoint"] == "none"


def test_qa_host_text_arm_has_no_images(tmp_path):
    host, _, _ = _stub_host(tmp_path, arm="text")
    text, images = host.initial_message()
    assert images == [] and "grid t0000.s0" in text
    assert "inspect" not in host.tool_names() and "read_values" in host.tool_names()


def test_runner_settings_fit_the_runner_config():
    from vista_spm.staticbench.qa_host import (ANSWER_TOOL, QA_SYSTEM_PROMPT, QA_SYSTEM_PROMPT_IMAGES_ONLY,
                                                    runner_settings)

    s = runner_settings(model="claude-opus-5-5")
    assert s["progress_tools"] == (ANSWER_TOOL,) and s["act_tools"] == () and s["compaction"] == "native"
    assert s["system_prompt"] == QA_SYSTEM_PROMPT
    assert runner_settings(images_only=True)["system_prompt"] == QA_SYSTEM_PROMPT_IMAGES_ONLY
    cr = pytest.importorskip("vista_spm.claude_runner")
    cfg = cr.RunnerConfig(**s)
    assert cfg.progress_tools == (ANSWER_TOOL,)


def test_answer_ends_with_a_runner_outcome(tmp_path):
    """A terminal host is reported by its stop_reason; it must be one of the runner's outcomes."""
    host, _, _ = _stub_host(tmp_path)
    host.call("answer", {"value": "good"})
    st = host.status()
    assert st["terminal"] and st["stop_reason"] == "done"
    cr = pytest.importorskip("vista_spm.claude_runner")
    assert st["stop_reason"] in cr.OUTCOMES


def test_images_only_host_says_so(tmp_path):
    host, _, _ = _stub_host(tmp_path, arm="img")
    assert host.tool_names() == ["answer"] and host.images_only
    text, images = host.initial_message()
    assert "no tool except `answer`" in text and "tool calls" not in text and len(images) == 2
    assert "X = cx + cos(a)*u - sin(a)*v" in text          # pixel → scan-nm conversion spelled out


def test_infra_error_and_spend(tmp_path):
    from vista_spm.staticbench.claude_qa import infra_error, spent_usd

    limit = [{"end_reason": "api_error", "results": [{"is_error": True, "api_error_status": 429}]}]
    assert infra_error(limit, answered=False) == "api_429"
    assert infra_error(limit, answered=True) is None                 # it answered before the limit: a result
    assert infra_error([{"end_reason": "terminal", "results": [{"is_error": False}]}], answered=False) is None
    # the runner's own classification of a subscription usage limit (end_reason / error text)
    usage = [{"end_reason": "usage_limit", "results": [{"is_error": True, "api_error_status": 429}]}]
    assert infra_error(usage, answered=False) == "usage_limit"
    assert infra_error([], answered=False, runner_failed=True, runner_error="usage_limit: hit your session limit") \
        == "usage_limit"
    assert infra_error([], answered=False, preflight={"ok": False, "problems": ["usage limit: resets 8:30pm"]}) \
        == "usage_limit"
    assert infra_error([], answered=False, preflight={"ok": False, "problems": ["no system/init envelope"]}) \
        == "preflight_failed"
    # spend: manifests (cut-off sessions too) or answers files, whichever is larger
    io = tmp_path / "full" / "work" / "x.y.0000000a" / "session0" / "claude_io"
    io.mkdir(parents=True)
    (io / "manifest.json").write_text(json.dumps({"totals": {"cost_usd": 0.25, "preflight_cost_usd": 0.05}}),
                                      encoding="utf-8")
    assert spent_usd(tmp_path) == pytest.approx(0.30)
    I.write_jsonl(tmp_path / "image" / "answers.jsonl", [{"id": "a", "answer": None,
                                                          "driver": {"usage": {"cost_usd": 0.5}}}])
    assert spent_usd(tmp_path) == pytest.approx(0.5)


def test_api_key_sessions_are_flagged():
    from vista_spm.staticbench.claude_qa import api_key_problem, api_key_sources, infra_error

    ok = [{"init": {"apiKeySource": "none"}, "problems": []}]
    assert api_key_problem(ok) is None and api_key_sources(ok) == ["none"]
    billed = [{"init": {"apiKeySource": "ANTHROPIC_API_KEY"}, "problems": []}]
    assert "ANTHROPIC_API_KEY" in api_key_problem(billed)
    refused = [{"init": {}, "end_reason": "init_invalid",
                "problems": ["init #1: apiKeySource 'ANTHROPIC_API_KEY': this session would bill an API key"]}]
    assert api_key_problem(refused) and infra_error(refused, answered=False) == "auth_api_key"
    assert api_key_problem([], {"ok": False, "problems": ["apiKeySource 'x': refused"]})


def test_score_out_leaves_out_infra_failures(tmp_path):
    from vista_spm.staticbench.claude_qa import score_out

    ds = tmp_path / "ds"
    a, b = _item("bool", id="move.detect.0000000a"), _item("count", id="adatoms.count.0000000b")
    I.write_jsonl(ds / "items.jsonl", [a, b])
    I.write_jsonl(ds / "answers_baseline.jsonl", [{"id": a["id"], "answer": {"value": True}},
                                                  {"id": b["id"], "answer": {"value": 4}}])
    I.write_jsonl(tmp_path / "out" / "full" / "answers.jsonl", [
        {"id": a["id"], "answer": {"value": True}, "driver": {"infra_error": None}},
        {"id": b["id"], "answer": None, "driver": {"infra_error": "api_429"}}])
    res = score_out(ds, tmp_path / "out")
    assert res["not_attempted"] == {"claude-full": [b["id"]]}
    assert res["overall"]["claude-full"]["n"] == 1 and res["overall"]["claude-full"]["correct"] == 1
    assert res["overall"]["baseline (classical)"]["n"] == 1        # the baseline on the same attempted items


def test_pilot_ids_one_per_subtype():
    from vista_spm.staticbench.claude_qa import pilot_ids

    its = [_item("choice", id="tip.condition.00000002"), _item("choice", id="tip.condition.00000001", difficulty="hard"),
           _item("count", id="adatoms.count.0000000a"), _item("vector", id="drift.vector.0000000b")]
    assert pilot_ids(its) == ["tip.condition.00000002", "adatoms.count.0000000a", "drift.vector.0000000b"]


def test_qa_host_refuses_invalid_item(tmp_path):
    from vista_spm.staticbench.qa_host import QAHost

    with pytest.raises(ValueError):
        QAHost(_item("choice", truth={"value": "fuzzy"}), tmp_path, tmp_path / "w", archive=_StubArchive(),
               view_tools={}, measure_tools={})


# ── end to end: simulator → items → baseline → score (and the real QA host) ─
@requires_mast
def test_end_to_end_tiny_dataset(tmp_path):
    from vista_spm.staticbench.baseline import answer_item
    from vista_spm.staticbench.generate import Recipe, run_recipe

    recipes = [Recipe("adatoms", 0, 0, {"tip": "good"}), Recipe("drift", 0, 0, {"substrate": "cu", "compensate": False})]
    outs = {}
    for name in ("a", "b"):
        items = []
        for r in recipes:
            res = run_recipe(r.as_dict(), str(tmp_path / name))
            assert res["error"] is None, res["error"]
            items += res["items"]
        outs[name] = items
    items = outs["a"]
    assert [it["subtype"] for it in items] == ["count", "positions", "vector"]
    for it in items:
        assert I.validate_item(it, root=tmp_path / "a") == []
    # deterministic: a second run in another directory gives the same items (ids, truth, provenance)
    # and bit-identical frame data (the header differs only in the file path it records)
    assert json.dumps(outs["a"], sort_keys=True) == json.dumps(outs["b"], sort_keys=True)
    from stmsim.io.sxm_reader import read_sxm

    for f in {f for it in items for f in it["frames"]}:
        sa, sb = read_sxm(str(tmp_path / "a" / f)), read_sxm(str(tmp_path / "b" / f))
        for ch, dirs in sa["channels"].items():
            for d in ("forward", "backward"):
                assert np.array_equal(np.asarray(dirs[d]), np.asarray(sb["channels"][ch][d]), equal_nan=True), (f, ch, d)
        # nothing in a frame's header says where or how it was made: the file names itself neutrally
        raw = (tmp_path / "a" / f).read_bytes()
        assert raw == (tmp_path / "b" / f).read_bytes()
        assert f":SCAN_FILE:\n{f}\n".encode() in raw[:raw.find(b"\x1a\x04")]
    assert not (tmp_path / "a" / "_work").exists() or not any((tmp_path / "a" / "_work").iterdir())
    recs = [score_item(it, answer_item(it, tmp_path / "a")["answer"]) for it in items]
    by = {r["subtype"]: r for r in recs}
    assert all(r["answered"] for r in recs)
    assert by["count"]["within_near"]
    assert by["positions"]["score"] >= 0.6
    assert by["vector"]["error"] < 0.6


@requires_mast
def test_end_to_end_qa_host_with_real_archive(tmp_path):
    pytest.importorskip("vista_spm.archive")
    pytest.importorskip("vista_spm.measure")
    from vista_spm.staticbench.generate import Recipe, run_recipe
    from vista_spm.staticbench.qa_host import QAHost

    res = run_recipe(Recipe("move", 1, 1, {"moved": False}).as_dict(), str(tmp_path / "ds"))
    assert res["error"] is None, res["error"]
    it = res["items"][0]
    host = QAHost(it, tmp_path / "ds", tmp_path / "qa")
    text, images = host.initial_message()
    assert len(images) == 2 and "frame 2 = t0000.s1" in text
    with pytest.raises(FileExistsError):                 # a reused archive would renumber the frames
        QAHost(it, tmp_path / "ds", tmp_path / "qa")
    r = host.call("diff", {"frame_a": host.frame_ids[0], "frame_b": host.frame_ids[1]})
    assert not r.is_error, r.text
    assert not host.call("answer", {"value": False}).is_error
    assert score_item(it, host.result()["answer"])["correct"]
    # the batch helper: one session per item, results appended as an answers file, resumable
    from vista_spm.staticbench.qa_host import run_items

    I.write_jsonl(tmp_path / "ds" / "items.jsonl", [it])
    out = tmp_path / "answers.jsonl"
    rows = run_items(tmp_path / "ds", out, lambda h: h.call("answer", {"value": True}), work_root=tmp_path / "w")
    assert len(rows) == 1 and rows[0]["answer"]["value"] is True
    assert run_items(tmp_path / "ds", out, lambda h: None, work_root=tmp_path / "w") == []
    recs = score_all([it], I.load_answers(out))
    assert recs[0]["answered"] and not recs[0]["correct"]


FAKE_CLAUDE = Path(__file__).resolve().parent / "fake_claude.py"


@requires_mast
def test_claude_driver_against_the_fake_cli(tmp_path):
    """QAHost + RpcServer + ClaudeCodeRunner + answers file + scoring, with WP-B's scripted
    stand-in for ``claude -p`` (real bridge, real hooks, no API call)."""
    import sys

    pytest.importorskip("vista_spm.claude_runner")
    if not FAKE_CLAUDE.is_file():
        pytest.skip("tests/fake_claude.py not present")
    from vista_spm.staticbench.claude_qa import run_claude_items, score_out, spent_usd
    from vista_spm.staticbench.generate import Recipe, run_recipe

    ds = tmp_path / "ds"
    res = run_recipe(Recipe("move", 1, 1, {"moved": False}).as_dict(), str(ds))
    assert res["error"] is None, res["error"]
    (it,) = res["items"]
    I.write_jsonl(ds / "items.jsonl", [it])

    def overrides(name: str, turns: list) -> dict:
        scenario = tmp_path / f"{name}.json"
        scenario.write_text(json.dumps({"sessions": [turns]}), encoding="utf-8")
        return {"claude_bin": [sys.executable, "-I", "-X", "utf8", str(FAKE_CLAUDE)], "python_exe": sys.executable,
                "extra_env": {"FAKE_CLAUDE_SCENARIO": str(scenario)}, "end_grace_s": 15.0, "startup_timeout_s": 60.0}

    full = overrides("full", [[{"tool": "diff", "args": {"frame_a": "t0000.s0", "frame_b": "t0000.s1"}},
                               {"tool": "answer", "args": {"value": False, "confidence": 0.9}},
                               {"text": "No atom moved. [DONE]"}]])
    out = tmp_path / "out"
    summary = run_claude_items(ds, out, ids=[it["id"]], tools="full", preflight="none", runner_overrides=full,
                               log=lambda s: None)
    assert [r["outcome"] for r in summary["run"]] == ["done"]
    (row,) = I.read_jsonl(out / "full" / "answers.jsonl")
    assert row["answer"]["value"] is False and row["tools_used"] == {"diff": 1, "answer": 1}
    assert row["driver"]["usage"]["cost_usd"] > 0 and row["driver"]["tool_counts"]["answer"] == 1
    assert row["driver"]["max_budget_usd_per_segment"] <= 2.0
    assert (out / row["driver"]["io_dir"] / "manifest.json").is_file()
    # resumable: an answered item is not run again
    again = run_claude_items(ds, out, ids=[it["id"]], tools="full", preflight="none", runner_overrides=full,
                             log=lambda s: None)
    assert again["run"] == [] and again["skipped_done"] == [it["id"]]
    # the images-only arm offers nothing but answer; a spent budget stops before anything starts
    image = overrides("image", [[{"tool": "answer", "args": {"value": True}}, {"text": "[DONE]"}]])
    stopped = run_claude_items(ds, out, ids=[it["id"]], tools="image", preflight="none", runner_overrides=image,
                               budget_usd=spent_usd(out), log=lambda s: None)
    assert stopped["stopped_for_budget"] and stopped["run"] == []
    ran = run_claude_items(ds, out, ids=[it["id"]], tools="image", preflight="none", runner_overrides=image,
                           log=lambda s: None)
    (img_row,) = I.read_jsonl(out / "image" / "answers.jsonl")
    assert ran["run"][0]["outcome"] == "done" and img_row["tools_used"] == {"answer": 1}
    scored = score_out(ds, out)
    assert scored["overall"]["claude-full"]["correct"] == 1 and scored["overall"]["claude-image"]["correct"] == 0
    assert (out / "report" / "report.md").is_file() and (out / "full" / "scores.jsonl").is_file()
