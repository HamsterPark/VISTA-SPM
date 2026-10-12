"""VistaHost and the V0 action layer, with fakes (no MAST, no simulator)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from vista_spm.actions import ACTIONS, ActionError, ActionRunner, validate_params
from vista_spm.arms import ARMS, Arm
from vista_spm.host import GATE_REFUSAL, VistaHost
from vista_spm.protocol import ImageBlock, frame_id


# ── fakes ───────────────────────────────────────────────────────────────────
@dataclass
class FakeResult:
    success: bool = True
    error: str = ""
    data: dict = field(default_factory=dict)


class FakeClock:
    def __init__(self):
        self.depth = 0
        self.t = 0.0
        self.resumes = 0

    def pause(self):
        self.depth += 1

    def resume(self):
        self.depth -= 1
        self.resumes += 1

    def sim(self):
        return self.t


class FakeCtx:
    """Records skill calls; ScanAt writes a file into the session dir (autosave)."""

    def __init__(self, session: Path, clock: FakeClock, autosave: bool = True):
        self.session = session
        self.clock = clock
        self.autosave = autosave
        self.calls: list[tuple[str, dict]] = []
        self.n = 0
        self.fail: set[str] = set()

    def run(self, name, params):
        self.calls.append((name, dict(params)))
        if name in self.fail:
            return FakeResult(False, "refused by SafetyGate")
        if name in ("ScanAt", "SaveScan"):
            if name == "SaveScan" or self.autosave:
                self.n += 1
                (self.session / f"frame_{self.n:03d}.sxm").write_bytes(b"x")
            if name == "ScanAt":
                self.clock.t += 100.0
            return FakeResult(True, data={"line_time_s": 0.5, "requested_pixels": 128})
        if name == "AcquireSTS":
            self.n += 1
            (self.session / f"spec_{self.n:03d}.dat").write_bytes(b"x")
            return FakeResult(True)
        readbacks = {"GetBias": {"bias_v": 0.1}, "GetSetpoint": {"setpoint_a": 5e-11},
                     "GetCurrent": {"current_a": 4.9e-11}, "GetZPosition": {"z_pos_m": 1.2e-9},
                     "GetScanXYPosition": {"x_m": 1e-9, "y_m": -2e-9},
                     "GetDriftCompensation": {"enabled": True, "vx": 2e-11, "vy": 0.0, "vz": 0.0,
                                              "x_saturated": False, "y_saturated": False, "z_saturated": False}}
        if name in readbacks:
            return FakeResult(True, data=readbacks[name])
        self.clock.t += 1.0
        return FakeResult(True)


@dataclass
class FakeEntry:
    fid: str
    kind: str
    turn: int
    nx: int = 128
    ny: int = 128

    def summary(self):
        return {"frame": self.fid, "kind": self.kind}


class FakeArchive:
    def __init__(self):
        self.items: list[FakeEntry] = []

    def _add(self, turn, kind):
        i = sum(1 for e in self.items if e.turn == turn and e.kind == kind)
        e = FakeEntry(frame_id(turn, kind, i), kind, turn)
        self.items.append(e)
        return e

    def add_scan(self, turn, path, sim_s=None):
        return self._add(turn, "s")

    def add_spectrum(self, turn, path, sim_s=None):
        return self._add(turn, "d")

    def latest(self, kind="s"):
        xs = [e for e in self.items if e.kind == kind]
        return xs[-1] if xs else None

    def get(self, fid):
        for e in self.items:
            if e.fid == fid:
                return e
        raise KeyError(f"no frame {fid!r}")

    def index(self):
        return [{"frame": e.fid, "kind": e.kind} for e in self.items]

    def frames(self, turn=None, kind=None):
        return [e for e in self.items if (turn is None or e.turn == turn) and (kind is None or e.kind == kind)]

    def observation(self, fid):
        return [ImageBlock(b"\x89PNG fake", f'<visual frame="{fid}"/>')]

    def text_grid(self, fid):
        return "1 2\n3 4"


class FakeNotes:
    def __init__(self):
        self.g, self.w = "No reliable model yet.\n", ""

    def read_guide(self):
        return self.g

    def write_guide(self, content):
        self.g = content

    def read_working(self):
        return self.w

    def write_working(self, content):
        self.w = content


NO_DEPS = Arm("t", "image", notes=False, history=True, inspect=False, read_values=False, measure=False,
              advisor=False)


def make_host(tmp_path, arm=NO_DEPS, claims=("target_site",), max_sim_s=1000.0, notes=None, autosave=True,
              **kw):
    session = tmp_path / "session"
    session.mkdir(parents=True, exist_ok=True)
    clock = FakeClock()
    clock.pause()
    ctx = FakeCtx(session, clock, autosave=autosave)
    host = VistaHost(arm=arm, vista_dir=tmp_path / "vista", task="Move one atom.", claim_ids=list(claims),
                     ctx=ctx, clock=clock, session_dir=session, time_scale=20.0,
                     budget_probe=lambda: (clock.t, 0), max_sim_s=max_sim_s, max_cmds=None,
                     archive=FakeArchive(), notes=notes, **kw)
    return host, ctx, clock


def body(reply):
    return json.loads(reply.text)


SCAN = {"action": "scan", "params": {"center": {"x_nm": 0, "y_nm": 0}, "size_nm": 50}}


# ── tests ───────────────────────────────────────────────────────────────────
def test_tool_surface_follows_the_arm(tmp_path):
    host, _, _ = make_host(tmp_path)
    assert host.tool_names() == ["act", "report_result", "report_tip_state", "history"]
    img = ARMS["img"]
    host2, _, _ = make_host(tmp_path / "b", arm=img, claims=())
    assert host2.tool_names() == ["act", "report_tip_state"]
    tools = {t["name"]: t for t in host.list_tools()}
    assert tools["act"]["annotations"]["readOnlyHint"] is False
    assert tools["act"]["inputSchema"]["additionalProperties"] is False
    assert set(tools["act"]["inputSchema"]["properties"]["action"]["enum"]) >= {"scan", "move_atom", "wait"}


def test_scan_produces_one_frame_and_an_image(tmp_path):
    host, ctx, clock = make_host(tmp_path)
    r = host.call("act", SCAN)
    b = body(r)
    assert not r.is_error and b["ok"]
    assert [f["frame"] for f in b["new_frames"]] == ["t0001.s0"]
    assert len(r.images) == 1
    assert [c[0] for c in ctx.calls if c[0] in ("ScanAt", "SaveScan")] == ["ScanAt"]   # autosaved: no SaveScan
    assert clock.depth == 1 and clock.resumes == 1        # ran with the clock, paused again
    assert b["instrument"]["bias_v"] == 0.1 and b["instrument"]["current_pa"] == pytest.approx(49.0)
    assert b["instrument"]["drift_compensation"]["vx_nm_s"] == pytest.approx(0.02)


def test_scan_saves_explicitly_without_autosave(tmp_path):
    host, ctx, _ = make_host(tmp_path, autosave=False)
    host.call("act", SCAN)
    assert [c[0] for c in ctx.calls if c[0] in ("ScanAt", "SaveScan")] == ["ScanAt", "SaveScan"]


def test_one_action_one_observation_gate(tmp_path):
    host, _, _ = make_host(tmp_path)
    assert not host.call("act", SCAN).is_error
    r = host.call("act", {"action": "set_bias", "params": {"bias_v": 0.2}})
    assert r.is_error and GATE_REFUSAL in body(r)["error"]
    host.acknowledge_act()
    assert not host.call("act", {"action": "set_bias", "params": {"bias_v": 0.2}}).is_error


def test_gate_fallback_after_timeout(tmp_path):
    host, _, _ = make_host(tmp_path, gate_fallback_s=0.0)
    host.call("act", SCAN)
    assert not host.call("act", {"action": "set_bias", "params": {"bias_v": 0.2}}).is_error
    assert host.gate_fallbacks == 1


def test_validation_errors_do_not_execute_or_close_the_gate(tmp_path):
    host, ctx, _ = make_host(tmp_path)
    r = host.call("act", {"action": "set_bias", "params": {"bias": 1}})
    assert r.is_error and "unknown parameter" in body(r)["error"]
    r = host.call("act", {"action": "teleport", "params": {}})
    assert r.is_error and "unknown action" in body(r)["error"]
    assert not any(c[0] == "SetBias" for c in ctx.calls)
    assert not host.call("act", {"action": "set_bias", "params": {"bias_v": 0.3}}).is_error


def test_failed_skill_is_an_error_reply_with_state(tmp_path):
    host, ctx, _ = make_host(tmp_path)
    ctx.fail.add("SetBias")
    r = host.call("act", {"action": "set_bias", "params": {"bias_v": 0.3}})
    b = body(r)
    assert r.is_error and b["ok"] is False and "SafetyGate" in b["error"]
    assert "instrument" in b


def test_budget_closes_the_instrument_but_not_the_episode(tmp_path):
    host, _, _ = make_host(tmp_path, max_sim_s=150.0)
    host.call("act", SCAN)
    host.acknowledge_act()
    r = host.call("act", SCAN)                      # 200 s used >= 150
    assert host.instrument_closed and host.stop_reason == "budget_sim"
    assert host.terminal is False and host.status()["instrument_closed"] is True
    assert "episode" in body(r)
    host.acknowledge_act()
    r = host.call("act", SCAN)
    assert r.is_error and "no longer available" in body(r)["error"]
    assert not host.call("report_result", {"claim_id": "target_site", "value": 1.0}).is_error
    host.finish("done", "[DONE]")
    assert host.terminal and host.status()["terminal"] is True


def test_spectrum_action_archives_a_spectrum(tmp_path):
    host, _, _ = make_host(tmp_path)
    r = host.call("act", {"action": "sts", "params": {"start_v": -0.5, "end_v": 0.5}})
    assert [f["frame"] for f in body(r)["new_frames"]] == ["t0001.d0"]


def test_reports_fold_like_the_judge(tmp_path):
    host, _, _ = make_host(tmp_path)
    assert host.call("report_result", {"claim_id": "nope", "value": 1}).is_error
    assert host.call("report_result", {"claim_id": "target_site", "value": "1"}).is_error
    host.call("report_result", {"claim_id": "target_site", "value": 1.0, "x_nm": 1.0, "y_nm": 2.0})
    host.call("report_result", {"claim_id": "target_site", "value": 2.0, "x_nm": 3.0, "y_nm": 4.0})
    assert host.results_folded()["claims"]["target_site"]["x_nm"] == 3.0
    assert host.call("report_tip_state", {"junction": "clean", "atomic_resolution": False,
                                          "tip_state": "double", "reason": "ghosts"}).effect == "report"
    assert host.call("report_tip_state", {"junction": "shiny", "atomic_resolution": False,
                                          "tip_state": "double", "reason": "x"}).is_error
    assert host.diagnosis()["tip_state"] == "double"


def test_history_views(tmp_path):
    host, _, _ = make_host(tmp_path)
    for v in (0.1, 0.2, 0.3):
        host.call("act", {"action": "set_bias", "params": {"bias_v": v}})
        host.acknowledge_act()
    ev = body(host.call("history", {"view": "events", "start_turn": 2, "limit": 5}))
    assert [e["turn"] for e in ev["events"]] == [2, 3]
    assert "steps" not in ev["events"][0]
    att = body(host.call("history", {"view": "attempts"}))
    assert att["actions"]["set_bias"]["n"] == 3 and att["results_missing"] == ["target_site"]


def test_checkpoint_gates_everything_but_notes(tmp_path, monkeypatch):
    import vista_spm.host as H
    from vista_spm.protocol import ToolDef

    fake_notes_tools = {n: ToolDef(n, n, {"type": "object", "properties": {}, "additionalProperties": True})
                        for n in H.NOTES_TOOL_NAMES}
    import types
    mod = types.ModuleType("vista_spm.notes")
    mod.NOTES_TOOLS = fake_notes_tools
    monkeypatch.setitem(__import__("sys").modules, "vista_spm.notes", mod)
    arm = Arm("n", "image", notes=True, history=True, inspect=False, read_values=False, measure=False,
              advisor=False)
    host, _, _ = make_host(tmp_path, arm=arm, notes=FakeNotes())
    host.request_checkpoint("context")
    r = host.call("act", SCAN)
    assert r.is_error and "WORKING.md" in body(r)["error"]
    assert not host.call("read_guide", {}).is_error
    r = host.call("write_working", {"content": "plan: scan again"})
    assert not r.is_error and host.status()["checkpoint"] == "ready"
    host.begin_segment(1, "sid")
    assert host.status()["checkpoint"] == "none" and host.images_delivered == 0
    host.request_checkpoint("context")            # a crash while requested: the next segment starts clean
    host.begin_segment(2, "sid2")
    assert host.status()["checkpoint"] == "none"
    text, images = host.recovery_message()
    assert "plan: scan again" in text and "Move one atom." in text


def test_point_resolution_needs_a_known_scan_frame(tmp_path):
    host, _, _ = make_host(tmp_path)
    r = host.call("act", {"action": "move_tip", "params": {"to": {"frame": "t0009.s0", "x": 1, "y": 2}}})
    assert r.is_error and "t0009.s0" in body(r)["error"]
    r = host.call("act", {"action": "move_tip", "params": {"to": {"x_nm": 1}}})
    assert r.is_error


def test_validate_params_ranges_and_defaults():
    spec = ACTIONS["sts"]
    out = validate_params(spec, {"start_v": -0.5, "end_v": 0.5})
    assert out["num_points"] == 256
    with pytest.raises(ActionError):
        validate_params(spec, {"start_v": -9, "end_v": 0.5})
    with pytest.raises(ActionError):
        validate_params(ACTIONS["scan"], {"center": {"x_nm": 0, "y_nm": 0}, "size_nm": 10, "pixels": 12.5})


def test_qplus_only_actions_are_hidden_on_an_stm_tip(tmp_path):
    host, _, _ = make_host(tmp_path)
    assert "df_curve" not in host.actions and "pll" not in host.actions
    host2, _, _ = make_host(tmp_path / "q", is_qplus=True)
    assert "df_curve" in host2.actions


def test_action_runner_never_raises():
    class Boom:
        def run(self, name, params):
            raise RuntimeError("socket closed")
    r = ActionRunner(Boom(), resolve_point=lambda s: (0.0, 0.0, {}), time_scale=1.0)
    out = r.run(ACTIONS["set_bias"], {"bias_v": 0.1})
    assert out.ok is False and "socket closed" in out.error
