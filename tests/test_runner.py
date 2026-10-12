"""Claude Code segment runner (vista_spm.claude_runner, DESIGN.md §7).

Most tests run the runner against ``fake_claude.py`` — a scripted stand-in for
``claude -p`` that drives the real MCP bridge and hook script — and the toy host from
``fakes.py``. Tests marked ``live`` run the real Claude Code CLI; they are skipped
unless ``VISTA_LIVE_CLAUDE=1`` (they cost a few cents each; ``VISTA_LIVE_OUT`` keeps their
files in a chosen directory).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from vista_spm.claude_runner import (
    DEFAULT_CONTINUE_TEXT, PREFLIGHT_PROMPT, ClaudeCodeRunner, RunnerConfig, build_command,
    child_environment, classify_result, hook_settings, mcp_config, result_usage, user_message,
    validate_init)
from vista_spm.compact_hook import CHECKPOINT_INSTRUCTION
from vista_spm.protocol import ImageBlock
from vista_spm.rpc import RpcServer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import ToyHost, png_squares  # noqa: E402

PY = sys.executable
FAKE = Path(__file__).resolve().parent / "fake_claude.py"
LIVE = os.environ.get("VISTA_LIVE_CLAUDE") == "1"
live = pytest.mark.skipif(not LIVE, reason="real Claude Code CLI test: set VISTA_LIVE_CLAUDE=1 (costs money)")

ACT = {"tool": "act", "args": {"action": "count"}}


# ── pure helpers ─────────────────────────────────────────────────────────────
def test_config_validation_and_summary(tmp_path):
    with pytest.raises(ValueError):
        RunnerConfig(effort="huge")
    with pytest.raises(ValueError):
        RunnerConfig(compact_window_tokens=50_000)
    with pytest.raises(ValueError):
        RunnerConfig(compact_pct=0)
    with pytest.raises(ValueError):
        RunnerConfig(image_limit_per_session=10, image_reserve=10)
    cfg = RunnerConfig(system_prompt="secret prompt", io_dir=tmp_path, extra_env={"A": "1"})
    summary = cfg.summary()
    assert "system_prompt" not in summary and summary["system_prompt_chars"] == len("secret prompt")
    assert summary["extra_env"] == ["A"] and json.dumps(summary)
    assert cfg.model == "claude-opus-5-5" and cfg.effort == "high" and cfg.image_limit_per_session == 500
    assert cfg.compact_window_tokens == 200_000 and cfg.compact_pct == 90 and cfg.max_segments == 20


def test_child_environment(tmp_path):
    base = {"PATH": "p", "CLAUDECODE": "1", "CLAUDE_PID": "9", "CLAUDE_EFFORT": "max",
            "CLAUDE_CODE_SESSION_ID": "x", "CLAUDE_CODE_MESSAGING_SOCKET": "pipe", "CLAUDE_CODE_ENTRYPOINT": "cli",
            "CLAUDE_CONFIG_DIR": "keep", "DISABLE_AUTO_COMPACT": "1", "ANTHROPIC_LOG": "keep"}
    cfg = RunnerConfig(compact_window_tokens=150_000, compact_pct=80, extra_env={"MAX_MCP_OUTPUT_TOKENS": "5"})
    env = child_environment(base, cfg, rpc_host="127.0.0.1", rpc_port=1234, rpc_token="tok", hook_dir=tmp_path)
    for gone in ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_SESSION_ID",
                 "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_ENTRYPOINT", "DISABLE_AUTO_COMPACT"):
        assert gone not in env
    assert env["PATH"] == "p" and env["CLAUDE_CONFIG_DIR"] == "keep" and env["ANTHROPIC_LOG"] == "keep"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1" and env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1" and env["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "150000" and env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "80"
    assert env["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false" and env["ENABLE_TOOL_SEARCH"] == "false"
    assert env["VISTA_RPC_PORT"] == "1234" and env["VISTA_RPC_TOKEN"] == "tok" and env["VISTA_HOOK_DIR"] == str(tmp_path)
    assert env["MAX_MCP_OUTPUT_TOKENS"] == "5"                 # extra_env wins


def test_config_files_and_command(tmp_path):
    python = str(tmp_path / "venv" / "python.exe")
    cfg = RunnerConfig(python_exe=python, turn_timeout_s=120, max_budget_usd=0.25, effort="low")
    mcp = mcp_config(cfg, rpc_host="127.0.0.1", rpc_port=99, rpc_token="t")["mcpServers"]["stm"]
    assert mcp["type"] == "stdio" and mcp["command"] == python
    assert mcp["args"][:3] == ["-I", "-X", "utf8"] and mcp["args"][3].endswith("mcp_bridge.py")
    assert mcp["env"]["VISTA_RPC_PORT"] == "99" and mcp["timeout"] == 120_000 and mcp["alwaysLoad"] is True
    hooks = hook_settings(cfg)["hooks"]
    assert set(hooks) == {"PreCompact", "PostCompact", "Stop"}
    hook = hooks["Stop"][0]["hooks"][0]
    assert hook["type"] == "command" and hook["args"][-1].endswith("compact_hook.py") and hook["timeout"] == 30
    cmd = build_command(cfg, claude=["claude.exe"], system_prompt_path=tmp_path / "sp.md",
                        settings_path=tmp_path / "s.json", mcp_path=tmp_path / "m.json",
                        allowed_tools=["mcp__stm__act", "mcp__stm__read_guide"], session_id="abc")
    joined = " ".join(cmd)
    for fragment in ("-p --input-format stream-json --output-format stream-json --verbose --include-hook-events",
                     "--effort low", "--strict-mcp-config", "--permission-mode default",
                     "--allowedTools mcp__stm__act,mcp__stm__read_guide", "--session-id abc",
                     "--max-budget-usd 0.25", "--permission-prompts none", "--disable-slash-commands"):
        assert fragment in joined, fragment
    assert cmd[cmd.index("--tools") + 1] == "" and cmd[cmd.index("--setting-sources") + 1] == ""


def test_user_message_blocks():
    msg = user_message("go", [ImageBlock(png=b"\x89PNG", label="<visual/>")])
    content = msg["message"]["content"]
    assert msg["type"] == "user" and msg["parent_tool_use_id"] is None
    assert [b["type"] for b in content] == ["text", "text", "image"]
    assert content[2]["source"] == {"type": "base64", "media_type": "image/png", "data": "iVBORw=="}


def test_validate_init():
    good = {"tools": ["mcp__stm__act", "mcp__stm__read_guide"], "mcp_servers": [{"name": "stm", "status": "connected"}],
            "permissionMode": "default", "model": "claude-opus-5-5", "skills": ["x"], "plugins": [{"name": "p"}]}
    expected = ["mcp__stm__read_guide", "mcp__stm__act"]
    assert validate_init(good, expected) == []               # skills/plugins are recorded, not fatal
    assert validate_init(good, expected, expected_model="claude-opus-5-5") == []
    assert "extra" in validate_init({**good, "tools": good["tools"] + ["mcp__claude_ai_Drive__x"]}, expected)[0]
    assert validate_init({**good, "tools": ["mcp__stm__act"]}, expected)
    assert validate_init({**good, "mcp_servers": [{"name": "stm", "status": "pending"}]}, expected)
    assert validate_init({**good, "mcp_servers": good["mcp_servers"] + [{"name": "claude.ai Drive", "status": "connected"}]}, expected)
    assert validate_init({**good, "permissionMode": "auto"}, expected)
    assert validate_init({**good, "model": "opus"}, expected)
    assert validate_init({**good, "model": "claude-sonnet-5-5"}, expected, expected_model="claude-opus-5-5")
    assert validate_init({**good, "mcp_server_errors": [{"name": "x"}]}, expected)


def test_classify_and_usage():
    assert classify_result({"subtype": "success", "is_error": False}) == "ok"
    assert classify_result({"subtype": "error_max_budget_usd", "is_error": True}) == "budget_usd"
    assert classify_result({"subtype": "success", "is_error": True, "api_error_status": 529}) == "retryable"
    assert classify_result({"subtype": "success", "is_error": True, "api_error_status": 429,
                            "result": "You've hit your session limit · resets 8:30pm (Asia/Shanghai)"}) == "usage_limit"
    assert classify_result({"subtype": "error_during_execution", "is_error": True,
                            "errors": ["429 {\"type\":\"usage_limit_reached\"}"]}) == "usage_limit"
    assert classify_result({"subtype": "success", "is_error": True, "api_error_status": 429,
                            "result": "API Error: rate limit"}) == "retryable"
    assert classify_result({"subtype": "success", "is_error": True, "result": "API Error: Overloaded"}) == "retryable"
    assert classify_result({"subtype": "error_during_execution", "is_error": True,
                            "terminal_reason": "aborted_streaming"}) == "error"
    usage = result_usage({"total_cost_usd": 0.5, "usage": {"input_tokens": 1},
                          "modelUsage": {"a": {"inputTokens": 2, "outputTokens": 3, "cacheReadInputTokens": 4,
                                               "cacheCreationInputTokens": 5},
                                         "b": {"inputTokens": 1, "outputTokens": 1}}})
    assert usage == {"input_tokens": 3, "output_tokens": 4, "cache_creation_input_tokens": 5,
                     "cache_read_input_tokens": 4, "cost_usd": 0.5}
    assert result_usage(None) is None


# ── runs against the fake claude ─────────────────────────────────────────────
def make_runner(tmp_path, sessions, *, host=None, gate_wait_s=1.0, **overrides):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"sessions": sessions}), encoding="utf-8")
    host = host or ToyHost(tmp_path / "host")
    rpc = RpcServer(host, gate_wait_s=gate_wait_s).start()
    settings = dict(claude_bin=[PY, "-I", "-X", "utf8", str(FAKE)], python_exe=PY, io_dir=tmp_path / "io",
                    effort="low", preflight=False, extra_env={"FAKE_CLAUDE_SCENARIO": str(scenario)},
                    end_grace_s=15.0, startup_timeout_s=60.0)
    settings.update(overrides)
    sleeps: list[float] = []
    runner = ClaudeCodeRunner(host, rpc, RunnerConfig(**settings), sleep=sleeps.append)
    return runner, host, rpc, sleeps


def finish(rpc: RpcServer) -> None:
    deadline = time.monotonic() + 10
    while rpc._conns and time.monotonic() < deadline:        # bridges exit when claude does
        time.sleep(0.05)
    leftover = len(rpc._conns)
    rpc.stop()
    assert leftover == 0, "an MCP bridge outlived its Claude Code process"


def stdout_lines(io_dir: Path, stem: str) -> list[dict]:
    return [json.loads(line) for line in (io_dir / f"{stem}.stdout.jsonl").read_text(encoding="utf-8").splitlines()]


def test_run_done_with_continuation(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[
        [ACT, {"text": "I see 1 square."}],
        [ACT, {"tool": "report_result", "args": {"claim_id": "squares", "value": 2}}, {"text": "Two squares. [DONE]"}],
    ]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done" and result.error is None
    assert host.finished == ("done", "Two squares. [DONE]")
    assert len(host.names("act")) == 2 and len(host.names("ack")) == 2 and host.reports == [{"claim_id": "squares", "value": 2}]
    assert [e[0] for e in host.events if e[0] in ("begin_segment", "end_segment", "finish")] == \
        ["begin_segment", "end_segment", "finish"]
    (seg,) = result.segments
    assert seg["end_reason"] == "marker" and seg["kind"] == "initial" and seg["turns"] == 2
    assert seg["messages_sent"] == 2 and seg["images_sent"] == 3 and seg["images_seen"] == 2
    assert seg["acts_acknowledged"] == 2 and seg["tool_counts"] == {"act": 2, "report_result": 1}
    assert seg["models"] == ["claude-fake-1"] and seg["exit_code"] == 0 and seg["usage_complete"]
    assert result.usage["cost_usd"] == pytest.approx(0.02) and result.usage["num_turns"] == 2
    io = tmp_path / "io"
    inputs = json.loads((io / "segment_000.input.json").read_text(encoding="utf-8"))
    assert inputs[1]["message"]["content"][0]["text"] == DEFAULT_CONTINUE_TEXT
    assert "sha256" in json.dumps(inputs[0]) and png_squares(0).hex() not in json.dumps(inputs)
    assert "<redacted>" in (io / "segment_000.mcp.json").read_text(encoding="utf-8")
    assert rpc.token not in (io / "segment_000.mcp.json").read_text(encoding="utf-8")
    assert '"--session-id"' in (io / "segment_000.command.json").read_text(encoding="utf-8")
    assert (io / "segment_000.bridge.jsonl").stat().st_size > 0
    assert all("sha256" in json.dumps(m) for m in stdout_lines(io, "segment_000")
               if m.get("type") == "user" and "image" in json.dumps(m))
    manifest = json.loads((io / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["outcome"] == "done" and len(manifest["segments"]) == 1
    assert manifest["totals"]["cost_usd"] == pytest.approx(0.02) and manifest["config"]["effort"] == "low"


def test_parallel_acts_second_refused(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"tools": [ACT, ACT]}, {"text": "[DONE]"}]]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done"
    assert len(host.names("act")) == 1 and len(host.names("act_refused")) == 1


def test_marker_in_body_and_stall(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[
        [{"text": "I will write [DONE] at the end, not yet."}], [{"text": "thinking"}], [{"text": "x"}]]],
        stall_turns=2)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "stalled" and host.finished[0] == "stalled"
    (seg,) = result.segments
    assert seg["end_reason"] == "stalled" and seg["messages_sent"] == 2
    assert any("marker_in_body" in n for n in seg["notes"])


def test_image_limit_checkpoint_handoff(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [
        [[ACT, ACT, ACT, {"tool": "write_working", "args": {"content": "after 2 acts: continue counting"}},
          {"wait_interrupt": 15}]],
        [[ACT, {"text": "[DONE]"}]],
    ], image_limit_per_session=3, image_reserve=0)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done", result
    seg0, seg1 = result.segments
    assert seg0["end_reason"] == "checkpoint" and seg1["kind"] == "checkpoint"
    assert seg0["checkpoint_requests"][0]["reason"] == "image_limit" and host.checkpoint_reasons == ["image_limit"]
    assert seg0["usage_complete"] and seg0["exit_code"] == 0         # interrupted, not killed
    assert len(host.names("act")) == 3 and len(host.names("act_refused")) == 0   # 2 + 1 after the handoff
    assert seg0["act_errors"] == 1 and seg0["acts_acknowledged"] == 3   # the refused act is acknowledged too
    lines = stdout_lines(tmp_path / "io", "segment_000")
    results = [json.dumps(m) for m in lines if m.get("type") == "user"]
    assert CHECKPOINT_INSTRUCTION in results[1]                       # appended to the reply that hit the limit
    assert "Nothing else was executed" in results[2]                 # the third act was refused by the host
    assert host.notes.read_working() == "after 2 acts: continue counting\n"
    begins = host.names("begin_segment")
    assert len(begins) == 2 and begins[0][2] != begins[1][2]
    assert host.names("recovery_message") and seg1["images_sent"] == 2


def test_precompact_checkpoint_via_hook(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [
        [{"steps": [ACT, {"precompact": True}, {"text": "continuing"}],
          "on_block": [{"tool": "write_working", "args": {"content": "state for the next context"}},
                       {"wait_interrupt": 15}]}],
        [[ACT, {"text": "[DONE]"}]],
    ])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done", result
    seg0 = result.segments[0]
    assert seg0["end_reason"] == "checkpoint" and seg0["compaction_attempts"] == 1
    assert seg0["compact_boundaries"] == [] and seg0["stop_hook_blocks"] >= 1
    assert host.checkpoint_reasons == ["context_limit"]
    hooks = [json.loads(line) for line in (tmp_path / "io" / "segment_000.hooks.jsonl").read_text(encoding="utf-8").splitlines()]
    pre = [h for h in hooks if h["event"]["hook_event_name"] == "PreCompact"]
    assert pre[0]["effect"] == "request_checkpoint" and pre[0]["output"]["decision"] == "block"
    assert any(h["event"]["hook_event_name"] == "Stop" and h["output"].get("decision") == "block" for h in hooks)


def test_native_compaction_is_error(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"compact_boundary": True}, {"wait_interrupt": 15}]]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "error" and "native_compact" in result.error
    assert result.segments[0]["end_reason"] == "native_compact"


def test_invalid_init_is_error(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"extra_tools": ["mcp__claude_ai_Google_Drive__search"]},
                                                     {"wait_interrupt": 15}]]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "error" and "unexpected tool surface" in result.error
    assert not host.names("call")


def test_crash_recovery(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[ACT, {"exit": 3}]], [[ACT, {"text": "[DONE]"}]]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done"
    seg0, seg1 = result.segments
    assert seg0["end_reason"] == "process_exit" and seg0["exit_code"] == 3 and not seg0["usage_complete"]
    assert seg1["kind"] == "recovery"
    assert len(host.names("ack")) == 3                 # act 1, the recovery message, act 2


def test_api_error_backoff_then_continue(tmp_path):
    runner, host, rpc, sleeps = make_runner(tmp_path, [[
        [{"result": {"is_error": True, "api_error_status": 529, "result": "API Error: 529 Overloaded"}}],
        [ACT, {"text": "[DONE]"}]]])
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done" and sleeps == [10.0]
    assert result.segments[0]["messages_sent"] == 2


def test_decision_lease_expiry_recovers(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"wait_interrupt": 30}]], [[ACT, {"text": "[DONE]"}]]],
                                       turn_timeout_s=1.5)
    t0 = time.monotonic()
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done" and time.monotonic() - t0 < 25
    assert result.segments[0]["end_reason"] == "lease_expired"


def test_terminal_host_ends_with_budget(tmp_path):
    host = ToyHost(tmp_path / "host", max_acts=2)
    runner, host, rpc, _ = make_runner(tmp_path, [[[ACT, ACT, {"wait_interrupt": 15}]]], host=host)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "budget" and host.finished[0] == "budget"
    assert result.segments[0]["end_reason"] == "terminal"


def test_max_segments(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [
        [[ACT, ACT, ACT, {"tool": "write_working", "args": {"content": "s"}}, {"wait_interrupt": 15}]]],
        image_limit_per_session=3, image_reserve=0, max_segments=1)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "max_segments" and len(result.segments) == 1


def test_preflight_and_run(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"text": "READY"}]], [[ACT, {"text": "[DONE]"}]]],
                                       preflight=True)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done"
    pf = result.preflight
    assert pf["ok"], pf["problems"]
    assert pf["model"] == "claude-fake-1" and pf["permission_mode"] == "default" and pf["hook_events"] >= 1
    assert sorted(pf["tools"]) == sorted(f"mcp__stm__{t['name']}" for t in host.list_tools())
    inputs = json.loads((tmp_path / "io" / "preflight.input.json").read_text(encoding="utf-8"))
    assert inputs[0]["message"]["content"][0]["text"] == PREFLIGHT_PROMPT
    assert [e[0] for e in host.events].count("begin_segment") == 1       # the preflight is not a segment


def test_preflight_failure_refuses_tools_and_ends_episode(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[ACT, {"text": "READY"}]]], preflight=True, gate_wait_s=0.3)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "error" and "preflight failed" in result.error
    assert not host.names("act") and host.finished[0] == "error"
    assert any("called tools" in p for p in result.preflight["problems"])


# ── live tests (real Claude Code) ────────────────────────────────────────────
LIVE_SYSTEM_PROMPT = ("You are testing a tool harness for a scientific instrument. Use only the provided tools, "
                      "follow the user's instructions exactly, and keep every reply short.")


def _live_dir(tmp_path: Path, name: str) -> Path:
    root = os.environ.get("VISTA_LIVE_OUT")
    out = Path(root) / name if root else tmp_path
    out.mkdir(parents=True, exist_ok=True)
    return out


def live_runner(out: Path, host: ToyHost, **overrides):
    rpc = RpcServer(host).start()
    settings = dict(model=os.environ.get("VISTA_LIVE_MODEL", "claude-opus-5-5"), effort="low",
                    system_prompt=LIVE_SYSTEM_PROMPT, io_dir=out / "io", preflight=False, max_budget_usd=0.5,
                    python_exe=PY, turn_timeout_s=600, max_segments=4)
    settings.update(overrides)
    return ClaudeCodeRunner(host, rpc, RunnerConfig(**settings)), rpc


@live
def test_live_preflight(tmp_path):
    out = _live_dir(tmp_path, "preflight")
    host = ToyHost(out / "host")
    runner, rpc = live_runner(out, host)
    try:
        pf = runner.preflight()
    finally:
        finish(rpc)
    print(json.dumps({k: pf[k] for k in ("ok", "problems", "model", "claude_code_version", "permission_mode",
                                         "skills", "plugins", "cost_usd")}, indent=1))
    assert pf["ok"], pf["problems"]
    assert pf["model"].startswith("claude-") and pf["permission_mode"] == "default"
    assert sorted(pf["tools"]) == sorted(f"mcp__stm__{t['name']}" for t in host.list_tools())
    assert pf["mcp_servers"][0]["name"] == "stm" and len(pf["mcp_servers"]) == 1
    assert pf["cost_usd"] < 0.2
    assert not host.names("call")


@live
def test_live_episode_continuation_and_done(tmp_path):
    out = _live_dir(tmp_path, "episode")
    task = ("Harness test with a toy instrument. Each `act` call with action \"count\" returns an image of white "
            "squares on black.\nStep 1: call `act` once with action \"count\", look at the image, then end your "
            "turn with one short sentence saying how many white squares you see. Do NOT write [DONE] in this "
            "turn; a continuation message will follow.\nStep 2 (after the continuation message): call `act` once "
            "more with action \"count\", then call `report_result` with claim_id \"squares\" and value equal to "
            "the number of white squares in the newest image, then reply with one line that contains the "
            "codeword 针尖扫描 and ends with [DONE].")
    host = ToyHost(out / "host", max_acts=4, task_text=task)
    runner, rpc = live_runner(out, host)
    try:
        result = runner.run()
    finally:
        finish(rpc)
    print(json.dumps({"outcome": result.outcome, "usage": result.usage, "models": result.models,
                      "reports": host.reports, "segments": [{k: s[k] for k in ("end_reason", "turns", "tool_counts",
                                                             "images_seen", "acts_acknowledged")} for s in result.segments]},
                     indent=1))
    assert result.outcome == "done", result.error
    assert host.acts == 2 and len(host.names("ack")) == 2 and not host.names("act_refused")
    assert host.reports and host.reports[-1]["value"] == 2           # the model saw the second image
    seg = result.segments[0]
    assert seg["turns"] >= 2 and seg["messages_sent"] >= 2           # the continuation message was used
    assert "针尖扫描" in result.final_text                              # UTF-8 in and out of stream-json
    assert result.usage["cost_usd"] < 0.5


@live
def test_live_image_limit_checkpoint(tmp_path):
    out = _live_dir(tmp_path, "image_limit")
    task = ("Harness test with a toy instrument. Each `act` call with action \"count\" returns an image of white "
            "squares on black; its JSON says how many acts were made so far (`turn`). Make exactly 3 `act` calls "
            "in total over the whole task, one call per message. After the third act, call `report_result` with "
            "claim_id \"squares\" and value equal to the number of white squares in the newest image, then reply "
            "with one line ending with [DONE]. If a tool result asks you to update WORKING.md, do that first "
            "(write the number of acts made so far and the plan), and then end your response.")
    host = ToyHost(out / "host", max_acts=5, task_text=task)
    runner, rpc = live_runner(out, host, image_limit_per_session=3, image_reserve=0)
    try:
        result = runner.run()
    finally:
        finish(rpc)
    print(json.dumps({"outcome": result.outcome, "usage": result.usage, "reports": host.reports,
                      "checkpoints": host.checkpoint_reasons, "working": host.notes.read_working(),
                      "segments": [{k: s[k] for k in ("kind", "end_reason", "turns", "tool_counts", "images_sent")}
                                   for s in result.segments]}, indent=1, ensure_ascii=False))
    assert host.checkpoint_reasons[:1] == ["image_limit"]
    assert result.segments[0]["end_reason"] == "checkpoint" and result.segments[1]["kind"] == "checkpoint"
    assert host.notes.read_working() is not None
    assert result.outcome == "done", result.error
    assert host.reports and host.reports[-1]["value"] == host.acts == 3


@live
def test_live_precompact_checkpoint(tmp_path):
    """A tiny auto-compact threshold (window 100k, 6 %: ~4000 context tokens) makes Claude Code
    attempt compaction after one or two 768 px observations; the PreCompact hook must block
    it, request the checkpoint, and the runner must hand off to a fresh session (measured
    2026-10-09: four segments, one act each, ~$0.12)."""
    out = _live_dir(tmp_path, "precompact")
    task = ("Harness test with a toy instrument. Each `act` call with action \"count\" returns an image of white "
            "squares on black; its JSON says how many acts were made so far (`turn`). Make exactly 3 `act` calls "
            "in total over the whole task, one call per message. After the third act, call `report_result` with "
            "claim_id \"squares\" and value equal to the number of white squares in the newest image, then reply "
            "with one line ending with [DONE]. If a tool result or a stop-hook message asks you to update "
            "WORKING.md, do that first (write the number of acts made so far and the plan), and then end your "
            "response.")
    host = ToyHost(out / "host", max_acts=5, task_text=task, image_size=768)
    runner, rpc = live_runner(out, host, compact_window_tokens=100_000, compact_pct=6)
    try:
        result = runner.run()
    finally:
        finish(rpc)
    hooks = []
    for path in sorted((out / "io").glob("segment_*.hooks.jsonl")):
        hooks += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(json.dumps({"outcome": result.outcome, "error": result.error, "usage": result.usage,
                      "reports": host.reports, "checkpoints": host.checkpoint_reasons,
                      "hook_events": [(h["event"].get("hook_event_name"), h.get("checkpoint"), h["output"]) for h in hooks],
                      "segments": [{k: s[k] for k in ("kind", "end_reason", "turns", "tool_counts", "compaction_attempts",
                                                       "compact_boundaries", "stop_hook_blocks", "exit_code")}
                                   for s in result.segments]}, indent=1, ensure_ascii=False))
    assert "context_limit" in host.checkpoint_reasons
    assert any(h["event"].get("hook_event_name") == "PreCompact" and h["output"].get("decision") == "block" for h in hooks)
    assert all(not s["compact_boundaries"] for s in result.segments)       # compaction never ran
    assert result.segments[0]["end_reason"] == "checkpoint"
    assert result.outcome == "done", result.error
    assert host.reports and host.reports[-1]["value"] == host.acts == 3


# ── compaction modes ─────────────────────────────────────────────────────────
class NoNotesHost(ToyHost):
    """The toy host as an arm without notes tools (like the ``img`` arm)."""

    def tool_defs(self):
        return [t for t in super().tool_defs() if t.name in ("act", "report_result")]


def test_native_mode_allows_compaction_and_hands_off_at_image_limit(tmp_path):
    host = NoNotesHost(tmp_path / "host")
    runner, host, rpc, _ = make_runner(tmp_path, [
        [[ACT, {"precompact": True}, ACT, ACT, ACT, {"wait_interrupt": 15}]],
        [[ACT, {"text": "[DONE]"}]],
    ], host=host, compaction="native", image_limit_per_session=3, image_reserve=0)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "done", result
    seg0, seg1 = result.segments
    assert len(seg0["compact_boundaries"]) == 1 and seg0["problems"] == []
    assert seg0["compact_boundary_count"] == 1 and seg1["compact_boundary_count"] == 0
    assert result.usage["compact_boundaries"] == 1
    assert seg0["end_reason"] == "image_limit" and seg1["kind"] == "handoff"
    assert host.checkpoint_reasons == []                       # no checkpoint protocol in native mode
    hooks = [json.loads(line) for line in (tmp_path / "io" / "segment_000.hooks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [h["output"] for h in hooks if h["event"]["hook_event_name"] == "PreCompact"] == [{}]
    assert any(h["event"]["hook_event_name"] == "PostCompact" for h in hooks)
    assert len(host.names("act")) == 4 + 1


def test_checkpoint_mode_requires_write_working(tmp_path):
    runner, host, rpc, _ = make_runner(tmp_path, [[[{"text": "[DONE]"}]]], host=NoNotesHost(tmp_path / "host"))
    result = runner.run()
    finish(rpc)
    assert result.outcome == "error" and "write_working" in result.error
    assert result.segments == [] and host.finished[0] == "error"


def test_checkpoint_loop_guard(tmp_path):
    session = [{"steps": [{"precompact": True}, {"text": "ok"}],
                "on_block": [{"tool": "write_working", "args": {"content": "nothing new"}}, {"wait_interrupt": 15}]}]
    runner, host, rpc, _ = make_runner(tmp_path, [session] * 4)
    result = runner.run()
    finish(rpc)
    assert result.outcome == "error" and "consecutive checkpoint handoffs" in result.error
    assert len(result.segments) == 3 and all(s["end_reason"] == "checkpoint" for s in result.segments)


def test_subscription_auth_drops_api_credentials_and_refuses_api_billing(tmp_path):
    from vista_spm.claude_runner import RunnerConfig, auth_problem, child_environment
    base = {"PATH": "x", "ANTHROPIC_API_KEY": "sk-test", "ANTHROPIC_AUTH_TOKEN": "t", "HOME": "h"}
    env = child_environment(base, RunnerConfig(io_dir=tmp_path), rpc_host="127.0.0.1", rpc_port=1,
                            rpc_token="k", hook_dir=tmp_path)
    assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env and env["PATH"] == "x"
    env = child_environment(base, RunnerConfig(io_dir=tmp_path, auth="env"), rpc_host="127.0.0.1", rpc_port=1,
                            rpc_token="k", hook_dir=tmp_path)
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert auth_problem({"apiKeySource": "ANTHROPIC_API_KEY"}, "subscription")
    assert auth_problem({"apiKeySource": "none"}, "subscription") is None
    assert auth_problem({}, "subscription") is None
    assert auth_problem({"apiKeySource": "ANTHROPIC_API_KEY"}, "env") is None
    with pytest.raises(ValueError):
        RunnerConfig(io_dir=tmp_path, auth="apikey")
