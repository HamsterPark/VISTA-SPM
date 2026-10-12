"""RpcServer, the stdio MCP bridge and the compaction hook (vista_spm.{rpc,mcp_bridge,compact_hook}).

The bridge and the hook run as real subprocesses (``python -I -X utf8 <script>``), exactly
as Claude Code starts them.
"""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from vista_spm import compact_hook
from vista_spm.compact_hook import CHECKPOINT_INSTRUCTION, HANDOFF_REASON, decide
from vista_spm.protocol import ImageBlock, ToolDef, ToolReply, error_reply, json_reply
from vista_spm.rpc import RpcServer, sanitize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fakes import ToyHost, png_squares  # noqa: E402

PY = sys.executable
PACKAGE = Path(__file__).resolve().parents[1] / "vista_spm"
BRIDGE = PACKAGE / "mcp_bridge.py"
HOOK = PACKAGE / "compact_hook.py"


class EchoHandler:
    """Minimal ToolHandler: echo, big image, slow, boom."""

    def __init__(self) -> None:
        self.active = 0
        self.overlap = False
        self.calls: list[str] = []

    def list_tools(self) -> list[dict]:
        schema = {"type": "object", "additionalProperties": True, "properties": {}}
        return [ToolDef(name=n, description=f"{n} tool", input_schema=schema).to_mcp()
                for n in ("echo", "image", "slow", "boom")]

    def call(self, name: str, args: dict) -> ToolReply:
        self.calls.append(name)
        self.active += 1
        if self.active > 1:
            self.overlap = True
        try:
            if name == "echo":
                return json_reply({"echo": args})
            if name == "image":
                size = int(args.get("bytes", 1000))
                return json_reply({"n": 1}, images=[ImageBlock(png=os.urandom(size), label="<visual kind=\"current\"/>")])
            if name == "slow":
                time.sleep(float(args.get("s", 0.3)))
                return json_reply({"slept": args.get("s", 0.3)})
            if name == "boom":
                raise RuntimeError("kaboom")
            return error_reply(f"unknown tool {name}")
        finally:
            self.active -= 1


class Client:
    """Test client for the RpcServer wire protocol."""

    def __init__(self, server: RpcServer, token: str | None = None, auth: bool = True) -> None:
        self.sock = socket.create_connection((server.host, server.port), timeout=30)
        self.reader = self.sock.makefile("rb")
        self.next_id = 0
        if auth:
            response = self.request("auth", {"token": token or server.token})
            assert response.get("result", {}).get("server") == "stm", response

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        line = json.dumps({"id": self.next_id, "method": method, "params": params or {}}) + "\n"
        self.sock.sendall(line.encode("utf-8"))
        raw = self.reader.readline()
        return json.loads(raw) if raw else {}

    def close(self) -> None:
        self.reader.close()
        self.sock.close()


@pytest.fixture
def server(tmp_path):
    handler = EchoHandler()
    srv = RpcServer(handler, log_path=tmp_path / "rpc.jsonl", gate_wait_s=0.5).start()
    srv.handler_obj = handler
    yield srv
    srv.stop()


# ── RpcServer ────────────────────────────────────────────────────────────────
def test_auth_required(server):
    bad = Client(server, auth=False)
    response = bad.request("auth", {"token": "wrong"})
    assert response["error"]["code"] == -32001
    assert bad.reader.readline() == b""          # connection closed
    bad.close()
    first = Client(server, auth=False)
    response = first.request("tools/list")
    assert "error" in response
    first.close()


def test_tools_list_adds_result_size_meta(server):
    client = Client(server)
    before = server.tools_listed
    tools = client.request("tools/list")["result"]["tools"]
    assert [t["name"] for t in tools] == ["echo", "image", "slow", "boom"]
    assert all(t["_meta"]["anthropic/maxResultSizeChars"] == 200_000 for t in tools)
    assert server.tools_listed == before + 1
    client.close()


def test_tools_call_roundtrip_and_sanitized_log(server, tmp_path):
    client = Client(server)
    result = client.request("tools/call", {"name": "echo", "arguments": {"x": "扫描"}, "call_id": 3})["result"]
    assert result["isError"] is False and json.loads(result["content"][0]["text"]) == {"echo": {"x": "扫描"}}
    big = client.request("tools/call", {"name": "image", "arguments": {"bytes": 3_000_000}})["result"]
    kinds = [b["type"] for b in big["content"]]
    assert kinds == ["text", "text", "image"]
    assert len(base64.b64decode(big["content"][2]["data"])) == 3_000_000
    client.close()
    log = (tmp_path / "rpc.jsonl").read_text(encoding="utf-8")
    assert server.token not in log and "<redacted>" in log
    assert "sha256" in log and len(log) < 200_000            # the 4 MB of base64 is not in the log
    assert server.activity()[0] == 2


def test_handler_exception_becomes_error_reply(server):
    client = Client(server)
    result = client.request("tools/call", {"name": "boom", "arguments": {}})["result"]
    assert result["isError"] is True and "kaboom" in result["content"][0]["text"]
    assert client.request("ping")["result"] == {"pong": True}       # server still alive
    assert client.request("nope")["error"]["code"] == -32601
    client.close()


def test_gate_refuses_then_releases(server):
    server.disable_calls("validation pending")
    client = Client(server)
    t0 = time.monotonic()
    result = client.request("tools/call", {"name": "echo", "arguments": {}})["result"]
    assert result["isError"] and "validation pending" in result["content"][0]["text"]
    assert time.monotonic() - t0 >= 0.4                       # waited gate_wait_s
    assert server.handler_obj.calls == []
    threading.Timer(0.2, server.enable_calls).start()          # enabled while the call waits
    result = client.request("tools/call", {"name": "echo", "arguments": {}})["result"]
    assert result["isError"] is False
    client.close()


def test_calls_are_serialized_across_connections(server):
    clients = [Client(server) for _ in range(4)]
    threads = [threading.Thread(target=c.request, args=("tools/call", {"name": "slow", "arguments": {"s": 0.15}}))
               for c in clients]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    time.sleep(0.05)
    assert server.activity()[1] >= 1
    for t in threads:
        t.join()
    assert time.monotonic() - t0 >= 0.55                       # 4 x 0.15 s, one at a time
    assert server.handler_obj.overlap is False
    assert server.wait_idle(1.0)
    for c in clients:
        c.close()


def test_hook_callback_and_observer(server):
    client = Client(server)
    assert client.request("episode/hook", {"event": {"hook_event_name": "Stop"}})["result"]["output"] == {}
    seen = []
    server.set_hook_callback(lambda event: (seen.append(event), {"decision": "block", "reason": "r"})[1])
    out = client.request("episode/hook", {"event": {"hook_event_name": "PreCompact"}})["result"]["output"]
    assert out == {"decision": "block", "reason": "r"} and seen[0]["hook_event_name"] == "PreCompact"
    assert "error" in client.request("episode/hook", {"event": "nope"})
    server.set_call_observer(lambda name, args, reply: ToolReply(text=reply.text + " [observed]"))
    result = client.request("tools/call", {"name": "echo", "arguments": {}})["result"]
    assert result["content"][0]["text"].endswith("[observed]")
    client.close()


def test_disconnect_mid_call_keeps_server_alive(server):
    client = Client(server)
    line = json.dumps({"id": 9, "method": "tools/call", "params": {"name": "slow", "arguments": {"s": 0.3}}}) + "\n"
    client.sock.sendall(line.encode())
    client.close()                                             # gone before the reply
    other = Client(server)
    assert other.request("tools/call", {"name": "echo", "arguments": {}})["result"]["isError"] is False
    other.close()


def test_stop_closes_connections(tmp_path):
    srv = RpcServer(EchoHandler()).start()
    client = Client(srv)
    srv.stop()
    assert client.reader.readline() == b""
    client.close()


def test_sanitize():
    value = {"token": "secret", "content": [{"type": "image", "data": "A" * 100}], "data": "short"}
    out = sanitize(value)
    assert out["token"] == "<redacted>" and out["data"] == "short"
    assert out["content"][0]["data"]["chars"] == 100 and len(out["content"][0]["data"]["sha256"]) == 64


# ── MCP bridge (real subprocess) ─────────────────────────────────────────────
class Bridge:
    def __init__(self, env: dict) -> None:
        full = dict(os.environ)
        for key in ("VISTA_RPC_PORT", "VISTA_RPC_TOKEN", "VISTA_RPC_HOST"):
            full.pop(key, None)
        full.update(env)
        self.proc = subprocess.Popen([PY, "-I", "-X", "utf8", str(BRIDGE)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=full)
        self.lines: list[dict] = []
        self.lock = threading.Lock()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        for raw in self.proc.stdout:
            with self.lock:
                self.lines.append(json.loads(raw.decode("utf-8")))

    def send(self, message: dict | bytes) -> None:
        data = message if isinstance(message, bytes) else (json.dumps(message) + "\n").encode()
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def wait_for(self, rid, timeout: float = 30.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                for message in self.lines:
                    if message.get("id") == rid:
                        return message
            time.sleep(0.01)
        raise AssertionError(f"no response for id {rid}; got {self.lines}")

    def close(self) -> tuple[int, str]:
        self.proc.stdin.close()
        code = self.proc.wait(timeout=20)
        return code, self.proc.stderr.read().decode("utf-8", "replace")


@pytest.fixture
def toy_server(tmp_path):
    host = ToyHost(tmp_path / "host", act_delay_s=0.0)
    srv = RpcServer(host, log_path=tmp_path / "rpc.jsonl").start()
    srv.toy = host
    yield srv
    srv.stop()


def _bridge_env(srv: RpcServer) -> dict:
    return {"VISTA_RPC_PORT": str(srv.port), "VISTA_RPC_TOKEN": srv.token, "VISTA_RPC_HOST": srv.host}


def test_bridge_protocol(toy_server):
    bridge = Bridge(_bridge_env(toy_server))
    bridge.send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}})
    init = bridge.wait_for(1)["result"]
    assert init["protocolVersion"] == "2025-06-18" and init["serverInfo"]["name"] == "stm"
    assert init["capabilities"] == {"tools": {"listChanged": False}}
    bridge.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    bridge.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert bridge.wait_for(2)["result"] == {}
    bridge.send({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    names = [t["name"] for t in bridge.wait_for(3)["result"]["tools"]]
    assert names == ["act", "report_result", "read_guide", "write_guide", "read_working", "write_working"]
    bridge.send({"jsonrpc": "2.0", "id": "a", "method": "tools/call",
                 "params": {"name": "act", "arguments": {"action": "count", "note": "第一次"}}})
    result = bridge.wait_for("a")["result"]
    assert result["isError"] is False
    assert [b["type"] for b in result["content"]] == ["text", "text", "image"]
    assert base64.b64decode(result["content"][2]["data"]) == png_squares(1)
    bridge.send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "act", "arguments": {"action": "count"}}})
    refused = bridge.wait_for(4)["result"]                     # not acknowledged yet
    assert refused["isError"] is True and "Observe the previous act" in refused["content"][0]["text"]
    bridge.send({"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
    assert bridge.wait_for(5)["error"]["code"] == -32601
    bridge.send(b"{not json\n")
    bridge.send({"jsonrpc": "2.0", "id": 6, "method": "initialize", "params": {"protocolVersion": "2099-01-01"}})
    assert bridge.wait_for(6)["result"]["protocolVersion"] == "2025-06-18"
    code, stderr = bridge.close()
    assert code == 0
    assert any(m.get("error", {}).get("code") == -32700 for m in bridge.lines)
    # notifications produced no output: every line is a response to a request id or a parse error
    assert all("id" in m for m in bridge.lines)
    assert "started" in stderr


def test_bridge_ping_answered_during_long_call(tmp_path):
    host = ToyHost(tmp_path / "host", act_delay_s=1.5)
    srv = RpcServer(host).start()
    try:
        bridge = Bridge(_bridge_env(srv))
        bridge.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "act", "arguments": {"action": "count"}}})
        time.sleep(0.3)
        t0 = time.monotonic()
        bridge.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
        bridge.wait_for(2)
        assert time.monotonic() - t0 < 1.0                     # not stuck behind the act
        assert bridge.wait_for(1)["result"]["isError"] is False
        bridge.close()
    finally:
        srv.stop()


def test_bridge_cancel_before_start(tmp_path):
    host = ToyHost(tmp_path / "host", act_delay_s=1.0)
    srv = RpcServer(host).start()
    try:
        bridge = Bridge(_bridge_env(srv))
        bridge.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "act", "arguments": {"action": "count"}}})
        bridge.send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "read_guide", "arguments": {}}})
        bridge.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}})
        bridge.wait_for(1)
        time.sleep(0.5)
        assert all(m.get("id") != 2 for m in bridge.lines)
        assert [e[1] for e in host.names("call")] == ["act"]
        bridge.close()
    finally:
        srv.stop()


def test_bridge_large_reply_and_unicode(server):
    bridge = Bridge(_bridge_env(server))
    bridge.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "image", "arguments": {"bytes": 4_000_000}}})
    result = bridge.wait_for(1, timeout=60)["result"]
    assert len(base64.b64decode(result["content"][2]["data"])) == 4_000_000
    bridge.send({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "echo", "arguments": {"s": "针尖 Å ✓"}}})
    assert json.loads(bridge.wait_for(2)["result"]["content"][0]["text"]) == {"echo": {"s": "针尖 Å ✓"}}
    bridge.close()


def test_bridge_host_unreachable_and_misconfigured():
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    bridge = Bridge({"VISTA_RPC_PORT": str(port), "VISTA_RPC_TOKEN": "x", "VISTA_RPC_CONNECT_TIMEOUT": "2"})
    bridge.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "act", "arguments": {}}})
    result = bridge.wait_for(1)["result"]
    assert result["isError"] is True and "unreachable" in result["content"][0]["text"]
    bridge.send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert "unreachable" in bridge.wait_for(2)["error"]["message"]
    bridge.close()
    bad = Bridge({})
    bad.send({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert "misconfigured" in bad.wait_for(1)["error"]["message"]
    bad.close()


def test_bridge_imports_nothing_from_the_harness():
    source = BRIDGE.read_text(encoding="utf-8")
    head = source.split('"""', 2)[2].split("SERVER_NAME", 1)[0]
    assert "vista_spm" not in head and "stmbench" not in head
    for line in source.splitlines():
        if line.startswith(("import ", "from ")):
            assert line.split()[1].split(".")[0] in {"__future__", "json", "os", "queue", "socket", "sys",
                                                     "threading", "typing"}, line


# ── compaction hook ──────────────────────────────────────────────────────────
def test_decide_native_never_blocks():
    for event in ("PreCompact", "Stop", "PostCompact"):
        for state in ("none", "requested", "ready"):
            assert decide(event, state, native=True) == ({}, None)


@pytest.mark.parametrize("event,state,terminal,decision,effect", [
    ("PreCompact", "none", False, CHECKPOINT_INSTRUCTION, "request_checkpoint"),
    ("PreCompact", "requested", False, CHECKPOINT_INSTRUCTION, None),
    ("PreCompact", "ready", False, HANDOFF_REASON, None),
    ("PreCompact", "none", True, HANDOFF_REASON, None),
    ("Stop", "requested", False, CHECKPOINT_INSTRUCTION, None),
    ("Stop", "none", False, None, None),
    ("Stop", "ready", False, None, None),
    ("Stop", "requested", True, None, None),
    ("PostCompact", "none", False, None, None),
    (None, "bogus", False, None, None),
])
def test_decide(event, state, terminal, decision, effect):
    output, got_effect = decide(event, state, terminal=terminal)
    assert got_effect == effect
    if decision is None:
        assert output == {}
    else:
        assert output == {"decision": "block", "reason": decision}


def _run_hook(event: dict | bytes, env: dict) -> tuple[dict, int]:
    full = dict(os.environ)
    for key in ("VISTA_RPC_PORT", "VISTA_RPC_TOKEN", "VISTA_RPC_HOST", "VISTA_HOOK_DIR"):
        full.pop(key, None)
    full.update(env)
    data = event if isinstance(event, bytes) else json.dumps(event).encode()
    proc = subprocess.run([PY, "-I", "-X", "utf8", str(HOOK)], input=data, capture_output=True, env=full, timeout=60)
    return json.loads(proc.stdout.decode("utf-8")), proc.returncode


def test_hook_via_rpc(server, tmp_path):
    seen = []
    server.set_hook_callback(lambda event: (seen.append(event), decide(event.get("hook_event_name"), "none")[0])[1])
    hook_dir = tmp_path / "hooks"
    event = {"session_id": "s", "transcript_path": "t", "cwd": "c", "prompt_id": "p",
             "hook_event_name": "PreCompact", "trigger": "auto", "custom_instructions": None}
    output, code = _run_hook(event, {**_bridge_env(server), "VISTA_HOOK_DIR": str(hook_dir)})
    assert code == 0 and output == {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}
    assert seen == [{"session_id": "s", "hook_event_name": "PreCompact", "trigger": "auto",
                     "custom_instructions": None, "prompt_id": "p"}]
    log = [json.loads(line) for line in (hook_dir / "hook_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert log[-1]["path"] == "rpc" and log[-1]["output"]["decision"] == "block"
    assert not (hook_dir / compact_hook.REQUEST_MARKER).exists()     # the RPC path does not touch markers


def test_hook_marker_fallback(tmp_path):
    hook_dir = tmp_path / "hooks"
    env = {"VISTA_RPC_PORT": "1", "VISTA_RPC_TOKEN": "x", "VISTA_HOOK_DIR": str(hook_dir),
           "VISTA_HOOK_RPC_TIMEOUT": "1"}
    stop = {"hook_event_name": "Stop", "stop_hook_active": False, "last_assistant_message": "hi"}
    assert _run_hook(stop, env) == ({}, 0)
    out, _ = _run_hook({"hook_event_name": "PreCompact", "trigger": "auto"}, env)
    assert out == {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}
    assert (hook_dir / compact_hook.REQUEST_MARKER).exists()
    out, _ = _run_hook(stop, env)
    assert out == {"decision": "block", "reason": CHECKPOINT_INSTRUCTION}
    (hook_dir / compact_hook.READY_MARKER).touch()
    assert _run_hook(stop, env) == ({}, 0)
    out, _ = _run_hook({"hook_event_name": "PreCompact", "trigger": "auto"}, env)
    assert out == {"decision": "block", "reason": HANDOFF_REASON}
    log = [json.loads(line) for line in (hook_dir / "hook_events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(log) == 5 and all(r["path"] == "markers" for r in log)
    assert log[0]["event"]["last_assistant_message"] == "hi"
    (hook_dir / compact_hook.READY_MARKER).unlink()
    native = {**env, "VISTA_COMPACTION": "native"}           # arms without notes never block
    assert _run_hook({"hook_event_name": "PreCompact", "trigger": "auto"}, native) == ({}, 0)
    assert _run_hook(stop, native) == ({}, 0)


def test_hook_malformed_input_allows(tmp_path):
    assert _run_hook(b"\xff not json", {"VISTA_HOOK_DIR": str(tmp_path)}) == ({}, 0)
    assert _run_hook(b"", {}) == ({}, 0)


def test_stale_generation_calls_are_refused(server):
    old = Client(server)
    server.new_generation()
    refused = old.request("tools/call", {"name": "echo", "arguments": {}})["result"]
    assert refused["isError"] and "has ended" in refused["content"][0]["text"]
    fresh = Client(server)
    assert fresh.request("tools/call", {"name": "echo", "arguments": {}})["result"]["isError"] is False
    # a call waiting at the closed gate must not execute when a LATER segment opens the gate
    server.disable_calls("segment ended")
    waiting = {}
    thread = threading.Thread(target=lambda: waiting.update(fresh.request("tools/call", {"name": "echo", "arguments": {}})))
    thread.start()
    time.sleep(0.1)
    server.new_generation()
    server.enable_calls()
    thread.join()
    assert waiting["result"]["isError"] and "has ended" in waiting["result"]["content"][0]["text"]
    assert server.handler_obj.calls == ["echo"]
    old.close()
    fresh.close()
