"""A fake ``claude -p`` for the VISTA runner tests (standard library only).

It speaks stream-json on stdin/stdout the way Claude Code 2.1.295 does (as recorded on
2026-10-09) and drives the REAL MCP bridge and hook script that the runner configures:
the ``stm`` server from ``--mcp-config`` is started and initialised over stdio, and the
PreCompact/Stop hooks from ``--settings`` are executed with the documented input.

Scenario: ``FAKE_CLAUDE_SCENARIO`` = path of a JSON file ``{"sessions": [[turn, ...], ...]}``;
process number k (counted in ``<scenario>.count``) plays ``sessions[k]``. A user message
starts the next turn; a turn is a list of steps:

* ``{"tool": name, "args": {...}}`` — assistant ``tool_use``, MCP ``tools/call``, then the
  ``user`` ``tool_result`` (images as base64 ``source`` blocks, like Claude Code).
* ``{"tools": [{"tool":..., "args":...}, ...]}`` — parallel tool uses in one message.
* ``{"text": "..."}`` — assistant text (the last one becomes the result text).
* ``{"precompact": true}`` — run the PreCompact hook; if it does not block, emit a
  ``compact_boundary`` (a native compaction).
* ``{"compact_boundary": true}`` — emit a native ``compact_boundary`` without any hook.
* ``{"sleep": s}``, ``{"exit": code}`` (crash), ``{"wait_interrupt": s}`` (block until an
  interrupt control request arrives).
* ``{"result": {...}}`` — fields merged into this turn's result event.
* ``{"extra_tools": [...]}`` — extra tool names in this turn's ``system/init``.

At the end of a turn the Stop hook runs; when it blocks, the fake emits the synthetic
"Stop hook feedback" user message and plays the turn's ``on_block`` steps (default: write
WORKING.md), at most 8 times. A user message beyond the script gets the reply
``(no more script)``. Interrupts abort the running turn with a
``result`` of subtype ``error_during_execution``. EOF on stdin ends the process (code 0).
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid

MODEL = "claude-fake-1"
STEP_DELAY = float(os.environ.get("FAKE_CLAUDE_STEP_DELAY", "0.05"))
DEFAULT_ON_BLOCK = [{"tool": "write_working", "args": {"content": "Plan: keep counting. State saved by the fake."}},
                    {"text": "WORKING.md saved."}]


def emit(obj: dict) -> None:
    sys.stdout.buffer.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def arg_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


class Mcp:
    """Minimal MCP client over stdio for one server."""

    def __init__(self, entry: dict) -> None:
        env = dict(os.environ)
        env.update(entry.get("env") or {})
        self.proc = subprocess.Popen([entry["command"], *entry.get("args", [])], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)
        self.next_id = 0
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "fake-claude", "version": "0"}})
        self.protocol = init.get("result", {}).get("protocolVersion")
        self.notify("notifications/initialized", {})
        listing = self.request("tools/list", {})
        self.tools = [t["name"] for t in listing.get("result", {}).get("tools", [])]

    def notify(self, method: str, params: dict) -> None:
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method: str, params: dict) -> dict:
        self.next_id += 1
        rid = self.next_id
        self.proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": rid, "method": method,
                                           "params": params}) + "\n").encode())
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return {"error": {"message": "MCP server closed stdout"}}
            message = json.loads(line.decode("utf-8"))
            if message.get("id") == rid:
                return message

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


class Fake:
    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.session_id = arg_value(argv, "--session-id") or str(uuid.uuid4())
        with open(arg_value(argv, "--settings"), encoding="utf-8") as fh:
            self.settings = json.load(fh)
        with open(arg_value(argv, "--mcp-config"), encoding="utf-8") as fh:
            self.mcp_config = json.load(fh)
        scenario_path = os.environ["FAKE_CLAUDE_SCENARIO"]
        with open(scenario_path, encoding="utf-8") as fh:
            sessions = json.load(fh)["sessions"]
        counter = scenario_path + ".count"
        index = int(open(counter).read()) if os.path.exists(counter) else 0
        with open(counter, "w") as fh:
            fh.write(str(index + 1))
        self.turns = sessions[index] if index < len(sessions) else []
        self.mcp = Mcp(self.mcp_config["mcpServers"]["stm"])
        self.inbox: "queue.Queue[dict | None]" = queue.Queue()
        self.interrupted = False
        self.cost = 0.0
        self.result_index = 0
        self.turn_no = 0
        threading.Thread(target=self._read_stdin, daemon=True).start()

    def _read_stdin(self) -> None:
        for raw in sys.stdin.buffer:
            raw = raw.strip()
            if not raw:
                continue
            message = json.loads(raw.decode("utf-8"))
            if message.get("type") == "control_request":
                if message.get("request", {}).get("subtype") == "interrupt":
                    self.interrupted = True
                emit({"type": "control_response", "response": {"subtype": "success",
                                                               "request_id": message.get("request_id"),
                                                               "response": {}}})
                continue
            self.inbox.put(message)
        self.inbox.put(None)

    # ── events ───────────────────────────────────────────────────────────────
    def init(self, extra: list[str]) -> None:
        emit({"type": "system", "subtype": "init", "session_id": self.session_id, "model": MODEL,
              "tools": [f"mcp__stm__{t}" for t in self.mcp.tools] + list(extra),
              "mcp_servers": [{"name": "stm", "status": "connected"}], "permissionMode": "default",
              "claude_code_version": "0.0.0-fake", "skills": [], "plugins": [], "slash_commands": []})

    def assistant(self, blocks: list[dict]) -> None:
        emit({"type": "assistant", "message": {"model": MODEL, "role": "assistant", "content": blocks},
              "parent_tool_use_id": None, "session_id": self.session_id})

    def call_tools(self, calls: list[dict]) -> None:
        uses = [{"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:12]}", "name": f"mcp__stm__{c['tool']}",
                 "input": c.get("args", {})} for c in calls]
        self.assistant(uses)
        results = []
        for use, call in zip(uses, calls):
            response = self.mcp.request("tools/call", {"name": call["tool"], "arguments": call.get("args", {})})
            result = response.get("result") or {"content": [{"type": "text", "text": json.dumps(response.get("error"))}],
                                                "isError": True}
            content = []
            for block in result.get("content", []):
                if block.get("type") == "image":
                    content.append({"type": "image", "source": {"type": "base64", "media_type": block.get("mimeType"),
                                                                "data": block.get("data")}})
                else:
                    content.append(block)
            results.append({"type": "tool_result", "tool_use_id": use["id"], "content": content,
                            "is_error": bool(result.get("isError"))})
        emit({"type": "user", "message": {"role": "user", "content": results}, "parent_tool_use_id": None,
              "session_id": self.session_id})

    def hook(self, event_name: str, extra: dict) -> dict:
        groups = self.settings.get("hooks", {}).get(event_name) or []
        output: dict = {}
        for group in groups:
            for hook in group.get("hooks", []):
                event = {"session_id": self.session_id, "transcript_path": "fake.jsonl", "cwd": os.getcwd(),
                         "prompt_id": str(uuid.uuid4()), "hook_event_name": event_name, **extra}
                proc = subprocess.run([hook["command"], *hook.get("args", [])], input=json.dumps(event).encode(),
                                      capture_output=True, timeout=hook.get("timeout", 30))
                if event_name == "Stop":
                    emit({"type": "system", "subtype": "hook_response", "hook_event": "Stop",
                          "output": proc.stdout.decode("utf-8", "replace"), "exit_code": proc.returncode})
                if proc.returncode == 2:
                    return {"decision": "block", "reason": proc.stderr.decode("utf-8", "replace")}
                text = proc.stdout.decode("utf-8", "replace").strip()
                if text:
                    output = json.loads(text)
        return output

    def result(self, text: str, overrides: dict, num_turns: int) -> None:
        self.cost += 0.01
        message = {"type": "result", "subtype": "success", "is_error": False, "result": text,
                   "num_turns": num_turns, "total_cost_usd": round(self.cost, 6), "session_id": self.session_id,
                   "result_index": self.result_index, "terminal_reason": "completed", "stop_reason": "end_turn",
                   "api_error_status": None,
                   "usage": {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 0},
                   "modelUsage": {MODEL: {"inputTokens": 10 * (self.result_index + 1),
                                          "outputTokens": 5 * (self.result_index + 1),
                                          "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
                                          "costUSD": round(self.cost, 6)}}}
        message.update(overrides)
        if (message.get("subtype") != "success" or message.get("is_error")) and "result" not in overrides:
            message.pop("result", None)
        self.result_index += 1
        emit(message)

    def abort_turn(self) -> None:
        self.interrupted = False
        emit({"type": "user", "message": {"role": "user", "content": [{"type": "text",
              "text": "[Request interrupted by user]"}]}, "parent_tool_use_id": None})
        self.result_index += 1
        emit({"type": "result", "subtype": "error_during_execution", "is_error": True, "num_turns": 1,
              "total_cost_usd": round(self.cost, 6), "session_id": self.session_id,
              "result_index": self.result_index - 1, "terminal_reason": "aborted_streaming",
              "errors": ["interrupted"], "usage": {},
              "modelUsage": {MODEL: {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 0,
                                     "cacheCreationInputTokens": 0, "costUSD": round(self.cost, 6)}}})

    # ── turns ────────────────────────────────────────────────────────────────
    def play_steps(self, steps: list[dict], state: dict) -> bool:
        """False if the turn was aborted (interrupt) — the caller then stops the turn."""
        for step in steps:
            if self.interrupted:
                self.abort_turn()
                return False
            if "tool" in step:
                self.call_tools([step])
            elif "tools" in step:
                self.call_tools(step["tools"])
            elif "text" in step:
                state["text"] = step["text"]
                self.assistant([{"type": "text", "text": step["text"]}])
            elif "precompact" in step:
                emit({"type": "system", "subtype": "status", "status": "compacting"})
                out = self.hook("PreCompact", {"trigger": "auto", "custom_instructions": None})
                emit({"type": "system", "subtype": "status", "status": None})
                if out.get("decision") != "block":
                    emit({"type": "system", "subtype": "compact_boundary",
                          "compact_metadata": {"trigger": "auto", "pre_tokens": 1000, "post_tokens": 100}})
                    self.hook("PostCompact", {"trigger": "auto", "compact_summary": "fake summary"})
            elif "compact_boundary" in step:
                emit({"type": "system", "subtype": "compact_boundary",
                      "compact_metadata": {"trigger": "auto", "pre_tokens": 1000, "post_tokens": 100}})
            elif "sleep" in step:
                time.sleep(float(step["sleep"]))
            elif "exit" in step:
                os._exit(int(step["exit"]))
            elif "wait_interrupt" in step:
                deadline = time.monotonic() + float(step["wait_interrupt"])
                while time.monotonic() < deadline and not self.interrupted:
                    time.sleep(0.02)
            elif "result" in step:
                state["result"] = step["result"]
            time.sleep(STEP_DELAY)
        if self.interrupted:
            self.abort_turn()
            return False
        return True

    def play_turn(self, turn: dict | list) -> None:
        if isinstance(turn, list):
            turn = {"steps": turn}
        extra = []
        for step in turn.get("steps", []):
            if "extra_tools" in step:
                extra = list(step["extra_tools"])
        self.init(extra)
        state: dict = {"text": "", "result": {}}
        steps = [s for s in turn.get("steps", []) if "extra_tools" not in s]
        if not self.play_steps(steps, state):
            return
        model_turns = 1
        for _ in range(8):
            decision = self.hook("Stop", {"permission_mode": "default", "effort": {"level": "low"},
                                          "stop_hook_active": model_turns > 1,
                                          "last_assistant_message": state["text"]})
            if decision.get("decision") != "block":
                break
            emit({"type": "user", "message": {"role": "user", "content": [{"type": "text",
                  "text": "Stop hook feedback:\n" + decision.get("reason", "")}]}, "isSynthetic": True})
            model_turns += 1
            if not self.play_steps(turn.get("on_block", DEFAULT_ON_BLOCK), state):
                return
        self.result(state["text"], state["result"], model_turns)

    def run(self) -> int:
        while True:
            message = self.inbox.get()
            if message is None:
                break
            if message.get("type") != "user":
                continue
            turn = self.turns[self.turn_no] if self.turn_no < len(self.turns) else [{"text": "(no more script)"}]
            self.turn_no += 1
            self.play_turn(turn)
        self.mcp.close()
        return 0


if __name__ == "__main__":
    sys.exit(Fake(sys.argv[1:]).run())
