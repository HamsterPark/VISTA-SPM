#!/usr/bin/env python3
# Derived from VISTA (MIT), github.com/joshhhhhan/VISTA, backends/claude/mcp_bridge.py
"""Stdio MCP server that forwards every tool call to the episode host's RpcServer.

README
------
Claude Code starts this file from the segment's ``--mcp-config`` as
``<venv python> -I -X utf8 mcp_bridge.py`` (server name ``stm``, so the model sees
``mcp__stm__<tool>``). It is deliberately standard-library only and imports nothing from
``stmbench`` — the few constants it needs are inlined below. ``-I`` ignores every
``PYTHON*`` variable, so all I/O here is explicit UTF-8 bytes.

Environment: ``VISTA_RPC_PORT`` and ``VISTA_RPC_TOKEN`` (required), ``VISTA_RPC_HOST``
(default ``127.0.0.1``), ``VISTA_RPC_CONNECT_TIMEOUT`` (seconds, default 10).

MCP side (newline-delimited JSON-RPC 2.0 on stdin/stdout):

* ``initialize`` → ``protocolVersion`` (the client's if it is one we speak, else
  ``2025-06-18``), ``capabilities.tools``, ``serverInfo``; ``notifications/initialized``
  and other notifications are accepted silently; ``ping`` → ``{}``.
* ``tools/list`` and ``tools/call`` are forwarded to the RpcServer (one authenticated TCP
  connection, re-established once if it drops). They run on ONE worker thread in arrival
  order, so the main thread keeps answering ``ping`` while a long ``act`` runs, and
  ``notifications/cancelled`` can drop a call that has not started yet.
* If the host cannot be reached, ``tools/call`` answers a tool result with
  ``isError: true`` that says so (the model sees why nothing happened); ``tools/list``
  answers a JSON-RPC error.
* Unknown requests get ``-32601``; unparsable lines ``-32700``. stdout carries protocol
  messages only; diagnostics go to stderr. Messages of several MB (images) are fine.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import sys
import threading
from typing import Any

SERVER_NAME = "stm"                    # = vista_spm.protocol.SERVER_NAME
SERVER_VERSION = "0.1.0"
DEFAULT_PROTOCOL = "2025-06-18"        # = vista_spm.protocol.MCP_PROTOCOL
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
ERR_PARSE, ERR_INVALID, ERR_METHOD, ERR_INTERNAL = -32700, -32600, -32601, -32603


def log(message: str) -> None:
    try:
        sys.stderr.buffer.write(f"[vista-mcp-bridge] {message}\n".encode("utf-8", "replace"))
        sys.stderr.buffer.flush()
    except (OSError, ValueError):
        pass


class HostUnavailable(RuntimeError):
    pass


class RpcClient:
    """One authenticated JSON-lines connection to the RpcServer."""

    def __init__(self, host: str, port: int, token: str, connect_timeout: float) -> None:
        self.host, self.port, self.token = host, port, token
        self.connect_timeout = connect_timeout
        self._sock: socket.socket | None = None
        self._reader: Any = None
        self._next_id = 0
        self._lock = threading.Lock()

    def _connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        sock.settimeout(None)                 # an act may run for a long time
        self._sock, self._reader = sock, sock.makefile("rb")
        result = self._roundtrip("auth", {"token": self.token})
        if not isinstance(result, dict) or result.get("server") != SERVER_NAME:
            raise HostUnavailable(f"unexpected auth reply {result!r}")

    def close(self) -> None:
        for item in (self._reader, self._sock):
            try:
                if item is not None:
                    item.close()
            except OSError:
                pass
        self._sock = self._reader = None

    def _roundtrip(self, method: str, params: dict) -> Any:
        assert self._sock is not None
        self._next_id += 1
        rid = self._next_id
        line = json.dumps({"id": rid, "method": method, "params": params}, ensure_ascii=False,
                          separators=(",", ":")) + "\n"
        self._sock.sendall(line.encode("utf-8"))
        raw = self._reader.readline()
        if not raw:
            raise ConnectionError("the episode host closed the connection")
        response = json.loads(raw.decode("utf-8"))
        if not isinstance(response, dict):
            raise HostUnavailable("the episode host sent an invalid response")
        if response.get("error") is not None:
            err = response["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            raise HostUnavailable(f"host error: {message}")
        return response.get("result")

    def request(self, method: str, params: dict) -> Any:
        """One request; reconnects once. A ``tools/call`` whose line was already sent is
        never replayed (it may have executed)."""
        with self._lock:
            last: Exception | None = None
            for attempt in (1, 2):
                sent = False
                try:
                    if self._sock is None:
                        self._connect()
                    sent = True
                    return self._roundtrip(method, params)
                except (OSError, ConnectionError, ValueError, HostUnavailable) as exc:
                    self.close()
                    last = exc
                    log(f"rpc {method} attempt {attempt} failed: {type(exc).__name__}: {exc}")
                    if isinstance(exc, HostUnavailable) and str(exc).startswith("host error"):
                        break
                    if method == "tools/call" and sent:
                        break
            raise HostUnavailable(f"{type(last).__name__}: {last}")


class Bridge:
    def __init__(self, rpc: RpcClient | None, config_error: str | None = None) -> None:
        self.rpc = rpc
        self.config_error = config_error
        self._out_lock = threading.Lock()
        self._jobs: "queue.Queue[dict | None]" = queue.Queue()
        self._cancelled: set = set()
        self._cancel_lock = threading.Lock()
        self._worker = threading.Thread(target=self._work, name="bridge-worker", daemon=True)
        self._worker.start()
        self.alive = True

    # ── output ───────────────────────────────────────────────────────────────
    def send(self, message: dict) -> None:
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._out_lock:
            try:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            except (OSError, ValueError) as exc:
                self.alive = False
                log(f"stdout closed: {exc}")

    def reply(self, rid: Any, result: dict) -> None:
        self.send({"jsonrpc": "2.0", "id": rid, "result": result})

    def error(self, rid: Any, code: int, message: str) -> None:
        self.send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})

    # ── dispatch (main thread) ───────────────────────────────────────────────
    def handle(self, message: Any) -> None:
        if not isinstance(message, dict):
            self.error(None, ERR_INVALID, "a JSON-RPC message must be an object")
            return
        method = message.get("method")
        rid = message.get("id")
        is_request = "id" in message and rid is not None
        if method == "initialize":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            requested = params.get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL
            self.reply(rid, {"protocolVersion": version,
                             "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION}})
            return
        if method == "ping":
            if is_request:
                self.reply(rid, {})
            return
        if method == "notifications/cancelled":
            params = message.get("params") if isinstance(message.get("params"), dict) else {}
            with self._cancel_lock:
                self._cancelled.add(params.get("requestId"))
            return
        if isinstance(method, str) and method.startswith("notifications/"):
            return
        if method in ("tools/list", "tools/call"):
            if not is_request:
                return
            self._jobs.put(message)
            return
        if is_request:
            self.error(rid, ERR_METHOD, f"Method not found: {method}")

    # ── RPC work (worker thread, FIFO) ───────────────────────────────────────
    def _work(self) -> None:
        while True:
            message = self._jobs.get()
            if message is None:
                return
            rid = message.get("id")
            with self._cancel_lock:
                if rid in self._cancelled:
                    self._cancelled.discard(rid)
                    log(f"request {rid!r} cancelled before it started")
                    continue
            try:
                if message.get("method") == "tools/list":
                    self._tools_list(rid)
                else:
                    self._tools_call(rid, message.get("params"))
            except Exception as exc:  # noqa: BLE001 — a bridge bug must still answer
                log(f"internal error on {message.get('method')}: {type(exc).__name__}: {exc}")
                self.error(rid, ERR_INTERNAL, f"{type(exc).__name__}: {exc}")

    def _tools_list(self, rid: Any) -> None:
        if self.rpc is None:
            self.error(rid, ERR_INTERNAL, f"bridge misconfigured: {self.config_error}")
            return
        try:
            result = self.rpc.request("tools/list", {})
        except HostUnavailable as exc:
            self.error(rid, ERR_INTERNAL, f"the episode host is unreachable: {exc}")
            return
        if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
            self.error(rid, ERR_INTERNAL, "the episode host returned no tool list")
            return
        self.reply(rid, {"tools": result["tools"]})

    def _tools_call(self, rid: Any, params: Any) -> None:
        if not isinstance(params, dict) or not isinstance(params.get("name"), str):
            self.error(rid, ERR_INVALID, "tools/call needs params.name")
            return
        if self.rpc is None:
            self.reply(rid, _error_result(f"bridge misconfigured: {self.config_error}; nothing executed"))
            return
        arguments = params.get("arguments")
        try:
            result = self.rpc.request("tools/call", {"name": params["name"],
                                                     "arguments": {} if arguments is None else arguments,
                                                     "call_id": rid})
        except HostUnavailable as exc:
            self.reply(rid, _error_result(f"the episode host is unreachable ({exc}); "
                                          "the call may not have executed"))
            return
        if not isinstance(result, dict) or not isinstance(result.get("content"), list):
            self.reply(rid, _error_result("the episode host returned an invalid tool result"))
            return
        self.reply(rid, result)

    def close(self) -> None:
        self._jobs.put(None)


def _error_result(text: str) -> dict:
    return {"content": [{"type": "text", "text": json.dumps({"error": text}, ensure_ascii=False)}],
            "isError": True}


def _make_client() -> tuple[RpcClient | None, str | None]:
    port = os.environ.get("VISTA_RPC_PORT", "")
    token = os.environ.get("VISTA_RPC_TOKEN", "")
    host = os.environ.get("VISTA_RPC_HOST", "127.0.0.1") or "127.0.0.1"
    if not port.isdigit() or not token:
        return None, "VISTA_RPC_PORT and VISTA_RPC_TOKEN must be set"
    try:
        timeout = float(os.environ.get("VISTA_RPC_CONNECT_TIMEOUT", "10"))
    except ValueError:
        timeout = 10.0
    return RpcClient(host, int(port), token, timeout), None


def main() -> int:
    rpc, config_error = _make_client()
    if config_error:
        log(config_error)
    bridge = Bridge(rpc, config_error)
    log(f"started pid={os.getpid()}")
    try:
        for raw in sys.stdin.buffer:
            if not bridge.alive:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                bridge.error(None, ERR_PARSE, f"Parse error: {exc}")
                continue
            bridge.handle(message)
    finally:
        bridge.close()
        if rpc is not None:
            rpc.close()
        log("stdin closed; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
