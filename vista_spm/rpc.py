# Derived from VISTA (MIT), github.com/joshhhhhan/VISTA, backends/claude/mcp_host.py
"""Host-side TCP JSON-lines server behind the stdio MCP bridge (DESIGN.md §2, §7).

README
------
``RpcServer(handler)`` listens on ``127.0.0.1:<port>`` (``port=0`` picks a free port).
The MCP bridge (``mcp_bridge.py``, a separate stdlib-only process started by Claude Code)
and the hook script (``compact_hook.py``) connect to it. One JSON object per line, UTF-8::

    -> {"id": 1, "method": "auth", "params": {"token": "<token>"}}          (first line)
    <- {"id": 1, "result": {"server": "stm", "rpc_protocol": 1}}
    -> {"id": 2, "method": "tools/list"}
    <- {"id": 2, "result": {"tools": [<ToolDef.to_mcp() dicts>]}}
    -> {"id": 3, "method": "tools/call", "params": {"name": "act", "arguments": {...}, "call_id": 7}}
    <- {"id": 3, "result": {"content": [...], "isError": false}}           (ToolReply.to_mcp())
    -> {"id": 4, "method": "episode/hook", "params": {"event": {...}}}
    <- {"id": 4, "result": {"output": {"decision": "block", "reason": "..."}}}
    -> {"id": 5, "method": "ping"}
    <- {"id": 5, "result": {"pong": true}}
    errors: {"id": n, "error": {"code": -32601, "message": "..."}}

* The first line of every connection must authenticate with the token (``hmac``
  comparison); anything else gets an error and the connection is closed.
* ``tools/call`` runs ``handler.call`` under ONE lock (calls are serialized across all
  connections), then the optional *call observer* (``observer(name, args, reply) ->
  ToolReply | None``) under the same lock — the runner uses it to count delivered images
  and to append the checkpoint instruction to the reply that crosses the image limit.
  A handler that raises (it must not) produces an ``isError`` reply that names the
  exception; the traceback goes to the log.
* Calls can be gated: ``disable_calls(reason)`` makes ``tools/call`` wait up to
  ``gate_wait_s`` for ``enable_calls()`` and otherwise answer an ``isError`` reply with
  ``reason`` (the runner keeps the gate closed until it has validated Claude Code's init
  envelope, and closes it again when a segment ends). ``tools/list`` is never gated.
* Generations: :meth:`new_generation` (the runner calls it before each Claude Code
  process) makes every connection authenticated earlier stale; a stale connection's
  ``tools/call`` is refused without executing, even if it was waiting at the gate when a
  later segment opened it. Without ``new_generation`` all connections stay valid.
* ``tools/list`` adds ``_meta["anthropic/maxResultSizeChars"]`` (``result_size_chars``,
  default 200000) to every tool: Claude Code 2.1.295 otherwise saves any text result
  over 50000 characters to a file and shows the model only the path — and the model has
  no tool to read files.
* ``episode/hook`` hands the hook event to the *hook callback* (``callback(event) ->
  dict``; the returned dict is what the hook prints to Claude Code) under a separate lock.
* Every request and response is appended to ``log_path`` (switchable with
  :meth:`set_log_path`, e.g. one file per segment) with tokens redacted and every base64
  ``data`` string replaced by ``{"sha256", "chars"}``.
* Client disconnects only end that connection's thread; ``stop()`` closes everything.

Counters for the runner: :meth:`activity` (completed calls, calls in flight, monotonic
time of the last completion) and :attr:`tools_listed` (number of ``tools/list`` served).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .protocol import SERVER_NAME, ToolHandler, ToolReply, compact_json

RPC_PROTOCOL = 1
DEFAULT_RESULT_SIZE_CHARS = 200_000
ERR_PARSE, ERR_INVALID, ERR_METHOD, ERR_PARAMS, ERR_INTERNAL, ERR_AUTH = (
    -32700, -32600, -32601, -32602, -32603, -32001)

HookCallback = Callable[[dict], dict]
CallObserver = Callable[[str, dict, ToolReply], "ToolReply | None"]


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def sanitize(value: Any) -> Any:
    """Copy of ``value`` for logs: ``token`` redacted, long base64 ``data`` hashed."""
    if isinstance(value, dict):
        out: dict = {}
        for key, item in value.items():
            if key == "token" and isinstance(item, str):
                out[key] = "<redacted>"
            elif key == "data" and isinstance(item, str) and len(item) > 64:
                out[key] = {"sha256": hashlib.sha256(item.encode("utf-8")).hexdigest(),
                            "chars": len(item)}
            else:
                out[key] = sanitize(item)
        return out
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


class RpcServer:
    """See the module docstring."""

    def __init__(self, handler: ToolHandler, *, host: str = "127.0.0.1", port: int = 0,
                 token: str | None = None, log_path: str | Path | None = None,
                 hook_callback: HookCallback | None = None,
                 result_size_chars: int | None = DEFAULT_RESULT_SIZE_CHARS,
                 gate_wait_s: float = 30.0) -> None:
        self.handler = handler
        self.host = host
        self._requested_port = int(port)
        self.token = token or secrets.token_urlsafe(32)
        self.result_size_chars = result_size_chars
        self.gate_wait_s = float(gate_wait_s)
        self._log_path: Path | None = Path(log_path) if log_path else None
        self._hook_callback = hook_callback
        self._observer: CallObserver | None = None
        self._call_lock = threading.Lock()
        self._hook_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._gate = threading.Event()
        self._gate.set()
        self._gate_reason = ""
        self._stop = threading.Event()
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._conns: dict[int, socket.socket] = {}
        self._conn_threads: list[threading.Thread] = []
        self._next_conn = 0
        self._completed = 0
        self._in_flight = 0
        self._last_completed: float | None = None
        self._tools_listed = 0
        self._hooks_served = 0
        self._generation = 0
        self.port: int = 0

    # ── life cycle ───────────────────────────────────────────────────────────
    def start(self) -> "RpcServer":
        if self._accept_thread is not None:
            raise RuntimeError("RpcServer already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((self.host, self._requested_port))
        listener.listen(16)
        listener.settimeout(0.2)
        self._listener = listener
        self.port = listener.getsockname()[1]
        self._accept_thread = threading.Thread(target=self._accept_loop, name="vista-rpc-accept",
                                               daemon=True)
        self._accept_thread.start()
        self._log_event({"event": "started", "host": self.host, "port": self.port})
        return self

    def stop(self) -> None:
        self._stop.set()
        self._gate.set()                     # release waiters; they see _stop and refuse
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._state_lock:
            conns = list(self._conns.values())
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=5)
        for thread in list(self._conn_threads):
            thread.join(timeout=2)
        self._log_event({"event": "stopped"})

    def __enter__(self) -> "RpcServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    @property
    def address(self) -> tuple[str, int]:
        return self.host, self.port

    # ── configuration used by the runner ─────────────────────────────────────
    def set_log_path(self, path: str | Path | None) -> None:
        with self._log_lock:
            self._log_path = Path(path) if path else None

    def set_hook_callback(self, callback: HookCallback | None) -> None:
        with self._hook_lock:
            self._hook_callback = callback

    def set_call_observer(self, observer: CallObserver | None) -> None:
        with self._call_lock:
            self._observer = observer

    def enable_calls(self) -> None:
        self._gate_reason = ""
        self._gate.set()

    def disable_calls(self, reason: str) -> None:
        self._gate_reason = str(reason)
        self._gate.clear()

    @property
    def calls_enabled(self) -> bool:
        return self._gate.is_set()

    def new_generation(self) -> int:
        """Invalidate tool calls from every connection authenticated before now."""
        with self._state_lock:
            self._generation += 1
            return self._generation

    @property
    def generation(self) -> int:
        with self._state_lock:
            return self._generation

    def activity(self) -> tuple[int, int, float | None]:
        """``(completed calls, calls in flight, monotonic time of the last completion)``."""
        with self._state_lock:
            return self._completed, self._in_flight, self._last_completed

    @property
    def tools_listed(self) -> int:
        with self._state_lock:
            return self._tools_listed

    @property
    def hooks_served(self) -> int:
        with self._state_lock:
            return self._hooks_served

    def wait_idle(self, timeout: float) -> bool:
        """Wait until no tool call is in flight. ``True`` if idle within ``timeout``."""
        deadline = time.monotonic() + timeout
        while True:
            if self.activity()[1] == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)

    # ── networking ───────────────────────────────────────────────────────────
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                conn, peer = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                time.sleep(0.05)
                continue
            conn.settimeout(None)
            with self._state_lock:
                self._next_conn += 1
                cid = self._next_conn
                self._conns[cid] = conn
            thread = threading.Thread(target=self._serve, args=(cid, conn), name=f"vista-rpc-{cid}",
                                      daemon=True)
            self._conn_threads.append(thread)
            thread.start()

    def _serve(self, cid: int, conn: socket.socket) -> None:
        self._log_event({"event": "connected", "conn": cid})
        reader = conn.makefile("rb")
        authenticated = False
        conn_generation = -1
        try:
            while not self._stop.is_set():
                try:
                    raw = reader.readline()
                except OSError:
                    break
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                request: Any = None
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    response = {"id": None, "error": {"code": ERR_PARSE, "message": f"invalid JSON: {exc}"}}
                    self._log("request", cid, {"unparsable": line[:200]})
                    if not self._send(conn, cid, response):
                        break
                    if not authenticated:
                        break
                    continue
                self._log("request", cid, request)
                if not isinstance(request, dict):
                    response = {"id": None, "error": {"code": ERR_INVALID, "message": "request must be an object"}}
                    self._send(conn, cid, response)
                    break
                rid = request.get("id")
                if not authenticated:
                    ok = self._authenticate(request)
                    if ok:
                        authenticated = True
                        conn_generation = self.generation
                        response = {"id": rid, "result": {"server": SERVER_NAME, "rpc_protocol": RPC_PROTOCOL}}
                        if not self._send(conn, cid, response):
                            break
                        continue
                    self._send(conn, cid, {"id": rid, "error": {"code": ERR_AUTH, "message": "unauthorized"}})
                    break
                try:
                    result = self._dispatch(request, conn_generation)
                    response = {"id": rid, "result": result}
                except RpcError as exc:
                    response = {"id": rid, "error": {"code": exc.code, "message": exc.message}}
                except Exception as exc:  # noqa: BLE001 — never let a request kill the server
                    self._log_event({"event": "internal_error", "conn": cid,
                                     "traceback": traceback.format_exc()})
                    response = {"id": rid, "error": {"code": ERR_INTERNAL,
                                                     "message": f"{type(exc).__name__}: {exc}"}}
                if not self._send(conn, cid, response):
                    break
        finally:
            try:
                reader.close()
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
            with self._state_lock:
                self._conns.pop(cid, None)
            self._log_event({"event": "disconnected", "conn": cid})

    def _authenticate(self, request: dict) -> bool:
        if request.get("method") != "auth":
            return False
        params = request.get("params")
        token = params.get("token") if isinstance(params, dict) else None
        return isinstance(token, str) and hmac.compare_digest(token.encode("utf-8"),
                                                              self.token.encode("utf-8"))

    def _send(self, conn: socket.socket, cid: int, response: dict) -> bool:
        self._log("response", cid, response)
        data = (json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            conn.sendall(data)
            return True
        except OSError as exc:
            self._log_event({"event": "send_failed", "conn": cid, "error": f"{type(exc).__name__}: {exc}"})
            return False

    # ── methods ──────────────────────────────────────────────────────────────
    def _dispatch(self, request: dict, conn_generation: int = 0) -> dict:
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise RpcError(ERR_PARAMS, "params must be an object")
        if method == "ping":
            return {"pong": True}
        if method == "auth":
            return {"server": SERVER_NAME, "rpc_protocol": RPC_PROTOCOL}
        if method == "tools/list":
            return self._tools_list()
        if method == "tools/call":
            return self._tools_call(params, conn_generation)
        if method == "episode/hook":
            return self._episode_hook(params)
        raise RpcError(ERR_METHOD, f"unknown method {method!r}")

    def _tools_list(self) -> dict:
        tools = []
        for tool in self.handler.list_tools():
            entry = dict(tool)
            if self.result_size_chars:
                meta = dict(entry.get("_meta") or {})
                meta.setdefault("anthropic/maxResultSizeChars", int(self.result_size_chars))
                entry["_meta"] = meta
            tools.append(entry)
        with self._state_lock:
            self._tools_listed += 1
        return {"tools": tools}

    def _tools_call(self, params: dict, conn_generation: int = 0) -> dict:
        name = params.get("name")
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(name, str) or not name:
            raise RpcError(ERR_PARAMS, "tools/call needs a tool name")
        if not isinstance(args, dict):
            return ToolReply(text=compact_json({"error": "arguments must be a JSON object"}),
                             is_error=True).to_mcp()
        if conn_generation != self.generation:
            return self._stale(name, conn_generation)
        if not self._gate.is_set():
            self._gate.wait(self.gate_wait_s)
        if conn_generation != self.generation:
            return self._stale(name, conn_generation)
        if self._stop.is_set() or not self._gate.is_set():
            reason = self._gate_reason or "the episode host is not accepting tool calls"
            return ToolReply(text=compact_json({"error": reason, "executed": False}), is_error=True).to_mcp()
        with self._state_lock:
            self._in_flight += 1
        try:
            with self._call_lock:
                try:
                    reply = self.handler.call(name, args)
                    if not isinstance(reply, ToolReply):
                        raise TypeError(f"handler returned {type(reply).__name__}, not ToolReply")
                except Exception as exc:  # noqa: BLE001 — the protocol says handlers never raise
                    self._log_event({"event": "handler_raised", "tool": name,
                                     "traceback": traceback.format_exc()})
                    reply = ToolReply(text=compact_json({"error": f"internal error in tool {name}: "
                                                                  f"{type(exc).__name__}: {exc}"}),
                                      is_error=True)
                observer = self._observer
                if observer is not None:
                    try:
                        changed = observer(name, args, reply)
                        if isinstance(changed, ToolReply):
                            reply = changed
                    except Exception:  # noqa: BLE001
                        self._log_event({"event": "observer_raised", "tool": name,
                                         "traceback": traceback.format_exc()})
                return reply.to_mcp()
        finally:
            with self._state_lock:
                self._in_flight -= 1
                self._completed += 1
                self._last_completed = time.monotonic()

    def _stale(self, name: str, conn_generation: int) -> dict:
        self._log_event({"event": "stale_call_refused", "tool": name, "generation": conn_generation})
        return ToolReply(text=compact_json({"error": "this call came from a Claude Code session that has "
                                                     "ended; nothing was executed", "executed": False}),
                         is_error=True).to_mcp()

    def _episode_hook(self, params: dict) -> dict:
        event = params.get("event")
        if not isinstance(event, dict):
            raise RpcError(ERR_PARAMS, "episode/hook needs an 'event' object")
        with self._state_lock:
            self._hooks_served += 1
        with self._hook_lock:
            callback = self._hook_callback
            if callback is None:
                return {"output": {}, "note": "no hook callback registered"}
            output = callback(event)
        if output is None:
            output = {}
        if not isinstance(output, dict):
            raise RpcError(ERR_INTERNAL, "hook callback returned a non-dict")
        return {"output": output}

    # ── logging ──────────────────────────────────────────────────────────────
    def _log(self, direction: str, cid: int, message: Any) -> None:
        self._write_log({"t": round(time.time(), 3), "dir": direction, "conn": cid,
                         "msg": sanitize(message)})

    def _log_event(self, record: dict) -> None:
        self._write_log({"t": round(time.time(), 3), **record})

    def _write_log(self, record: dict) -> None:
        with self._log_lock:
            path = self._log_path
            if path is None:
                return
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                        default=str) + "\n")
            except OSError:
                pass
