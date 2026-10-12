# Derived from VISTA (MIT), github.com/joshhhhhan/VISTA, backends/claude/{runner,lifecycle,preflight}.py
"""Claude Code segment runner: one episode = a chain of ``claude -p`` sessions (DESIGN.md §7).

README
------
``ClaudeCodeRunner(host, rpc, cfg).run()`` plays one episode against a
:class:`~vista_spm.protocol.RunnerHost` and returns a :class:`RunResult`;
``preflight()`` runs VISTA's throw-away "Reply exactly READY" segment and returns a dict
(``ok``, ``problems``, resolved ``model``, init envelope summary, cost).

A *segment* is one Claude Code process (= one session, always fresh: ``--session-id
<uuid4>``). The runner

1. writes ``segment_NNN.{settings,mcp,command}.json`` (hooks in exec form, the ``stm``
   MCP server = ``<python> -I -X utf8 mcp_bridge.py`` with the RpcServer's port/token),
   starts a new RpcServer generation (tool calls still queued from an earlier process
   are refused, never executed later) and closes the call gate, starts ``claude`` in an
   empty directory with a cleaned environment (:func:`child_environment`) and waits until
   the bridge has listed the tools (Claude Code connects it at startup, before stdin);
2. calls ``host.begin_segment(i, session_id)`` and sends the first user message
   (``host.initial_message()`` or ``host.recovery_message()``: text + label/PNG image
   blocks) as stream-json on stdin, which stays open;
3. validates EVERY ``system/init`` envelope (Claude Code emits one per user turn): the
   tool list must be exactly ``mcp__stm__<each host tool>``, the only MCP server ``stm``
   connected, ``permissionMode`` ``default``; then opens the call gate;
4. reads stdout on a thread: an ``assistant`` ``tool_use`` of ``act`` is remembered and
   the ``user`` ``tool_result`` for it calls ``host.acknowledge_act()`` immediately (on
   the reader thread, before the model can issue the next call). Images in tool replies
   are counted by an RpcServer call observer; when ``image_limit_per_session -
   image_reserve`` is reached the runner calls ``host.request_checkpoint("image_limit")``
   and appends the checkpoint instruction to that reply;
5. at each ``result`` (end of a turn): final text ending with ``[DONE]``/``[ABORT]``
   (``ic_driver.marker_outcome``) ⇒ end; terminal host ⇒ end; checkpoint ``ready`` ⇒
   hand off; API error ⇒ back off and continue (bounded); otherwise send
   ``cfg.continue_text`` (plus the checkpoint instruction while one is requested) as the
   next user message of the same process; ``stall_turns`` consecutive turns without an
   ``act``/report call ⇒ ``stalled``;
6. ends a segment by an ``interrupt`` control request (if a turn is running) and closing
   stdin, so Claude Code still emits its ``result`` with the cumulative cost (such a
   process exits with code 1, because its last result is ``error_during_execution``); a
   process that does not exit within ``end_grace_s`` is killed with its whole tree
   (``taskkill /T /F`` on Windows). Then it waits until no tool call is in flight and
   calls ``host.end_segment(i, info)``. The host's checkpoint or terminal state is
   polled every 0.5 s, so a handoff or a budget stop interrupts a running turn.

Compaction modes (``cfg.compaction``): ``"checkpoint"`` (default, needs the
``write_working`` tool) runs VISTA's checkpoint protocol below; ``"native"`` (arms without
notes, VISTA's ``native_compaction``) lets Claude Code compact on its own (the hooks only
record), resets the session image count at each ``compact_boundary``, and at the image
limit ends the segment and continues in a fresh session (``kind="handoff"``) with
``host.recovery_message()``.

Checkpoint protocol (VISTA): the PreCompact/Stop hooks (``compact_hook.py``) reach the
runner's hook callback through the RpcServer; a PreCompact without a ready checkpoint
blocks compaction and calls ``host.request_checkpoint("context_limit")``; the host then
executes only the notes tools until ``write_working`` makes ``status()["checkpoint"] ==
"ready"``; the runner ends the segment and starts a fresh session with
``host.recovery_message()``. After every recovery message the runner calls
``host.acknowledge_act()`` (the fresh context has seen the current observation). A native
``compact_boundary`` ends the episode with outcome ``error``; so does an invalid init
envelope, and so do three consecutive checkpoint handoffs without an executed act (VISTA's
``MAX_COMPACT_CYCLES_WITHOUT_ACTION``: the recovery context alone already crosses the
compaction threshold). A segment that crashes, stops responding for ``turn_timeout_s`` (no tool call
completed and no turn ended — the decision lease), or exhausts its API retries is
replaced by a fresh recovery session, at most ``max_recoveries`` times.

Next-segment kinds: ``checkpoint`` (after a handoff), ``recovery`` (after a failure),
``handoff`` (native mode, image limit). Episode outcomes are ``protocol.RunnerHost.finish``'s
set; a terminal host maps to its ``stop_reason`` when that is one of them, else ``budget``.

Host contract (in addition to ``protocol.RunnerHost``): host methods are called from
three threads — the RPC thread (``list_tools``, ``call``, ``status`` and
``request_checkpoint`` from the image observer and the hook callback), the stdout reader
(``acknowledge_act``) and the runner thread (everything else) — so they must be
thread-safe. ``begin_segment`` must reset the checkpoint state to ``"none"`` (every
segment is a fresh context); ``recovery_message`` must be self-contained (task text
included); ``status()["acts"]`` feeds the checkpoint-loop guard. The runner also calls
``acknowledge_act`` for an act whose result was an error (refused or failed): the result
reached the model either way.

Logs under ``cfg.io_dir``: ``system_prompt.md``, ``segment_NNN.{command.json,
settings.json, mcp.json (token redacted after the segment), input.json (image data
replaced by sha256 + size), stdout.jsonl (likewise), stderr.txt, bridge.jsonl (RpcServer
log), hooks.jsonl}``, ``hooks/`` (hook script log + fallback markers), ``cwd/`` (the
empty working directory) and ``manifest.json`` (per segment: session id, models, exit
code, end reason, usage, cost_usd, num_turns, images, acts, ``compact_boundaries`` and
``compact_boundary_count``; totals incl. ``compact_boundaries``). ``RunResult.segments``
are the same records and ``RunResult.usage`` the totals. The preflight uses the stem
``preflight``.

Verified with Claude Code 2.1.295 on Windows 11 (2026-10-09, see the tests): stream-json
input accepts further user messages after a ``result`` in the same process (each turn
emits a new ``system/init``; ``total_cost_usd`` and ``modelUsage`` in ``result`` are
cumulative per process, ``usage`` and ``num_turns`` per turn); ``{"type":
"control_request", "request": {"subtype": "interrupt"}}`` ends a running turn with a
``result`` of subtype ``error_during_execution``; claude.ai connectors join the session
from the second turn on unless ``ENABLE_CLAUDEAI_MCP_SERVERS=false``; MCP image results
reach the model as ``text, image, text`` (Claude Code appends an ``[Image: source: <saved
copy>]`` note); with ``CLAUDE_CODE_AUTO_COMPACT_WINDOW=W`` and
``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=P`` compaction is attempted at roughly (W - 33000) x P/100
context tokens (fitted to two measurements with W=100000: P=5 between 3252 and 4020 tokens,
P=6 at ~4020), i.e. ~150k tokens for the defaults (200000, 90) — an estimate, not a spec.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from stmbench.harness.ic_driver import marker_outcome

from . import compact_hook
from .compact_hook import CHECKPOINT_INSTRUCTION
from .protocol import SERVER_NAME, ImageBlock, RunnerHost, ToolReply
from .rpc import RpcServer, sanitize

BRIDGE_SCRIPT = Path(__file__).resolve().with_name("mcp_bridge.py")
HOOK_SCRIPT = Path(__file__).resolve().with_name("compact_hook.py")

PREFLIGHT_PROMPT = "This is a runtime preflight, not a task. Do not call tools. Reply exactly READY."
DEFAULT_CONTINUE_TEXT = (
    "The episode is still active. Context limits are handled automatically. Continue. "
    "End your reply with [DONE] when the task is complete, or [ABORT] if it cannot be completed."
)
OUTCOMES = ("done", "abort", "budget", "error", "stalled", "max_segments")
COMPACTION_MODES = ("checkpoint", "native")
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MODEL_ALIASES = ("opus", "sonnet", "haiku", "fable", "default")
PERMISSION_MODE = "default"
RETRYABLE_API_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
SESSION_ENV_NAMES = ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT")
#: Credentials that make Claude Code bill an API account instead of the logged-in subscription.
#: Dropped from the child environment unless ``RunnerConfig.auth == "env"``: a host process (MAST's
#: runtime exports the repository's provider keys) must not decide who pays for an episode.
API_CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
AUTH_MODES = ("subscription", "env")
#: Variables that would silently disable the compaction hook path.
COMPACTION_ENV_BLOCKLIST = ("DISABLE_AUTO_COMPACT", "DISABLE_COMPACT")
INTERRUPT_WAIT_S = 20.0
#: VISTA's MAX_COMPACT_CYCLES_WITHOUT_ACTION: consecutive checkpoint handoffs without an executed act.
MAX_CHECKPOINTS_WITHOUT_ACT = 3
POLL_S = 0.2
STATUS_EVERY_S = 0.5


class RunnerError(RuntimeError):
    pass


# ── configuration ────────────────────────────────────────────────────────────
@dataclass
class RunnerConfig:
    """Everything the runner needs besides the host and the RpcServer."""

    model: str = "claude-opus-5-5"
    effort: str = "high"
    system_prompt: str = ""
    #: ``"claude"`` (resolved on PATH) or a full command prefix, e.g. ``[python, fake_claude.py]``.
    claude_bin: str | Sequence[str] = "claude"
    python_exe: str = sys.executable
    io_dir: str | Path = "claude_io"
    compact_window_tokens: int = 200_000
    compact_pct: int = 90
    image_limit_per_session: int = 500
    #: Request the checkpoint this many images before the limit (an inspect returns up to 16).
    image_reserve: int = 20
    max_segments: int = 20
    #: Decision lease: no tool call completed and no turn ended for this long ⇒ recover.
    turn_timeout_s: float = 3600.0
    stall_turns: int = 3
    continue_text: str = DEFAULT_CONTINUE_TEXT
    preflight: bool = True
    extra_env: dict[str, str] = field(default_factory=dict)
    #: Tool names (without the ``mcp__stm__`` prefix) whose result must be acknowledged.
    act_tools: tuple[str, ...] = ("act",)
    #: Tool names that count as progress for the stall rule.
    progress_tools: tuple[str, ...] = ("act", "report_result", "report_tip_state")
    #: If set, the model id Claude Code resolves must equal it.
    expected_model: str | None = None
    #: ``--max-budget-usd`` per segment (a safety cap for tests); ``None`` = no cap.
    max_budget_usd: float | None = None
    startup_timeout_s: float = 120.0
    end_grace_s: float = 30.0
    hook_timeout_s: int = 30
    #: Per-server MCP tool timeout (hard wall clock per call); ``None`` = ``turn_timeout_s``.
    mcp_tool_timeout_s: float | None = None
    #: Waits before continuing after a retryable API error (one entry per retry).
    api_retry_backoff_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    max_recoveries: int = 5
    max_description_chars: int = 8192
    #: ``"checkpoint"`` (VISTA's WORKING.md handoff) or ``"native"`` (Claude Code compacts itself).
    compaction: str = "checkpoint"
    #: ``"subscription"``: drop API credentials from the child environment and refuse a session whose
    #: init reports an API key source. ``"env"``: pass the caller's credentials through (explicit opt-in
    #: to API billing).
    auth: str = "subscription"

    def __post_init__(self) -> None:
        if self.auth not in AUTH_MODES:
            raise ValueError(f"auth {self.auth!r} not in {AUTH_MODES}")
        if self.compaction not in COMPACTION_MODES:
            raise ValueError(f"compaction {self.compaction!r} not in {COMPACTION_MODES}")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort {self.effort!r} not in {EFFORT_LEVELS}")
        if not 100_000 <= int(self.compact_window_tokens) <= 1_000_000:
            raise ValueError("compact_window_tokens must be in [100000, 1000000] "
                             "(Claude Code clamps CLAUDE_CODE_AUTO_COMPACT_WINDOW to that range)")
        if not 1 <= int(self.compact_pct) <= 100:
            raise ValueError("compact_pct must be in [1, 100]")
        if int(self.image_limit_per_session) <= int(self.image_reserve) or int(self.image_reserve) < 0:
            raise ValueError("need image_limit_per_session > image_reserve >= 0")
        if int(self.max_segments) < 1 or int(self.stall_turns) < 1:
            raise ValueError("max_segments and stall_turns must be >= 1")
        if float(self.turn_timeout_s) <= 0:
            raise ValueError("turn_timeout_s must be positive")
        self.io_dir = Path(self.io_dir)

    def summary(self) -> dict:
        """JSON-safe description for the manifest (the system prompt only as hash + size)."""
        out: dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            value = getattr(self, key)
            if key == "system_prompt":
                out["system_prompt_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
                out["system_prompt_chars"] = len(value)
            elif key == "extra_env":
                out[key] = sorted(value)
            elif isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
            else:
                out[key] = value
        return out


@dataclass
class RunResult:
    outcome: str
    final_text: str
    segments: list[dict]
    usage: dict
    models: list[str]
    preflight: dict | None = None
    manifest_path: str | None = None
    error: str | None = None


# ── pure helpers (unit-tested) ────────────────────────────────────────────────
def auth_problem(init: Mapping[str, Any], auth: str) -> str | None:
    """Why this session must not run under ``auth``, or ``None``: in subscription mode an init that
    reports an API key source would bill that key."""
    src = init.get("apiKeySource")
    if auth == "subscription" and isinstance(src, str) and src not in ("", "none"):
        return (f"apiKeySource {src!r}: this session would bill an API key, but RunnerConfig.auth is "
                "'subscription'")
    return None


def child_environment(base: Mapping[str, str], cfg: RunnerConfig, *, rpc_host: str, rpc_port: int,
                      rpc_token: str, hook_dir: Path) -> dict[str, str]:
    """The ``claude`` process environment: the caller's minus everything that ties a
    process to a calling Claude Code session, plus the harness settings."""
    env = {k: v for k, v in base.items()
           if not k.startswith("CLAUDE_CODE_") and k not in SESSION_ENV_NAMES
           and k not in COMPACTION_ENV_BLOCKLIST
           and not (cfg.auth == "subscription" and k in API_CREDENTIAL_ENV)}
    env.update({
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
        "CLAUDE_CODE_DISABLE_CRON": "1",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
        "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": str(int(cfg.max_description_chars)),
        "MAX_MCP_OUTPUT_TOKENS": "100000",
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(int(cfg.compact_window_tokens)),
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": str(int(cfg.compact_pct)),
        # Verified 2026-10-09: without this, claude.ai connectors (Docs, Drive ...) join the
        # session from the second turn on, --strict-mcp-config notwithstanding.
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
        "ENABLE_TOOL_SEARCH": "false",            # MCP tools up front, no ToolSearch indirection
        "MCP_CONNECTION_NONBLOCKING": "0",        # stream-json input does not wait otherwise
        "VISTA_RPC_HOST": rpc_host,
        "VISTA_RPC_PORT": str(int(rpc_port)),
        "VISTA_RPC_TOKEN": rpc_token,
        "VISTA_HOOK_DIR": str(hook_dir),
        "VISTA_COMPACTION": cfg.compaction,
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    env.update({str(k): str(v) for k, v in cfg.extra_env.items()})
    return env


def mcp_config(cfg: RunnerConfig, *, rpc_host: str, rpc_port: int, rpc_token: str) -> dict:
    timeout_s = cfg.mcp_tool_timeout_s if cfg.mcp_tool_timeout_s is not None else cfg.turn_timeout_s
    return {"mcpServers": {SERVER_NAME: {
        "type": "stdio",
        "command": str(cfg.python_exe),
        "args": ["-I", "-X", "utf8", str(BRIDGE_SCRIPT)],
        "env": {"VISTA_RPC_HOST": rpc_host, "VISTA_RPC_PORT": str(int(rpc_port)),
                "VISTA_RPC_TOKEN": rpc_token, "PYTHONUTF8": "1"},
        "timeout": int(float(timeout_s) * 1000),
        "alwaysLoad": True,
    }}}


def hook_settings(cfg: RunnerConfig) -> dict:
    hook = {"type": "command", "command": str(cfg.python_exe),
            "args": ["-I", "-X", "utf8", str(HOOK_SCRIPT)], "timeout": int(cfg.hook_timeout_s)}
    return {"autoCompactEnabled": True,
            "hooks": {event: [{"hooks": [hook]}] for event in ("PreCompact", "PostCompact", "Stop")}}


def resolve_claude(claude_bin: str | Sequence[str]) -> list[str]:
    if isinstance(claude_bin, (list, tuple)):
        if not claude_bin:
            raise RunnerError("claude_bin is an empty command")
        return [str(part) for part in claude_bin]
    found = shutil.which(str(claude_bin))
    if not found:
        raise RunnerError(f"Claude Code executable {claude_bin!r} not found on PATH")
    return [found]


def build_command(cfg: RunnerConfig, *, claude: Sequence[str], system_prompt_path: Path,
                  settings_path: Path, mcp_path: Path, allowed_tools: Sequence[str],
                  session_id: str) -> list[str]:
    cmd = [*claude, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
           "--verbose", "--include-hook-events", "--model", cfg.model, "--effort", cfg.effort,
           "--system-prompt-file", str(system_prompt_path), "--setting-sources", "",
           "--settings", str(settings_path), "--mcp-config", str(mcp_path), "--strict-mcp-config",
           "--tools", "", "--allowedTools", ",".join(allowed_tools),
           "--permission-mode", PERMISSION_MODE, "--permission-prompts", "none",
           "--disable-slash-commands", "--prompt-suggestions", "false", "--no-chrome",
           "--session-id", session_id]
    if cfg.max_budget_usd is not None:
        cmd += ["--max-budget-usd", f"{float(cfg.max_budget_usd):g}"]
    return cmd


def user_message(text: str, images: Sequence[ImageBlock] = ()) -> dict:
    """One stream-json user message: the text, then per image its label and the PNG."""
    content: list[dict] = [{"type": "text", "text": text}]
    for image in images:
        if image.label:
            content.append({"type": "text", "text": image.label})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                    "data": image.b64()}})
    return {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}


def validate_init(envelope: Mapping[str, Any], expected_tools: Sequence[str], *,
                  expected_model: str | None = None) -> list[str]:
    """Problems with a ``system/init`` envelope (empty list = valid)."""
    problems: list[str] = []
    tools = envelope.get("tools")
    expected = sorted(expected_tools)
    if not isinstance(tools, list) or len(tools) != len(set(tools)) or sorted(tools) != expected:
        extra = sorted(set(tools or []) - set(expected)) if isinstance(tools, list) else tools
        missing = sorted(set(expected) - set(tools or [])) if isinstance(tools, list) else expected
        problems.append(f"unexpected tool surface (extra {extra}, missing {missing})")
    servers = envelope.get("mcp_servers")
    if not isinstance(servers, list) or len(servers) != 1 or not isinstance(servers[0], dict) \
            or servers[0].get("name") != SERVER_NAME:
        problems.append(f"MCP servers must be exactly [{SERVER_NAME!r}], got "
                        f"{[s.get('name') for s in servers] if isinstance(servers, list) else servers}")
    elif str(servers[0].get("status", "")).lower() != "connected":
        problems.append(f"MCP server {SERVER_NAME!r} status is {servers[0].get('status')!r}")
    if envelope.get("mcp_server_errors"):
        problems.append(f"mcp_server_errors: {envelope.get('mcp_server_errors')}")
    if envelope.get("permissionMode") != PERMISSION_MODE:
        problems.append(f"permissionMode {envelope.get('permissionMode')!r} != {PERMISSION_MODE!r}")
    model = envelope.get("model")
    if not isinstance(model, str) or not model.strip():
        problems.append("no resolved model id")
    elif model.strip().lower() in MODEL_ALIASES:
        problems.append(f"model {model!r} is an alias, not a resolved id")
    elif expected_model is not None and model != expected_model:
        problems.append(f"model {model!r} != expected {expected_model!r}")
    return problems


def init_summary(envelope: Mapping[str, Any] | None) -> dict | None:
    if envelope is None:
        return None
    keys = ("session_id", "model", "tools", "mcp_servers", "permissionMode", "claude_code_version",
            "skills", "plugins", "slash_commands", "output_style", "apiKeySource", "capabilities")
    return {k: envelope[k] for k in keys if k in envelope}


USAGE_LIMIT_MARKERS = ("usage_limit_reached", "hit your session limit", "hit your weekly limit",
                       "hit your usage limit", "usage limit reached", "session limit · resets")


def usage_limit_text(message: Mapping[str, Any]) -> str | None:
    """The account's usage-limit message when this ``result`` is one, else ``None``.

    A subscription usage limit is a 429 that no retry within the episode can outlast (it resets
    hours later), so it is not ``retryable``: the run ends at once and says so."""
    if message.get("subtype") == "success" and message.get("is_error") is not True:
        return None
    raw = [message.get("result"), message.get("errors")]
    text = json.dumps(raw, ensure_ascii=False).lower()
    if any(m in text for m in USAGE_LIMIT_MARKERS):
        r = message.get("result")
        return (r if isinstance(r, str) and r else json.dumps(raw, ensure_ascii=False))[:300]
    return None


def classify_result(message: Mapping[str, Any]) -> str:
    """``ok`` | ``budget_usd`` | ``usage_limit`` | ``retryable`` | ``error`` for a stream ``result``."""
    subtype = message.get("subtype")
    if subtype == "success" and message.get("is_error") is not True:
        return "ok"
    if subtype == "error_max_budget_usd":
        return "budget_usd"
    if usage_limit_text(message) is not None:
        return "usage_limit"
    status = message.get("api_error_status")
    if isinstance(status, int) and (status in RETRYABLE_API_STATUS or status >= 500):
        return "retryable"
    if message.get("terminal_reason") == "api_error":
        return "retryable"
    text = json.dumps([message.get("result"), message.get("errors")], ensure_ascii=False).lower()
    if "overloaded" in text or "rate limit" in text or "529" in text:
        return "retryable"
    return "error"


def result_usage(message: Mapping[str, Any] | None) -> dict | None:
    """Cumulative usage of the process from a ``result``'s ``modelUsage`` and ``total_cost_usd``."""
    if not isinstance(message, Mapping):
        return None
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
              "cache_read_input_tokens": 0}
    model_usage = message.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        for item in model_usage.values():
            if not isinstance(item, dict):
                continue
            totals["input_tokens"] += int(item.get("inputTokens") or 0)
            totals["output_tokens"] += int(item.get("outputTokens") or 0)
            totals["cache_creation_input_tokens"] += int(item.get("cacheCreationInputTokens") or 0)
            totals["cache_read_input_tokens"] += int(item.get("cacheReadInputTokens") or 0)
    else:
        usage = message.get("usage") if isinstance(message.get("usage"), dict) else {}
        for key in totals:
            totals[key] = int(usage.get(key) or 0)
    cost = message.get("total_cost_usd")
    totals["cost_usd"] = float(cost) if isinstance(cost, (int, float)) else None
    return totals


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill ``proc`` and its children (Windows: ``taskkill /T /F``)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True,
                           timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def _atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _short(name: str) -> str:
    prefix = f"mcp__{SERVER_NAME}__"
    return name[len(prefix):] if isinstance(name, str) and name.startswith(prefix) else str(name)


# ── per-segment state ────────────────────────────────────────────────────────
@dataclass
class _Segment:
    index: int | None                 # None for the preflight
    stem: str
    kind: str                         # initial | checkpoint | recovery | preflight
    session_id: str
    expected_tools: list[str]
    started: float = field(default_factory=time.monotonic)
    started_wall: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    proc: subprocess.Popen | None = None
    queue: "queue.Queue[tuple[str, Any]]" = field(default_factory=queue.Queue)
    markers: str = "none"             # checkpoint state last mirrored into the hook markers
    init: dict | None = None
    init_count: int = 0
    claude_code_version: str | None = None
    models: set = field(default_factory=set)
    problems: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    results: list = field(default_factory=list)
    last_result: dict | None = None
    turn_open: bool = False
    turn_progress: bool = False
    stall: int = 0
    api_failures: int = 0
    final_text: str = ""
    last_text: str = ""
    end_reason: str | None = None
    marker: str | None = None
    pending_acts: set = field(default_factory=set)
    acts_acknowledged: int = 0
    act_errors: int = 0
    images_input: int = 0
    images_sent: int = 0
    images_seen: int = 0
    tool_uses: int = 0
    tool_counts: dict = field(default_factory=dict)
    compaction_attempts: int = 0
    compact_boundaries: list = field(default_factory=list)
    hook_events: list = field(default_factory=list)
    stop_hook_blocks: int = 0
    api_retries: list = field(default_factory=list)
    permission_denials: list = field(default_factory=list)
    rate_limit: dict | None = None
    checkpoint_armed: bool = True
    begun: bool = False
    image_handoff: bool = False       # native mode: the image limit was reached
    checkpoint_requests: list = field(default_factory=list)
    interrupt_sent: bool = False
    interrupt_deadline: float = 0.0
    kill_deadline: float = 0.0
    stdin_closed: bool = False
    killed: bool = False
    exit_code: int | None = None
    last_activity: float = field(default_factory=time.monotonic)
    messages_sent: int = 0
    inputs: list = field(default_factory=list)
    stderr_tail: bytes = b""
    ended_wall: str | None = None
    duration_s: float | None = None

    @property
    def preflight(self) -> bool:
        return self.kind == "preflight"

    def record(self) -> dict:
        usage = result_usage(self.last_result)
        return {
            "index": self.index, "stem": self.stem, "kind": self.kind, "session_id": self.session_id,
            "started_at": self.started_wall, "ended_at": self.ended_wall, "duration_s": self.duration_s,
            "claude_code_version": self.claude_code_version, "models": sorted(self.models),
            "init": init_summary(self.init), "init_events": self.init_count,
            "exit_code": self.exit_code, "killed": self.killed, "end_reason": self.end_reason,
            "boundary": self.end_reason, "marker": self.marker,
            "usage": usage, "cost_usd": usage.get("cost_usd") if usage else None,
            "usage_complete": bool(self.last_result) and not self.turn_open,
            "num_turns": sum(int(r.get("num_turns") or 0) for r in self.results),
            "turns": len(self.results), "results": self.results,
            "messages_sent": self.messages_sent, "images_input": self.images_input,
            "images_sent": self.images_sent, "images_seen": self.images_seen,
            "tool_uses": self.tool_uses, "tool_counts": dict(self.tool_counts),
            "acts_acknowledged": self.acts_acknowledged, "act_errors": self.act_errors,
            "unacknowledged_acts": len(self.pending_acts),
            "compaction_attempts": self.compaction_attempts, "compact_boundaries": self.compact_boundaries,
            "compact_boundary_count": len(self.compact_boundaries),
            "hook_events": len(self.hook_events), "stop_hook_blocks": self.stop_hook_blocks,
            "checkpoint_requests": self.checkpoint_requests, "api_retries": self.api_retries,
            "permission_denials": self.permission_denials, "rate_limit": self.rate_limit,
            "problems": self.problems, "notes": self.notes,
            "stderr_tail": self.stderr_tail.decode("utf-8", "replace")[-2000:],
        }


# ── the runner ───────────────────────────────────────────────────────────────
class ClaudeCodeRunner:
    """Drives one episode through Claude Code segments (see the module docstring)."""

    def __init__(self, host: RunnerHost, rpc: RpcServer, cfg: RunnerConfig | None = None, *,
                 sleep: Callable[[float], None] = time.sleep,
                 base_env: Mapping[str, str] | None = None) -> None:
        self.host = host
        self.rpc = rpc
        self.cfg = cfg or RunnerConfig()
        self.io_dir = Path(self.cfg.io_dir)
        self.io_dir.mkdir(parents=True, exist_ok=True)
        self.hook_dir = self.io_dir / "hooks"
        self.hook_dir.mkdir(exist_ok=True)
        # A neutral working directory: Claude Code derives its per-project storage path from the
        # cwd and may show that path to the model (e.g. where it saved a returned image). A cwd
        # under the run directory would carry the scenario id, seed and arm into the observation.
        self.cwd = Path(tempfile.mkdtemp(prefix="vista_cc_"))
        self._sleep = sleep
        self._base_env = dict(os.environ if base_env is None else base_env)
        self._rpc_log_default = getattr(rpc, "_log_path", None)
        self._seg: _Segment | None = None
        self._seg_lock = threading.RLock()
        self._img_lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self.segments: list[dict] = []
        self.preflight_result: dict | None = None
        self.models: set = set()
        self.claude_code_version: str | None = None
        self._started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._outcome: str | None = None
        self._final_text = ""
        self._error: str | None = None
        for path in (BRIDGE_SCRIPT, HOOK_SCRIPT):
            if not path.is_file():
                raise RunnerError(f"missing runtime file {path}")
        self.system_prompt_path = self.io_dir / "system_prompt.md"
        self.system_prompt_path.write_text(self.cfg.system_prompt or "", encoding="utf-8", newline="\n")

    # ── public API ───────────────────────────────────────────────────────────
    def preflight(self) -> dict:
        """Throw-away segment: same flags and tools, tool calls refused, reply must be READY."""
        seg = _Segment(index=None, stem="preflight", kind="preflight", session_id=str(uuid.uuid4()),
                       expected_tools=self._expected_tools())
        problems: list[str] = []
        try:
            self._run_segment(seg, PREFLIGHT_PROMPT, [])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{type(exc).__name__}: {exc}")
        reply = (seg.final_text or "").strip()
        problems += list(seg.problems)
        if seg.init is None:
            problems.append("no system/init envelope")
        if seg.end_reason != "preflight_done":
            problems.append(f"preflight segment ended with {seg.end_reason!r}")
        if reply != "READY":
            problems.append(f"reply {reply[:200]!r} is not 'READY'")
        if seg.tool_uses:
            problems.append(f"the model called tools during the preflight: {seg.tool_counts}")
        if not any(e.get("event", {}).get("hook_event_name") == "Stop" for e in seg.hook_events):
            problems.append("no Stop hook event reached the RpcServer (hooks inactive or failing)")
        init = init_summary(seg.init) or {}
        usage = result_usage(seg.last_result)
        result = {
            "ok": not problems, "problems": problems, "reply": reply,
            "model": init.get("model"), "models": sorted(seg.models),
            "claude_code_version": init.get("claude_code_version"),
            "tools": init.get("tools"), "mcp_servers": init.get("mcp_servers"),
            "permission_mode": init.get("permissionMode"), "skills": init.get("skills"),
            "plugins": init.get("plugins"), "slash_commands": init.get("slash_commands"),
            "session_id": seg.session_id, "cost_usd": usage.get("cost_usd") if usage else None,
            "usage": usage, "hook_events": len(seg.hook_events), "duration_s": seg.duration_s,
            "segment": seg.record(),
        }
        self.preflight_result = result
        self._write_manifest()
        return result

    def run(self) -> RunResult:
        outcome: str | None = None
        try:
            if self.cfg.compaction == "checkpoint" and \
                    f"mcp__{SERVER_NAME}__write_working" not in self._expected_tools():
                raise RunnerError("compaction='checkpoint' needs the write_working tool; "
                                  "use compaction='native' for arms without notes")
            if self.cfg.preflight:
                pf = self.preflight()
                if not pf["ok"]:
                    outcome = "error"
                    self._error = "preflight failed: " + "; ".join(pf["problems"])
            if outcome is None:
                outcome = self._episode()
        except Exception as exc:  # noqa: BLE001 — the episode must always be finished
            outcome = "error"
            self._error = f"{type(exc).__name__}: {exc}"
            (self.io_dir / "runner_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            self.rpc.disable_calls("the episode has finished; no further tool calls are executed")
            self.rpc.set_call_observer(None)
            self.rpc.set_hook_callback(None)
        if outcome not in OUTCOMES:
            self._error = (self._error or "") + f" (unknown outcome {outcome!r})"
            outcome = "error"
        self._outcome = outcome
        try:
            self.host.finish(outcome, self._final_text)
        finally:
            manifest = self._write_manifest()
        return RunResult(outcome=outcome, final_text=self._final_text, segments=list(self.segments),
                         usage=self._totals(), models=sorted(self.models),
                         preflight=self.preflight_result, manifest_path=str(manifest), error=self._error)

    # ── episode life cycle ───────────────────────────────────────────────────
    def _episode(self) -> str:
        text, images = self.host.initial_message()
        kind = "initial"
        recoveries = 0
        idle_checkpoints = 0
        while True:
            if len(self.segments) >= self.cfg.max_segments:
                return "max_segments"
            index = len(self.segments)
            seg = _Segment(index=index, stem=f"segment_{index:03d}", kind=kind,
                           session_id=str(uuid.uuid4()), expected_tools=self._expected_tools())
            acts_before = self._status().get("acts")
            self._run_segment(seg, text, images)
            if seg.final_text:
                self._final_text = seg.final_text
            status = self._status()
            reason = seg.end_reason
            if reason == "marker" and seg.marker in ("done", "abort"):
                return seg.marker
            if status.get("terminal") or reason == "terminal":
                stop = status.get("stop_reason")
                return stop if stop in OUTCOMES else "budget"
            if reason == "stalled":
                return "stalled"
            if reason == "usage_limit":
                self._error = f"usage_limit: {'; '.join(seg.problems) or reason}"
                return "error"
            if reason in ("init_invalid", "native_compact", "max_budget_usd"):
                self._error = f"segment {index} ended with {reason}: {'; '.join(seg.problems) or reason}"
                return "error"
            if reason == "image_limit":
                kind = "handoff"
            elif reason == "checkpoint":
                acts_after = status.get("acts")
                idle = isinstance(acts_before, int) and acts_after == acts_before
                idle_checkpoints = idle_checkpoints + 1 if idle else 0
                if idle_checkpoints >= MAX_CHECKPOINTS_WITHOUT_ACT:
                    self._error = (f"{idle_checkpoints} consecutive checkpoint handoffs without an executed act "
                                   "(does the recovery context already exceed the compaction threshold?)")
                    return "error"
                kind = "checkpoint"
            else:                                   # lease_expired, process_exit, api_error, ...
                recoveries += 1
                if recoveries > self.cfg.max_recoveries:
                    self._error = f"too many recoveries (last: segment {index} ended with {reason})"
                    return "error"
                if reason == "api_error" and self.cfg.api_retry_backoff_s:
                    self._sleep(float(self.cfg.api_retry_backoff_s[-1]))
                kind = "recovery"
            text, images = self.host.recovery_message()

    def _status(self) -> dict:
        try:
            status = self.host.status()
        except Exception as exc:  # noqa: BLE001
            return {"terminal": False, "checkpoint": "none", "status_error": f"{type(exc).__name__}: {exc}"}
        return status if isinstance(status, dict) else {}

    def _expected_tools(self) -> list[str]:
        return [f"mcp__{SERVER_NAME}__{tool['name']}" for tool in self.host.list_tools()]

    # ── one segment ──────────────────────────────────────────────────────────
    def _paths(self, stem: str) -> dict[str, Path]:
        return {name: self.io_dir / f"{stem}.{suffix}" for name, suffix in (
            ("command", "command.json"), ("settings", "settings.json"), ("mcp", "mcp.json"),
            ("input", "input.json"), ("stdout", "stdout.jsonl"), ("stderr", "stderr.txt"),
            ("bridge", "bridge.jsonl"), ("hooks", "hooks.jsonl"))}

    def _run_segment(self, seg: _Segment, text: str, images: Sequence[ImageBlock]) -> None:
        cfg = self.cfg
        paths = self._paths(seg.stem)
        rpc_host, rpc_port, token = self.rpc.host, self.rpc.port, self.rpc.token
        if not rpc_port:
            raise RunnerError("the RpcServer is not started")
        _atomic_json(paths["settings"], hook_settings(cfg))
        _atomic_json(paths["mcp"], mcp_config(cfg, rpc_host=rpc_host, rpc_port=rpc_port, rpc_token=token))
        cmd = build_command(cfg, claude=resolve_claude(cfg.claude_bin),
                            system_prompt_path=self.system_prompt_path, settings_path=paths["settings"],
                            mcp_path=paths["mcp"], allowed_tools=seg.expected_tools,
                            session_id=seg.session_id)
        _atomic_json(paths["command"], {"argv": cmd, "cwd": str(self.cwd)})
        env = child_environment(self._base_env, cfg, rpc_host=rpc_host, rpc_port=rpc_port,
                                rpc_token=token, hook_dir=self.hook_dir)
        for marker in (compact_hook.REQUEST_MARKER, compact_hook.READY_MARKER):
            (self.hook_dir / marker).unlink(missing_ok=True)
        with self._seg_lock:
            self._seg = seg
        self.rpc.new_generation()            # calls from earlier Claude Code processes are refused
        self.rpc.set_log_path(paths["bridge"])
        self.rpc.disable_calls("Tools are unavailable during the runtime preflight." if seg.preflight
                               else "runtime validation pending; nothing was executed")
        self.rpc.set_hook_callback(self._hook_callback)
        self.rpc.set_call_observer(None if seg.preflight else self._observe_call)
        listed_before = self.rpc.tools_listed
        threads: list[threading.Thread] = []
        try:
            creation = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            try:
                seg.proc = subprocess.Popen(cmd, cwd=str(self.cwd), env=env, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=creation)
            except OSError as exc:
                seg.problems.append(f"could not start Claude Code: {exc}")
                seg.end_reason = "startup_failed"
                return
            threads = [
                threading.Thread(target=self._read_stdout, args=(seg, seg.proc.stdout, paths["stdout"]),
                                 name=f"{seg.stem}-stdout", daemon=True),
                threading.Thread(target=self._read_stderr, args=(seg, seg.proc.stderr, paths["stderr"]),
                                 name=f"{seg.stem}-stderr", daemon=True)]
            for thread in threads:
                thread.start()
            if not self._wait_for_bridge(seg, listed_before):
                return
            if not seg.preflight:
                self.host.begin_segment(seg.index, seg.session_id)
                seg.begun = True
                if self._status().get("checkpoint") == "ready":
                    seg.checkpoint_armed = False
                    seg.problems.append("host checkpoint was still 'ready' after begin_segment")
            seg.images_sent = len(images)
            self._send_user(seg, text, images, paths["input"])
            if not seg.preflight and seg.kind != "initial":
                self.host.acknowledge_act()          # the recovery message shows the current state
            self._loop(seg, paths)
        finally:
            self._shutdown(seg, threads, paths)
            self.rpc.disable_calls(f"segment {seg.stem} has ended; nothing was executed")
            idle = self.rpc.wait_idle(max(float(cfg.turn_timeout_s), 1.0))
            if not idle:
                seg.problems.append("a tool call was still running when the segment ended")
            seg.ended_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            seg.duration_s = round(time.monotonic() - seg.started, 3)
            self._redact_mcp(paths["mcp"])
            self.models |= seg.models
            if seg.claude_code_version:
                self.claude_code_version = seg.claude_code_version
            record = seg.record()
            if not seg.preflight:
                self.segments.append(record)
                try:
                    if seg.begun:
                        self.host.end_segment(seg.index, {
                            "session_id": seg.session_id, "exit_code": seg.exit_code,
                            "boundary": seg.end_reason, "usage": record["usage"],
                            "cost_usd": record["cost_usd"], "num_turns": record["num_turns"],
                            "images_sent": seg.images_sent, "acts_acknowledged": seg.acts_acknowledged})
                finally:
                    self._write_manifest()
            with self._seg_lock:
                self._seg = None
            self.rpc.set_log_path(self._rpc_log_default)

    def _wait_for_bridge(self, seg: _Segment, listed_before: int) -> bool:
        """Wait until the bridge has listed the tools (Claude Code connected the MCP server)."""
        deadline = time.monotonic() + min(float(self.cfg.startup_timeout_s), 60.0)
        while time.monotonic() < deadline:
            if self.rpc.tools_listed > listed_before:
                return True
            if seg.proc is not None and seg.proc.poll() is not None:
                seg.problems.append(f"Claude Code exited during startup (code {seg.proc.returncode})")
                seg.end_reason = "startup_failed"
                return False
            time.sleep(0.05)
        seg.notes.append("the MCP bridge did not list tools before the first message; sending anyway")
        return True

    def _loop(self, seg: _Segment, paths: dict[str, Path]) -> None:
        closed = 0
        startup_deadline = time.monotonic() + float(self.cfg.startup_timeout_s)
        next_status = 0.0
        exited_at: float | None = None
        while True:
            try:
                kind, payload = seg.queue.get(timeout=POLL_S)
            except queue.Empty:
                kind, payload = "", None
            if kind == "msg":
                self._handle(seg, payload, paths)
            elif kind == "closed":
                closed += 1
            proc = seg.proc
            now = time.monotonic()
            if exited_at is None and proc is not None and proc.poll() is not None:
                exited_at = now
            # both readers at EOF, or the process is gone and a pipe is held open elsewhere
            if closed >= 2 or (exited_at is not None and now - exited_at > 10.0):
                if proc is not None and proc.poll() is None:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                if seg.end_reason is None:
                    code = proc.poll() if proc is not None else None
                    seg.problems.append(f"Claude Code exited unexpectedly (code {code})")
                    seg.end_reason = "process_exit" if seg.init is not None else "startup_failed"
                break
            if seg.end_reason is None:
                if seg.init is None and now > startup_deadline:
                    seg.problems.append("no system/init envelope within startup_timeout_s")
                    self._begin_end(seg, "startup_failed")
                elif not seg.preflight and now >= next_status:
                    next_status = now + STATUS_EVERY_S
                    self._periodic(seg, now)
            else:
                if seg.interrupt_sent and seg.turn_open and now > seg.interrupt_deadline:
                    seg.notes.append("no result after the interrupt; closing stdin")
                    seg.turn_open = False
                    self._close_stdin(seg)
                if proc is not None and proc.poll() is None and seg.kill_deadline and now > seg.kill_deadline:
                    seg.problems.append("Claude Code did not exit after stdin closed; killed")
                    seg.killed = True
                    kill_tree(proc)

    def _periodic(self, seg: _Segment, now: float) -> None:
        status = self._status()
        checkpoint = status.get("checkpoint", "none")
        if status.get("terminal"):
            self._begin_end(seg, "terminal")
            return
        if seg.image_handoff:
            self._begin_end(seg, "image_limit")
            return
        if checkpoint != "ready":
            seg.checkpoint_armed = True
        elif seg.checkpoint_armed:
            self._begin_end(seg, "checkpoint")
            return
        self._sync_markers(seg, checkpoint)
        completed, in_flight, last_completed = self.rpc.activity()
        activity = max(seg.last_activity, last_completed or 0.0)
        if in_flight == 0 and now - activity > float(self.cfg.turn_timeout_s):
            seg.problems.append(f"decision lease expired: no tool call or turn end for "
                                f"{self.cfg.turn_timeout_s:g} s")
            self._begin_end(seg, "lease_expired")

    def _sync_markers(self, seg: _Segment, checkpoint: str) -> None:
        """Mirror the host checkpoint state into the hook's fallback markers, and pick up a
        request the hook made through them when the RpcServer was unreachable."""
        request = self.hook_dir / compact_hook.REQUEST_MARKER
        ready = self.hook_dir / compact_hook.READY_MARKER
        if checkpoint == "none" and seg.markers == "none" and request.exists():
            seg.notes.append("checkpoint requested through the hook's marker file")
            self._request_checkpoint(seg, "context_limit")
            checkpoint = "requested"
        seg.markers = checkpoint
        try:
            if checkpoint in ("requested", "ready"):
                request.touch()
            else:
                request.unlink(missing_ok=True)
            if checkpoint == "ready":
                ready.touch()
            else:
                ready.unlink(missing_ok=True)
        except OSError:
            pass

    def _request_checkpoint(self, seg: _Segment, reason: str) -> None:
        self.host.request_checkpoint(reason)
        seg.checkpoint_requests.append({"reason": reason, "t": round(time.monotonic() - seg.started, 3),
                                        "images_sent": seg.images_sent})

    # ── stream handling ──────────────────────────────────────────────────────
    def _read_stdout(self, seg: _Segment, stream: Any, path: Path) -> None:
        act_names = {f"mcp__{SERVER_NAME}__{name}" for name in self.cfg.act_tools}
        with path.open("w", encoding="utf-8", newline="\n") as log:
            for raw in stream:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    log.write(json.dumps({"_unparsable": line[:4000]}, ensure_ascii=False) + "\n")
                    log.flush()
                    continue
                log.write(json.dumps(sanitize(message), ensure_ascii=False, separators=(",", ":")) + "\n")
                log.flush()
                if isinstance(message, dict):
                    if not seg.preflight:
                        self._fast_path(seg, message, act_names)
                    seg.queue.put(("msg", message))
        seg.queue.put(("closed", "stdout"))

    def _fast_path(self, seg: _Segment, message: dict, act_names: set) -> None:
        """Act bookkeeping on the reader thread, so the next act cannot overtake the ack."""
        kind = message.get("type")
        body = message.get("message") if isinstance(message.get("message"), dict) else {}
        blocks = body.get("content") if isinstance(body.get("content"), list) else []
        if kind == "assistant":
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_use" \
                        and block.get("name") in act_names and isinstance(block.get("id"), str):
                    seg.pending_acts.add(block["id"])
        elif kind == "user":
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "tool_result" \
                        and block.get("tool_use_id") in seg.pending_acts:
                    seg.pending_acts.discard(block["tool_use_id"])
                    if block.get("is_error"):
                        seg.act_errors += 1      # refused or failed; acknowledging is still correct
                    try:
                        self.host.acknowledge_act()
                        seg.acts_acknowledged += 1
                    except Exception as exc:  # noqa: BLE001
                        seg.problems.append(f"acknowledge_act raised {type(exc).__name__}: {exc}")

    def _read_stderr(self, seg: _Segment, stream: Any, path: Path) -> None:
        with path.open("wb") as log:
            for raw in stream:
                log.write(raw)
                log.flush()
                seg.stderr_tail = (seg.stderr_tail + raw)[-4096:]
        seg.queue.put(("closed", "stderr"))

    def _handle(self, seg: _Segment, message: dict, paths: dict[str, Path]) -> None:
        kind = message.get("type")
        subtype = message.get("subtype")
        if kind == "system":
            if subtype == "init":
                self._on_init(seg, message)
            elif subtype == "compact_boundary":
                seg.compact_boundaries.append({k: message.get(k) for k in ("uuid", "compact_metadata")})
                if self.cfg.compaction == "native":
                    with self._img_lock:
                        seg.images_sent = 0          # the summary replaced every earlier image
                    seg.notes.append("native compaction (expected in native mode)")
                else:
                    seg.problems.append("native compaction happened (compact_boundary) before the "
                                        "checkpoint handoff")
                    if seg.end_reason is None:
                        self._begin_end(seg, "native_compact")
            elif subtype == "status":
                if message.get("status") == "compacting":
                    seg.compaction_attempts += 1
            elif subtype == "api_retry":
                seg.api_retries.append({k: message.get(k) for k in
                                        ("attempt", "max_retries", "retry_delay_ms", "error_status", "error")})
            elif subtype == "hook_response" and message.get("hook_event") == "Stop":
                if '"block"' in str(message.get("output") or ""):
                    seg.stop_hook_blocks += 1
            elif subtype == "permission_denied":
                seg.permission_denials.append({k: v for k, v in message.items() if k not in ("uuid",)})
        elif kind == "assistant":
            body = message.get("message") if isinstance(message.get("message"), dict) else {}
            model = body.get("model")
            if isinstance(model, str) and model and not model.startswith("<"):
                seg.models.add(model)
            for block in body.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    name = _short(block.get("name", ""))
                    seg.tool_uses += 1
                    seg.tool_counts[name] = seg.tool_counts.get(name, 0) + 1
                    if name in self.cfg.progress_tools:
                        seg.turn_progress = True
                elif block.get("type") == "text" and isinstance(block.get("text"), str):
                    seg.last_text = block["text"]
        elif kind == "user":
            body = message.get("message") if isinstance(message.get("message"), dict) else {}
            for block in body.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_result" \
                        and isinstance(block.get("content"), list):
                    seg.images_seen += sum(1 for b in block["content"]
                                           if isinstance(b, dict) and b.get("type") == "image")
        elif kind == "result":
            self._on_result(seg, message, paths)
        elif kind == "control_request":
            self._write_stdin(seg, {"type": "control_response", "response": {
                "subtype": "error", "request_id": message.get("request_id"),
                "error": "control requests are not supported by the VISTA runner"}})
            seg.notes.append(f"refused control_request {message.get('request', {}).get('subtype')!r}")
        elif kind == "rate_limit_event":
            info = message.get("rate_limit_info")
            if isinstance(info, dict):
                seg.rate_limit = info

    def _on_init(self, seg: _Segment, message: dict) -> None:
        seg.init_count += 1
        first = seg.init is None
        if first:
            seg.init = message
            version = message.get("claude_code_version")
            seg.claude_code_version = version if isinstance(version, str) else None
        model = message.get("model")
        if isinstance(model, str) and model:
            seg.models.add(model)
        problems = validate_init(message, seg.expected_tools, expected_model=self.cfg.expected_model)
        billing = auth_problem(message, self.cfg.auth)
        if billing:
            problems.append(billing)
        if first and message.get("session_id") not in (None, seg.session_id):
            problems.append(f"session id {message.get('session_id')!r} != requested {seg.session_id!r}")
        if problems:
            seg.problems.extend(f"init #{seg.init_count}: {p}" for p in problems)
            if seg.end_reason is None:
                self._begin_end(seg, "init_invalid")
            return
        if first and not seg.preflight and seg.end_reason is None:
            self.rpc.enable_calls()

    def _on_result(self, seg: _Segment, message: dict, paths: dict[str, Path]) -> None:
        seg.turn_open = False
        seg.last_activity = time.monotonic()
        seg.last_result = message
        text = message.get("result") if isinstance(message.get("result"), str) else ""
        seg.results.append({k: message.get(k) for k in (
            "subtype", "is_error", "num_turns", "total_cost_usd", "terminal_reason", "result_index",
            "stop_reason", "api_error_status", "duration_ms")} | {"result_chars": len(text)})
        denials = message.get("permission_denials")
        if isinstance(denials, list) and denials:
            seg.permission_denials.extend(denials)
        if text:
            seg.final_text = text
        elif seg.last_text and classify_result(message) == "ok":
            seg.final_text = seg.last_text
        if seg.end_reason is not None:
            self._close_stdin(seg)
            return
        if seg.preflight:
            self._begin_end(seg, "preflight_done")
            return
        outcome, note = marker_outcome(text)
        if outcome in ("done", "abort"):              # the agent's own decision wins over a budget stop
            seg.marker = outcome
            self._begin_end(seg, "marker")
            return
        status = self._status()
        if status.get("terminal"):
            self._begin_end(seg, "terminal")
            return
        if note:
            seg.notes.append(f"turn {len(seg.results)}: {note}")
        checkpoint = status.get("checkpoint", "none")
        if checkpoint == "ready" and seg.checkpoint_armed:
            self._begin_end(seg, "checkpoint")
            return
        verdict = classify_result(message)
        if verdict == "budget_usd":
            seg.problems.append("Claude Code reached --max-budget-usd")
            self._begin_end(seg, "max_budget_usd")
            return
        if verdict == "usage_limit":
            seg.problems.append(f"usage limit: {usage_limit_text(message)}")
            self._begin_end(seg, "usage_limit")
            return
        if verdict == "retryable":
            seg.api_failures += 1
            if seg.api_failures > len(self.cfg.api_retry_backoff_s):
                seg.problems.append(f"API errors persisted after {seg.api_failures - 1} retries")
                self._begin_end(seg, "api_error")
                return
            delay = float(self.cfg.api_retry_backoff_s[seg.api_failures - 1])
            seg.notes.append(f"retryable API error (status {message.get('api_error_status')}); "
                             f"waiting {delay:g} s")
            self._sleep(delay)
            self._send_user(seg, self.cfg.continue_text, (), paths["input"])
            return
        if verdict == "error":
            seg.problems.append(f"turn ended with {message.get('subtype')!r} "
                                f"({message.get('terminal_reason')!r}): {str(message.get('errors') or text)[:300]}")
            self._begin_end(seg, "turn_error")
            return
        seg.api_failures = 0
        seg.stall = 0 if seg.turn_progress else seg.stall + 1
        if seg.stall >= int(self.cfg.stall_turns):
            seg.notes.append(f"{seg.stall} consecutive turns without act or report")
            self._begin_end(seg, "stalled")
            return
        follow_up = self.cfg.continue_text
        if checkpoint == "requested":
            follow_up = f"{follow_up}\n\n{CHECKPOINT_INSTRUCTION}"
        self._send_user(seg, follow_up, (), paths["input"])

    # ── RPC callbacks (RpcServer threads) ────────────────────────────────────
    def _observe_call(self, name: str, args: dict, reply: ToolReply) -> ToolReply | None:
        with self._seg_lock:
            seg = self._seg
        if seg is None or seg.preflight or not reply.images:
            return None
        with self._img_lock:
            seg.images_sent += len(reply.images)
            total = seg.images_sent
        threshold = int(self.cfg.image_limit_per_session) - int(self.cfg.image_reserve)
        if total < threshold:
            return None
        status = self._status()
        if self.cfg.compaction == "native":
            if not seg.image_handoff and not status.get("terminal"):
                seg.image_handoff = True
                seg.notes.append(f"image limit reached ({total} images); handing off to a fresh session")
            return None
        if status.get("terminal") or status.get("checkpoint", "none") != "none":
            return None
        try:
            self._request_checkpoint(seg, "image_limit")
        except Exception as exc:  # noqa: BLE001
            seg.problems.append(f"request_checkpoint raised {type(exc).__name__}: {exc}")
            return None
        return replace(reply, text=f"{reply.text}\n\n{CHECKPOINT_INSTRUCTION}")

    def _hook_callback(self, event: dict) -> dict:
        with self._seg_lock:
            seg = self._seg
        name = event.get("hook_event_name")
        record: dict[str, Any] = {"t": round(time.time(), 3), "event": event}
        stale = seg is not None and event.get("session_id") not in (None, seg.session_id)
        if seg is None or seg.preflight or stale:
            output: dict = {}
            record["note"] = ("stale session" if stale else "preflight" if seg is not None
                              else "no active segment")
        else:
            status = self._status()
            checkpoint = status.get("checkpoint", "none")
            output, effect = compact_hook.decide(name, checkpoint, terminal=bool(status.get("terminal")),
                                                 native=self.cfg.compaction == "native")
            record.update({"checkpoint": checkpoint, "effect": effect})
            if effect == "request_checkpoint":
                try:
                    self._request_checkpoint(seg, "context_limit")
                except Exception as exc:  # noqa: BLE001
                    record["error"] = f"{type(exc).__name__}: {exc}"
        record["output"] = output
        if seg is not None:
            seg.hook_events.append(record)
            try:
                with (self.io_dir / f"{seg.stem}.hooks.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
        return output

    # ── process I/O ──────────────────────────────────────────────────────────
    def _send_user(self, seg: _Segment, text: str, images: Sequence[ImageBlock], input_path: Path) -> None:
        message = user_message(text, images)
        seg.inputs.append(sanitize(message))
        _atomic_json(input_path, seg.inputs)
        if self._write_stdin(seg, message):
            seg.turn_open = True
            seg.turn_progress = False
            seg.messages_sent += 1
            seg.images_input += len(images)
            seg.last_activity = time.monotonic()

    def _write_stdin(self, seg: _Segment, message: dict) -> bool:
        proc = seg.proc
        if proc is None or proc.stdin is None or seg.stdin_closed:
            return False
        data = (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self._stdin_lock:
            try:
                proc.stdin.write(data)
                proc.stdin.flush()
                return True
            except (OSError, ValueError) as exc:
                seg.notes.append(f"stdin write failed: {type(exc).__name__}: {exc}")
                return False

    def _close_stdin(self, seg: _Segment) -> None:
        if seg.stdin_closed or seg.proc is None or seg.proc.stdin is None:
            return
        seg.stdin_closed = True
        with self._stdin_lock:
            try:
                seg.proc.stdin.close()
            except (OSError, ValueError):
                pass
        seg.kill_deadline = time.monotonic() + float(self.cfg.end_grace_s)

    def _begin_end(self, seg: _Segment, reason: str) -> None:
        if seg.end_reason is not None:
            return
        seg.end_reason = reason
        self.rpc.disable_calls(f"segment {seg.stem} is ending ({reason}); nothing was executed")
        proc = seg.proc
        if proc is not None and proc.poll() is None and seg.turn_open and not seg.stdin_closed:
            seg.interrupt_sent = self._write_stdin(seg, {
                "type": "control_request", "request_id": f"vista_interrupt_{uuid.uuid4().hex[:8]}",
                "request": {"subtype": "interrupt"}})
            seg.interrupt_deadline = time.monotonic() + INTERRUPT_WAIT_S
            seg.kill_deadline = seg.interrupt_deadline + float(self.cfg.end_grace_s)
            if not seg.interrupt_sent:
                self._close_stdin(seg)
        else:
            self._close_stdin(seg)

    def _shutdown(self, seg: _Segment, threads: list[threading.Thread], paths: dict[str, Path]) -> None:
        proc = seg.proc
        if proc is None:
            return
        if proc.poll() is None:
            self._close_stdin(seg)
            try:
                proc.wait(timeout=max(1.0, float(self.cfg.end_grace_s)))
            except subprocess.TimeoutExpired:
                seg.problems.append("Claude Code still running at segment shutdown; killed")
                seg.killed = True
                kill_tree(proc)
        for thread in threads:
            thread.join(timeout=10)
        seg.exit_code = proc.poll()
        # drain what the readers queued after the loop ended (a late result carries usage)
        if seg.end_reason is None:
            seg.end_reason = "process_exit"
        while True:
            try:
                kind, payload = seg.queue.get_nowait()
            except queue.Empty:
                break
            if kind == "msg" and isinstance(payload, dict):
                self._handle(seg, payload, paths)

    def _redact_mcp(self, path: Path) -> None:
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            for server in config.get("mcpServers", {}).values():
                if isinstance(server.get("env"), dict) and "VISTA_RPC_TOKEN" in server["env"]:
                    server["env"]["VISTA_RPC_TOKEN"] = "<redacted>"
            _atomic_json(path, config)
        except (OSError, ValueError, AttributeError):
            pass

    # ── manifest ─────────────────────────────────────────────────────────────
    def _totals(self) -> dict:
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": 0, "cost_usd": 0.0, "num_turns": 0, "segments": len(self.segments),
                  "compact_boundaries": 0, "usage_complete": True}
        for record in self.segments:
            usage = record.get("usage") or {}
            for key in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
                totals[key] += int(usage.get(key) or 0)
            totals["cost_usd"] += float(record.get("cost_usd") or 0.0)
            totals["num_turns"] += int(record.get("num_turns") or 0)
            totals["compact_boundaries"] += int(record.get("compact_boundary_count") or 0)
            if not record.get("usage_complete"):
                totals["usage_complete"] = False
        totals["cost_usd"] = round(totals["cost_usd"], 6)
        if self.preflight_result is not None:
            totals["preflight_cost_usd"] = self.preflight_result.get("cost_usd")
        return totals

    def _write_manifest(self) -> Path:
        path = self.io_dir / "manifest.json"
        _atomic_json(path, {
            "manifest_version": 1, "started_at": self._started_wall,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "claude_code_version": self.claude_code_version, "models": sorted(self.models),
            "config": self.cfg.summary(), "preflight": self.preflight_result,
            "segments": self.segments, "totals": self._totals(), "outcome": self._outcome,
            "final_text": self._final_text, "error": self._error})
        return path


def image_digest(png: bytes) -> dict:
    """``{"sha256", "bytes"}`` of an image, for logs."""
    return {"sha256": hashlib.sha256(png).hexdigest(), "bytes": len(png)}


def decode_image_block(block: Mapping[str, Any]) -> bytes:
    """PNG bytes of a stream-json/MCP image block (helper for tests and log readers)."""
    data = block.get("data")
    if data is None and isinstance(block.get("source"), Mapping):
        data = block["source"].get("data")
    return base64.b64decode(data or b"")
