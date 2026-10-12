"""Answering STM-BabyVision items with Claude Code through WP-B's runner.

README
------
::

    python -m vista_spm.staticbench qa --driver claude --dataset tmp/staticbench/v1 \
        --out tmp/staticbench/pilot1 --pilot --tools full --model claude-opus-5-5 --effort high

For every selected item: a fresh :class:`.qa_host.QAHost` (its own work dir), an
``rpc.RpcServer`` on it, and a ``claude_runner.ClaudeCodeRunner`` configured with
:func:`.qa_host.runner_settings`. The runner ends as soon as the host turns terminal after a
valid ``answer`` (outcome ``done``). Each item's ``host.result()`` plus the runner's usage is
appended to ``<out>/<tools>/answers.jsonl`` the moment it exists, so a crash keeps what was
answered and a rerun skips it. ``--tools full`` offers inspect, read_values and the measurement
tools; ``--tools image`` offers only ``answer`` (the images in the first message are all the
model gets).

Spend is bounded twice: no new item starts once the dollars recorded under ``<out>`` (every
arm, preflights included) reach ``budget_usd``, and each Claude Code process gets
``--max-budget-usd`` = the smaller of ``segment_cap_usd`` and what is left of the budget divided by
the runner's segment limit. ``score_out`` scores every arm under ``<out>`` with
:mod:`.score` and writes ``scores.jsonl`` per arm and ``report/`` (with the classical baseline
restricted to the same items when the dataset has ``answers_baseline.jsonl``).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from . import items as I
from .qa_host import QAHost, runner_settings

#: ``--tools`` value → the ablation arm the QA host is built with
ARM_FOR_TOOLS = {"full": "full", "image": "img"}
ANSWERS_NAME = "answers.jsonl"


def pilot_ids(items: list[dict], per_subtype: int = 1) -> list[str]:
    """``per_subtype`` items of every subtype, in the report's type order: the first ones in id
    order whose difficulty is ``normal`` (hard ones only when a subtype has nothing else)."""
    order = {t: k for k, t in enumerate(I.ITEM_TYPES)}
    groups: dict[str, list[dict]] = {}
    for it in sorted(items, key=lambda x: (order.get(x["type"], 99), x["id"])):
        groups.setdefault(f"{it['type']}.{it['subtype']}", []).append(it)
    out = []
    for key in sorted(groups, key=lambda k: (order.get(k.split(".")[0], 99), k)):
        g = groups[key]
        pick = [it for it in g if it.get("difficulty", "normal") == "normal"][:per_subtype]
        if len(pick) < per_subtype:
            pick += [it for it in g if it not in pick][: per_subtype - len(pick)]
        out += [it["id"] for it in pick]
    return out


def spent_usd(out_dir: str | Path) -> float:
    """Dollars spent under ``out_dir``, preflights included: the sum over every Claude Code
    manifest (sessions that were cut off count too), or over the answers files if that is larger."""
    out_dir = Path(out_dir)
    from_manifests = 0.0
    for path in out_dir.glob("*/work/*/session*/claude_io/manifest.json"):
        try:
            totals = json.loads(path.read_text(encoding="utf-8")).get("totals") or {}
        except (OSError, ValueError):
            continue
        from_manifests += float(totals.get("cost_usd") or 0.0) + float(totals.get("preflight_cost_usd") or 0.0)
    from_answers = 0.0
    for path in out_dir.glob(f"*/{ANSWERS_NAME}"):
        for row in I.read_jsonl(path):
            usage = (row.get("driver") or {}).get("usage") or {}
            from_answers += float(usage.get("cost_usd") or 0.0) + float(usage.get("preflight_cost_usd") or 0.0)
    return max(from_manifests, from_answers)


def api_key_problem(segments: list[dict], preflight: dict | None = None) -> str | None:
    """Why a session ran (or was refused) on an API key instead of the subscription, or ``None``:
    an init whose ``apiKeySource`` is not ``"none"``, or the runner's refusal of one (its
    ``auth_problem`` text names ``apiKeySource``)."""
    for s in segments:
        src = (s.get("init") or {}).get("apiKeySource")
        if isinstance(src, str) and src not in ("", "none"):
            return f"apiKeySource {src!r}"
        for p in s.get("problems") or []:
            if "apiKeySource" in str(p):
                return str(p)
    for p in (preflight or {}).get("problems") or []:
        if "apiKeySource" in str(p):
            return str(p)
    return None


def api_key_sources(segments: list[dict]) -> list[str]:
    """Every ``apiKeySource`` the sessions' init envelopes reported (for the audit trail)."""
    return sorted({str((s.get("init") or {}).get("apiKeySource")) for s in segments
                   if (s.get("init") or {}).get("apiKeySource") is not None})


def infra_error(segments: list[dict], answered: bool, *, preflight: dict | None = None,
                runner_failed: bool = False, runner_error: str | None = None) -> str | None:
    """Why an unanswered item was never really attempted, or ``None``: ``"usage_limit"`` (the
    runner's ``end_reason``/error for a subscription usage limit), ``"api_<status>"`` when its
    session ended on other API errors (a rate limit, an overload), ``"preflight_failed"``,
    ``"runner_failed"``. Such an item is retried on the next run, is left out of the scores, and
    stops the current run (every later item would meet the same error)."""
    if answered:
        return None
    if api_key_problem(segments, preflight):
        return "auth_api_key"
    pre_problems = " ".join(str(p) for p in ((preflight or {}).get("problems") or [])).lower()
    if (any(s.get("end_reason") == "usage_limit" for s in segments)
            or str(runner_error or "").startswith("usage_limit") or "usage limit" in pre_problems):
        return "usage_limit"
    statuses = [r.get("api_error_status") for s in segments for r in (s.get("results") or [])
                if r.get("api_error_status")]
    if statuses or any(s.get("end_reason") == "api_error" for s in segments):
        return f"api_{statuses[-1] if statuses else 'error'}"
    if preflight is not None and not preflight.get("ok", True):
        return "preflight_failed"
    if runner_failed and not segments:
        return "runner_failed"
    return None


def run_claude_items(dataset: str | Path, out_dir: str | Path, *, ids: list[str], tools: str = "full",
                     model: str = "claude-opus-5-5", effort: str = "high", budget_usd: float = 15.0,
                     segment_cap_usd: float = 2.0, max_tool_calls: int = 60, preflight: str = "first",
                     retry_unanswered: bool = False, runner_overrides: dict | None = None,
                     log: Callable[[str], None] | None = None) -> dict:
    """Run ``ids`` through Claude Code with the ``tools`` arm; returns a summary of this call.

    ``preflight``: ``"first"`` runs the runner's READY preflight before the first item this call
    starts, ``"all"`` before every item, ``"none"`` never. Items already in the answers file are
    skipped; with ``retry_unanswered`` those whose recorded answer is null run again (the newer
    line wins when scoring). ``runner_overrides`` are passed to ``RunnerConfig`` last (tests point
    ``claude_bin`` at a fake)."""
    from ..claude_runner import ClaudeCodeRunner, RunnerConfig
    from ..rpc import RpcServer

    if tools not in ARM_FOR_TOOLS:
        raise ValueError(f"tools must be one of {sorted(ARM_FOR_TOOLS)}, got {tools!r}")
    log = log or (lambda s: print(s, flush=True))
    # absolute: Claude Code runs in its own cwd (<io_dir>/cwd), so every path the runner writes
    # into its command line (settings, MCP config, system prompt) must not be relative
    dataset, out_dir = Path(dataset).resolve(), Path(out_dir).resolve()
    arm_dir = out_dir / tools
    arm_dir.mkdir(parents=True, exist_ok=True)
    answers_path = arm_dir / ANSWERS_NAME
    items = {it["id"]: it for it in I.load_items(dataset)}
    unknown = [i for i in ids if i not in items]
    if unknown:
        raise KeyError(f"item ids not in {dataset}: {unknown}")
    recorded = I.load_answers(answers_path) if answers_path.exists() else {}
    infra = {}
    if answers_path.exists():
        for row in I.read_jsonl(answers_path):          # later lines win, as in load_answers
            infra[str(row.get("id"))] = (row.get("driver") or {}).get("infra_error")
    done = {i for i, a in recorded.items()
            if a is not None or (not retry_unanswered and not infra.get(i))}
    summary: dict[str, Any] = {"tools": tools, "arm": ARM_FOR_TOOLS[tools], "model": model, "effort": effort,
                               "answers": str(answers_path), "run": [], "skipped_done": sorted(done & set(ids)),
                               "stopped_for_budget": False, "stopped_for_api_error": None, "stopped_for_auth": None,
                               "spent_before_usd": round(spent_usd(out_dir), 6)}
    first = True
    cfg_overrides = dict(runner_overrides or {})
    for iid in ids:
        if iid in done:
            continue
        spent = spent_usd(out_dir)
        if spent >= budget_usd:
            summary["stopped_for_budget"] = True
            log(f"budget reached: ${spent:.2f} >= ${budget_usd:.2f}; not starting {iid}")
            break
        item = items[iid]
        base = arm_dir / "work" / iid
        n_prev = sum(1 for p in base.iterdir() if p.is_dir()) if base.is_dir() else 0
        work = base / f"session{n_prev}"
        host = QAHost(item, dataset, work / "host", arm=ARM_FOR_TOOLS[tools], max_tool_calls=max_tool_calls)
        settings = runner_settings(images_only=host.images_only, model=model, effort=effort,
                                   io_dir=work / "claude_io",
                                   preflight=(preflight == "all") or (preflight == "first" and first))
        cap = min(float(segment_cap_usd), max(0.05, (budget_usd - spent) / max(1, int(settings["max_segments"]))))
        settings["max_budget_usd"] = round(cap, 2)
        settings.update(cfg_overrides)
        fields = {f for f in RunnerConfig.__dataclass_fields__}
        cfg = RunnerConfig(**{k: v for k, v in settings.items() if k in fields})
        rpc = RpcServer(host, log_path=work / "claude_io" / "rpc.jsonl")
        rpc.start()
        t0 = time.perf_counter()
        error = None
        try:
            rr = ClaudeCodeRunner(host, rpc, cfg).run()
        except Exception as exc:  # noqa: BLE001 — one failed session must not end the run
            rr = None
            error = f"{type(exc).__name__}: {exc}"
            host.finish("error", error)
        finally:
            rpc.stop()
        wall = time.perf_counter() - t0
        first = False
        record = host.result()
        segments = rr.segments if rr else []
        record["driver"] = {
            "name": "claude_code", "model": model, "effort": effort, "tools": tools,
            "outcome": rr.outcome if rr else "error", "error": rr.error if rr else error,
            "infra_error": infra_error(segments, record["answer"] is not None,
                                       preflight=rr.preflight if rr else None,
                                       runner_failed=rr is None or rr.outcome == "error",
                                       runner_error=rr.error if rr else error),
            "api_key_sources": api_key_sources(segments),
            "api_key_problem": api_key_problem(segments, rr.preflight if rr else None),
            "usage": rr.usage if rr else {}, "models": rr.models if rr else [],
            "n_segments": len(segments),
            "preflight_ok": (rr.preflight or {}).get("ok") if rr and rr.preflight else None,
            "max_budget_usd_per_segment": cfg.max_budget_usd, "wall_s": round(wall, 2),
            "turns": sum(int(s.get("turns") or 0) for s in segments),
            "tool_counts": _merge_counts(segments),
            "io_dir": str((work / "claude_io").relative_to(out_dir)).replace("\\", "/")}
        with open(answers_path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        cost = float((record["driver"]["usage"] or {}).get("cost_usd") or 0.0)
        summary["run"].append({"id": iid, "outcome": record["driver"]["outcome"], "answer": record["answer"],
                               "cost_usd": cost, "wall_s": round(wall, 1), "tool_calls": record["tool_calls"],
                               "infra_error": record["driver"]["infra_error"]})
        log(f"[{tools}] {iid}: {record['driver']['outcome']} answer={json.dumps(record['answer'])[:120]} "
            f"tools={record['tool_calls']} ${cost:.3f} {wall:.0f}s")
        if record["driver"]["api_key_problem"]:
            # subscription only: a session that ran, or tried to run, on an API key ends the run
            summary["stopped_for_auth"] = record["driver"]["api_key_problem"]
            log(f"aborting: {iid} reported {record['driver']['api_key_problem']}")
            break
        if record["driver"]["infra_error"]:
            summary["stopped_for_api_error"] = record["driver"]["infra_error"]
            log(f"stopping: {iid} met {record['driver']['infra_error']} (it will be retried on the next run)")
            break
    summary["spent_after_usd"] = round(spent_usd(out_dir), 6)
    return summary


def _merge_counts(segments: list[dict]) -> dict:
    out: dict[str, int] = {}
    for s in segments:
        for k, v in (s.get("tool_counts") or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def score_out(dataset: str | Path, out_dir: str | Path, *, title: str = "STM-BabyVision: Claude pilot") -> dict:
    """Score every arm under ``out_dir`` on the items it attempted; write ``<arm>/scores.jsonl`` and
    ``<out>/report/`` (with the classical baseline on the same items when available). An item whose
    latest record is an infrastructure failure (``driver.infra_error``) was never really attempted:
    it is left out and listed under ``not_attempted``."""
    from .report import write_report
    from .score import aggregate, score_all

    dataset, out_dir = Path(dataset), Path(out_dir)
    items = {it["id"]: it for it in I.load_items(dataset)}
    runs: dict[str, list[dict]] = {}
    attempted: set[str] = set()
    not_attempted: dict[str, list[str]] = {}
    for answers_path in sorted(out_dir.glob(f"*/{ANSWERS_NAME}")):
        answers = I.load_answers(answers_path)
        last = {str(r.get("id")): r for r in I.read_jsonl(answers_path)}
        failed = sorted(i for i, r in last.items() if (r.get("driver") or {}).get("infra_error"))
        ids = [i for i in answers if i in items and i not in failed]
        attempted |= set(ids)
        name = f"claude-{answers_path.parent.name}"
        not_attempted[name] = failed
        recs = score_all([items[i] for i in ids], answers)
        I.write_jsonl(answers_path.with_name("scores.jsonl"), recs)
        runs[name] = recs
    base_path = dataset / "answers_baseline.jsonl"
    if base_path.exists() and attempted:
        base = I.load_answers(base_path)
        ids = sorted(attempted)
        runs["baseline (classical)"] = score_all([items[i] for i in ids], {i: base.get(i) for i in ids})
    res = write_report(runs, out_dir / "report", title=title, dataset=str(dataset)) if runs else {}
    return {"report": res, "overall": {name: aggregate(recs)["overall"] for name, recs in runs.items()},
            "not_attempted": not_attempted}
