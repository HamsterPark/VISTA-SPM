"""CLI: ``python -m vista_spm.staticbench <command> ...``

generate  build a dataset            generate --out tmp/staticbench/v1 --types tip,adatoms,move,drift,herringbone --seeds 0-9 [--jobs 8] [--overlays]
validate  check items and files      validate tmp/staticbench/v1
answer    answer every item          answer tmp/staticbench/v1 --mode baseline [--out answers.jsonl]   (modes: baseline, oracle, abstain)
score     score an answers file      score tmp/staticbench/v1 answers.jsonl [--out scores.jsonl]
report    CSV + Markdown tables      report tmp/staticbench/v1 --run baseline=answers.jsonl [--run claude=...] --out <dir>
qa        the QA host                qa --dataset tmp/staticbench/v1 --item <id> [--arm full]            (dry run: first message + tools)
                                     qa --driver claude --dataset tmp/staticbench/v1 --out <dir> (--items id,id | --pilot)
                                        [--tools full|image] [--model claude-opus-5-5] [--effort high] [--budget-usd 15]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import items as I


def _cmd_generate(a) -> int:
    from .generate import generate, parse_seeds, parse_types

    m = generate(a.out, parse_types(a.types), parse_seeds(a.seeds), jobs=a.jobs, keep_work=a.keep_work,
                 overlays=a.overlays)
    print(json.dumps({k: m[k] for k in ("n_items", "items_per_subtype", "class_balance", "difficulty",
                                         "n_frames", "wall_s")}, indent=1))
    if m["failures"]:
        print(f"{len(m['failures'])} recipe(s) failed:", file=sys.stderr)
        for f in m["failures"]:
            print(f"  {f['recipe']}: {str(f['error']).splitlines()[0]}", file=sys.stderr)
        return 2
    return 0


def _cmd_validate(a) -> int:
    items = I.load_items(a.dataset)
    bad = 0
    ids = set()
    for it in items:
        errs = I.validate_item(it, root=a.dataset)
        if it.get("id") in ids:
            errs.append("duplicate id")
        ids.add(it.get("id"))
        if errs:
            bad += 1
            print(f"{it.get('id')}: {errs}")
    print(f"{len(items)} items, {bad} invalid")
    return 1 if bad else 0


def _oracle(it: dict) -> dict:
    t = it["truth"]
    return {"value": t["value"], "abstain": False}


def _cmd_answer(a) -> int:
    items = I.load_items(a.dataset)
    types = [t for t in (a.types or "").split(",") if t] or None
    out = Path(a.out) if a.out else Path(a.dataset) / f"answers_{a.mode}.jsonl"
    if a.mode == "baseline":
        from .baseline import run_baseline

        rows = run_baseline(a.dataset, out, types=types)
    else:
        rows = []
        for it in items:
            if types and it["type"] not in types:
                continue
            ans = _oracle(it) if a.mode == "oracle" else {"abstain": True}
            rows.append({"id": it["id"], "answer": ans})
        I.write_jsonl(out, rows)
    n_abs = sum(1 for r in rows if (r.get("answer") or {}).get("abstain"))
    print(f"wrote {len(rows)} answers ({n_abs} abstained) to {out}")
    return 0


def _score(dataset: str, answers_path: str) -> list[dict]:
    from .score import score_all

    items = I.load_items(dataset)
    answers = I.load_answers(answers_path)
    unknown = sorted(set(answers) - {it["id"] for it in items})
    if unknown:
        print(f"warning: {len(unknown)} answer id(s) not in the dataset, ignored: {unknown[:5]}", file=sys.stderr)
    return score_all(items, answers)


def _cmd_score(a) -> int:
    from .score import aggregate

    recs = _score(a.dataset, a.answers)
    out = Path(a.out) if a.out else Path(a.answers).with_name(Path(a.answers).stem + "_scores.jsonl")
    I.write_jsonl(out, recs)
    agg = aggregate(recs)
    print(json.dumps(agg["overall"], indent=1))
    for row in agg["subtypes"]:
        e = (row.get("error") or {}).get("median")
        print(f"  {row['subtype']:<26} n={row['n']:<4} acc={row['accuracy']:.3f}"
              f"{'' if e is None else f'  err_median={e:.3f}'}")
    print(f"scores -> {out}")
    return 0


def _cmd_report(a) -> int:
    from .report import write_report

    runs = {}
    for spec in a.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects name=answers.jsonl, got {spec!r}")
        name, path = spec.split("=", 1)
        runs[name] = _score(a.dataset, path)
    res = write_report(runs, a.out, dataset=str(a.dataset), title=a.title)
    print(json.dumps(res, indent=1))
    return 0


def _cmd_qa(a) -> int:
    dataset = a.dataset or a.dataset_pos
    if not dataset:
        raise SystemExit("qa needs --dataset <dir>")
    if a.driver == "claude":
        return _qa_claude(a, dataset)
    from .qa_host import QAHost

    if not a.item:
        raise SystemExit("the dry run needs --item <id>")
    items = {it["id"]: it for it in I.load_items(dataset)}
    if a.item not in items:
        raise SystemExit(f"no item {a.item!r} in {dataset}")
    import time as _time

    work = Path(a.work) if a.work else Path(dataset) / "_qa" / a.item / _time.strftime("%Y%m%dT%H%M%S")
    host = QAHost(items[a.item], dataset, work, arm=a.arm)
    text, images = host.initial_message()
    print(text)
    print(f"\n[{len(images)} image(s): " + ", ".join(f"{im.label} {len(im.png)} B" for im in images) + "]")
    print("tools:", [t["name"] for t in host.list_tools()])
    return 0


def _qa_claude(a, dataset: str) -> int:
    from .claude_qa import pilot_ids, run_claude_items, score_out

    if not a.out:
        raise SystemExit("--driver claude needs --out <dir>")
    items = I.load_items(dataset)
    if a.pilot:
        ids = pilot_ids(items)
    elif a.items:
        ids = [s.strip() for s in a.items.split(",") if s.strip()]
    elif a.difficulty:
        order = {t: k for k, t in enumerate(I.ITEM_TYPES)}
        ids = [it["id"] for it in sorted(items, key=lambda x: (order.get(x["type"], 99), x["id"]))
               if it.get("difficulty", "normal") == a.difficulty]
    else:
        raise SystemExit("choose the items: --items id,id, --pilot or --difficulty <level>")
    summary = run_claude_items(dataset, a.out, ids=ids, tools=a.tools, model=a.model, effort=a.effort,
                               budget_usd=a.budget_usd, segment_cap_usd=a.segment_cap_usd,
                               max_tool_calls=a.max_tool_calls, preflight=a.preflight,
                               retry_unanswered=a.retry_unanswered)
    scored = score_out(dataset, a.out)
    print(json.dumps({"run": summary, "scores": scored}, indent=1, default=str))
    if summary.get("stopped_for_auth"):
        return 4
    return 3 if (summary.get("stopped_for_budget") or summary.get("stopped_for_api_error")) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vista_spm.staticbench",
                                 description="STM-BabyVision static perception benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="build a dataset")
    g.add_argument("--out", required=True)
    g.add_argument("--types", default="tip,adatoms,move,drift,herringbone",
                   help="comma list or 'all' (" + ",".join(I.ITEM_TYPES) + ")")
    g.add_argument("--seeds", default="0-9")
    g.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    g.add_argument("--overlays", action="store_true", help="also render truth overlays into <out>/_overlays")
    g.add_argument("--keep-work", action="store_true", help="keep the per-recipe session dirs (<out>/_work)")
    g.set_defaults(fn=_cmd_generate)
    v = sub.add_parser("validate", help="schema-check a dataset")
    v.add_argument("dataset")
    v.set_defaults(fn=_cmd_validate)
    an = sub.add_parser("answer", help="answer every item without an LLM")
    an.add_argument("dataset")
    an.add_argument("--mode", choices=("baseline", "oracle", "abstain"), default="baseline")
    an.add_argument("--out")
    an.add_argument("--types", default="")
    an.set_defaults(fn=_cmd_answer)
    s = sub.add_parser("score", help="score an answers file")
    s.add_argument("dataset")
    s.add_argument("answers")
    s.add_argument("--out")
    s.set_defaults(fn=_cmd_score)
    r = sub.add_parser("report", help="CSV + Markdown tables for one or more answer sets")
    r.add_argument("dataset")
    r.add_argument("--run", action="append", required=True, help="name=answers.jsonl (repeatable)")
    r.add_argument("--out", required=True)
    r.add_argument("--title", default="STM-BabyVision results")
    r.set_defaults(fn=_cmd_report)
    q = sub.add_parser("qa", help="the QA host: a dry run for one item, or answer items with Claude Code")
    q.add_argument("dataset_pos", nargs="?", metavar="dataset", help="the dataset dir (or --dataset)")
    q.add_argument("--dataset")
    q.add_argument("--driver", choices=("dry", "claude"), default="dry")
    q.add_argument("--item", help="dry run: the item id")
    q.add_argument("--arm", default="full", help="dry run: the ablation arm")
    q.add_argument("--work", help="dry run: work dir (default <dataset>/_qa/<item>/<time>)")
    q.add_argument("--out", help="claude: output dir (answers per arm, Claude Code logs, report)")
    q.add_argument("--items", help="claude: comma-separated item ids")
    q.add_argument("--pilot", action="store_true", help="claude: one item per subtype")
    q.add_argument("--difficulty", choices=I.DIFFICULTIES, help="claude: every item of this difficulty")
    q.add_argument("--tools", choices=("full", "image"), default="full",
                   help="claude: full = view + measurement tools + answer; image = answer only")
    q.add_argument("--model", default="claude-opus-5-5")
    q.add_argument("--effort", default="high")
    q.add_argument("--budget-usd", type=float, default=15.0,
                   help="claude: no new item starts once the spend recorded under --out reaches this")
    q.add_argument("--segment-cap-usd", type=float, default=2.0,
                   help="claude: --max-budget-usd for each Claude Code process (also limited by what is left)")
    q.add_argument("--max-tool-calls", type=int, default=60)
    q.add_argument("--preflight", choices=("first", "all", "none"), default="first")
    q.add_argument("--retry-unanswered", action="store_true",
                   help="claude: run again the items whose recorded answer is null")
    q.set_defaults(fn=_cmd_qa)
    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
