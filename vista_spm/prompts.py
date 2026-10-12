"""Prompt texts of the harness. Kept minimal on purpose, following VISTA's prompt
(benchmarks/arc3/claude/prompt.md in the VISTA repository, MIT): an objective, a revisable
model, expect-before / report-after, notes. Everything task-specific comes from the
scenario's own task text; nothing here teaches STM technique.
"""
from __future__ import annotations

from .arms import Arm

SYSTEM_TEMPLATE = """# Instrument task

You are operating a scanning tunnelling microscope through the tools of this session.

Complete the task with as little instrument time and as few instrument actions as possible.

Build and use a compact, revisable model of the instrument, the tip and the sample, and of their current state. Update it as new evidence changes what is supported.

Before each `act`, briefly state what you expect to {verb}. Afterward, briefly state all {noun} changes, expected or not.
{notes}"""

NOTES_LINE = ("\nKeep concise, durable, revisable understanding in `GUIDE.md`; use `WORKING.md` as a scratchpad "
              "when useful.\n")

END_RULES = ("When the task is complete, end your reply with [DONE]. If it cannot be completed, end your reply "
             "with [ABORT] and say why. The marker must be the last word of the reply; nobody reads the replies "
             "in between, so keep acting until then.")

CONTINUE_TEXT = ("Nobody is at the console. Continue with the task: act while the instrument is available, report "
                 "results with report_result, and end your reply with [DONE] when the task is complete or [ABORT] if "
                 "it cannot be completed.")

from .compact_hook import CHECKPOINT_INSTRUCTION  # noqa: E402,F401  (one source for the wording)


def system_prompt(arm: Arm) -> str:
    seeing = arm.observation == "image"
    return SYSTEM_TEMPLATE.format(verb="see" if seeing else "observe",
                                  noun="visible" if seeing else "observed",
                                  notes=NOTES_LINE if arm.notes else "\n").rstrip() + "\n"


def task_block(task: str, *, claim_ids: list[str], budget_s: float | None, diagnosis: bool = True) -> str:
    lines = ["Task:", task.strip(), ""]
    if claim_ids:
        lines.append("Report each result with `report_result` (claim ids: " + ", ".join(claim_ids) + "). "
                     "A result never reported counts as not reproduced; the last report of a claim counts. "
                     "Positions are scan-frame nm.")
    if diagnosis:
        lines.append("You may state your judgement of the junction and the tip with `report_tip_state`.")
    if budget_s:
        lines.append(f"Instrument budget: {budget_s / 3600.0:.2f} h ({budget_s:.0f} s) of instrument time. "
                     "Instrument time passes only while an action executes; thinking and the read-only tools "
                     "are free.")
    lines.append(END_RULES)
    return "\n".join(lines)
