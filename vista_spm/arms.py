"""Ablation arms: which observation the model gets and which tools it has.

The arms mirror VISTA's ablation (arXiv 2610.02200, Table/Fig. 5) translated to the STM:
official text interface → images → notes → lossless visual memory + inspect → exact
read-out (+ measurement tools, which the STM needs and ARC did not). ``vigil-text`` is
the representation MAST used before this harness (vision-model summaries as text), run
through the same harness and model so the comparison isolates the representation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .protocol import DISPLAY_MAX_DEFAULT


@dataclass(frozen=True)
class Arm:
    name: str
    observation: str            # "image" | "text" | "vigil"
    notes: bool
    history: bool
    inspect: bool
    read_values: bool
    measure: bool
    advisor: bool
    display_max: int = DISPLAY_MAX_DEFAULT
    description: str = ""

    @property
    def compaction(self) -> str:
        """``checkpoint``: VISTA's WORKING.md handoff (needs the notes tools); ``native``:
        Claude Code's own compaction."""
        return "checkpoint" if self.notes else "native"


ARMS: dict[str, Arm] = {a.name: a for a in (
    Arm("full", "image", notes=True, history=True, inspect=True, read_values=True, measure=True,
        advisor=False, description="images + notes + visual memory/inspect + read_values + measurement tools"),
    Arm("mem", "image", notes=True, history=True, inspect=True, read_values=False, measure=False,
        advisor=False, description="images + notes + visual memory/inspect"),
    Arm("notes", "image", notes=True, history=True, inspect=False, read_values=False, measure=False,
        advisor=False, description="images + notes"),
    Arm("img", "image", notes=False, history=False, inspect=False, read_values=False, measure=False,
        advisor=False, description="images only"),
    Arm("text", "text", notes=True, history=True, inspect=False, read_values=True, measure=False,
        advisor=False, description="numeric text grids instead of images + notes + read_values"),
    Arm("vigil-text", "vigil", notes=True, history=True, inspect=False, read_values=False, measure=False,
        advisor=False, description="MAST vision-model summaries as text, no images + notes"),
    Arm("advisor", "image", notes=True, history=True, inspect=True, read_values=True, measure=True,
        advisor=True, description="full + MAST detectors/VIGIL on request"),
)}


def get_arm(name: str) -> Arm:
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; choose from {sorted(ARMS)}") from None
