"""Ablation arms: which observation the model gets and which tools it has.

The arms mirror VISTA's ablation (arXiv 2610.02200, Table/Fig. 5) translated to the STM:
official text interface → images → notes → lossless visual memory + inspect → exact
read-out (``read``: VISTA's whole tool set) → measurement tools, which the STM needs and ARC
did not (``full``). ``vigil-text`` is the representation MAST used before this harness
(vision-model summaries as text), run through the same harness and model so the comparison
isolates the representation.

``grid``: scan images carry the metric grid, VISTA's graph paper (``frames.view_grid``). Every
image arm has it since 2026-10-12. The first matrix (m1) ran without it; ``with_options(arm,
grid=False)`` renders without it under the name ``<arm>-nogrid``. m1 itself, surface and tool
behaviour, is reproduced from the tag ``m1-surface`` (docs/notes/2026-10-13-reference-fixes.zh-CN.md).
"""
from __future__ import annotations

import dataclasses
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
    grid: bool = True           # the metric grid on scan images (image arms only)
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
    Arm("read", "image", notes=True, history=True, inspect=True, read_values=True, measure=False,
        advisor=False, description="images + notes + visual memory/inspect + read_values (VISTA's tool set)"),
    Arm("mem", "image", notes=True, history=True, inspect=True, read_values=False, measure=False,
        advisor=False, description="images + notes + visual memory/inspect"),
    Arm("notes", "image", notes=True, history=True, inspect=False, read_values=False, measure=False,
        advisor=False, description="images + notes"),
    Arm("img", "image", notes=False, history=False, inspect=False, read_values=False, measure=False,
        advisor=False, description="images only"),
    Arm("text", "text", notes=True, history=True, inspect=False, read_values=True, measure=False,
        advisor=False, grid=False, description="numeric text grids instead of images + notes + read_values"),
    Arm("vigil-text", "vigil", notes=True, history=True, inspect=False, read_values=False, measure=False,
        advisor=False, grid=False, description="MAST vision-model summaries as text, no images + notes"),
    Arm("advisor", "image", notes=True, history=True, inspect=True, read_values=True, measure=True,
        advisor=True, description="full + MAST detectors/VIGIL on request"),
)}


def get_arm(name: str) -> Arm:
    try:
        return ARMS[name]
    except KeyError:
        raise KeyError(f"unknown arm {name!r}; choose from {sorted(ARMS)}") from None


def with_options(arm: Arm | str, *, display_max: int | None = None, grid: bool | None = None) -> Arm:
    """``arm`` with run-time overrides. Turning the grid off renames the arm ``<name>-nogrid``,
    so its run directories and tables never mix with the gridded runs of the same arm."""
    a = get_arm(arm) if isinstance(arm, str) else arm
    if display_max is not None:
        a = dataclasses.replace(a, display_max=int(display_max))
    if grid is not None and bool(grid) != a.grid:
        if a.observation != "image":
            raise ValueError(f"arm {a.name!r} shows no images; the grid does not apply")
        a = dataclasses.replace(a, grid=bool(grid), name=a.name if grid else f"{a.name}-nogrid")
    return a
