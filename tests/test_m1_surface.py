"""The first matrix (m1) stays reproducible: with the grid off, every arm offers the model the
system prompt and tools it offered in m1, byte for byte.

The hashes are sha256 (first 16 hex digits) of ``python -m vista_spm tools --arm <arm> [--qplus]
--json`` as it printed for m1's code, recorded on 2026-10-12 just before the graph paper was
added. A deliberate change to the model-facing surface must update them and say which runs it
separates from m1 (AGENTS.md, invariant 2).
"""
from __future__ import annotations

import hashlib
import json

import pytest

from vista_spm.arms import with_options
from vista_spm.cli import _offline_host
from vista_spm.prompts import system_prompt

M1_SURFACE = {          # arm: (without qPlus, with qPlus)
    "full": ("98140a8d2b79d853", "6179652feeaabb65"),
    "mem": ("8c0597f70f0a3e71", "ad11f19af99a3ab1"),
    "notes": ("ec7380e9da0d3fad", "361b0dc13083ce70"),
    "img": ("67fd045e1bde22fe", "29c20de81320aa48"),
    "text": ("fdb6ff1467869c44", "d086df22455209e7"),
    "vigil-text": ("dbc1e6c31136c497", "1684d369c65154ef"),
    "advisor": ("8ae2856c03da4706", "2b8fdc50c3ddf88d"),
}


def _surface(arm_name: str, qplus: bool, grid: bool | None) -> str:
    arm = with_options(arm_name, grid=grid)
    host = _offline_host(arm, qplus, ["example_claim"])
    text = json.dumps({"system_prompt": system_prompt(arm), "tools": host.list_tools()},
                      ensure_ascii=False, indent=1) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("arm", sorted(M1_SURFACE))
def test_grid_off_reproduces_the_m1_surface(arm):
    assert (_surface(arm, False, False), _surface(arm, True, False)) == M1_SURFACE[arm]


@pytest.mark.parametrize("arm", ["full", "mem", "notes", "img", "advisor"])
def test_by_default_image_arms_describe_the_grid(arm):
    assert _surface(arm, False, None) != M1_SURFACE[arm][0]


@pytest.mark.parametrize("arm", ["text", "vigil-text"])
def test_text_arms_are_unchanged_by_default(arm):
    assert _surface(arm, False, None) == M1_SURFACE[arm][0]
