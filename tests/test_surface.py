"""The model-facing surface changes only on purpose (AGENTS.md, invariant 2).

Every hash is the sha256 (first 16 hex digits) of what ``python -m vista_spm tools --arm <arm>
[--qplus] [--grid off] --json`` offers the model: the system prompt and the tool list.

* ``SURFACE`` pins the current surface of every arm: with the grid off (image arms only) and
  by default, each without and with qPlus. A deliberate change to the surface updates these pins
  and records, in ``docs/notes/``, which runs it separates.
* ``M1_SURFACE`` is the surface the first matrix (m1) offered, recorded on 2026-10-12 just before
  the graph paper was added. m1 is reproduced, surface and tool behaviour, from the tag
  ``m1-surface`` with ``--grid off``. ``M1_UNCHANGED`` lists the arms whose current surface still
  equals m1's (grid off for image arms); a change that touches one of them removes it there.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from vista_spm.arms import ARMS, with_options
from vista_spm.cli import _offline_host
from vista_spm.prompts import system_prompt

M1_SURFACE = {          # arm: (without qPlus, with qPlus), grid off
    "full": ("98140a8d2b79d853", "6179652feeaabb65"),
    "mem": ("8c0597f70f0a3e71", "ad11f19af99a3ab1"),
    "notes": ("ec7380e9da0d3fad", "361b0dc13083ce70"),
    "img": ("67fd045e1bde22fe", "29c20de81320aa48"),
    "text": ("fdb6ff1467869c44", "d086df22455209e7"),
    "vigil-text": ("dbc1e6c31136c497", "1684d369c65154ef"),
    "advisor": ("8ae2856c03da4706", "2b8fdc50c3ddf88d"),
}

SURFACE = {             # arm: (grid off, grid off + qPlus, default, default + qPlus)
    "full": ("0988707445b0385a", "71e5aa3a8ecc9538", "9254e28b00d6c261", "e13f03efed6b976b"),
    "read": ("ac6457e6fa5e3b88", "b9d7db692dcbeb26", "8bca9399a9d8aea9", "39955e065d63e146"),
    "mem": ("1a709ad3a198888e", "6ef0604706c5a502", "6f28dbea8b711b67", "d0810ba35839084d"),
    "notes": ("ec7380e9da0d3fad", "361b0dc13083ce70", "c3fbebb6d195eccd", "d00e7393a2e7bfa4"),
    "img": ("67fd045e1bde22fe", "29c20de81320aa48", "541abb11d243be7f", "ea416f110bab4ff9"),
    "text": (None, None, "fdb6ff1467869c44", "d086df22455209e7"),
    "vigil-text": (None, None, "dbc1e6c31136c497", "1684d369c65154ef"),
    "advisor": ("fc550b0cd070a70a", "3fa2d85efbe511d8", "2dc3f69123525d66", "27bf04fe9e46270e"),
}

M1_UNCHANGED = ("notes", "img", "text", "vigil-text")


def _surface(arm_name: str, qplus: bool, grid: bool | None) -> str:
    arm = with_options(arm_name, grid=grid)
    host = _offline_host(arm, qplus, ["example_claim"])
    text = json.dumps({"system_prompt": system_prompt(arm), "tools": host.list_tools()},
                      ensure_ascii=False, indent=1) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_every_arm_is_pinned():
    assert set(SURFACE) == set(ARMS)


@pytest.mark.parametrize("arm", sorted(SURFACE))
def test_the_surface_is_pinned(arm):
    image = ARMS[arm].observation == "image"
    got = ((_surface(arm, False, False), _surface(arm, True, False)) if image else (None, None)) + \
          (_surface(arm, False, None), _surface(arm, True, None))
    assert got == SURFACE[arm]


@pytest.mark.parametrize("arm", M1_UNCHANGED)
def test_unchanged_arms_still_offer_m1(arm):
    grid = False if ARMS[arm].observation == "image" else None
    assert (_surface(arm, False, grid), _surface(arm, True, grid)) == M1_SURFACE[arm]


@pytest.mark.parametrize("arm", ["full", "mem", "notes", "img", "advisor"])
def test_by_default_image_arms_describe_the_grid(arm):
    assert SURFACE[arm][2] != SURFACE[arm][0]
