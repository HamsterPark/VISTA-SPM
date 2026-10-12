"""Test configuration.

STM-Bench (``stmbench`` + ``stmsim``) must be importable: install it editable in the active
environment. MAST is optional. Set ``MAST_ROOT`` (or put ``MASTv2`` on ``PYTHONPATH``); everything
that imports ``mast`` carries ``requires_mast`` and is skipped, not failed, when MAST is absent,
so ``pytest`` is green on a machine without MAST. The selection mirrors STM-Bench's own
``tests/conftest.py``::

    from tests.conftest import requires_mast

    pytestmark = requires_mast          # whole module

Markers (registered in ``pyproject.toml``):

* ``requires_mast``: MAST + its patched controller client importable (``mast_available()``).
* ``isolated``: must be the only RuntimeHost of its process (MAST keeps process-level
  singletons); run with a separate ``pytest -m isolated`` invocation.

The live tests in ``tests/test_runner.py`` run the real Claude Code CLI on the subscription and
are skipped unless ``VISTA_LIVE_CLAUDE=1``.
"""
from __future__ import annotations

import sys

import pytest

from stmsim.paths import mast_root

MAST_ROOT = mast_root()
MASTV2 = MAST_ROOT / "MASTv2" if MAST_ROOT is not None else None
if MASTV2 is not None and MASTV2.exists() and str(MASTV2) not in sys.path:
    sys.path.insert(0, str(MASTV2))


def mast_available() -> bool:
    try:
        import nanonis_spm  # noqa: F401
        import mast.core.nanonis_patch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


requires_mast = pytest.mark.requires_mast

_SKIP_REASON = "MAST + its controller client not importable (set MAST_ROOT or put MASTv2 on PYTHONPATH)"


def pytest_collection_modifyitems(config, items):
    if mast_available():
        return
    for item in items:
        if item.get_closest_marker("requires_mast") is not None:
            item.add_marker(pytest.mark.skip(reason=_SKIP_REASON))
