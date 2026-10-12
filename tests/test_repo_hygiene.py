"""Repository guards, ported from STM-Bench's ``tests/test_no_machine_paths.py``.

* No machine-specific path literal (``D:/...``) in source: everything outside the checkout is
  reached through ``stmbench.paths`` / ``stmsim.paths`` or a command-line argument.
* No second reader of ``STM_BENCH_*`` / ``MAST_ROOT``: only ``stmsim.paths`` reads them.
* The controller vendor's name appears only inside MAST's own identifiers and the .sxm header
  token, as in STM-Bench.
* Nothing still points at the harness's former home inside STM-Bench.

Scope: the package, the scripts, the tests and the maintained docs. ``docs/notes/`` (research
notes kept verbatim) and ``results/`` (outputs of past runs) are records, not maintained text.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("vista_spm", "scripts")
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml"}
MAINTAINED_TEXT = ("README.md", "README.zh-CN.md", "AGENTS.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml",
                   "docs/DESIGN.md", "docs/staticbench.md", "scripts/README.md", "results/README.md")
# any drive-letter path (D:/Users/researcher, C:\data, ...) not preceded by an identifier character
DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
ENV_READ = re.compile(r"""(?:environ(?:\.get\(|\[)|getenv\()\s*["'](?:STM_BENCH_\w+|MAST_ROOT)["']""")
VENDOR = re.compile(r"nanonis", re.IGNORECASE)
#: MAST's module and setting names and the .sxm header token; anything else that says it is a leak
VENDOR_ALLOWED = ("nanonis_spm", "nanonis_files", "nanonis_patch", "NanonisConfig", "MAST_NANONIS_PORT_",
                  "NANONIS_VERSION")
FORMER_HOME = re.compile(r"stmbench[./\\]vista")


def _code_files(root: Path) -> list[Path]:
    out = []
    for d in CODE_DIRS:
        base = root / d
        if base.is_dir():
            out += [p for p in sorted(base.rglob("*"))
                    if p.is_file() and p.suffix in CODE_SUFFIXES and "__pycache__" not in p.parts]
    return out


def _text_files(root: Path) -> list[Path]:
    """Code, tests and maintained docs, without this guard (its patterns name what they look for)."""
    tests = [p for p in sorted((root / "tests").glob("*.py")) if p.name != Path(__file__).name]
    return _code_files(root) + tests + [root / f for f in MAINTAINED_TEXT if (root / f).is_file()]


def scan(files: list[Path], pattern: re.Pattern, root: Path = ROOT, *, skip_comments: bool = True,
         allowed: tuple[str, ...] = ()) -> list[tuple[str, int, str]]:
    """(relative path, 1-based line, stripped line) for every offending line."""
    out = []
    for p in files:
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            s = line.strip()
            if not s or (skip_comments and s.startswith("#")):
                continue
            for tok in allowed:
                line = line.replace(tok, "")
            if pattern.search(line):
                out.append((p.relative_to(root).as_posix(), i, s[:140]))
    return out


def _report(offenders, what: str) -> str:
    return "\n".join([f"{len(offenders)} {what}:"] + [f"  {f}:{n}: {s}" for f, n, s in offenders])


def test_no_machine_paths_in_source():
    offenders = scan(_code_files(ROOT), DRIVE_PATH)
    assert not offenders, _report(offenders, "machine-path literal(s); use stmbench.paths or an argument")


def test_no_parallel_env_readers_in_source():
    offenders = scan(_code_files(ROOT), ENV_READ)
    assert not offenders, _report(offenders, "direct STM_BENCH_*/MAST_ROOT read(s); use stmbench.paths")


def test_the_vendor_name_appears_only_as_mast_api_or_file_format_tokens():
    offenders = scan(_text_files(ROOT), VENDOR, skip_comments=False, allowed=VENDOR_ALLOWED)
    assert not offenders, _report(offenders, "vendor name occurrence(s)")


def test_nothing_points_at_the_former_home_in_stm_bench():
    offenders = scan(_text_files(ROOT), FORMER_HOME, skip_comments=False)
    assert not offenders, _report(offenders, "reference(s) to stmbench.vista; the package is vista_spm")


@pytest.mark.parametrize("text, n", [('X = "E:/stm_bench/x"\n# E:/comment is fine\nY = 1\n', 1),
                                     ('C = "C:\\\\Users\\\\researcher"\n', 1),
                                     ('SIZE_E = "SIZE:/"\nname = "TYPE:/x"\n', 0)])
def test_scanner_finds_a_planted_literal(tmp_path, text, n):
    """The guard must see a path when one is there (a scanner that never fires would make the
    green above vacuous)."""
    (tmp_path / "vista_spm").mkdir()
    (tmp_path / "vista_spm" / "bad.py").write_text(text, encoding="utf-8")
    assert len(scan(_code_files(tmp_path), DRIVE_PATH, tmp_path)) == n


def test_vendor_scanner_strips_only_the_allowed_tokens(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("from mast.io.nanonis_files import read_sxm\nname = 'Nanonis'\n", encoding="utf-8")
    hits = scan([p], VENDOR, tmp_path, skip_comments=False, allowed=VENDOR_ALLOWED)
    assert [(f, n) for f, n, _ in hits] == [("x.py", 2)]
