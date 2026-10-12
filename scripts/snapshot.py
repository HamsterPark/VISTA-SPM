"""Freeze the code a matrix runs on: copy vista_spm/, stmbench/ and stmsim/ to <dest>/ with a hash manifest.

usage: python scripts/snapshot.py <dest> [--stm-bench ROOT]
vista_spm comes from this checkout; stmbench and stmsim from the STM-Bench checkout that Python
imports (or --stm-bench). Run the matrix with --cwd <dest> so that `python -m vista_spm` imports
the frozen copies (python -m puts the cwd first on sys.path); both checkouts may keep changing
meanwhile. The manifest records each checkout's commit and whether it had uncommitted changes.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def git_state(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest")
    ap.add_argument("--stm-bench", help="STM-Bench checkout (default: the one stmbench is imported from)")
    a = ap.parse_args()
    if a.stm_bench:
        stm = Path(a.stm_bench).resolve()
    else:
        import stmbench
        stm = Path(stmbench.__file__).resolve().parents[1]
    dest = Path(a.dest).resolve()
    if dest.exists():
        raise SystemExit(f"{dest} exists")
    dest.mkdir(parents=True)
    manifest = {}
    for root, pkg in ((HERE, "vista_spm"), (stm, "stmbench"), (stm, "stmsim")):
        shutil.copytree(root / pkg, dest / pkg, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for f in sorted((dest / pkg).rglob("*")):
            if f.is_file():
                manifest[f.relative_to(dest).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    sources = {"vista-spm": git_state(HERE), "stm-bench": git_state(stm)}
    (dest / "MANIFEST.json").write_text(json.dumps({"digest": digest, "sources": sources, "files": manifest},
                                                   indent=1), encoding="utf-8")
    print(f"snapshot {digest}: {len(manifest)} files -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
