"""Pre-release packaging integrity check.

Run this before every PyPI upload:
    python CI/pre_release_check.py

Checks performed:
1. Every eegunity subpackage directory contains __init__.py.
2. find_packages() discovers all expected packages.
3. The newest wheel in dist/ contains every discovered package.
4. twine check passes on all dist/ artifacts.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "eegunity"
DIST_DIR = PROJECT_ROOT / "dist"


def _all_package_dirs() -> list[Path]:
    """Return every directory under eegunity/ that is (or should be) a package."""
    return [p for p in PACKAGE_ROOT.rglob("*") if p.is_dir() and "__pycache__" not in p.parts]


def check_init_files() -> list[str]:
    """Fail if any subpackage directory is missing __init__.py."""
    errors: list[str] = []
    for pkg_dir in _all_package_dirs():
        if not (pkg_dir / "__init__.py").exists():
            rel = pkg_dir.relative_to(PROJECT_ROOT)
            errors.append(f"MISSING __init__.py: {rel}")
    return errors


def check_find_packages() -> tuple[list[str], list[str]]:
    """Return (found_packages, errors)."""
    result = subprocess.run(
        [sys.executable, "-c",
         "from setuptools import find_packages; "
         "pkgs=find_packages(include=['eegunity','eegunity.*']); "
         "print('\\n'.join(sorted(pkgs)))"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], [f"find_packages() failed: {result.stderr.strip()}"]
    found = [line for line in result.stdout.splitlines() if line.strip()]
    return found, []


def check_wheel_contents(found_packages: list[str]) -> list[str]:
    """Verify the newest wheel contains every package discovered by find_packages."""
    wheels = sorted(DIST_DIR.glob("*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        return ["No wheel found in dist/ — run 'python -m build' first."]

    wheel = wheels[0]
    print(f"  Checking wheel: {wheel.name}")

    with zipfile.ZipFile(wheel) as zf:
        wheel_dirs = {Path(n).parts[0] for n in zf.namelist() if "/" in n}
        wheel_files = set(zf.namelist())

    errors: list[str] = []
    for pkg in found_packages:
        pkg_path = pkg.replace(".", "/")
        init_path = f"{pkg_path}/__init__.py"
        if init_path not in wheel_files:
            errors.append(f"Wheel missing: {init_path}  (package: {pkg})")

    return errors


def check_twine() -> list[str]:
    """Run twine check on all dist/ artifacts."""
    artifacts = list(DIST_DIR.glob("*"))
    if not artifacts:
        return ["No artifacts in dist/ to check."]

    result = subprocess.run(
        [sys.executable, "-m", "twine", "check"] + [str(a) for a in artifacts],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [f"twine check failed:\n{result.stdout}\n{result.stderr}"]
    return []


def main() -> int:
    all_errors: list[str] = []

    print("[1/4] Checking __init__.py in all subpackage dirs...")
    errs = check_init_files()
    all_errors.extend(errs)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
    else:
        print("  OK")

    print("[2/4] Running find_packages()...")
    found, errs = check_find_packages()
    all_errors.extend(errs)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
    else:
        print(f"  OK — {len(found)} packages: {found}")

    print("[3/4] Checking wheel contents...")
    errs = check_wheel_contents(found)
    all_errors.extend(errs)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
    else:
        print("  OK")

    print("[4/4] Running twine check...")
    errs = check_twine()
    all_errors.extend(errs)
    if errs:
        for e in errs:
            print(f"  ERROR: {e}")
    else:
        print("  OK")

    print()
    if all_errors:
        print(f"PRE-RELEASE CHECK FAILED — {len(all_errors)} error(s). Fix before uploading.")
        return 1
    print("PRE-RELEASE CHECK PASSED. Safe to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
