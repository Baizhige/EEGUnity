"""Run local EEGUnity CI scripts with the current Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _collect_scripts(ci_dir: Path) -> list[str]:
    """Collect scripts in deterministic order for local CI execution."""
    batch_scripts = sorted(p.name for p in ci_dir.glob("eeg_batch_*.py"))
    scripts = [
        "import_testing.py",
        "eeg_parser_generate_locator.py",
        *batch_scripts,
        "eeg_kernel_test.py",
    ]

    return [name for name in scripts if (ci_dir / name).exists()]


def run_tests() -> int:
    """Execute CI scripts and return process exit code."""
    ci_dir = Path(__file__).resolve().parent
    scripts = _collect_scripts(ci_dir)
    failed: list[str] = []

    print("[run_all_CI_test] Scripts:", scripts)

    for script_name in scripts:
        script_path = ci_dir / script_name
        print(f"[run_all_CI_test] Running: {script_name}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ci_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            failed.append(script_name)
            print(f"[run_all_CI_test] FAILED: {script_name}")
            print(result.stdout)
            print(result.stderr)
        else:
            print(f"[run_all_CI_test] PASSED: {script_name}")
            if result.stdout.strip():
                print(result.stdout)

    if failed:
        print("\n[run_all_CI_test] Failed scripts:")
        for name in failed:
            print(name)
        return 1

    print("\n[run_all_CI_test] All scripts passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_tests())
