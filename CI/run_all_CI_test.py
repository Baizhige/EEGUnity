"""Run local EEGUnity CI scripts with the current Python interpreter."""

from __future__ import annotations

import argparse
import os
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


def run_tests(config_path: Path | None = None) -> int:
    """Execute CI scripts and return their combined process exit code.

    Parameters
    ----------
    config_path : pathlib.Path or None
        Optional machine-local configuration passed to every CI script.

    Returns
    -------
    int
        Zero when every script passes, otherwise one.

    Examples
    --------
    >>> run_tests(Path("local_ci_config.json"))  # doctest: +SKIP
    0
    """
    ci_dir = Path(__file__).resolve().parent
    scripts = _collect_scripts(ci_dir)
    failed: list[str] = []
    child_env = os.environ.copy()

    if config_path is not None:
        resolved_config = config_path.resolve()
        if not resolved_config.is_file():
            raise FileNotFoundError(f"CI config not found: {resolved_config}")
        child_env["EEGUNITY_CI_CONFIG"] = str(resolved_config)
        print(f"[run_all_CI_test] Config: {resolved_config}")

    print("[run_all_CI_test] Scripts:", scripts)

    for script_name in scripts:
        script_path = ci_dir / script_name
        print(f"[run_all_CI_test] Running: {script_name}")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ci_dir),
            capture_output=True,
            text=True,
            env=child_env,
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


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local CI runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a machine-local CI JSON configuration.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_tests(_parse_args().config))
