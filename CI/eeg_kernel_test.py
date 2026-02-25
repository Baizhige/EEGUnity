"""CI test script for external EEGUnity kernels.

This script is modeled after the existing resample CI script. It loads test
locators, binds an external kernel to each UnifiedDataset, triggers parsing,
and validates that kernels update ``raw.info['description']`` as JSON.

Expected CI_config.json fields
------------------------------
Required:
- test_data_list: list[str]
- locator_base_path: str
- CI_output_path: str

Optional:
- kernel_spec_map: dict[str, str]
    Mapping from domain_tag to kernel spec (path/module + object name).
    Example:
        {
          "bcic_iv_2a": "/abs/path/bcic_iv_2a_kernel.py:KERNEL",
          "figshare_largemi": "/abs/path/figshare_largemi_kernel.py:KERNEL"
        }

If kernel_spec_map is not provided, this script will try:
- kernels/{domain_tag}_kernel.py:KERNEL
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Dict, Optional

# Get the parent directory of the script (same style as resample CI script).
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from eegunity.unifieddataset import UnifiedDataset  # noqa: E402


def _get_kernel_spec(domain_tag: str, config: Dict) -> Optional[str]:
    """Resolve kernel spec for a domain tag.

    The spec is either a file path (extension optional) or a module import path.
    The module must expose a single kernel object named ``KERNEL``.
    """
    kernel_map = config.get("kernel_spec_map", None)
    if isinstance(kernel_map, dict) and domain_tag in kernel_map:
        return kernel_map[domain_tag]

    # Default convention: kernels/{domain}_kernel ('.py' optional)
    base = os.path.join("kernels", f"{domain_tag}_kernel")
    if os.path.isfile(base) or os.path.isfile(f"{base}.py"):
        return base

    return None


def _validate_description_json(raw) -> None:
    """Validate that raw.info['description'] is JSON with expected keys."""
    desc = raw.info.get("description", None)
    if desc is None:
        raise AssertionError("raw.info['description'] is missing.")

    try:
        payload = json.loads(desc)
    except Exception as e:
        raise AssertionError(f"raw.info['description'] is not valid JSON: {e}") from e

    if not isinstance(payload, dict):
        raise AssertionError("raw.info['description'] JSON must be a dict.")

    if "original_description" not in payload:
        raise AssertionError("Missing key: 'original_description' in description JSON.")

    if "eegunity_description" not in payload:
        raise AssertionError("Missing key: 'eegunity_description' in description JSON.")


def main() -> None:
    """Run kernel CI tests."""
    # Obtain base config from file (same as resample CI script).
    with open("CI_config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    remain_list = config["test_data_list"]
    locator_base_path = config["locator_base_path"]
    ci_output_base = config["CI_output_path"]
    ci_output_path = os.path.join(ci_output_base, "kernel")

    # Create output directory if it doesn't exist (keep the same convention).
    os.makedirs(ci_output_path, exist_ok=True)

    # Test each dataset.
    for domain_tag in remain_list:
        locator_path = f"{locator_base_path}/{domain_tag}.csv"
        unified_dataset = UnifiedDataset(
            domain_tag=domain_tag,
            locator_path=locator_path,
            is_unzip=False,
        )

        kernel_spec = _get_kernel_spec(domain_tag, config)
        if not kernel_spec:
            warnings.warn(
                (
                    f"[Kernel CI] No kernel spec found for domain '{domain_tag}'. "
                    "Provide 'kernel_spec_map' in CI_config.json or place a file at "
                    f"'kernels/{domain_tag}_kernel.py'. Skipping kernel test for this dataset."
                )
            )
            continue

        # Bind kernel (constructed or after instantiation are both supported).
        unified_dataset.load_kernel(kernel_spec)

        # Trigger parsing (and kernel application) for a few rows.
        # Keep it small for CI stability.
        locator_len = len(unified_dataset.get_shared_attr()["locator"])
        test_indices = [0] if locator_len <= 1 else [0, min(1, locator_len - 1)]

        for idx in test_indices:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                raw = unified_dataset.eeg_parser.get_data(idx, preload=False)

            # If the kernel raised internally, EEGUnity should warn and return raw unchanged.
            # We still validate description is present and JSON-formatted, because kernel
            # is expected to write it when compatible.
            try:
                _validate_description_json(raw)
                print(
                    f"[Kernel CI] Domain '{domain_tag}' index {idx}: "
                    "description JSON validated."
                )
            except AssertionError as e:
                # If there was a kernel compatibility warning, treat this as a soft-fail
                # and print the warning message for debugging.
                warn_msgs = [str(w.message) for w in caught]
                if any("Kernel '" in m and "not compatible" in m for m in warn_msgs):
                    print(
                        f"[Kernel CI] Domain '{domain_tag}' index {idx}: "
                        "kernel reported incompatibility; returned raw without valid "
                        f"description JSON. Warning: {warn_msgs[-1]}"
                    )
                else:
                    # Hard fail: no incompatibility warning, but still invalid.
                    raise

        print(f"[Kernel CI] Completed kernel tests for {domain_tag}.")

    print("Successfully completed all kernel tests.")


if __name__ == "__main__":
    main()