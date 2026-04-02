"""Shared helpers for local CI scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CI_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eegunity.unifieddataset import UnifiedDataset


def load_ci_config() -> dict:
    """Load CI configuration and validate required fields."""
    config_path = CI_DIR / "CI_config.json"
    with config_path.open("r", encoding="utf-8-sig") as f:
        config = json.load(f)

    required = ["test_data_list", "data_root_path", "locator_base_path", "CI_output_path"]
    for key in required:
        if key not in config:
            raise KeyError(f"Missing required CI config key: {key}")

    return config


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if needed and return the path object."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def domains(config: dict) -> list[str]:
    """Return configured test dataset domains."""
    return list(config["test_data_list"])


def dataset_path(config: dict, domain: str) -> Path:
    """Return the source dataset path for a domain."""
    return Path(config["data_root_path"]) / domain


def locator_path(config: dict, domain: str) -> Path:
    """Return locator CSV path for a domain."""
    return Path(config["locator_base_path"]) / f"{domain}.csv"


def output_dir(config: dict, *parts: str) -> Path:
    """Return output directory under CI_output_path, creating it."""
    base = Path(config["CI_output_path"])
    return ensure_dir(base.joinpath(*parts))


def _apply_row_limit(u_ds: UnifiedDataset, config: dict) -> None:
    """Trim locator rows for faster local CI if configured."""
    max_rows = int(config.get("max_rows_per_test", 0) or 0)
    if max_rows > 0:
        loc = u_ds.get_locator().head(max_rows).copy().reset_index(drop=True)
        u_ds.set_locator(loc)


def dataset_from_path(config: dict, domain: str) -> UnifiedDataset:
    """Build UnifiedDataset from source files and apply optional row limit."""
    src_path = dataset_path(config, domain)
    if not src_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {src_path}")

    u_ds = UnifiedDataset(dataset_path=str(src_path), domain_tag=domain)
    _apply_row_limit(u_ds, config)
    return u_ds


def dataset_from_locator(config: dict, domain: str) -> UnifiedDataset:
    """Build UnifiedDataset from locator CSV and apply optional row limit."""
    loc_path = locator_path(config, domain)
    if not loc_path.exists():
        raise FileNotFoundError(
            f"Locator not found for domain '{domain}': {loc_path}. "
            "Run eeg_parser_generate_locator.py first."
        )

    u_ds = UnifiedDataset(domain_tag=domain, locator_path=str(loc_path), is_unzip=False)
    _apply_row_limit(u_ds, config)
    return u_ds
