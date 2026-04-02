"""Generate local locators for configured datasets."""

from __future__ import annotations

from ci_runtime import (
    dataset_from_path,
    domains,
    ensure_dir,
    load_ci_config,
    locator_path,
)


def main() -> None:
    """Parse configured datasets and save locators to configured directory."""
    config = load_ci_config()

    for domain in domains(config):
        u_ds = dataset_from_path(config, domain)
        out_path = locator_path(config, domain)
        ensure_dir(out_path.parent)
        u_ds.save_locator(str(out_path))
        print(f"[eeg_parser_generate_locator] {domain}: locator saved -> {out_path}")


if __name__ == "__main__":
    main()
