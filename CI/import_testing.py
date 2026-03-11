"""Basic import and dataset smoke tests."""

from __future__ import annotations

from eegunity import Pipeline, UnifiedDataset, get_data_row

from ci_runtime import dataset_from_path, domains, load_ci_config


def main() -> None:
    """Run basic import and single-row data-loading checks."""
    config = load_ci_config()

    # Keep imported symbols alive for smoke validation.
    _ = Pipeline
    _ = UnifiedDataset

    for domain in domains(config):
        u_ds = dataset_from_path(config, domain)
        locator = u_ds.get_locator()
        if locator.empty:
            raise RuntimeError(f"Empty locator for dataset: {domain}")

        # Locator rows generated directly from source files may not yet use
        # typed channel names (type:name), so keep this smoke test conservative.
        raw = get_data_row(locator.iloc[0], is_set_channel_type=False, preload=False)
        if raw is None:
            raise RuntimeError(f"Failed to load first row raw object for dataset: {domain}")

        print(f"[import_testing] {domain}: locator rows={len(locator)}; first raw loaded.")


if __name__ == "__main__":
    main()
