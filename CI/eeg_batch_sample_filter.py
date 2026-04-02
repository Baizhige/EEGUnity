"""CI test for sample_filter."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config


def main() -> None:
    """Run sample_filter with representative parameter combinations."""
    config = load_ci_config()

    for domain in domains(config):
        cases = [
            {},
            {"channel_number": (1, 64)},
            {"sampling_rate": (1.0, 4096.0)},
            {"duration": (0.0, 999999.0)},
            {"completeness_check": "Completed"},
            {"domain_tag": domain},
            {"file_type": "standard_data"},
        ]

        for params in cases:
            u_ds = dataset_from_locator(config, domain)
            u_ds.eeg_batch.sample_filter(**params)
            print(f"[eeg_batch_sample_filter] {domain}: params={params} -> rows={len(u_ds.get_locator())}")


if __name__ == "__main__":
    main()
