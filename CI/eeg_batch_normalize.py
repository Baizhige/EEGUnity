"""CI test for normalize."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run normalize in sample-wise and channel-wise modes."""
    config = load_ci_config()
    out_dir = output_dir(config, "normalize")

    for domain in domains(config):
        cases = [
            {},
            {"norm_type": "sample-wise"},
            {"norm_type": "channel-wise"},
            {"miss_bad_data": True},
        ]

        for params in cases:
            u_ds = dataset_from_locator(config, domain)
            u_ds.eeg_batch.normalize(output_path=str(out_dir), **params)
            print(f"[eeg_batch_normalize] {domain}: params={params}")


if __name__ == "__main__":
    main()
