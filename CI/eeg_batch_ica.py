"""CI test for ICA."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run ICA-based artifact correction."""
    config = load_ci_config()
    out_dir = output_dir(config, "ica")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.ica(output_path=str(out_dir), miss_bad_data=True)
        print(f"[eeg_batch_ica] {domain}: completed")


if __name__ == "__main__":
    main()
