"""CI test for save_as_other."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Validate save_as_other success and expected error paths."""
    config = load_ci_config()
    out_dir = output_dir(config, "save_as_other")

    for domain in domains(config):
        # valid fif
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.save_as_other(output_path=str(out_dir), format="fif", miss_bad_data=True)

        # valid csv
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.save_as_other(output_path=str(out_dir), format="csv", miss_bad_data=True)

        # with explicit domain tag
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.save_as_other(
            output_path=str(out_dir),
            domain_tag=domain,
            format="fif",
            miss_bad_data=True,
        )

        # invalid format should raise
        u_ds = dataset_from_locator(config, domain)
        try:
            u_ds.eeg_batch.save_as_other(output_path=str(out_dir), format="unsupported_format")
            raise AssertionError("Expected ValueError for unsupported format")
        except ValueError:
            pass

        # invalid output path should raise
        u_ds = dataset_from_locator(config, domain)
        try:
            u_ds.eeg_batch.save_as_other(output_path="Z:/__definitely_missing__/", format="fif")
            raise AssertionError("Expected FileNotFoundError for invalid output path")
        except FileNotFoundError:
            pass

        print(f"[eeg_batch_save_as_other] {domain}: completed")


if __name__ == "__main__":
    main()
