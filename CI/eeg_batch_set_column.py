"""CI test for set_metadata."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Set a new metadata column and verify values."""
    config = load_ci_config()
    out_dir = output_dir(config, "set_column")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        locator_len = len(u_ds.get_locator())
        test_values = [1] * locator_len

        u_ds.eeg_batch.set_metadata(col_name="new_column", value=test_values)
        if not (u_ds.get_locator()["new_column"] == 1).all():
            raise AssertionError(f"set_metadata failed for {domain}")

        out_file = out_dir / f"{domain}_set_metadata.csv"
        u_ds.save_locator(str(out_file))
        print(f"[eeg_batch_set_column] {domain}: saved -> {out_file}")


if __name__ == "__main__":
    main()
