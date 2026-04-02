"""CI test for export_h5Dataset."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Validate HDF5 export and basic error paths."""
    config = load_ci_config()

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        out_dir = output_dir(config, "export_h5Dataset", domain)

        export_name = f"{domain}_export"
        export_file = out_dir / f"{export_name}.hdf5"
        if export_file.exists():
            export_file.unlink()

        u_ds.eeg_batch.export_h5Dataset(output_path=str(out_dir), name=export_name, miss_bad_data=True)
        if not export_file.exists():
            raise AssertionError(f"Expected export file not found: {export_file}")

        # output_path does not exist should raise FileNotFoundError
        bad_dir = out_dir / "__missing_dir__"
        if bad_dir.exists():
            raise RuntimeError(f"Unexpected path exists: {bad_dir}")
        try:
            u_ds.eeg_batch.export_h5Dataset(output_path=str(bad_dir), name="bad")
            raise AssertionError("Expected FileNotFoundError for missing output dir")
        except FileNotFoundError:
            pass

        # invalid name type should raise TypeError
        try:
            u_ds.eeg_batch.export_h5Dataset(output_path=str(out_dir), name=12345)
            raise AssertionError("Expected TypeError for non-string name")
        except TypeError:
            pass

        print(f"[eeg_batch_export_h5Dataset] {domain}: completed")


if __name__ == "__main__":
    main()
