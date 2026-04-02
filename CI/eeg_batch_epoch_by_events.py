"""CI test for process_epochs."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run event-based epoch processing into HDF5."""
    config = load_ci_config()
    out_dir = output_dir(config, "epoch_by_events_hdf5")

    epoch_params = {
        "tmin": 0,
        "tmax": 2,
        "baseline": None,
        "event_repeated": "merge",
    }

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.process_epochs(
            output_path=str(out_dir),
            long_event=False,
            use_hdf5=True,
            epoch_params=epoch_params,
            miss_bad_data=True,
        )
        print(f"[eeg_batch_epoch_by_events] {domain}: completed")


if __name__ == "__main__":
    main()
