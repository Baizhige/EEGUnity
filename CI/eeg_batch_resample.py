"""CI test for resample."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run resample for downsample and upsample paths."""
    config = load_ci_config()
    out_dir = output_dir(config, "resample")

    for domain in domains(config):
        for sfreq in (100.0, 512.0):
            u_ds = dataset_from_locator(config, domain)
            u_ds.eeg_batch.resample(
                output_path=str(out_dir),
                resample_params={"sfreq": sfreq},
            )
            print(f"[eeg_batch_resample] {domain}: sfreq={sfreq}")


if __name__ == "__main__":
    main()
