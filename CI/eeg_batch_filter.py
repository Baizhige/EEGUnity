"""CI test for filter."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run filter with representative filter modes."""
    config = load_ci_config()
    out_dir = output_dir(config, "filter")

    for domain in domains(config):
        cases = [
            {"filter_type": "bandpass", "l_freq": 1.0, "h_freq": 40.0},
            {"filter_type": "lowpass", "h_freq": 40.0},
            {"filter_type": "highpass", "l_freq": 1.0},
            {"filter_type": "notch", "notch_freq": 50.0},
            {"filter_type": "bandpass", "l_freq": 0.1, "h_freq": 256.0, "auto_adjust_h_freq": True},
            {"miss_bad_data": True},
        ]

        for params in cases:
            u_ds = dataset_from_locator(config, domain)
            u_ds.eeg_batch.filter(output_path=str(out_dir), **params)
            print(f"[eeg_batch_filter] {domain}: params={params}")


if __name__ == "__main__":
    main()
