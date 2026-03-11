"""CI test for align_channel."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Align channels with normal and missing-channel target orders."""
    config = load_ci_config()
    out_dir = output_dir(config, "align_channel")

    channel_order = ["C3", "C4", "Cz"]
    channel_order_miss = ["C3", "C4", "Cz", "AFp6"]

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.format_channel_names()
        u_ds.eeg_batch.align_channel(
            output_path=str(out_dir),
            channel_order=channel_order,
            get_data_row_params={"is_set_channel_type": True},
        )
        print(f"[eeg_batch_align_channel] {domain}: aligned with channel_order")

        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.format_channel_names()
        u_ds.eeg_batch.align_channel(
            output_path=str(out_dir),
            channel_order=channel_order_miss,
            get_data_row_params={"is_set_channel_type": True},
            miss_bad_data=True,
        )
        print(f"[eeg_batch_align_channel] {domain}: aligned with channel_order_miss")


if __name__ == "__main__":
    main()
