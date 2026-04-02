"""CI test for format_channel_names."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Validate channel-name formatting and locator persistence."""
    config = load_ci_config()
    out_dir = output_dir(config, "format_channel_names")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.format_channel_names()

        channel_names = u_ds.get_locator().loc[:, "Channel Names"]
        if not channel_names.notnull().all():
            raise AssertionError(f"Channel Names column contains null values for {domain}")

        out_file = out_dir / f"{domain}_format_channel_names.csv"
        u_ds.save_locator(str(out_file))
        print(f"[eeg_batch_format_channel_names] {domain}: saved -> {out_file}")


if __name__ == "__main__":
    main()
