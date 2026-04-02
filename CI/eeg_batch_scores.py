"""CI test for quality scores."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run both shady and ICA quality scoring."""
    config = load_ci_config()
    out_dir = output_dir(config, "batch_scores")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.get_quality(miss_bad_data=True, method="shady", save_name="shady_scores")
        u_ds.eeg_batch.get_quality(miss_bad_data=True, method="ica", save_name="ica_scores")

        out_file = out_dir / f"{domain}_scores.csv"
        u_ds.save_locator(str(out_file))
        print(f"[eeg_batch_scores] {domain}: saved -> {out_file}")


if __name__ == "__main__":
    main()
