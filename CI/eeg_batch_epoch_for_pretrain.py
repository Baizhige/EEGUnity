"""CI test for epoch_for_pretraining."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run epoch_for_pretraining with representative options."""
    config = load_ci_config()
    out_dir = output_dir(config, "epoch_for_pretraining")

    for domain in domains(config):
        cases = [
            {"seg_sec": 2.0},
            {"seg_sec": 2.0, "overlap": 0.5},
            {"seg_sec": 2.0, "resample": 100},
            {"seg_sec": 2.0, "exclude_bad": True},
            {"seg_sec": 2.0, "baseline": (None, 0.2)},
            {"seg_sec": 2.0, "miss_bad_data": True},
        ]

        for params in cases:
            u_ds = dataset_from_locator(config, domain)
            u_ds.eeg_batch.epoch_for_pretraining(output_path=str(out_dir), **params)
            print(f"[eeg_batch_epoch_for_pretrain] {domain}: params={params}")


if __name__ == "__main__":
    main()
