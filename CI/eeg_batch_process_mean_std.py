"""CI test for process_mean_std."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run process_mean_std for domain and per-file modes."""
    config = load_ci_config()
    out_dir = output_dir(config, "process_mean_std")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.process_mean_std(domain_mean=True)
        u_ds.eeg_batch.process_mean_std(domain_mean=False)

        out_file = out_dir / f"{domain}_mean_std.csv"
        u_ds.save_locator(str(out_file))
        print(f"[eeg_batch_process_mean_std] {domain}: saved -> {out_file}")


if __name__ == "__main__":
    main()
