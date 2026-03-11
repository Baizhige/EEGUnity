"""CI test for hash generation."""

from __future__ import annotations

from ci_runtime import dataset_from_locator, domains, load_ci_config, output_dir


def main() -> None:
    """Run source/data hash generation and persist locator."""
    config = load_ci_config()
    out_dir = output_dir(config, "batch_hash")

    for domain in domains(config):
        u_ds = dataset_from_locator(config, domain)
        u_ds.eeg_batch.get_file_hashes(data_stream=False)
        u_ds.eeg_batch.get_file_hashes(data_stream=True)

        out_file = out_dir / f"{domain}_hashes.csv"
        u_ds.save_locator(str(out_file))
        print(f"[eeg_batch_get_hash] {domain}: saved -> {out_file}")


if __name__ == "__main__":
    main()
