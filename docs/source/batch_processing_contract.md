# Batch Processing Contract

This document defines how EEGUnity batch methods interact with source files,
derived files, and the in-memory locator. The locator remains the preferred
metadata source: loading and processing must not silently replace user-maintained
metadata with a fresh directory scan.

## Operation categories

### Derived-dataset operations

Derived-dataset operations transform signals into new files that remain usable
as an EEGUnity dataset. They must replace the in-memory locator with rows that
describe the successfully created files and set `dataset_path` to the output
directory.

Current examples include:

- `filter`
- `resample`
- `normalize`
- `ica`
- `align_channel`
- `save_as_other`

These operations must never modify the original source files. They must build
the new locator from the existing locator rows rather than rescanning the output
directory.

### Metadata operations

Metadata operations inspect data or apply user rules, then update locator
columns without creating replacement EEG files or changing `dataset_path`.

Current examples include:

- `format_channel_names`
- `process_mean_std`
- `get_events`
- `get_meta`
- `infer_units`
- `get_quality`
- `set_metadata`
- `set_column`

### Locator-selection operations

Locator-selection operations only change which rows are active. They do not
write signal files or change row metadata. `sample_filter` is the current
example.

### Export-only operations

Export-only operations create downstream artifacts but do not replace the
active EEGUnity dataset or locator.

Current examples include:

- `export_h5Dataset`
- `epoch_for_pretraining`
- `epoch_by_event_hdf5`
- `epoch_by_segmentation_hdf5`
- `epoch_by_long_event_hdf5`

## Derived locator rules

Every derived-dataset method must follow these rules:

1. Start from a copy of the source locator row so domain, subject, provenance,
   kernel, and user-defined columns survive.
2. Always update `File Path`, `File Type`, and `Completeness Check` for a
   successfully written file.
3. Update only metadata that the operation materially changes.
4. Preserve an existing `type:name` channel schema when the channel layout is
   unchanged.
5. If an operation or kernel changes channel names, order, count, or types,
   regenerate `Channel Names` as MNE-compatible `type:name` entries from the
   final Raw object.
6. Never count a CSV timestamp column as a signal channel. Event columns must
   use a `stim` type, for example `stim:marker`.
7. Replace the active locator only after batch processing has completed. With
   `miss_bad_data=False`, an error aborts the operation and leaves the active
   locator unchanged. With `miss_bad_data=True`, the derived locator contains
   only rows whose output files were created successfully.
8. Set `dataset_path` to the derived output directory and clear stale locator
   path state when the active dataset switches to derived files.
9. Do not rescan the output directory. Rescanning loses user metadata and may
   include unrelated files that were already present.

The fields changed by each operation should be explicit:

| Operation | Required locator updates |
| --- | --- |
| `filter` | Path, file type, completeness; structural signal metadata remains unchanged unless a kernel changed it |
| `normalize` | Path, file type, completeness; structural signal metadata remains unchanged unless a kernel changed it |
| `ica` | Path, file type, completeness; structural signal metadata remains unchanged unless a kernel changed it |
| `resample` | Path, file type, completeness, sampling rate, sample count/data shape, and duration |
| `align_channel` | Path, file type, completeness, channel schema, channel count, and data shape |
| `save_as_other` | Path, target file type, completeness, and target-format structural metadata |

## Implementation status

The 0.8.6 development branch applies this contract through a shared derived-row
update and locator commit path:

- `filter`, `resample`, `save_as_other`, `align_channel`, `normalize`, and
  `ica` build their locator from copied source rows without rescanning.
- Every derived-dataset operation switches `dataset_path` to its output
  directory and clears the stale `locator_path`.
- Typed channel schemas survive operations that leave the layout unchanged.
- Kernel, channel-selection, and alignment changes regenerate typed schemas
  from the final Raw object.
- CSV output excludes timestamps from signal metadata and records event columns
  as `stim:marker` when the locator uses a typed schema.
- `get_meta` persists post-kernel channels as `type:name` and retains the
  separate `Channel Types` column for validation.
- With `miss_bad_data=True`, failed rows are excluded from derived locators.
  With `miss_bad_data=False`, an exception prevents the new locator from being
  committed.

## Required regression coverage

Tests for derived-dataset methods must verify:

- Original source files are not modified.
- Custom locator columns survive successful derivation.
- Typed `type:name` channel schemas survive operations that do not alter the
  channel layout.
- Channel-changing operations regenerate a typed schema from the final Raw.
- Resampling updates sampling rate, duration, and shape without losing channel
  types.
- CSV locator metadata excludes timestamps and uses `stim:marker` for events.
- Unrelated files in the output directory never appear in the new locator.
- Successful and failed row handling follows the documented
  `miss_bad_data` policy.
