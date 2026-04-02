import os
import shutil

import h5py
import numpy as np


def _get_group_size(group):
    """
    Calculate the total storage size of datasets under a given group.

    This function recursively iterates through all datasets within the group,
    summing their storage sizes.

    Parameters
    ----------
    group : h5py.Group
        The HDF5 group whose total dataset size we want to calculate.

    Returns
    -------
    int
        The total storage size in bytes of all datasets in this group.
    """
    total_size = 0

    def size_visitor_func(name, obj):
        nonlocal total_size
        if isinstance(obj, h5py.Dataset):
            total_size += obj.id.get_storage_size()

    group.visititems(size_visitor_func)
    return total_size


def _split_v1(input_path, max_file_size, output_dir, base_name):
    """
    Split a v1-format HDF5 file into multiple parts by top-level group.

    The minimal splitting unit is one top-level group (one source file).
    Output files are named ``{base_name}_s1.hdf5``, ``{base_name}_s2.hdf5``, etc.

    Parameters
    ----------
    input_path : str
        Path to the input v1 HDF5 file.
    max_file_size : int
        Maximum size in bytes for each output file.
    output_dir : str
        Directory where output files will be saved.
    base_name : str
        Base name (without extension) for output files.

    Returns
    -------
    list of str
        Paths to the generated HDF5 files.
    """
    with h5py.File(input_path, 'r') as f:
        groups_info = []
        for name in f.keys():
            obj = f[name]
            if isinstance(obj, h5py.Group):
                grp_size = _get_group_size(obj)
            else:
                grp_size = obj.id.get_storage_size()
            groups_info.append((name, grp_size))

    total_size = sum(s for _, s in groups_info)
    if total_size <= max_file_size:
        output_path = os.path.join(output_dir, f"{base_name}_s1.hdf5")
        shutil.copyfile(input_path, output_path)
        return [output_path]

    output_files = []
    current_file_index = 1
    current_output_path = os.path.join(output_dir, f"{base_name}_s{current_file_index}.hdf5")
    current_out_file = h5py.File(current_output_path, 'w')
    current_used_space = 0

    with h5py.File(input_path, 'r') as in_f:
        for grp_name, grp_size in groups_info:
            if current_used_space + grp_size > max_file_size:
                current_out_file.close()
                output_files.append(current_output_path)
                current_file_index += 1
                current_output_path = os.path.join(
                    output_dir, f"{base_name}_s{current_file_index}.hdf5")
                current_out_file = h5py.File(current_output_path, 'w')
                current_used_space = 0

            in_f.copy(grp_name, current_out_file)
            current_used_space += grp_size

    current_out_file.close()
    output_files.append(current_output_path)
    return output_files


def _split_v2(input_path, max_file_size, output_dir, base_name):
    """
    Split a v2-format HDF5 file into multiple parts by source file.

    The minimal splitting unit is all epochs from one source file, keeping
    epochs from the same source together in the same output file.
    Each output file is a valid standalone v2 HDF5 with the same root
    attributes, the global ``label_map``, and only the ``source_meta``
    entries for sources present in that split.

    Size estimation uses the uncompressed epoch data size
    (``n_epochs × n_channels × n_times × 4`` bytes, float32), which is a
    conservative upper bound relative to the gzip-compressed on-disk size.

    Parameters
    ----------
    input_path : str
        Path to the input v2 HDF5 file.
    max_file_size : int
        Maximum size in bytes (estimated uncompressed) for each output file.
    output_dir : str
        Directory where output files will be saved.
    base_name : str
        Base name (without extension) for output files.

    Returns
    -------
    list of str
        Paths to the generated HDF5 files.
    """
    with h5py.File(input_path, 'r') as f:
        n_ch = int(f.attrs['n_channels'])
        n_times = int(f.attrs['n_times'])
        label_map = f.attrs['label_map']       # global JSON string; kept in every split
        root_attrs = dict(f.attrs)             # copy all root attrs

        source_groups_arr = f['epoch_meta/source_group'][:].astype(str)
        event_codes_arr = f['epoch_meta/event_code'][:]

        # Determine unique source files in order of first appearance
        seen = set()
        ordered_sources = []
        for sg in source_groups_arr:
            if sg not in seen:
                seen.add(sg)
                ordered_sources.append(sg)

        # Build per-source index and size estimate
        bytes_per_epoch = n_ch * n_times * 4  # float32, uncompressed
        source_info = {}
        for src in ordered_sources:
            indices = np.where(source_groups_arr == src)[0]
            source_info[src] = {
                'indices': indices,
                'estimated_size': len(indices) * bytes_per_epoch,
            }

        # Pack sources into splits greedily
        splits = []          # list[list[str]]
        current_split = []
        current_size = 0
        for src in ordered_sources:
            sz = source_info[src]['estimated_size']
            if current_split and current_size + sz > max_file_size:
                splits.append(current_split)
                current_split = [src]
                current_size = sz
            else:
                current_split.append(src)
                current_size += sz
        if current_split:
            splits.append(current_split)

        # No split needed — just copy
        if len(splits) == 1:
            output_path = os.path.join(output_dir, f"{base_name}_s1.hdf5")
            shutil.copyfile(input_path, output_path)
            return [output_path]

        # Write each split as a valid v2 file
        output_files = []
        for split_idx, sources_in_split in enumerate(splits, start=1):
            output_path = os.path.join(output_dir, f"{base_name}_s{split_idx}.hdf5")

            # Gather and sort epoch indices for this split
            all_indices = np.concatenate(
                [source_info[src]['indices'] for src in sources_in_split])
            all_indices.sort()
            n_epochs_split = len(all_indices)

            with h5py.File(output_path, 'w') as out_f:
                # ---- Root attributes ----
                for k, v in root_attrs.items():
                    if k == 'n_epochs_total':
                        continue          # written below with correct count
                    out_f.attrs[k] = v
                out_f.attrs['label_map'] = label_map   # global map (Q2: B)
                out_f.attrs['n_epochs_total'] = n_epochs_split

                # ---- data ----
                epoch_data = f['data'][all_indices]    # fancy indexing (sorted)
                out_f.create_dataset(
                    'data',
                    data=epoch_data.astype('float32'),
                    chunks=(1, n_ch, n_times),
                    compression='gzip',
                    compression_opts=1,
                )

                # ---- epoch_meta ----
                em = out_f.create_group('epoch_meta')
                em.create_dataset(
                    'source_group',
                    data=source_groups_arr[all_indices],
                    dtype=h5py.string_dtype(encoding='utf-8'),
                )
                em.create_dataset(
                    'event_code',
                    data=event_codes_arr[all_indices],
                )

                # ---- source_meta (only sources present in this split) ----
                sm_out = out_f.create_group('source_meta')
                sm_in = f['source_meta']
                for src in sources_in_split:
                    if src not in sm_in:
                        continue
                    f.copy(f'source_meta/{src}', sm_out)
                    # Overwrite n_epochs_in_source with the count in this split
                    sm_out[src].attrs['n_epochs_in_source'] = int(
                        len(source_info[src]['indices']))

            output_files.append(output_path)

    return output_files


def split_hdf5_file(input_path, max_file_size=10 * 1024 ** 3, output_dir="."):
    """
    Split an HDF5 file into multiple parts if its total size exceeds the given limit.

    Both v1 (file-per-group) and v2 (flat-array) EEGUnity HDF5 formats are
    supported.  The format is detected automatically from the root ``version``
    attribute.

    **v1 behaviour** — The minimal splitting unit is one top-level group (one
    source file).  Groups are accumulated greedily and a new output file is
    started whenever adding the next group would exceed ``max_file_size``.

    **v2 behaviour** — The minimal splitting unit is all epochs from one
    source file (epochs from the same source are never split across files).
    Size is estimated from uncompressed epoch data
    (``n_epochs × n_channels × n_times × 4`` bytes).  Each output file is a
    fully self-contained v2 HDF5 that shares the global ``label_map`` of the
    source file and contains only the ``source_meta`` entries relevant to that
    split.

    If the file fits within ``max_file_size`` a single ``_s1`` copy is
    returned without modification.

    Parameters
    ----------
    input_path : str
        Path to the input HDF5 file.
    max_file_size : int, optional
        Maximum size in bytes for each output HDF5 file (default is 10 GB).
        For v2 files this is compared against the *uncompressed* epoch data
        size, which is a conservative upper bound.
    output_dir : str, optional
        Directory where output files will be saved. Defaults to the current
        directory.

    Returns
    -------
    list of str
        A list of paths to the generated HDF5 files.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` does not exist.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Detect format from root attribute
    with h5py.File(input_path, 'r') as f:
        version = f.attrs.get('version', None)

    if version is not None and str(version).startswith('2'):
        return _split_v2(input_path, max_file_size, output_dir, base_name)
    else:
        return _split_v1(input_path, max_file_size, output_dir, base_name)
