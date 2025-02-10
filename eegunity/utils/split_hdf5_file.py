import os
import h5py
import shutil

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

def split_hdf5_file(input_path, max_file_size=10 * 1024**3, output_dir="."):
    """
    Split an HDF5 file into multiple parts if its total size exceeds the given limit.

    The minimal splitting unit is a top-level group. If the file size surpasses
    the specified max_file_size, this function creates multiple output HDF5 files
    and distributes the top-level groups among them without splitting any single group.
    Output files are named based on the input file's base name, with suffixes like
    `_s1.hdf5`, `_s2.hdf5`, etc.

    Parameters
    ----------
    input_path : str
        Path to the input HDF5 file.
    max_file_size : int, optional
        Maximum size in bytes for each output HDF5 file (default is 10GB).
    output_dir : str, optional
        Directory where output files will be saved. Defaults to the current directory.

    Returns
    -------
    list of str
        A list of paths to the generated HDF5 files.
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Calculate total size and group sizes
    with h5py.File(input_path, 'r') as f:
        # Collect top-level groups and their sizes
        groups_info = []
        for name in f.keys():
            obj = f[name]
            if isinstance(obj, h5py.Group):
                grp_size = _get_group_size(obj)
                groups_info.append((name, grp_size))
            else:
                # If top-level item is a dataset (not group), treat it as a "group-like" unit.
                grp_size = obj.id.get_storage_size()
                groups_info.append((name, grp_size))

    total_size = sum(s for _, s in groups_info)
    if total_size <= max_file_size:
        # Simply copy the original file as is, naming it with _s1 suffix
        output_path = os.path.join(output_dir, f"{base_name}_s1.hdf5")
        shutil.copyfile(input_path, output_path)
        return [output_path]

    # Otherwise, we need to split into multiple files
    output_files = []
    current_file_index = 1
    current_output_path = os.path.join(output_dir, f"{base_name}_s{current_file_index}.hdf5")
    current_out_file = h5py.File(current_output_path, 'w')
    current_used_space = 0

    # We'll re-open the input in read mode for copying groups
    with h5py.File(input_path, 'r') as in_f:
        for grp_name, grp_size in groups_info:
            # If adding this group exceeds the limit, start a new file
            if current_used_space + grp_size > max_file_size:
                current_out_file.close()
                output_files.append(current_output_path)
                current_file_index += 1
                current_output_path = os.path.join(output_dir, f"{base_name}_s{current_file_index}.hdf5")
                current_out_file = h5py.File(current_output_path, 'w')
                current_used_space = 0

            # Copy the group to the current output file
            in_f.copy(grp_name, current_out_file)
            current_used_space += grp_size

    current_out_file.close()
    output_files.append(current_output_path)

    return output_files


