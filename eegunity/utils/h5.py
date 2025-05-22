import h5py
import numpy as np
from pathlib import Path

class h5Dataset:
    """
   Handle HDF5 file operations in a format compatible with h5py.

    This class is adapted from:
    https://github.com/935963004/LaBraM/blob/main/dataset_maker/shock/utils/h5.py#L8.
    """

    def __init__(self, path: Path, name: str) -> None:
        """
        Initialize the HDF5 dataset handler.

        Parameters
        ----------
        path : Path
            The path to the directory containing the HDF5 file.
        name : str
            The name of the HDF5 file (without extension).
        """
        self.__name = name
        self.__f = h5py.File(path / f'{name}.hdf5', 'a')

    def addGroup(self, grpName: str):
        """
        Add a new group to the HDF5 file.

        Parameters
        ----------
        grpName : str
            The name of the group to create.

        Returns
        -------
        h5py.Group
            The created group object.
        """
        return self.__f.create_group(grpName)

    def addDataset(self, grp: h5py.Group, dsName: str, arr: np.array, chunks: tuple = None, **kwargs):
        """
        Add a dataset to a specified group.

        Parameters
        ----------
        grp : h5py.Group
            The group to which the dataset will be added.
        dsName : str
            The name of the dataset.
        arr : np.array
            The data to store in the dataset.
        chunks : tuple, optional
            The chunk shape to use when storing the dataset.
        **kwargs
            Additional keyword arguments passed to `create_dataset`.

        Returns
        -------
        h5py.Dataset
            The created dataset object.
        """
        return grp.create_dataset(dsName, data=arr, chunks=chunks, **kwargs)

    def addAttributes(self, src: 'h5py.Dataset|h5py.Group', attrName: str, attrValue):
        """
        Add an attribute to a dataset or group.

        Parameters
        ----------
        src : h5py.Dataset or h5py.Group
            The target object to which the attribute will be added.
        attrName : str
            The name of the attribute.
        attrValue : any
            The value of the attribute.
        """
        src.attrs[f'{attrName}'] = attrValue

    def save(self):
        """
        Close the HDF5 file.
        """
        self.__f.close()

    @property
    def name(self):
        """
        Get the name of the HDF5 dataset.

        Returns
        -------
        str
            The name of the HDF5 file.
        """
        return self.__name



