#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np

from .. import exceptions, transforms, types
from ..constants import Dimensions

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# Global variable to inform reader to use _read_delayed or _read_immediate
# Useful in sitations where you may be on a cluster with very few workers
# but still using Dask for other operations.
# I.E. GPU clusters where you don't care about the IO of the file so much as the
# training and application of a model
USE_DASK = True


def use_dask(setting: bool):
    """
    Enable or disable Dask for image reading.
    When True, image reads are first attempted to be handled by a distributed cluster.
    When False, image reads are never routed to a distributed cluster and are instead
    read immediately in the running process.
    """
    global USE_DASK

    # Check parameter
    if not isinstance(setting, bool):
        raise TypeError("The setting parameter provided to use_dask must be a boolean.")

    # Assign to global state
    USE_DASK = setting


###############################################################################


class Reader(ABC):
    _dask_data = None
    _data = None
    _dims = None
    _metadata = None

    @staticmethod
    def _resolve_image_path(img: Union[str, Path]) -> Path:
        # Convert pathlike to Path
        if isinstance(img, (str, Path)):
            # Strictly do not fully resolve the path because Mac is bad with mounted
            # drives
            img = Path(img).expanduser()

            # Check the file exists
            if not img.exists():
                raise FileNotFoundError(img)

            # Check path
            if img.is_dir():
                raise IsADirectoryError(
                    f"Please provide a single file to the `img` parameter. "
                    f"Received directory: {img}"
                )

        # Check that no other type was provided
        if not isinstance(img, Path):
            raise TypeError(
                f"Please provide a path to a file as a string, or an pathlib.Path, to "
                f"the `img` parameter. "
                f"Received type: {type(img)}"
            )

        return img

    def __init__(
        self, file: types.ImageLike, dask_kwargs: Dict[str, Any] = {}, **kwargs
    ):
        # This will both fully expand and enforce that the filepath exists
        file = self._resolve_image_path(file)

        # Check type
        if not self.is_this_type(file):
            raise exceptions.UnsupportedFileFormatError(
                f"Reader does not support file or object: {file}"
            )

        # Store this filepath
        self._file = file

        # Store dask client and cluster setup
        self._dask_kwargs = dask_kwargs
        self._client = None
        self._cluster = None

    @staticmethod
    def guess_dim_order(shape: Tuple[int]) -> str:
        return Dimensions.DefaultOrder[len(Dimensions.DefaultOrder) - len(shape) :]

    @classmethod
    def is_this_type(cls, data: types.ImageLike) -> bool:
        # Check path
        if isinstance(data, (str, Path)):
            # Resolve image path
            f = cls._resolve_image_path(data)

            # Return and close the open pointer
            with open(f, "rb") as read_bytes:
                return cls._is_this_type(read_bytes)

        # Convert bytes to BytesIO
        if isinstance(data, bytes):
            data = io.BytesIO(data)

        # Check type
        if isinstance(data, io.BytesIO):
            return cls._is_this_type(data)

        # Special cases
        if isinstance(data, (np.ndarray, da.core.Array)):
            return cls._is_this_type(data)

        # Raise because none of the above returned
        raise TypeError(
            f"Reader only accepts types: "
            f"[str, pathlib.Path, bytes, io.BytesIO, numpy or dask array]. "
            f"Received: {type(data)}"
        )

    @staticmethod
    @abstractmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        pass

    @abstractmethod
    def _read_delayed(self) -> da.core.Array:
        pass

    @abstractmethod
    def _read_immediate(self) -> np.ndarray:
        pass

    @property
    def dask_data(self) -> da.core.Array:
        if self._dask_data is None:
            self._dask_data = self._read_delayed()

        return self._dask_data

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            try:
                # Fast re-route to _read_immediate
                if not USE_DASK:
                    raise ValueError("USE_DASK marked False. Rerouting.")

                # Otherwise, assume the user want's to use their Dask cluster
                # if the following `get_client` call succeeds

                # These lines both check if distributed has been imported
                # and if a client connection has been created
                # If distributed hasn't been imported it will KeyError
                # If no client has been created it will ValueError
                # We can't import distributed due to it requiring network utilities:
                # https://github.com/AllenCellModeling/aicsimageio/issues/82
                sys.modules["distributed"].get_client()

                # No error means there is a cluster and client
                # available on this worker process
                # Use delayed dask reader
                self._dask_data = self._read_delayed()
                self._data = self._dask_data.compute()
            except (KeyError, ValueError):
                self._data = self._read_immediate()
                self._dask_data = da.from_array(self._data)

        return self._data

    def get_image_dask_data(
        self, out_orientation: Optional[str] = None, **kwargs
    ) -> da.core.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims`

        kwargs:
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the out_orientation.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indicies
              desired. D should be present in the out_orientation.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indicies
              desired. D should be present in the out_orientation.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the out_orientation.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the out_orientation.

        Returns
        -------
        data: dask array
            The read data with the dimension ordering that was specified with
            out_orientation.

        Examples
        --------
        Specific index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_dask_data("ZYX", C=1)

        List of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_dask_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_dask_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_dask_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if out_orientation is None:
            return self.dask_data

        # Transform and return
        return transforms.reshape_data(
            data=self.dask_data,
            given_dims=self.dims,
            return_dims=out_orientation,
            **kwargs,
        )

    def get_image_data(
        self, out_orientation: Optional[str] = None, **kwargs
    ) -> np.ndarray:
        """
        Read the image as a numpy array then return specific dimension image data.

        Parameters
        ----------
        out_orientation: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The current image dimensions. i.e. `self.dims`

        kwargs:
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the out_orientation.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indicies
              desired. D should be present in the out_orientation.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indicies
              desired. D should be present in the out_orientation.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the out_orientation.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the out_orientation.

        Examples
        --------
        Specific index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... c1 = img.get_image_data("ZYX", C=1)

        List of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_second = img.get_image_data("CZYX", C=[0, 1])

        Tuple of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_and_last = img.get_image_data("CZYX", C=(0, -1))

        Range of index selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... first_three = img.get_image_data("CZYX", C=range(3))

        Slice selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_data("CZYX", C=slice(0, -1, 2))

        Returns
        -------
        data: np.ndarray
            The read data with the dimension ordering that was specified with
            out_orientation.

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.
        * This will preload the entire image before returning the requested data.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if out_orientation is None:
            return self.data

        # Transform and return
        return transforms.reshape_data(
            data=self.data,
            given_dims=self.dims,
            return_dims=out_orientation,
            **kwargs,
        )

    @property
    @abstractmethod
    def dims(self) -> str:
        pass

    def size(self, dims: str = Dimensions.DefaultOrder) -> Tuple[int]:
        """
        Parameters
        ----------
        dims: str
            A string containing a list of dimensions being requested. The default is to
            return the six standard dims.

        Returns
        -------
        size: Tuple[int]
            A tuple with the requested dimensions filled in.
        """
        # Ensure dims is an uppercase string
        dims = dims.upper()

        # Check that the dims requested are in the image dims
        if not (all(d in self.dims for d in dims)):
            raise exceptions.InvalidDimensionOrderingError(
                f"Invalid dimensions requested: {dims}"
            )

        # Return the shape of the data for the dimensions requested
        return tuple([self.dask_data.shape[self.dims.index(dim)] for dim in dims])

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns
        -------
        shape: Tuple[int]
            A tuple with the size of all dimensions.
        """
        return self.dask_data.shape

    @property
    @abstractmethod
    def metadata(self) -> Any:
        pass

    def get_physical_pixel_size(self, scene: int = 0) -> Tuple[float]:
        """
        Attempts to retrieve physical pixel size for the specified scene.
        If none available, returns `1.0` for each spatial dimension.

        Parameters
        ----------
        scene: int
            The index of the scene for which to return physical pixel sizes.

        Returns
        -------
        sizes: Tuple[float]
            Tuple of floats representing the pixel sizes for X, Y, Z, in that order.
        """
        return (1.0, 1.0, 1.0)

    def get_channel_names(self, scene: int = 0) -> Optional[List[str]]:
        """
        Attempts to use the image's metadata to get the image's channel names.

        Parameters
        ----------
        scene: int
            The index of the scene for which to return channel names.

        Returns
        -------
        channels_names: Optional[List[str]]
            List of strings representing the channel names.
            If channel dimension not present in file, return None.
        """
        # Check for channels dimension
        if Dimensions.Channel not in self.dims:
            return None

        # Channel dimension in reader data, get default channel names
        channel_index = self.dims.index(Dimensions.Channel)
        channel_dim_size = self.dask_data.shape[channel_index]
        return [str(i) for i in range(channel_dim_size)]

    @property
    def cluster(self) -> Optional["distributed.LocalCluster"]:  # noqa: F821
        return self._cluster

    @property
    def client(self) -> Optional["distributed.Client"]:  # noqa: F821
        return self._client

    def close(self):
        """
        Always close the Dask Client connection.
        If connected to *strictly* a LocalCluster, close it down as well.
        """
        from .. import dask_utils

        self._cluster, self._client = dask_utils.shutdown_cluster_and_client(
            self.cluster, self.client
        )

    def __enter__(self):
        """
        If provided an address, create a Dask Client connection.
        If not provided an address, create a LocalCluster and Client connection.
        If not provided an address, other Dask kwargs are accepted and passed down to
        the LocalCluster object.
        """
        from .. import dask_utils

        self._cluster, self._client = dask_utils.spawn_cluster_and_client(
            **self._dask_kwargs
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Always close the Dask Client connection.
        If connected to *strictly* a LocalCluster, close it down as well.
        """
        self.close()
