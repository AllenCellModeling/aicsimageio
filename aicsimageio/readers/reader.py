#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from fsspec.spec import AbstractFileSystem

from .. import constants, transforms, types
from ..dimensions import DEFAULT_DIMENSION_ORDER, DimensionNames, Dimensions
from ..types import PhysicalPixelSizes
from ..utils import io_utils

###############################################################################


class Reader(ABC):
    _xarray_dask_data: Optional[xr.DataArray] = None
    _xarray_data: Optional[xr.DataArray] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene: Optional[str] = None
    fs: AbstractFileSystem
    path: str

    @staticmethod
    @abstractmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        """
        The per-Reader implementation used to validate that an image is supported or not
        by the Reader itself.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to used for reading.
        path: str
            The path to the file to read.
        kwargs: Any
            Any kwargs used for reading and validation of the file.

        Returns
        -------
        supported: bool
            Boolean value indicating if the file is supported by the reader.
        """
        pass

    @classmethod
    def is_supported_image(cls, image: types.ImageLike, **kwargs: Any) -> bool:
        """
        Asserts that the provided image like object is supported by the current Reader.

        Parameters
        ----------
        image: types.ImageLike
            The filepath or array to validate as a supported type.
        kwargs: Any
            Any kwargs used for reading and validation of the file.

        Returns
        -------
        supported: bool
            Boolean indicated if the provided data is or is not supported by the
            current Reader.

        Raises
        ------
        TypeError: Invalid type provided to image parameter.
        """
        # Check path
        if isinstance(image, (str, Path)):
            # Expand details of provided image
            fs, path = io_utils.pathlike_to_fs(image, enforce_exists=True)

            return cls._is_supported_image(fs, path, **kwargs)

        # Special cases
        if isinstance(image, (np.ndarray, da.core.Array)):
            return cls._is_supported_image(image, **kwargs)

        # Raise because none of the above returned
        raise TypeError(
            f"Reader only accepts types: {types.ImageLike}. Received: '{type(image)}'."
        )

    def __init__(self, image: types.ImageLike, **kwargs: Any):
        """
        A small class to build standardized image reader objects that deal with the raw
        image and metadata.

        Parameters
        ----------
        image: types.ImageLike
            The filepath or array to read.

        Notes
        -----
        While this base class accepts image as any "ImageLike", it is up to the
        implementer of the Reader to decide which types they would like to accept
        (certain readers may not support buffers for example).
        """
        pass

    @staticmethod
    def _guess_dim_order(shape: Tuple[int, ...]) -> str:
        """
        Given an image shape attempts to guess the dimension order.

        Can and should be overwritten by inhereting Reader classes to make more
        informed guesses based off the format when possible.

        Parameters
        ----------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.

        Returns
        -------
        dim_order: str
            The guessed dimension order.
        """
        return DEFAULT_DIMENSION_ORDER[len(DEFAULT_DIMENSION_ORDER) - len(shape) :]

    @property
    @abstractmethod
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """
        pass

    @property
    def current_scene(self) -> str:
        """
        Returns
        -------
        scene: str
            The current operating scene.
        """
        if self._current_scene is None:
            self._current_scene = self.scenes[0]

        return self._current_scene

    @property
    def current_scene_index(self) -> int:
        """
        Returns
        -------
        scene_index: int
            The current operating scene index in the file.
        """
        return self.scenes.index(self.current_scene)

    def set_scene(self, scene_id: str) -> None:
        """
        Set the operating scene.

        Parameters
        ----------
        scene_id: str
            The scene id to set as the operating scene.

        Raises
        ------
        IndexError: the provided scene id is not found in the available scene id list
        """
        # Only need to run when the scene id is different from current scene
        if scene_id != self.current_scene:

            # Validate scene id
            if scene_id not in self.scenes:
                raise IndexError(
                    f"Scene id: {scene_id} "
                    f"is not present in available image scenes: {self.scenes}"
                )

            # Update current scene
            self._current_scene = scene_id

            # Reset the data stored in the Reader object
            self._xarray_dask_data = None
            self._xarray_data = None
            self._dims = None
            self._metadata = None

    @abstractmethod
    def _read_delayed(self) -> xr.DataArray:
        """
        The delayed data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully constructed delayed DataArray.

            It is additionally recommended to closely monitor how dask array chunks are
            managed.

        Notes
        -----
        Requirements for the returned xr.DataArray:
        * Must have the `dims` populated.
        * If a channel dimension is present, please populate the channel dimensions
        coordinate array the respective channel coordinate values.
        """
        pass

    @abstractmethod
    def _read_immediate(self) -> xr.DataArray:
        """
        The immediate data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully read data array.

        Notes
        -----
        Requirements for the returned xr.DataArray:
        * Must have the `dims` populated.
        * If a channel dimension is present, please populate the channel dimensions
        coordinate array the respective channel coordinate values.
        """
        pass

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed image and metadata as an annotated data array.
        """
        if self._xarray_dask_data is None:
            self._xarray_dask_data = self._read_delayed()

        return self._xarray_dask_data

    @property
    def xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_data: xr.DataArray
            The fully read image and metadata as an annotated data array.
        """
        if self._xarray_data is None:
            self._xarray_data = self._read_immediate()

            # Remake the delayed xarray dataarray object using a rechunked dask array
            # from the just retrieved in-memory xarray dataarray
            self._xarray_dask_data = xr.DataArray(
                da.from_array(self._xarray_data.data),
                dims=self._xarray_data.dims,
                coords=self._xarray_data.coords,
                attrs=self._xarray_data.attrs,
            )

        return self._xarray_data

    @property
    def dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The image as a dask array with the native dimension ordering.
        """
        return self.xarray_dask_data.data

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The image as a numpy array with native dimension ordering.
        """
        return self.xarray_data.data

    @property
    def dtype(self) -> np.dtype:
        """
        Returns
        -------
        dtype: np.dtype
            Data-type of the image array's elements.
        """
        return self.xarray_dask_data.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.
        """
        return self.xarray_dask_data.shape

    @property
    def dims(self) -> Dimensions:
        """
        Returns
        -------
        dims: Dimensions
            Object with the paired dimension names and their sizes.
        """
        if self._dims is None:
            self._dims = Dimensions(dims=self.xarray_dask_data.dims, shape=self.shape)

        return self._dims

    def get_image_dask_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> da.Array:
        """
        Get specific dimension image data out of an image as a dask array.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The natural image dimension order.

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

        Returns
        -------
        data: dask array
            The image data with the specified dimension ordering.

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
        if dimension_order_out is None:
            return self.dask_data

        # Transform and return
        return transforms.reshape_data(
            data=self.dask_data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

    def get_image_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> da.Array:
        """
        Read the image as a numpy array then return specific dimension image data.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            A string containing the dimension ordering desired for the returned ndarray.
            Default: The natural image dimension order.

        kwargs: Any
            * C=1: specifies Channel 1
            * T=3: specifies the fourth index in T
            * D=n: D is Dimension letter and n is the index desired. D should not be
              present in the dimension_order_out.
            * D=[a, b, c]: D is Dimension letter and a, b, c is the list of indices
              desired. D should be present in the dimension_order_out.
            * D=(a, b, c): D is Dimension letter and a, b, c is the tuple of indices
              desired. D should be present in the dimension_order_out.
            * D=range(...): D is Dimension letter and range is the standard Python
              range function. D should be present in the dimension_order_out.
            * D=slice(...): D is Dimension letter and slice is the standard Python
              slice function. D should be present in the dimension_order_out.

        Returns
        -------
        data: numpy array
            The image data with the specified dimension ordering.

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
        ... first_three = img.get_image_dask_data("CZYX", C=range(3))

        Slice selection

        >>> img = Reader("s_1_t_1_c_10_z_20.ome.tiff")
        ... every_other = img.get_image_data("CZYX", C=slice(0, -1, 2))

        Notes
        -----
        * If a requested dimension is not present in the data the dimension is
          added with a depth of 1.
        * This will preload the entire image before returning the requested data.

        See `aicsimageio.transforms.reshape_data` for more details.
        """
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None:
            return self.data

        # Transform and return
        return transforms.reshape_data(
            data=self.data,
            given_dims=self.dims.order,
            return_dims=dimension_order_out,
            **kwargs,
        )

    @property
    def metadata(self) -> Any:
        """
        Returns
        -------
        metadata: Any
            The metadata for the formats supported by the inhereting Reader.

            If the inheriting Reader supports processing the metadata into a more useful
            format / Python object, this will return the result.

            For both the unprocessed and processed metadata from the file, use
            `xarray_dask_data.attrs` which will contain a dictionary with keys:
            `unprocessed` and `processed` that you can then select.
        """
        if self._metadata is None:
            if constants.METADATA_PROCESSED in self.xarray_dask_data.attrs:
                self._metadata = self.xarray_dask_data.attrs[
                    constants.METADATA_PROCESSED
                ]
            else:
                self._metadata = self.xarray_dask_data.attrs[
                    constants.METADATA_UNPROCESSED
                ]

        return self._metadata

    @property
    def channel_names(self) -> Optional[List[str]]:
        """
        Returns
        -------
        channel_names: List[str]
            Using available metadata, the list of strings representing channel names.
            If no channel dimension present in the data, returns None.
        """
        if DimensionNames.Channel in self.xarray_dask_data.dims:
            return list(self.xarray_dask_data[DimensionNames.Channel].values)

        return None

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return PhysicalPixelSizes(1.0, 1.0, 1.0)

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"[Image-is-in-Memory: {self._xarray_data is not None}]>"
        )

    def __repr__(self) -> str:
        return str(self)
