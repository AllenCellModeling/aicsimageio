#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import dask.array as da
import numpy as np
import xarray as xr
from fsspec.spec import AbstractFileSystem

from ... import constants, exceptions, transforms, types
from ...dimensions import DEFAULT_DIMENSION_ORDER_LIST, DimensionNames
from ...types import PhysicalPixelSizes
from ...utils import io_utils
from ..reader import Reader
from .sldy_image import SldyImage

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_DATA_FILE_PREFIX = "ImageData"


class SldyReader(Reader):
    """
    Read 3i slidebook (SLDY) images

    Parameters
    ----------
    image : Path or str
        path to file
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    data_file_prefix: str, default = "ImageData"
        Prefix to the data files within the image directories to extract.
        By default this will
        be set to "ImageData". However, specifying "HistogramData" would allow you
        to interact with the image's histogram data instead.

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by this reader.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            SldyReader._get_images_from_data_directory(
                fs, path, kwargs.get("data_file_prefix", DEFAULT_DATA_FILE_PREFIX)
            )
            return True
        except Exception:
            return False

    @staticmethod
    def _get_images_from_data_directory(
        fs: AbstractFileSystem, path: types.PathLike, data_file_prefix: str
    ) -> List[SldyImage]:
        """
        Retrieves list of image acquisitions found at path.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to used for reading.
        path: str
            The path to the file to read.
        data_file_prefix: str
            Prefix to the data files within the image directories to extract.

        Returns
        -------
        images: List[SldyImage]
            A list of the acquisitions found at the path given
            ordered by their names
        """
        data_directory = Path(path).with_suffix(".dir")
        images = [
            SldyImage(fs, image_dir, data_file_prefix=data_file_prefix)
            for image_dir in data_directory.glob("*.imgdir")
        ]
        if not images:
            raise ValueError("Unable to find any images within the image directory")

        # Prevent inconsistent scene (image) ordering
        images.sort(key=lambda img: img.id)
        return images

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
        data_file_prefix: str = DEFAULT_DATA_FILE_PREFIX,
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        try:
            self._images = SldyReader._get_images_from_data_directory(
                self._fs, self._path, data_file_prefix=data_file_prefix
            )
        except Exception:
            # Enforce valid image
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            self._scenes = tuple(image.id for image in self._images)

        return self._scenes

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        image = self._images[self.current_scene_index]
        return PhysicalPixelSizes(
            image.physical_pixel_size_z,
            image.physical_pixel_size_y,
            image.physical_pixel_size_x,
        )

    def _read_delayed(self) -> xr.DataArray:
        image = self._images[self.current_scene_index]

        # If no timepoints or channels available create a single
        # element list of `None` to represent the empty dimension
        timepoints = cast(list, image.timepoints) or [None]
        channels = cast(list, image.channels) or [None]

        # Iterate over each timepoint and channel retreiving data from the
        # image data file (lazily as a delayed read).
        # If no timepoints or channels are available this will fill
        # the otherwise empty Time/Channel dimension
        data_as_list: List[da.Array] = []
        for timepoint in timepoints:
            data_for_timepoint: List[da.Array] = []
            for channel in channels:
                data = image.get_data(
                    timepoint=timepoint, channel=channel, delayed=True
                )
                data_for_timepoint.append(data)

            data_as_list.append(da.stack(data_for_timepoint))

        image_data = da.stack(data_as_list)

        return self._create_data_array(image_data)

    def _read_immediate(self) -> xr.DataArray:
        image = self._images[self.current_scene_index]

        # If no timepoints or channels available create a single
        # element list of `None` to represent the empty dimension
        timepoints = cast(list, image.timepoints) or [None]
        channels = cast(list, image.channels) or [None]

        # Iterate over each timepoint and channel retreiving data from the
        # image data file.
        # If no timepoints or channels are available this will fill
        # the otherwise empty Time/Channel dimension
        data_as_list: List[np.ndarray] = []
        for timepoint in timepoints:
            data_for_timepoint: List[np.ndarray] = []
            for channel in channels:
                data = image.get_data(
                    timepoint=timepoint, channel=channel, delayed=False
                )
                data_for_timepoint.append(data)

            data_as_list.append(np.array(data_for_timepoint))

        image_data = np.array(data_as_list)

        return self._create_data_array(image_data)

    def _create_data_array(self, image_data: types.ArrayLike) -> xr.DataArray:
        """
        Given data representing an image this will create an xr.DataArray
        representation of the image data.

        This makes some assumptions about the dimensions of the image data
        that are based on how the data arrays were constructed in _read_immediate
        and _read_delayed.

        Parameters
        ----------
        image_data: types.ArrayLike
            Array like representation of the image data intended to be wrapped.

        Returns
        -------
        data_array: xr.DataArray
            xarray representation of the image data given
        """
        original_dims = [
            DimensionNames.Time,
            DimensionNames.Channel,
            DimensionNames.SpatialZ,
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
        ]
        intended_dims = DEFAULT_DIMENSION_ORDER_LIST

        # If the original dimensions of the data does not equal the dimensions
        # this needs to output then reshape the data
        if original_dims != intended_dims:
            image_data = transforms.reshape_data(
                data=image_data,
                given_dims="".join(original_dims),
                return_dims="".join(intended_dims),
            )

        return xr.DataArray(
            data=image_data,
            dims=intended_dims,
            coords=self._get_coords(image_data, intended_dims),
            attrs={
                constants.METADATA_UNPROCESSED: self._images[
                    self.current_scene_index
                ].metadata,
            },
        )

    def _get_coords(
        self, image_data: types.ArrayLike, dims: List[str]
    ) -> Dict[str, Any]:
        """
        Given data representing an image and the dimension order
        this will return a dictionary mapping representing the
        coordinates of the dimensions inside the image data.

        Parameters
        ----------
        image_data: types.ArrayLike
            Array like representation of the image data intended to be mapped.
        dims: List[str]
            Order of dimensions present in the image data.

        Returns
        -------
        coords: Dict[str, Any]
            Dictionary mapping dimensions to their coordinates within the given
            image data array
        """
        coords = {}
        image = self._images[self.current_scene_index]
        if image.channels:
            coords[DimensionNames.Channel] = [
                str(channel) for channel in image.channels
            ]

        if image.timepoints:
            timepoint_scale = 1
            coords[DimensionNames.Time] = Reader._generate_coord_array(
                0, len(image.timepoints), timepoint_scale
            )

        if self.physical_pixel_sizes.Z is not None:
            coords[DimensionNames.SpatialZ] = Reader._generate_coord_array(
                0,
                image_data.shape[dims.index(DimensionNames.SpatialZ)],
                self.physical_pixel_sizes.Z,
            )

        # Should never happen due to how SldyImage reads these, but here to typeguard
        if self.physical_pixel_sizes.Y is None or self.physical_pixel_sizes.X is None:
            raise ValueError(
                "Unable to determine physical pixel size of Y and/or X dimension"
            )

        coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
            0,
            image_data.shape[dims.index(DimensionNames.SpatialY)],
            self.physical_pixel_sizes.Y,
        )
        coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
            0,
            image_data.shape[dims.index(DimensionNames.SpatialX)],
            self.physical_pixel_sizes.X,
        )
        return coords
