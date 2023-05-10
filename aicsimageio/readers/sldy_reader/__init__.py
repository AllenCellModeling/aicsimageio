#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dask.array as da
import numpy as np
import xarray as xr
import yaml
from fsspec.spec import AbstractFileSystem

from ... import constants, exceptions, transforms, types
from ...dimensions import DEFAULT_DIMENSION_ORDER, DimensionNames
from ...metadata import utils as metadata_utils
from ...types import PhysicalPixelSizes
from ...utils import io_utils
from ..reader import Reader
from .sldy_image import SldyImage

###############################################################################

log = logging.getLogger(__name__)
yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, SldyImage.yaml_mapping, yaml.Loader
)

###############################################################################

# TODO: Add docstrings
# TODO: Make sure dims are ZYX order not XYZ or something else
# TODO: Test memory pulling in spatial dims


class SldyReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            SldyReader._get_images_from_data_directory(fs, path)
            return True
        except Exception:
            return False

    @staticmethod
    def _get_images_from_data_directory(
        fs: AbstractFileSystem, path: types.PathLike, data_file_prefix: str
    ) -> List[SldyImage]:
        data_directory = Path(path).with_suffix(".dir")
        images = [
            SldyImage(fs, image_dir, data_file_prefix=data_file_prefix)
            for image_dir in data_directory.glob("*.imgdir")
        ]

        # Prevent inconsistent scene (image) ordering
        images.sort(key=lambda img: img.id)
        return images

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
        data_file_prefix="ImageData",
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

        self._scenes = tuple(image.id for image in self._images)

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

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
        image = self._images[self.current_scene_index]
        return PhysicalPixelSizes(
            image.physical_pixel_size_z,
            image.physical_pixel_size_y,
            image.physical_pixel_size_x,
        )

    def _read_delayed(self) -> xr.DataArray:
        # Iterate over each timepoint and channel retreiving data from the
        # image data file (lazily as a delayed read).
        # If no timepoints or channels are available this will fill
        # the otherwise empty Time/Channel dimension
        data_as_list: List[da.Array] = []
        image = self._images[self.current_scene_index]
        for timepoint in image.timepoints or [None]:
            data_for_timepoint: List[da.Array] = []
            for channel in image.channels or [None]:
                data = image.get_data(
                    timepoint=timepoint, channel=channel, delayed=True
                )
                data_for_timepoint.append(data)

            data_as_list.append(da.stack(data_for_timepoint))

        image_data = da.stack(data_as_list)

        return self._create_data_array(image_data)

    def _read_immediate(self) -> xr.DataArray:
        # Iterate over each timepoint and channel retreiving data from the
        # image data file (lazily as a delayed read).
        # If no timepoints or channels are available this will fill
        # the otherwise empty Time/Channel dimension
        data_as_list = []
        image = self._images[self.current_scene_index]
        for timepoint in image.timepoints or [None]:
            data_for_timepoint = []
            for channel in image.channels or [None]:
                data = image.get_data(
                    timepoint=timepoint, channel=channel, delayed=False
                )
                data_for_timepoint.append(data)

            data_as_list.append(np.array(data_for_timepoint))

        image_data = np.array(data_as_list)

        return self._create_data_array(image_data)

    def _create_data_array(self, image_data: types.ArrayLike) -> xr.DataArray:
        original_dims = (
            DimensionNames.Time
            + DimensionNames.Channel
            + DimensionNames.SpatialZ
            + DimensionNames.SpatialY
            + DimensionNames.SpatialX
        )
        intended_dims = DEFAULT_DIMENSION_ORDER

        # If the original dimensions of the data does not equal the dimensions
        # this needs to output then reshape the data
        if original_dims != intended_dims:
            image_data = transforms.reshape_data(
                data=image_data,
                given_dims=original_dims,
                return_dims=intended_dims,
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

    def _get_coords(self, image_data: types.ArrayLike, dims: str) -> Dict[str, Any]:
        coords = {}
        image = self._images[self.current_scene_index]
        if image.channels:
            coords[DimensionNames.Channel] = [
                metadata_utils.generate_ome_channel_id(
                    image_id=image.id, channel_id=channel
                )
                for channel in image.channels
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
