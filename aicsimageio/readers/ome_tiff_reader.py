#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Union

import numpy as np
import xarray as xr
from fsspec.spec import AbstractFileSystem
from ome_types import from_xml
from ome_types.model.ome import OME
from tifffile import TiffFile, TiffFileError, TiffTag

from .. import constants, exceptions, transforms, types
from ..dimensions import DEFAULT_CHUNK_BY_DIMS, DEFAULT_DIMENSION_ORDER, DimensionNames
from ..metadata import utils as metadata_utils
from ..types import PhysicalPixelSizes
from ..utils import io_utils
from .tiff_reader import TiffReader

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


class OmeTiffReader(TiffReader):
    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            ome_xml = metadata_utils.clean_ome_xml_for_known_issues(ome_xml)

        return from_xml(ome_xml)

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, clean_metadata: bool = True, **kwargs
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    xml = tiff.pages[0].description
                    return OmeTiffReader._get_ome(xml, clean_metadata)

        # tifffile exception, tifffile exception, ome-types / etree exception
        except (TiffFileError, TypeError, ET.ParseError):
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_by_dims: List[str] = DEFAULT_CHUNK_BY_DIMS,
        clean_metadata: bool = True,
    ):
        """
        Wraps the tifffile and ome-types APIs to provide the same aicsimageio Reader
        API but for volumetric OME-TIFF images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        chunk_by_dims: List[str]
            Which dimensions to create chunks for.
            Default: DEFAULT_CHUNK_BY_DIMS
            Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
            will always be added to the list if not present during dask array
            construction.
        clean_metadata: bool
            Should the OME XML metadata found in the file be cleaned for known
            AICSImageIO 3.x and earlier created errors.
            Default: True (Clean the metadata for known errors)

        Notes
        -----
        If the OME metadata in your file isn't OME schema compilant or does not validate
        this will fail to read your file and raise an exception.

        If the OME metadata in your file doesn't use the latest OME schema (2016-06),
        this reader will make a request to the referenced remote OME schema to validate.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Store params
        self.chunk_by_dims = chunk_by_dims
        self.clean_metadata = clean_metadata

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path, clean_metadata):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self.path
            )

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    self._ome = self._get_ome(
                        tiff.pages[0].description, self.clean_metadata
                    )
                    self._scenes = tuple(
                        image_meta.id for image_meta in self._ome.images
                    )

        return self._scenes

    @staticmethod
    def _get_dims_and_coords_from_ome(
        ome: TiffTag,
        scene_index: int,
    ) -> Tuple[List[str], Dict[str, Union[List, types.ArrayLike]]]:
        """
        Process the OME metadata to retrieve the dimension names and coordinate planes.

        Parameters
        ----------
        ome: OME
            A constructed OME object to retrieve data from.
        scene_index: int
            The current operating scene index to pull metadata from.

        Returns
        -------
        dims: List[str]
            The dimension names pulled from the OME metadata.
        coords: Dict[str, Union[List, types.ArrayLike]]
            The coordinate planes / data for each dimension.
        """
        # Select scene
        scene_meta = ome.images[scene_index]

        # Create dimension order by getting the current scene's dimension order
        # and reversing it because OME store order vs use order is :shrug:
        dims = [d for d in scene_meta.pixels.dimension_order.value[::-1]]

        # Get coordinate planes
        coords = {}

        # Channels
        # Channel name isn't required by OME spec, so try to use it but
        # roll back to ID if not found
        coords[DimensionNames.Channel] = [
            channel.name if channel.name is not None else channel.id
            for channel in scene_meta.pixels.channels
        ]

        # Time
        # If global linear timescale we can np.linspace with metadata
        if scene_meta.pixels.time_increment is not None:
            coords[DimensionNames.Time] = np.linspace(
                0,
                scene_meta.pixels.time_increment_quantity,
                scene_meta.pixels.time_increment,
            )
        # If non global linear timescale, we need to create an array of every plane
        # time value
        elif scene_meta.pixels.size_t > 1:
            t_index_to_delta_map = {
                p.the_t: p.delta_t for p in scene_meta.pixels.planes
            }
            coords[DimensionNames.Time] = list(t_index_to_delta_map.values())

        # Handle Spatial Dimensions
        if scene_meta.pixels.physical_size_z is not None:
            coords[DimensionNames.SpatialZ] = np.arange(
                0,
                scene_meta.pixels.size_z * scene_meta.pixels.physical_size_z,
                scene_meta.pixels.physical_size_z,
            )

        if scene_meta.pixels.physical_size_y is not None:
            coords[DimensionNames.SpatialY] = np.arange(
                0,
                scene_meta.pixels.size_y * scene_meta.pixels.physical_size_y,
                scene_meta.pixels.physical_size_y,
            )

        if scene_meta.pixels.physical_size_x is not None:
            coords[DimensionNames.SpatialX] = np.arange(
                0,
                scene_meta.pixels.size_x * scene_meta.pixels.physical_size_x,
                scene_meta.pixels.physical_size_x,
            )

        return dims, coords

    @staticmethod
    def _expand_dims_to_match_ome(
        image_data: types.ArrayLike,
        ome: OME,
        dims: List[str],
        scene_index: int,
    ) -> types.ArrayLike:
        # Expand image_data for empty dimensions
        ome_shape = []
        for d in dims:
            ome_shape.append(
                getattr(ome.images[scene_index].pixels, f"size_{d.lower()}")
            )

        # Check for num samples and expand dims if greater than 1
        n_samples = ome.images[scene_index].pixels.channels[0].samples_per_pixel
        if n_samples > 1:
            # Append to the end, i.e. the last dimension
            dims.append("S")
            ome_shape.append(n_samples)

        # The file may not have all the data but OME requires certain dimensions
        # expand to fill
        expand_dim_ops = []
        for d_size in ome_shape:
            # Add empty dimension where OME requires dimension but no data exists
            if d_size == 1:
                expand_dim_ops.append(None)
            # Add full slice where data exists
            else:
                expand_dim_ops.append(slice(None, None, None))

        # Apply operators to dask array
        return image_data[tuple(expand_dim_ops)]

    def _general_data_array_constructor(
        self, image_data: types.ArrayLike
    ) -> xr.DataArray:
        # Get unprocessed metadata from tags
        tiff_tags = self._get_tiff_tags()

        # Unpack dims and coords from OME
        dims, coords = self._get_dims_and_coords_from_ome(
            ome=self._ome,
            scene_index=self.current_scene_index,
        )

        # Expand the image data to match the OME empty dimensions
        image_data = self._expand_dims_to_match_ome(
            image_data=image_data,
            ome=self._ome,
            dims=dims,
            scene_index=self.current_scene_index,
        )

        # Always order array
        if DimensionNames.Samples in dims:
            out_order = f"{DEFAULT_DIMENSION_ORDER}{DimensionNames.Samples}"
        else:
            out_order = DEFAULT_DIMENSION_ORDER

        # Transform into order
        image_data = transforms.reshape_data(
            image_data,
            "".join(dims),
            out_order,
        )

        # Reset dims after transform
        dims = [d for d in out_order]

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tiff_tags,
                constants.METADATA_PROCESSED: self._ome,
            },
        )

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs contains
            unprocessed tags and processed OME object.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        # Create the delayed dask array
        image_data = self._create_dask_array()

        return self._general_data_array_constructor(image_data)

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs
            contains unprocessed tags and processed OME object.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                return self._general_data_array_constructor(image_data)

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
        z = self.metadata.images[self.current_scene_index].pixels.physical_size_z
        y = self.metadata.images[self.current_scene_index].pixels.physical_size_y
        x = self.metadata.images[self.current_scene_index].pixels.physical_size_x

        return PhysicalPixelSizes(
            z if z is not None else 1.0,
            y if y is not None else 1.0,
            x if x is not None else 1.0,
        )
