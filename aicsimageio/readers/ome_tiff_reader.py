#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union
from xml.etree.ElementTree import ParseError

from fsspec.spec import AbstractFileSystem
from ome_types import from_xml
from ome_types.model.ome import OME
import requests
from tifffile import TiffFile, TiffFileError, TiffTag
import xarray as xr

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .tiff_reader import TiffReader

###############################################################################


class OmeTiffReader(TiffReader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    # Assert that it is a valid OME object
                    return isinstance(from_xml(tiff.pages[0].description), OME)

        # tifffile exception, tifffile exception, ome-types exception
        except (TiffFileError, TypeError, ParseError):
            return False

        # In the case where the OME metadata is not the latest version it will attempt
        # make a request to the remote XSD if this fails we need to catch the error.
        except requests.exceptions.HTTPError:
            raise exceptions.MalformedMetadataError(
                path=path,
                msg="Failed in retrieving the referenced remote OME schema",
            )

    def __init__(self, image: types.PathLike):
        """
        Wraps the tifffile and ome-types APIs to provide the same aicsimageio Reader
        API but for volumetric OME-TIFF images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension = self.path.split(".")[-1]

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path):
            raise exceptions.UnsupportedFileFormatError(self.extension)

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    ome = from_xml(tiff.pages[0].description)
                    self._scenes = tuple(image_meta.id for image_meta in ome.images)

        return self._scenes

    @staticmethod
    def _process_ome_metadata(
        image_description: TiffTag,
        scene_index: int,
    ) -> Tuple[OME, List[str], Dict[str, Union[List, types.ArrayLike]]]:
        """
        Process the image description tag and return the OME metadata object, the
        dimension names, and the coordinates for non spatial dimensions.

        Parameters
        ----------
        image_description: TiffTag
            The ImageDescription TiffTag as read from tifffile.
        scene_index: int
            The current operating scene index to pull metadata from.

        Returns
        -------
        ome: OME
            The metadata read as an OME object from ome-types.
            Will attempt to validate the metadata against the OME schema.
            If the OME schema is not the latest (2016-06), will attempt to validate
            against the remotely referenced XSD.

        # TODO
        ADD RAISES CLAUSE
        """
        # Create OME object from all xml
        ome = from_xml(image_description.value)
        scene_meta = ome.images[scene_index]

        # Create dimension order by getting the current scene's dimension order
        # and reversing it because OME store order vs use order is :shrug:
        dims = [d for d in scene_meta.pixels.dimension_order.value[::-1]]

        # Get coordinate planes
        coords = {}
        coords[DimensionNames.Channel] = [
            channel.name for channel in scene_meta.pixels.channels
        ]

        return ome, dims, coords

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

        # Get unprocessed metadata from tags
        tiff_tags = self._get_tiff_tags(
            fs=self.fs,
            path=self.path,
            scene=self.current_scene_index,
        )

        # Get and unpack OME, dims, and coords
        ome, dims, coords = self._process_ome_metadata(
            image_description=tiff_tags[270],  # image description tag index
            scene_index=self.current_scene_index,
        )

        # Expand image_data for empty dimensions
        ome_shape = []
        for d in dims:
            ome_shape.append(
                getattr(
                    ome.images[self.current_scene_index].pixels, f"size_{d.lower()}"
                )
            )

        # TODO handle Samples dimension attachment for RGB OME TIFF

        expand_dim_ops = []
        for d_size in ome_shape:
            # Add empty dimension where OME requires dimension but not data exists
            if d_size == 1:
                expand_dim_ops.append(None)
            # Add full slice where data exists
            else:
                expand_dim_ops.append(slice(None, None, None))

        # Apply operators to dask array
        image_data = image_data[tuple(expand_dim_ops)]

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tiff_tags,
                constants.METADATA_PROCESSED: ome,
            },
        )

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

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(
                    fs=self.fs,
                    path=self.path,
                    scene=self.current_scene_index,
                )

                # Get and unpack OME, dims, and coords
                ome, dims, coords = self._process_ome_metadata(
                    image_description=tiff_tags[270],  # image description tag index
                    scene_index=self.current_scene_index,
                )

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs={
                        constants.METADATA_UNPROCESSED: tiff_tags,
                        constants.METADATA_PROCESSED: ome,
                    },
                )
