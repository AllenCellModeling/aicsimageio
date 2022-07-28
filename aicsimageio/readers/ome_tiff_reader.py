#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import from_xml
from ome_types.model.ome import OME
from tifffile.tifffile import TiffFile, TiffFileError, TiffTags
from xmlschema import XMLSchemaValidationError

from .. import constants, exceptions, transforms, types
from ..dimensions import DEFAULT_CHUNK_DIMS, DEFAULT_DIMENSION_ORDER, DimensionNames
from ..metadata import utils as metadata_utils
from ..types import PhysicalPixelSizes
from ..utils import io_utils
from .tiff_reader import TiffReader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class OmeTiffReader(TiffReader):
    """
    Wraps the tifffile and ome-types APIs to provide the same aicsimageio Reader
    API but for volumetric OME-TIFF images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_dims: List[str]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.
    clean_metadata: bool
        Should the OME XML metadata found in the file be cleaned for known
        AICSImageIO 3.x and earlier created errors.
        Default: True (Clean the metadata for known errors)
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Notes
    -----
    If the OME metadata in your file isn't OME schema compilant or does not validate
    this will fail to read your file and raise an exception.

    If the OME metadata in your file doesn't use the latest OME schema (2016-06),
    this reader will make a request to the referenced remote OME schema to validate.
    """

    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            ome_xml = metadata_utils.clean_ome_xml_for_known_issues(ome_xml)

        return from_xml(ome_xml)

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem, path: str, clean_metadata: bool = True, **kwargs: Any
    ) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # Get first page description (aka the description tag in general)
                    xml = tiff.pages[0].description
                    ome = OmeTiffReader._get_ome(xml, clean_metadata)

                    # Handle no images in metadata
                    # this commonly means it is a "BinaryData" OME file
                    # i.e. a non-main OME-TIFF from MicroManager or similar
                    # in this case, because it's not the main file we want to just role
                    # back to TiffReader
                    if ome.binary_only:
                        return False

                    return True

        # tifffile exceptions
        except (TiffFileError, TypeError):
            return False

        # xml parse errors
        except ET.ParseError as e:
            log.debug(f"Failed to parse XML for the provided file. Error: {e}")
            return False

        # invalid OME XMl
        except XMLSchemaValidationError as e:
            log.debug(f"OME XML validation failed. Error: {e}")
            return False

        # cant connect to external schema resource (no internet conection)
        except URLError as e:
            log.debug(
                f"Could not validate OME XML against referenced schema "
                f"(no internet connection). "
                f"Error: {e}"
            )
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        clean_metadata: bool = True,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims
        self.clean_metadata = clean_metadata

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path, clean_metadata):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        # Get ome-types object and warn of other behaviors
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get and store OME
                self._ome = self._get_ome(
                    tiff.pages[0].description, self.clean_metadata
                )

                # Get and store scenes
                self._scenes: Tuple[str, ...] = tuple(
                    image_meta.id for image_meta in self._ome.images
                )

                # Log a warning stating that if this is a MM OME-TIFF, don't read
                # many series
                if tiff.is_micromanager and not isinstance(self._fs, LocalFileSystem):
                    log.warning(
                        "**Remote reading** (S3, GCS, HTTPS, etc.) of multi-image "
                        "(or scene) OME-TIFFs created by MicroManager has limited "
                        "support with the scene API. "
                        "It is recommended to use independent AICSImage or Reader "
                        "objects for each remote file instead of the `set_scene` API. "
                        "Track progress on support here: "
                        "https://github.com/AllenCellModeling/aicsimageio/issues/196"
                    )

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

    @staticmethod
    def _expand_dims_to_match_ome(
        image_data: types.ArrayLike,
        ome: OME,
        dims: List[str],
        scene_index: int,
    ) -> types.ArrayLike:
        # Expand image_data for empty dimensions
        ome_shape = []

        # need to correct channel count if this is a RGB image
        n_samples = ome.images[scene_index].pixels.channels[0].samples_per_pixel
        for d in dims:
            if d == "C" and n_samples is not None and n_samples > 1:
                count = len(ome.images[scene_index].pixels.channels)
            else:
                count = getattr(ome.images[scene_index].pixels, f"size_{d.lower()}")
            ome_shape.append(count)

        # Check for num samples and expand dims if greater than 1
        if n_samples is not None and n_samples > 1:
            # Append to the end, i.e. the last dimension
            dims.append("S")
            ome_shape.append(n_samples)

        # The file may not have all the data but OME requires certain dimensions
        # expand to fill
        expand_dim_ops: List[Optional[slice]] = []
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
        self,
        image_data: types.ArrayLike,
        dims: List[str],
        coords: Dict[str, Union[List[Any], types.ArrayLike]],
        tiff_tags: TiffTags,
    ) -> xr.DataArray:
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
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Unpack dims and coords from OME
                dims, coords = metadata_utils.get_dims_and_coords_from_ome(
                    ome=self._ome,
                    scene_index=self.current_scene_index,
                )

                # Grab the tifffile axes to use for dask array construction
                # If any of the non-"standard" dims are present
                # they will be filtered out during later reshape data calls
                strictly_read_dims = list(tiff.series[self.current_scene_index].axes)

                # Create the delayed dask array
                image_data = self._create_dask_array(tiff, strictly_read_dims)

                return self._general_data_array_constructor(
                    image_data,
                    dims,
                    coords,
                    tiff_tags,
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
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Unpack dims and coords from OME
                dims, coords = metadata_utils.get_dims_and_coords_from_ome(
                    ome=self._ome,
                    scene_index=self.current_scene_index,
                )

                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                return self._general_data_array_constructor(
                    image_data,
                    dims,
                    coords,
                    tiff_tags,
                )

    @property
    def ome_metadata(self) -> OME:
        return self.metadata

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
        return metadata_utils.physical_pixel_sizes(
            self.metadata, self.current_scene_index
        )
