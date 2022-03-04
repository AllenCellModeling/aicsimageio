#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, List, Optional, Tuple, Union

import dask.array as da
import xarray as xr
from bfio import BioReader
from fsspec.spec import AbstractFileSystem
from ome_types import OME
from tifffile.tifffile import TiffTags

from .. import constants, exceptions, types
from ..dimensions import DEFAULT_CHUNK_DIMS
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from .reader import Reader

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class BfioReader(Reader):
    """
    Abstract bfio reader to utilize optimized readers for ome tiled tiffs and ome zarr.

    Parameters
    ----------
    image: types.PathLike
        Path to image file.
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

    Notes
    -----
    If the OME metadata in your file isn't OME schema compilant or does not validate
    this will fail to read your file and raise an exception.

    If the OME metadata in your file doesn't use the latest OME schema (2016-06),
    this reader will make a request to the referenced remote OME schema to validate.
    """

    backend: Optional[str] = None

    def _general_data_array_constructor(
        self,
        image_data: types.ArrayLike,
        tiff_tags: Optional[TiffTags] = None,
    ) -> xr.DataArray:

        # Unpack dims and coords from OME
        dims, coords = metadata_utils.get_dims_and_coords_from_ome(
            ome=self._rdr.metadata,
            scene_index=0,
        )

        dims = [
            d
            for d in self._rdr.metadata.images[0].pixels.dimension_order.name[
                : len(image_data.shape)
            ]
        ]

        coords = {d: coords[d] for d in dims if d in coords}

        attrs = {constants.METADATA_PROCESSED: self._rdr.metadata}

        if tiff_tags is not None:
            attrs[constants.METADATA_UNPROCESSED] = tiff_tags

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,  # type: ignore
            attrs=attrs,
        )

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            # with fs.open(path) as open_resource:
            with BioReader(path, backend="python"):

                return True

        # tifffile exceptions
        except (TypeError, ValueError):
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        clean_metadata: bool = True,
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Enforce valid image
        if not self._is_supported_image(None, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        self._rdr = BioReader(self._path, backend=self.backend)

        # Add ndim attribute so _rdr can be passed directly to dask
        self._rdr.ndim = len(self._rdr.shape)

    @property
    def scenes(self) -> Tuple[str, ...]:
        return tuple(image_meta.id for image_meta in self._rdr.metadata.images)

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

        return self._general_data_array_constructor(
            da.from_array(self._rdr, chunks=(1024, 1024) + (1,) * (self._rdr.ndim - 2)),
            self._tiff_tags(),
        )

    def _tiff_tags(self) -> TiffTags:

        if self.backend == "python":
            tiff_tags = self._rdr._backend._rdr.pages[0].tags
        else:
            tiff_tags = None

        return tiff_tags

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
        return self._general_data_array_constructor(
            self._rdr[:],
            self._tiff_tags(),
        )

    @property
    def ome_metadata(self) -> OME:
        return self._rdr.metadata

    def __del__(self) -> None:
        """Try to do some cleanup when deleted or falls out of context."""

        self._rdr.close()


class OmeTiledTiffReader(BfioReader):
    """
    Wrapper around bfio.BioReader(backend="python").

    Parameters
    ----------
    image: types.PathLike
        Path to image file.
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

    Notes
    -----
    If the OME metadata in your file isn't OME schema compilant or does not validate
    this will fail to read your file and raise an exception.

    If the OME metadata in your file doesn't use the latest OME schema (2016-06),
    this reader will make a request to the referenced remote OME schema to validate.
    """

    backend: str = "python"

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            # with fs.open(path) as open_resource:
            with BioReader(path, backend="python"):

                return True

        # tifffile exceptions
        except (TypeError, ValueError):
            return False
