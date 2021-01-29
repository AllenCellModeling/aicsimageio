#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr

from .. import constants, exceptions
from ..dimensions import DimensionNames
from ..metadata import utils as metadata_utils
from ..types import ArrayLike
from .reader import Reader

###############################################################################


class ArrayLikeReader(Reader):
    """
    A catch all for numpy or dask to Reader interface.

    Useful for passing through raw data to Reader -> Writer.

    Parameters
    ----------
    image: types.ArrayLike
        A numpy ndarray or dask Array.

    known_dims: Optional[str]
        A string containing dimension identifiers to use instead of guesses.
        Default: None (guess dimensions)
    """

    @staticmethod
    def _is_supported_image(image: ArrayLike, **kwargs) -> bool:
        return isinstance(image, (np.ndarray, da.Array))

    def __init__(self, image: ArrayLike, known_dims: Optional[str] = None, **kwargs):
        # Enforce valid image
        if not self._is_supported_image(image):
            raise exceptions.UnsupportedFileFormatError(self.__class__.__name__, image)

        # Store dims
        if known_dims is None:
            dims = self._guess_dim_order(image.shape)
        else:
            dims = known_dims

        # Turn dims to list
        dims = [d for d in dims]

        # Catch dims not matching data
        if len(dims) != len(image.shape):
            raise exceptions.InvalidDimensionOrderingError(
                "The provided dims have a different length than the provided data."
            )

        # Coords
        coords = {}
        if DimensionNames.Channel in dims:
            num_channels = image.shape[dims.index(DimensionNames.Channel)]
            coords[DimensionNames.Channel] = [
                metadata_utils.generate_ome_channel_id(i) for i in range(num_channels)
            ]

        # Unpack data
        if isinstance(image, da.Array):
            self._xarray_dask_data = xr.DataArray(
                data=image,
                dims=dims,
                coords=coords,
                attrs={constants.METADATA_UNPROCESSED: None},
            )

        else:
            self._xarray_data = xr.DataArray(
                data=image,
                dims=dims,
                coords=coords,
                attrs={constants.METADATA_UNPROCESSED: None},
            )
            self._xarray_dask_data = xr.DataArray(
                data=da.from_array(image),
                dims=dims,
                coords=coords,
                attrs=self._xarray_data.attrs,
            )

    # We are required to have these as defined by the Reader spec
    def _read_delayed(self) -> xr.DataArray:
        pass

    def _read_immediate(self) -> xr.DataArray:
        return xr.DataArray(
            data=self._xarray_dask_data.data.compute(),
            dims=self._xarray_dask_data.dims,
            attrs=self._xarray_dask_data.attrs,
        )

    @property
    def scenes(self) -> Tuple[str]:
        # There is currently an assumption that DefaultReader will not encounter
        # files with multiple scenes. But, if we do encounter a file that DefaultReader
        # hits and a user wants scene management from that file type, we can update
        # this property then.
        return (metadata_utils.generate_ome_image_id(0),)
