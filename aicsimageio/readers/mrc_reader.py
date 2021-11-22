#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Tuple

import mrcfile
import dask.array as da
import xarray as xr
from numpy.lib.recfunctions import structured_to_unstructured
from fsspec.spec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from .. import exceptions, types
from ..utils import io_utils
from .reader import Reader


class MrcReader(Reader):
    """
    Read MRC files using mrcfile.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with mrcfile.open(path, permissive=True, header_only=True):
                return True
        except (ValueError, TypeError):
            return False

    def __init__(self, image: types.PathLike):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise NotImplementedError(
                f"mrc reader not yet implemented for non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        # TODO: treat mrcs as stacks of scenes?
        return ("Image:0",)

    def _read_delayed(self) -> xr.DataArray:
        return self._make_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._make_xarray(delayed=False)

    def _make_xarray(self, delayed: bool) -> xr.DataArray:
        reader = mrcfile.mmap if delayed else mrcfile.open
        with reader(self._path, permissive=True) as mrc:
            data = mrc.data
            if delayed:
                data = da.from_array(data)
        xarr = xr.DataArray(data, dims=['Z', 'Y', 'X'])
        return xarr

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
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
        with mrcfile.open(self._path, permissive=True, header_only=True) as mrc:
            sizes = structured_to_unstructured(mrc.voxel_size)[::-1]
            return types.PhysicalPixelSizes(*sizes)
