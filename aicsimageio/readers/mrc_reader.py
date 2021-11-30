#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Tuple

import dask.array as da
import mrcfile
import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from numpy.lib.recfunctions import structured_to_unstructured

from .. import exceptions, types
from ..dimensions import DimensionNames
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
        if self._scenes is None:
            # need full mmap open here because `is_image_stack` needs access to data shape
            with mrcfile.mmap(self._path, permissive=True) as mrc:
                if mrc.is_image_stack():
                    n_scenes = mrc.header.nz
                else:
                    n_scenes = mrc.header.nz / mrc.header.mz
                self._scenes = [f"Image:{i}" for i in range(int(n_scenes))]
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._make_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._make_xarray(delayed=False)

    def _make_xarray(self, delayed: bool) -> xr.DataArray:
        with mrcfile.mmap(self._path, permissive=True) as mrc:
            if mrc.is_single_image():
                scene = mrc.data
                dims = [DimensionNames.SpatialY, DimensionNames.SpatialX]
            elif mrc.is_image_stack():
                scene = mrc.data[self._current_scene_index]
                dims = [DimensionNames.SpatialY, DimensionNames.SpatialX]
            elif mrc.is_volume():
                scene = mrc.data
                dims = [
                    DimensionNames.SpatialZ,
                    DimensionNames.SpatialY,
                    DimensionNames.SpatialX,
                ]
            else:
                scene = mrc.data[self._current_scene_index]
                dims = [
                    DimensionNames.SpatialZ,
                    DimensionNames.SpatialY,
                    DimensionNames.SpatialX,
                ]

            # convert before exiting the context manager or scene will be None
            if delayed:
                data = da.from_array(scene)
            else:
                data = np.asarray(scene)

        return xr.DataArray(data, dims=dims)

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
