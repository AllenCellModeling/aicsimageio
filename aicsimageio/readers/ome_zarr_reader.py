#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from .. import exceptions, types
from ..utils import io_utils
from .reader import Reader

try:
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader as ZarrReader

except ImportError:
    raise ImportError(
        "ome_zarr is required for this reader. " "Install with `pip install 'ome_zarr'`"
    )

###############################################################################


class OmeZarrReader(Reader):
    """
    Wraps the readlif API to provide the same aicsimageio Reader API but for
    volumetric LIF images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    """

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            fs_kwargs=fs_kwargs,
        )

        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read OmeZarr from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                ZarrReader(parse_url(open_resource, mode="r"))
                return True

        except ValueError:
            return False

    def _read_delayed(self) -> xr.DataArray:
        return self._create_dask_array(self._path)

    def _read_immediate(self) -> xr.DataArray:
        return self._create_dask_array(self._path).compute()

    def _create_dask_array(self, path: str) -> xr.DataArray:
        image_location = parse_url(path, mode="r")
        reader = ZarrReader(image_location).zarr
        dask_array = reader.load("0")
        return dask_array

    """
    def _create_dask_array(self, path: str) -> xr.DataArray:
        reader = ZarrReader(parse_url(path, mode="r"))
        image_node = list(reader())[0]
        return image_node.data
    """
