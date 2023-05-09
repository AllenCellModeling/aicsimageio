#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, Tuple

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import OME, from_xml

from .. import constants, exceptions, types
from ..metadata import utils as metadata_utils
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
    Wraps the ome-zarr-py API to provide the same aicsimageio Reader API but for
    OmeZarr images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    """

    @staticmethod
    def _get_ome(ome_xml: str, clean_metadata: bool = True) -> OME:
        # To clean or not to clean, that is the question
        if clean_metadata:
            ome_xml = metadata_utils.clean_ome_xml_for_known_issues(ome_xml)

        return from_xml(ome_xml, parser="lxml")

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                ZarrReader(parse_url(open_resource, mode="r"))
                return True

        except ValueError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        clean_metadata: bool = True,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
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

        self.clean_metadata = clean_metadata

        # Get ome-types object and warn of other behaviors
        with ZarrReader(parse_url(self._path, mode="r")).zarr as OmeZarr:
            self._ome = self._get_ome(
                OmeZarr.root_attrs["WHATEVER KEY GIVES OME METADATA"],  # OMERO?
                self.clean_metadata,
            )

            self._scenes: Tuple[str, ...] = tuple(
                image_meta.id for image_meta in self._ome.images
            )

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(self._path, delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(self._path, delayed=False)

    def _xarr_format(self, path: str, delayed: bool) -> xr.DataArray:

        with ZarrReader(parse_url(path, mode="r")).zarr as OmeZarr:
            dask_array = OmeZarr.load(str(self.current_scene_index))
            axes = OmeZarr.root_attrs["multiscales"][self.current_scene_index].axes
            dims = [sub["name"].upper() for sub in axes]

            coords = metadata_utils.get_coords_from_ome(
                ome=self._ome,
                scene_index=self.current_scene_index,
            )

            if not delayed:
                dask_array = dask_array.compute()

            return xr.DataArray(
                dask_array,
                dims=dims,
                coords=coords,
                attrs={
                    constants.METADATA_PROCESSED: self._ome,
                    # ADD info from "multiscenes"?
                },
            )
