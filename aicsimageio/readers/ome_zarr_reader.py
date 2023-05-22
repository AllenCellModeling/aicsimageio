#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
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
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            ZarrReader(parse_url(path, mode="r"))
            return True

        except AttributeError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=False,
            fs_kwargs=fs_kwargs,
        )

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        self._zarr = ZarrReader(parse_url(self._path, mode="r")).zarr

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            scenes = self._zarr.root_attrs["multiscales"]

            # if (each scene has a name) and (that name is unique) use name.
            # otherwise generate scene names.
            if all("name" in scene for scene in scenes) and (
                len({scene["name"] for scene in scenes}) == len(scenes)
            ):
                self._scenes = tuple(str(scene["name"]) for scene in scenes)
            else:
                self._scenes = tuple(
                    metadata_utils.generate_ome_image_id(i)
                    for i in range(len(self._zarr.root_attrs["multiscales"]))
                )
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:
        image_data = self._zarr.load(str(self.current_scene_index))
        axes = self._zarr.root_attrs["multiscales"][self.current_scene_index]["axes"]
        dims = [sub["name"].upper() for sub in axes]

        if not delayed:
            image_data = image_data.compute()

        coords = self._get_coords(
            dims,
            image_data.shape,
            scene=self.current_scene,
            channel_names=self._get_channel_names_from_ome(),
        )

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={constants.METADATA_UNPROCESSED: self._zarr.root_attrs},
        )

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene: str,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        # Use dims for coord determination
        coords: Dict[str, Any] = {}

        if DimensionNames.Channel in dims:
            if channel_names is None:
                coords[DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(image_id=scene, channel_id=i)
                    for i in range(shape[dims.index(DimensionNames.Channel)])
                ]
            else:
                coords[DimensionNames.Channel] = channel_names

        return coords

    def _get_channel_names_from_ome(self) -> "List[str] | None":
        try:
            channels = [
                str(channel["label"])
                for channel in self._zarr.root_attrs["omero"]["channels"]
            ]
            return channels
        except KeyError:
            return None
