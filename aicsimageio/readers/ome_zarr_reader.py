#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple, Union

import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from .. import exceptions, types
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
            _ = fs  # TODO: How to use the FS to test for suppored image
            ZarrReader(parse_url(path, mode="r"))
            return True

        except ValueError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        channel_names: Optional[Union[List[str], List[List[str]]]] = None,
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

        self._channel_names = channel_names

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            OmeZarr = ZarrReader(parse_url(self._path, mode="r")).zarr
            self._scenes = tuple(
                metadata_utils.generate_ome_image_id(i)
                for i in range(len(OmeZarr.root_attrs["multiscales"]))
            )
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:

        OmeZarr = ZarrReader(parse_url(self._path, mode="r")).zarr
        image_data = OmeZarr.load(str(self.current_scene_index))
        axes = OmeZarr.root_attrs["multiscales"][self.current_scene_index]["axes"]
        dims = [sub["name"].upper() for sub in axes]

        if not delayed:
            image_data = image_data.compute()

        channels = self._get_channel_names_for_scene(image_data.shape, dims)

        coords = self._get_coords(
            dims,
            image_data.shape,
            scene_index=self.current_scene_index,
            channel_names=channels,
        )

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                # TODO: ADD info from "multiscales" "omero"?
            },
        )

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene_index: int,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        # Use dims for coord determination
        coords: Dict[str, Any] = {}

        if channel_names is None:
            # Get ImageId for channel naming
            image_id = metadata_utils.generate_ome_image_id(scene_index)

            # Use range for channel indices
            if DimensionNames.Channel in dims:
                coords[DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(
                        image_id=image_id, channel_id=i
                    )
                    for i in range(shape[dims.index(DimensionNames.Channel)])
                ]
        else:
            coords[DimensionNames.Channel] = channel_names

        return coords

    def _get_channel_names_for_scene(
        self, image_shape: Tuple[int], dims: List[str]
    ) -> Optional[List[str]]:
        # Fast return in None case
        if self._channel_names is None:
            return None

        # If channels was provided as a list of lists
        if isinstance(self._channel_names[0], list):
            scene_channels = self._channel_names[self.current_scene_index]
        elif all(isinstance(c, str) for c in self._channel_names):
            scene_channels = self._channel_names  # type: ignore
        else:
            return None

        # If scene channels isn't None and no channel dimension raise error
        if DimensionNames.Channel not in dims:
            raise exceptions.ConflictingArgumentsError(
                f"Provided channel names for scene with no channel dimension. "
                f"Scene dims: {dims}, "
                f"Provided channel names: {scene_channels}"
            )

        # If scene channels isn't the same length as the size of channel dim
        if len(scene_channels) != image_shape[dims.index(DimensionNames.Channel)]:
            raise exceptions.ConflictingArgumentsError(
                f"Number of channel names provided does not match the "
                f"size of the channel dimension for this scene. "
                f"Scene shape: {image_shape}, "
                f"Dims: {dims}, "
                f"Provided channel names: {self._channel_names}",
            )

        return scene_channels  # type: ignore
