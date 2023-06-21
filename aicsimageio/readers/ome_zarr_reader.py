#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
from typing import Any, Dict, List, Optional, Tuple

import xarray as xr
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..metadata import utils as metadata_utils
from ..types import PhysicalPixelSizes
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
        self._physical_pixel_sizes: Optional[PhysicalPixelSizes] = None
        self._multiresolution_level = 0
        self._channel_names: Optional[List[str]] = None

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

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """Return the physical pixel sizes of the image."""
        if self._physical_pixel_sizes is None:
            try:
                z_size, y_size, x_size = OmeZarrReader._get_pixel_size(
                    self._zarr,
                    list(self.dims.order),
                    self._current_scene_index,
                    self._multiresolution_level,
                )
            except Exception as e:
                warnings.warn(f"Could not parse zarr pixel size: {e}")
                z_size, y_size, x_size = None, None, None

            self._physical_pixel_sizes = PhysicalPixelSizes(z_size, y_size, x_size)
        return self._physical_pixel_sizes

    @property
    def channel_names(self) -> Optional[List[str]]:
        if self._channel_names is None:
            try:
                self._channel_names = [
                    str(channel["label"])
                    for channel in self._zarr.root_attrs["omero"]["channels"]
                ]
            except KeyError:
                self._channel_names = super().channel_names
        return self._channel_names

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_format(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_format(delayed=False)

    def _xarr_format(self, delayed: bool) -> xr.DataArray:
        image_data = self._zarr.load(str(self.current_scene_index))

        axes = self._zarr.root_attrs["multiscales"][self.current_scene_index].get(
            "axes"
        )
        if axes:
            dims = [sub["name"].upper() for sub in axes]
        else:
            dims = list(OmeZarrReader._guess_dim_order(image_data.shape))

        if not delayed:
            image_data = image_data.compute()

        coords = self._get_coords(
            dims,
            image_data.shape,
            scene=self.current_scene,
            channel_names=self.channel_names,
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

        coords: Dict[str, Any] = {}

        # Use dims for coord determination
        if DimensionNames.Channel in dims:
            # Generate channel names if no existing channel names
            if channel_names is None:
                coords[DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(image_id=scene, channel_id=i)
                    for i in range(shape[dims.index(DimensionNames.Channel)])
                ]
            else:
                coords[DimensionNames.Channel] = channel_names

        return coords

    @staticmethod
    def _get_pixel_size(
        reader: ZarrReader, dims: List[str], series_index: int, resolution_index: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        # OmeZarr file may contain an additional set of "coordinateTransformations"
        # these coefficents are applied to all resolution levels.
        if (
            "coordinateTransformations"
            in reader.root_attrs["multiscales"][series_index]
        ):
            universal_res_consts = reader.root_attrs["multiscales"][series_index][
                "coordinateTransformations"
            ][0]["scale"]
        else:
            universal_res_consts = [1.0 for _ in range(len(dims))]

        coord_transform = reader.root_attrs["multiscales"][series_index]["datasets"][
            resolution_index
        ]["coordinateTransformations"]

        spatial_coeffs = {}

        for dim in [
            DimensionNames.SpatialX,
            DimensionNames.SpatialY,
            DimensionNames.SpatialZ,
        ]:
            if dim in dims:
                dim_index = dims.index(dim)
                spatial_coeffs[dim] = (
                    coord_transform[0]["scale"][dim_index]
                    * universal_res_consts[dim_index]
                )
            else:
                spatial_coeffs[dim] = None

        return (
            spatial_coeffs[DimensionNames.SpatialZ],
            spatial_coeffs[DimensionNames.SpatialY],
            spatial_coeffs[DimensionNames.SpatialX],
        )
