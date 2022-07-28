#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import xarray as xr
from bfio import BioReader
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types import OME
from tifffile.tifffile import TiffFileError, TiffTags

from .. import constants, exceptions, transforms, types
from ..dimensions import DEFAULT_DIMENSION_ORDER
from ..exceptions import UnsupportedFileFormatError
from ..metadata import utils as metadata_utils
from ..types import PhysicalPixelSizes
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
        Note: Dimensions.SpatialY, and Dimensions.SpatialX will always be added to the
        list if not present during dask array construction.
    out_order: List[str]
        The output dimension ordering.
        Default: DEFAULT_DIMENSION_ORDER
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Notes
    -----
    If the OME metadata in your file isn't OME schema compliant or does not validate
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
        _, coords = metadata_utils.get_dims_and_coords_from_ome(
            ome=self._rdr.metadata,
            scene_index=0,
        )

        coords = {d: coords[d] for d in self.out_dim_order if d in coords}
        image_data = transforms.reshape_data(
            image_data, self.native_dim_order, "".join(self.out_dim_order)
        )

        attrs = {constants.METADATA_PROCESSED: self._rdr.metadata}

        if tiff_tags is not None:
            attrs[constants.METADATA_UNPROCESSED] = tiff_tags

        return xr.DataArray(
            image_data,
            dims=self.out_dim_order,
            coords=coords,
            attrs=attrs,
        )

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        """This method should be overwritten by a subclass."""
        try:
            with BioReader(path):
                return True

        except Exception:
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Optional[Union[str, List[str]]] = None,
        out_order: str = DEFAULT_DIMENSION_ORDER,
        fs_kwargs: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                "Cannot read .ome.tif from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        try:
            self._rdr = BioReader(self._path, backend=self.backend)
        except (TypeError, ValueError, TiffFileError):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        # Add ndim attribute so _rdr can be passed directly to dask
        self._rdr.ndim = len(self._rdr.shape)

        # Setup dimension ordering
        dims = "YXZCT"
        self.native_dim_order = dims[: len(self._rdr.shape)]
        assert all(d in out_order for d in dims)
        self.out_dim_order = [d for d in out_order if d in dims]

        # Currently do not support custom chunking, throw a warning.
        if chunk_dims is not None:
            log.warning(
                "OmeTiledTiffReader does not currently support custom chunking."
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        return tuple(image_meta.id for image_meta in self._rdr.metadata.images)

    @property
    def current_scene(self) -> str:
        return self.scenes[self._current_scene_index]

    @property
    def current_scene_index(self) -> int:
        return self._current_scene_index

    def set_scene(self, scene_id: Union[str, int]) -> None:
        """
        For all BfioReader subclasses, the only allowed value is the name of the first
        scene since only the first scene can be read by BioReader objects. This method
        exists primarily to help this Reader fit into existing unit test templates and
        in case BioReader is updated to support multiple scenes.

        Parameters
        ----------
        scene_id: Union[str, int]
            The scene id (if string) or scene index (if integer)
            to set as the operating scene.

        Raises
        ------
        IndexError
            The provided scene id or index does not reference the first scene.
        TypeError
            The provided value wasn't a string (scene id) or integer (scene index).
        """
        # Route to int or str setting
        if isinstance(scene_id, (str, int)):
            # Only need to run when the scene id is different from current scene
            if scene_id not in (self.current_scene, self.current_scene_index):

                raise IndexError(
                    "Scene id: Cannot change scene for "
                    + f"{self.__class__.__name__} objects."
                )

        else:
            raise TypeError(
                f"Must provide either a string (for scene id) "
                f"or integer (for scene index). Provided: {scene_id} ({type(scene_id)}."
            )

    @property
    def channel_names(self) -> Optional[List[str]]:

        return self._rdr.channel_names

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        return types.PhysicalPixelSizes(
            self._rdr.ps_z[0],
            self._rdr.ps_y[0],
            self._rdr.ps_x[0],
        )

    def _read_delayed(self) -> xr.DataArray:

        return self._general_data_array_constructor(
            da.from_array(self._rdr, chunks=(1024, 1024) + (1,) * (self._rdr.ndim - 2)),
            self._tiff_tags(),
        )

    def _tiff_tags(self) -> Optional[Dict[str, str]]:

        tiff_tags: Optional[Dict[str, str]] = None
        if self.backend == "python":
            # Create a copy since TiffTags are not serializable
            tiff_tags = {
                code: tag.value
                for code, tag in self._rdr._backend._rdr.pages[0].tags.items()
            }

        return tiff_tags

    def _read_immediate(self) -> xr.DataArray:
        return self._general_data_array_constructor(
            self._rdr[:],
            self._tiff_tags(),
        )

    @property
    def ome_metadata(self) -> OME:
        return self._rdr.metadata


class OmeTiledTiffReader(BfioReader):
    """
    Wrapper around bfio.BioReader(backend="python").

    The OmeTiledTiffReader is an optimized TIFF reader written in pure Python, built on
    top of tifffile. This reader is optimized for speed and scalability, but will only
    read TIFF files that meet the following requirements:

    1. TileWidth and TileHeight tags must both be set to 1024
    2. The Description tag must contain OMEXML
    3. The OMEXML channel ordering must be set to XYZCT
    4. Channels cannot be interleaved, meaning individual channels must be planes.

    The advantage of the reader for files that meet these requirements are improvements
    in reading speed, especially when accessing data using dask.

    This TIFF reader will only read the first image and pyramid layer. If pyramid layers
    or images beyond the first image in the file need to be read, use the OmeTiffReader.

    Parameters
    ----------
    image: types.PathLike
        Path to image file.
    chunk_dims: List[str]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, and Dimensions.SpatialX will always be added to the
        list if not present during dask array construction.
    out_order: List[str]
        The output dimension ordering.
        Default: DEFAULT_DIMENSION_ORDER

    Notes
    -----
    If the OME metadata in your file isn't OME schema compliant or does not validate
    this will fail to read your file and raise an exception.

    If the OME metadata in your file doesn't use the latest OME schema (2016-06),
    this reader will make a request to the referenced remote OME schema to validate.
    """

    backend: str = "python"

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            if not isinstance(fs, LocalFileSystem):
                return False

            with BioReader(path, backend="python") as br:

                # Fail fast if multi-image file
                if len(br.metadata.images) > 1:
                    raise UnsupportedFileFormatError(
                        path,
                        "This file contains more than one scene and only the first "
                        + "scene can be read by the OmeTiledTiffReader. "
                        + "To read additional scenes, use the TiffReader, "
                        + "OmeTiffReader, or BioformatsReader.",
                    )

                return True

        # tifffile exceptions
        except (TypeError, ValueError):
            return False
