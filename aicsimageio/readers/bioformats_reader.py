#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from ..utils.cached_property import cached_property

from typing import TYPE_CHECKING, Any, Tuple

import xarray as xr
from .. import exceptions
from ..constants import METADATA_PROCESSED, METADATA_UNPROCESSED
from ..metadata import utils as metadata_utils
from ..types import PhysicalPixelSizes
from ..utils import io_utils
from .reader import Reader
from .. import dimensions

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem
    from ome_types import OME

    from .. import types


try:
    from ._loci_reader import LociFile
except ImportError:
    raise ImportError(
        "bioformats_jar is required for this reader. "
        "Install with `pip install aicsimageio[bioformats]`"
    )


class BioformatsReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            # TODO: deal with remote data
            l = LociFile(path, meta=False, memoize=False)
            l.close()
            return True

        except Exception:
            return False

    def __init__(self, image: types.PathLike):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        try:
            with LociFile(self._path) as rdr:
                self._scenes: Tuple[str, ...] = tuple(
                    metadata_utils.generate_ome_image_id(i) for i in range(rdr._nseries)
                )
        except Exception:
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._to_xarray(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        with LociFile(self._path) as rdr:
            arr = self._to_xarray(delayed=False)
        return arr

    @cached_property
    def ome_metadata(self) -> OME:
        with LociFile(self._path) as rdr:
            meta = rdr.ome_metadata
        return meta

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        px = self.ome_metadata.images[self.current_scene_index].pixels
        return PhysicalPixelSizes(
            px.physical_size_z, px.physical_size_y, px.physical_size_x
        )

    def _to_xarray(self, delayed: bool = True) -> xr.DataArray:
        # TODO: put in a utils method somewhere?
        from .ome_tiff_reader import OmeTiffReader

        with LociFile(self._path) as rdr:
            image_data = rdr.to_dask() if delayed else rdr.to_numpy()
            xml = rdr.ome_xml
            ome = rdr.ome_metadata
            rgb = rdr.core_meta.is_rgb
            _, coords = OmeTiffReader._get_dims_and_coords_from_ome(
                ome=ome,
                scene_index=self.current_scene_index,
            )

        return xr.DataArray(
            image_data,
            dims=dimensions.DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
            if rgb
            else dimensions.DEFAULT_DIMENSION_ORDER_LIST,
            coords=coords,  # type: ignore
            attrs={
                METADATA_UNPROCESSED: xml,
                METADATA_PROCESSED: ome,
            },
        )
