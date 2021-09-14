#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from ..utils.cached_property import cached_property
from ..constants import METADATA_PROCESSED, METADATA_UNPROCESSED
from ..utils import io_utils
from .reader import Reader
from ..types import PhysicalPixelSizes

if TYPE_CHECKING:
    from ome_types import OME
    from bioformats_jar import loci
    from fsspec.spec import AbstractFileSystem

    from .. import types

    FormatReader = loci.formats.IFormatReader


class LociReader:
    def __init__(self, path: types.PathLike) -> None:
        try:
            from bioformats_jar import loci
        except ImportError:
            raise ImportError(
                "bioformats_jar is required for this reader. "
                "Install with `pip install aicsimageio[bioformats]`"
            )

        fmt = loci.formats
        self._rdr = fmt.ChannelSeparator(fmt.ChannelFiller())
        self._meta = fmt.MetadataTools.createOMEXMLMetadata()
        self._rdr.setMetadataStore(self._meta)
        self._rdr.setId(str(path))

    @cached_property
    def shape(self) -> Tuple[int, int, int, int, int]:
        from bioformats_jar.utils import shape

        return shape(self._rdr)

    @cached_property
    def dtype(self) -> str:
        from bioformats_jar.utils import dtype

        return dtype(self._rdr)

    @cached_property
    def xml(self) -> str:
        return str(self._meta.dumpXML())

    @cached_property
    def ome_metadata(self) -> OME:
        import ome_types

        return ome_types.from_xml(self.xml)

    def to_xarray(self, lazy: bool = True) -> xr.DataArray:
        scene_index = 0  # TODO
        # TODO: put in a utils method somewhere?
        from .ome_tiff_reader import OmeTiffReader

        _, coords = OmeTiffReader._get_dims_and_coords_from_ome(
            ome=self.ome_metadata,
            scene_index=scene_index,
        )
        image_data = self.to_dask() if lazy else self.to_numpy()
        # this is currently hardcoded in reader2dask
        dims = list("TCZYX")

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,  # type: ignore
            attrs={
                METADATA_UNPROCESSED: self.xml,
                METADATA_PROCESSED: self.ome_metadata,
            },
        )

    def to_dask(self) -> da.Array:
        from bioformats_jar.utils import reader2dask

        return reader2dask(self._rdr)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self.to_dask())


class BioformatsReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            # TODO: deal with remote data
            LociReader(path)
            return True

        except Exception:
            return False

    def __init__(self, image: types.PathLike):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # n_series = LociReader(image)._rdr.getSeriesCount()
        # self._scenes: Tuple[str, ...] = tuple(f"Image:{i}" for i in range(n_series))
        self._scenes: Tuple[str, ...] = ("Image:0",)

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return LociReader(self._path).to_xarray(lazy=True)

    def _read_immediate(self) -> xr.DataArray:
        return LociReader(self._path).to_xarray(lazy=False)

    @property
    def ome_metadata(self) -> OME:
        return LociReader(self._path).ome_metadata

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        z = self.ome_metadata.images[self.current_scene_index].pixels.physical_size_z
        y = self.ome_metadata.images[self.current_scene_index].pixels.physical_size_y
        x = self.ome_metadata.images[self.current_scene_index].pixels.physical_size_x

        return PhysicalPixelSizes(z, y, x)
