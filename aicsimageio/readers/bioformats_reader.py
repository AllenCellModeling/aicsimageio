#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from collections import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Iterable, List, Optional, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from attr import has
from ..utils.cached_property import cached_property
from ..constants import METADATA_PROCESSED, METADATA_UNPROCESSED
from ..dimensions import DimensionNames
from ..utils import io_utils
from .reader import Reader

if TYPE_CHECKING:
    import ome_types
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
    def ome_metadata(self) -> ome_types.OME:
        import ome_types

        return ome_types.from_xml(self.xml)

    def to_xarray(self, lazy: bool = True) -> xr.DataArray:
        return xr.DataArray(
            self.to_dask() if lazy else self.to_numpy(),
            dims=self._dims,
            coords=self._coords,
            attrs={
                METADATA_UNPROCESSED: self.xml,
                METADATA_PROCESSED: self.ome_metadata,
            },
        )

    @cached_property
    def _dims(self) -> List[str]:
        return list("TCZYX")

    @cached_property
    def _coords(self) -> dict:
        _coords = {}
        for d, s in zip(self._dims, self.shape):
            getpix = getattr(self._meta, f"getPixelsPhysicalSize{d}", None)
            size = 1.0
            if getpix is not None:
                _size = getpix(0)
                size = float(_size.value()) if _size is not None else 1
            _coords[d] = np.arange(0, s) * size
        return _coords

    def to_dask(self) -> da.Array:
        from bioformats_jar.utils import reader2dask

        return reader2dask(self._rdr)

    def to_numpy(self) -> np.ndarray:
        return self.to_dask().compute()


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
        self._loci_rdr = LociReader(image)

    @property
    def scenes(self) -> Tuple[str, ...]:
        # TODO: improve
        return tuple(f"Image:{i}" for i in range(self._loci_rdr._rdr.getSeriesCount()))

    def _read_delayed(self) -> xr.DataArray:
        return self._loci_rdr.to_xarray(lazy=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._loci_rdr.to_xarray(lazy=False)
