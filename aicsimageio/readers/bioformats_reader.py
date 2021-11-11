#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, NamedTuple, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from fsspec.implementations.local import LocalFileSystem
from ome_types import OME

from .. import constants, dimensions, exceptions
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from ..utils.cached_property import cached_property
from ..utils.dask_proxy import DaskArrayProxy
from .reader import Reader

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem

    from .. import types


try:
    import jpype
    from bioformats_jar import get_loci
except ImportError:
    raise ImportError(
        "bioformats_jar is required for this reader. "
        "Install with `pip install bioformats_jar`"
    )


class BioformatsReader(Reader):
    """Read files using bioformats.

    This reader requires `bioformats_jar` to be installed in the environment, and
    requires the java executable to be available on the path, or via the JAVA_HOME
    environment variable.

    To install java with conda, run `conda install -c conda-forge openjdk`.
    You may need to deactivate/reactivate your environment after installing.  If you
    are *still* getting a `JVMNotFoundException`, try setting JAVA_HOME as follows:

        # mac and linux:
        export JAVA_HOME=$CONDA_PREFIX

        # windows:
        set JAVA_HOME=%CONDA_PREFIX%\\Library

    Parameters
    ----------
    image : Path or str
        path to file
    original_meta : bool, optional
        whether to also retrieve the proprietary metadata as structured annotations in
        the OME output, by default False
    memoize : bool or int, optional
        threshold (in milliseconds) for memoizing the reader. If the the time
        required to call `reader.setId()` is larger than this number, the initialized
        reader (including all reader wrappers) will be cached in a memo file, reducing
        time to load the file on future reads.  By default, this results in a hidden
        `.bfmemo` file in the same directory as the file. The `BIOFORMATS_MEMO_DIR`
        environment can be used to change the memo file directory.
        Set `memoize` to greater than 0 to turn on memoization. by default it's off.
        https://downloads.openmicroscopy.org/bio-formats/latest/api/loci/formats/Memoizer.html
    options : Dict[str, bool], optional
        A mapping of option-name -> bool specifying additional reader-specific options.
        see: https://docs.openmicroscopy.org/bio-formats/latest/formats/options.html
        For example: to turn off chunkmap table reading for ND2 files, use
        `options={"nativend2.chunkmap": False}`
    dask_tiles: bool, optional
        Whether to chunk the bioformats dask array by tiles to easily read sub-regions
        with numpy-like array indexing
        Defaults to false and iamges are read by entire planes
    tile_size: Optional[Tuple[int, int]]
        Tuple that sets the tile size of y and x axis, respectively
        By default, it will use optimal values computed by bioformats itself
    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by bioformats.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            if not isinstance(fs, LocalFileSystem):
                return False
            f = BioFile(path, meta=False, memoize=False)
            f.close()
            return True

        except Exception:
            return False

    def __init__(
        self,
        image: types.PathLike,
        *,
        original_meta: bool = False,
        memoize: Union[int, bool] = 0,
        options: Dict[str, bool] = {},
        dask_tiles: bool = False,
        tile_size: Optional[Tuple[int, int]] = None,
    ):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read Bioformats from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        self._bf_kwargs = {
            "options": options,
            "original_meta": original_meta,
            "memoize": memoize,
            "dask_tiles": dask_tiles,
            "tile_size": tile_size,
        }
        try:
            with BioFile(self._path, **self._bf_kwargs) as rdr:  # type: ignore
                md = rdr._r.getMetadataStore()
                self._scenes: Tuple[str, ...] = tuple(
                    str(md.getImageName(i)) for i in range(md.getImageCount())
                )
        except jpype.JVMNotFoundException:
            raise
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
        return self._to_xarray(delayed=False)

    @cached_property
    def ome_metadata(self) -> OME:
        """Return OME object parsed by ome_types."""
        with BioFile(self._path, **self._bf_kwargs) as rdr:  # type: ignore
            meta = rdr.ome_metadata
        return meta

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
        return metadata_utils.physical_pixel_sizes(
            self.metadata, self.current_scene_index
        )

    def _to_xarray(self, delayed: bool = True) -> xr.DataArray:
        with BioFile(self._path, **self._bf_kwargs) as rdr:  # type: ignore
            image_data = rdr.to_dask() if delayed else rdr.to_numpy()
            _, coords = metadata_utils.get_dims_and_coords_from_ome(
                ome=rdr.ome_metadata,
                scene_index=self.current_scene_index,
            )

        return xr.DataArray(
            image_data,
            dims=dimensions.DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
            if rdr.core_meta.is_rgb
            else dimensions.DEFAULT_DIMENSION_ORDER_LIST,
            coords=coords,  # type: ignore
            attrs={
                constants.METADATA_UNPROCESSED: rdr.ome_xml,
                constants.METADATA_PROCESSED: rdr.ome_metadata,
            },
        )

    @staticmethod
    def bioformats_version() -> str:
        """The version of the bioformats_package.jar being used."""
        from bioformats_jar import get_loci

        return get_loci().__version__


class CoreMeta(NamedTuple):
    """NamedTuple with core bioformats metadata. (not OME meta)"""

    shape: Tuple[int, int, int, int, int, int]
    dtype: np.dtype
    series_count: int
    is_rgb: bool
    is_interleaved: bool
    dimension_order: str
    resolution_count: int


# by default, .bfmemo files will go into the same directory as the file.
# users can override this with BIOFORMATS_MEMO_DIR env var
BIOFORMATS_MEMO_DIR: Optional[Path] = None
_BFDIR = os.getenv("BIOFORMATS_MEMO_DIR")
if _BFDIR:
    BIOFORMATS_MEMO_DIR = Path(_BFDIR).expanduser().absolute()
    BIOFORMATS_MEMO_DIR.mkdir(exist_ok=True, parents=True)


class BioFile:
    """Read image and metadata from file supported by Bioformats.

    BioFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    BioFile instances are not thread-safe.

    Bio-Formats is licensed under GPLv2 and is not included in this package.

    Parameters
    ----------
    path : str or Path
        path to file
    series : int, optional
        the image series to read, by default 0
    meta : bool, optional
        whether to get metadata as well, by default True
    original_meta : bool, optional
        whether to also retrieve the proprietary metadata as structured annotations in
        the OME output, by default False
    memoize : bool or int, optional
        threshold (in milliseconds) for memoizing the reader. If the the time
        required to call `reader.setId()` is larger than this number, the initialized
        reader (including all reader wrappers) will be cached in a memo file, reducing
        time to load the file on future reads.  By default, this results in a hidden
        `.bfmemo` file in the same directory as the file. The `BIOFORMATS_MEMO_DIR`
        environment can be used to change the memo file directory.
        Set `memoize` to greater than 0 to turn on memoization. by default it's off.
        https://downloads.openmicroscopy.org/bio-formats/latest/api/loci/formats/Memoizer.html
    options : Dict[str, bool], optional
        A mapping of option-name -> bool specifying additional reader-specific options.
        see: https://docs.openmicroscopy.org/bio-formats/latest/formats/options.html
        For example: to turn off chunkmap table reading for ND2 files, use
        `options={"nativend2.chunkmap": False}`
    dask_tiles: bool, optional
        Whether to chunk the bioformats dask array by tiles to easily read sub-regions
        with numpy-like array indexing
        Defaults to false and iamges are read by entire planes
    tile_size: Optional[Tuple[int, int]]
        Tuple that sets the tile size of y and x axis, respectively
        By default, it will use optimal values computed by bioformats itself
    """

    def __init__(
        self,
        path: types.PathLike,
        series: int = 0,
        meta: bool = True,
        *,
        original_meta: bool = False,
        memoize: Union[int, bool] = 0,
        options: Dict[str, bool] = {},
        dask_tiles: bool = False,
        tile_size: Optional[Tuple[int, int]] = None,
    ):
        try:
            loci = get_loci()
        except jpype.JVMNotFoundException as e:
            raise type(e)(
                str(e) + "\n\nBioformatsReader requires a java executable to be "
                "available in your environment. If you are using conda, you can "
                "install with `conda install -c conda-forge openjdk`.\n\n"
                "Note: you may need to reactivate your conda environment after "
                "installing opendjk. If you still have this error, try:\n\n"
                "# mac and linux:\n"
                "export JAVA_HOME=$CONDA_PREFIX\n\n"
                "# windows:\n"
                "set JAVA_HOME=%CONDA_PREFIX%\\Library"
            )

        self._path = str(path)
        self._r = loci.formats.ImageReader()
        if meta:
            self._r.setMetadataStore(self._create_ome_meta())
        if original_meta:
            self._r.setOriginalMetadataPopulated(True)

        # memoize to save time on later re-openings of the same file.
        if memoize > 0:
            _hide_memoization_warning()
            if BIOFORMATS_MEMO_DIR is not None:
                self._r = loci.formats.Memoizer(self._r, memoize, BIOFORMATS_MEMO_DIR)
            else:
                self._r = loci.formats.Memoizer(self._r, memoize)

        if options:
            mo = loci.formats.in_.DynamicMetadataOptions()
            for name, value in options.items():
                mo.set(name, str(value))
            self._r.setMetadataOptions(mo)

        self.open()
        self._lock = Lock()
        self.set_series(series)

        self.dask_tiles = dask_tiles
        if self.dask_tiles:
            if tile_size is None:
                self.tile_size = (
                    self._r.getOptimalTileHeight(),
                    self._r.getOptimalTileWidth(),
                )
            else:
                self.tile_size = tile_size

    def set_series(self, series: int = 0) -> None:
        self._r.setSeries(series)
        self._core_meta = CoreMeta(
            (
                self._r.getSizeT(),
                self._r.getEffectiveSizeC(),
                self._r.getSizeZ(),
                self._r.getSizeY(),
                self._r.getSizeX(),
                self._r.getRGBChannelCount(),
            ),
            _pixtype2dtype(self._r.getPixelType(), self._r.isLittleEndian()),
            self._r.getSeriesCount(),
            self._r.isRGB(),
            self._r.isInterleaved(),
            self._r.getDimensionOrder(),
            self._r.getResolutionCount(),
        )

    @property
    def core_meta(self) -> CoreMeta:
        return self._core_meta

    def open(self) -> None:
        """Open file."""
        self._r.setId(self._path)

    def close(self) -> None:
        """Close file."""
        try:
            self._r.close()
        except (AttributeError, ImportError, RuntimeError):
            pass

    def to_numpy(self, series: Optional[int] = None) -> np.ndarray:
        """Create numpy array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        Parameters
        ----------
        series : int, optional
            The series index to retrieve, by default None
        """
        return np.asarray(self.to_dask(series))

    def to_dask(self, series: Optional[int] = None) -> DaskArrayProxy:
        """Create dask array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        The returned object is a `DaskArrayProxy`, which is a wrapper on a dask array
        that ensures the file is open when actually reading (computing) a chunk.  It
        has all the methods and behavior of a dask array.
        see :class:`aicsimageio.utils.dask_proxy.DaskArrayProxy`.

        Returns
        -------
        DaskArrayProxy
        """
        if series is not None:
            self._r.setSeries(series)

        nt, nc, nz, ny, nx, nrgb = self.core_meta.shape

        if self.dask_tiles:
            chunks = _get_dask_tile_chunks(nt, nc, nz, ny, nx, self.tile_size)
        else:
            chunks = ((1,) * nt, (1,) * nc, (1,) * nz, (ny,), (nx,))

        if nrgb > 1:
            chunks = chunks + (nrgb,)  # type: ignore
        arr = da.map_blocks(
            self._dask_chunk,
            chunks=chunks,
            dtype=self.core_meta.dtype,
        )
        return DaskArrayProxy(arr, self)

    @property
    def closed(self) -> bool:
        """Whether the underlying file is currently open"""
        return not bool(self._r.getCurrentFile())

    @property
    def filename(self) -> str:
        """Return name of file handle."""
        # return self._r.getCurrentFile()
        return self._path

    @property
    def ome_xml(self) -> str:
        """return OME XML string."""
        with self:
            store = self._r.getMetadataStore()

            return str(store.dumpXML()) if store else ""

    @property
    def ome_metadata(self) -> OME:
        """Return OME object parsed by ome_types."""
        xml = metadata_utils.clean_ome_xml_for_known_issues(self.ome_xml)
        return OME.from_xml(xml)

    def __enter__(self) -> BioFile:
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _get_plane(
        self,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        y: slice = slice(None),
        x: slice = slice(None),
    ) -> np.ndarray:
        """Load bytes from a single plane.

        Parameters
        ----------
        t : int, optional
            the time index, by default 0
        c : int, optional
            the channel index, by default 0
        z : int, optional
            the z index, by default 0
        y : slice, optional
            a slice object to select a Y subset of the plane, by default: full axis.
        x : slice, optional
            a slice object to select a X subset of the plane, by default: full axis.

        Returns
        -------
        np.ndarray
            array of requested bytes.
        """
        with self._lock:
            was_open = not self.closed
            if not was_open:
                self.open()

            *_, ny, nx, nrgb = self.core_meta.shape

            # get bytes from bioformats
            idx = self._r.getIndex(z, c, t)
            ystart, ywidth = _slice2width(y, ny)
            xstart, xwidth = _slice2width(x, nx)
            # read bytes using bioformats
            buffer = self._r.openBytes(idx, xstart, ystart, xwidth, ywidth)
            # convert buffer to numpy array
            im = np.frombuffer(bytes(buffer), self.core_meta.dtype)

            # reshape
            if nrgb > 1:
                if self.core_meta.is_interleaved:
                    im.shape = (ywidth, xwidth, nrgb)
                else:
                    im.shape = (nrgb, ywidth, xwidth)
                    im = np.transpose(im, (1, 2, 0))
            else:
                im.shape = (ywidth, xwidth)

            if not was_open:
                self.close()

        return im

    def _dask_chunk(self, block_id: Tuple[int, ...]) -> np.ndarray:
        """Retrieve `block_id` from array.

        This function is for map_blocks (called in `to_dask`).
        If someone indexes a 5D dask array as `arr[0, 1, 2]`, then 'block_id'
        will be (0, 1, 2, 0, 0)
        """
        # Our convention is that the final dask array is in the order TCZYX, so
        # block_id will be coming in as (T, C, Z, Y, X).
        t, c, z, y, x, *_ = block_id

        if self.dask_tiles:
            *_, ny, nx, _ = self.core_meta.shape
            y_slice = _axis_id_to_slice(y, self.tile_size[0], ny)
            x_slice = _axis_id_to_slice(x, self.tile_size[1], nx)
            im = self._get_plane(t, c, z, y_slice, x_slice)
        else:
            im = self._get_plane(t, c, z)

        return im[np.newaxis, np.newaxis, np.newaxis]

    _service: Any = None

    @classmethod
    def _create_ome_meta(cls) -> Any:
        """create an OMEXMLMetadata object to populate"""
        from bioformats_jar import get_loci

        loci = get_loci()
        if not cls._service:
            factory = loci.common.services.ServiceFactory()
            cls._service = factory.getInstance(loci.formats.services.OMEXMLService)
        return cls._service.createOMEXMLMetadata()


def _pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a loci.formats PixelType integer into a numpy dtype."""
    from bioformats_jar import get_loci

    FT = get_loci().formats.FormatTools
    fmt2type: Dict[int, str] = {
        FT.INT8: "i1",
        FT.UINT8: "u1",
        FT.INT16: "i2",
        FT.UINT16: "u2",
        FT.INT32: "i4",
        FT.UINT32: "u4",
        FT.FLOAT: "f4",
        FT.DOUBLE: "f8",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


def _chunk_by_tile_size(n_px: int, tile_length: int) -> Tuple[int, ...]:
    n_splits = n_px / tile_length
    n_full_tiles = np.floor(n_splits)

    if n_splits.is_integer():
        tile_chunks = (int(tile_length),) * int(n_full_tiles)
    else:
        edge_tile = n_px - (n_full_tiles * tile_length)
        tile_chunks = (int(tile_length),) * int(n_full_tiles) + (int(edge_tile),)
    return tile_chunks


def _get_dask_tile_chunks(
    nt: int, nc: int, nz: int, ny: int, nx: int, tile_size: Tuple[int, int]
) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]
]:
    """Returns chunking tuples (length of each chunk in each axis) after tiling.
    I.e., if nx == 2048 and tile_size == 1024, chunks for x axis will be (1024,1024)"""

    y_tile_size, x_tile_size = tile_size

    y_tiling_chunks = _chunk_by_tile_size(ny, y_tile_size)
    x_tiling_chunks = _chunk_by_tile_size(nx, x_tile_size)

    return ((1,) * nt, (1,) * nc, (1,) * nz, y_tiling_chunks, x_tiling_chunks)


def _axis_id_to_slice(axis_id: int, tile_length: int, n_px: int) -> slice:
    """Take the axis_id from a dask block_id and create the corresponding
    tile slice, taking into account edge tiles."""
    if (axis_id * tile_length) + tile_length <= n_px:
        return slice(axis_id * tile_length, (axis_id * tile_length) + tile_length)
    else:
        return slice(axis_id * tile_length, n_px)


def _slice2width(slc: slice, length: int) -> Tuple[int, int]:
    """Convert `slice` object into (start, width)"""
    if slc.stop is not None or slc.start is not None:
        # NOTE: we're ignoring step != 1 here
        start, stop, _ = slc.indices(length)
        return min(start, stop), abs(stop - start)
    return 0, length


@lru_cache(maxsize=1)
def _hide_memoization_warning() -> None:
    """HACK: this silences a warning about memoization for now

    An illegal reflective access operation has occurred
    https://github.com/ome/bioformats/issues/3659
    """
    import jpype

    System = jpype.JPackage("java").lang.System
    System.err.close()
