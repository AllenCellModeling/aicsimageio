from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from functools import lru_cache
import numpy as np
from numpy.core.shape_base import block
from wrapt import ObjectProxy
from typing import NamedTuple

if TYPE_CHECKING:
    from pathlib import Path
    from bioformats_jar import loci

    import dask.array as da
    from ome_types import OME


class CoreMeta(NamedTuple):
    shape: Tuple[int, int, int, int, int]
    dtype: str
    series_count: int
    is_rgb: bool
    is_interleaved: bool
    dimension_order: str


def get_ome_metadata(path) -> 'OME':
    """Helper to retrieve OME meta from any compatible file, using bioformats."""
    with LociFile(path) as lf:
        return lf.ome_metadata


class LociFile:
    """Read image and metadata from file supported by Bioformats.

    LociFile instances must be closed using the 'close' method, which is
    automatically called when using the 'with' context manager.

    LociFile instances are not thread-safe.

    Parameters
    ----------
    path : Union[str, Path]
        path to file
    series : int, optional
        the image series to read, by default 0
    meta : bool, optional
        whether to get metadata as well, by default True
    memoize : int, optional
        threshold for memoizing the reader. If the time required to call
        `reader.setId()` is larger than this number, the initialized reader
        (including all reader wrappers) will be cached in a memo file for later
        retrieval. by default 50.  set to 0 or less to turn off memoization.
        https://downloads.openmicroscopy.org/bio-formats/latest/api/loci/formats/Memoizer.html
    """

    def __init__(
        self,
        path: Union[str, "Path"],
        series=0,
        meta=True,
        memoize=50,
    ):
        from bioformats_jar import loci

        self._path = str(path)
        self._r = loci.formats.ImageReader()
        if meta:
            self._r.setMetadataStore(self._create_ome_meta())

        # memoize to save time on later re-openings of the same file.
        if memoize > 0:
            _hide_memoization_warning()
            self._r = loci.formats.Memoizer(self._r, memoize)

        self.open()
        self._lock = Lock()

        if "ND2" in str(self._r.getReader()):
            # https://github.com/openmicroscopy/bioformats/issues/2955
            mo = loci.formats.in_.DynamicMetadataOptions()
            mo.set("nativend2.chunkmap", "False")
            self._r.setMetadataOptions(mo)

        self._nseries = self._r.getSeriesCount()
        self.set_series(series)

    def set_series(self, series=0):
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
        )

    @property
    def core_meta(self) -> CoreMeta:
        return self._core_meta

    def open(self) -> None:
        """Open file."""
        self._r.setId(self._path)

    def close(self) -> None:
        """Close file."""
        self._r.close()

    def to_numpy(self, series: Optional[int] = None) -> np.ndarray:
        """Create numpy array for the current series."""
        return np.asarray(self.to_dask(series))
        # doesn't seem any faster ...
        # from itertools import product
        # nt, nc, nz, *_ = self._shape
        # out = np.empty(self._shape)
        # for z, c, t in product(range(nz), range(nc), range(nt)):
        #     out[t, c, z] = self._get_plane(z, c, t)
        # return out

    def to_dask(self, series: Optional[int] = None) -> "da.Array":
        """Create dask array for the current series."""
        import dask.array as da

        if series is not None:
            self._r.setSeries(series)

        nt, nc, nz, ny, nx, nrgb = self.core_meta.shape
        chunks = ((1,) * nt, (1,) * nc, (1,) * nz, ny, nx)
        if nrgb > 1:
            chunks = chunks + (nrgb,)
        arr = da.map_blocks(
            self._dask_chunk,
            chunks=chunks,
            dtype=self.core_meta.dtype,
        )
        return DaskArrayProxy(arr, self)

    @property
    def is_open(self) -> bool:
        """Whether the underlying file is currently open"""
        return bool(self._r.getCurrentFile())

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
    def ome_metadata(self) -> "OME":
        """Return OME object parsed by ome_types."""
        import ome_types

        from ..metadata import utils as metadata_utils

        xml = metadata_utils.clean_ome_xml_for_known_issues(self.ome_xml)
        return ome_types.from_xml(xml)

    def __enter__(self) -> "LociFile":
        if not self.is_open:
            self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if __import__("jpype").isJVMStarted():
                self.close()
        except (ImportError, RuntimeError):
            pass

    def _get_plane(
        self, t=0, c=0, z=0, y: slice = slice(None), x: slice = slice(None)
    ) -> np.ndarray:
        """Load a single plane."""
        idx = self._r.getIndex(z, c, t)
        *_, ny, nx, nrgb = self.core_meta.shape
        ystart, ywidth = _slice2width(y, ny)
        xstart, xwidth = _slice2width(x, nx)
        with self._lock:
            buffer = self._r.openBytes(idx, xstart, ystart, xwidth, ywidth)
            # TODO: check what the bare minimim amount of copying is to prevent
            # reading incorrect regions (doesnt' segfault, but can get weird)
            im = np.frombuffer(buffer[:], self.core_meta.dtype).copy()
        if nrgb > 1:
            # TODO: check this with some examples
            if self.core_meta.is_interleaved:
                im.shape = (ywidth, xwidth, nrgb)
            else:
                im.shape = (nrgb, ywidth, xwidth)
                im = np.transpose(im, (1, 2, 0))
        else:
            im.shape = (ywidth, xwidth)
        # TODO: check this... we might need to reshaped the non interleaved case
        return im

    def _dask_chunk(self, block_id: tuple) -> np.ndarray:
        # if someone indexes a 5D dask array as `arr[0,1,2]`, then the `info`
        # dict will contain the key: 'chunk-location': (0, 1, 2, 0, 0)
        # reader.getIndex() expects (Z, C, T) ...
        # We ASSUME that the final dask array is in the order TCZYX, so chunk-location
        # will be coming in as (T, C, Z, Y, X). `[2::-1]`` converts that to (Z, C, T)
        # TODO: we could either use the native getDimensionOrder? (or not) or let
        # the user
        t, c, z, *_ = block_id
        im = self._get_plane(t, c, z)
        return im[np.newaxis, np.newaxis, np.newaxis]

    _service: Optional["loci.common.services.ServiceFactory"] = None

    @classmethod
    def _create_ome_meta(cls) -> Any:
        from bioformats_jar import loci

        if not cls._service:
            factory = loci.common.services.ServiceFactory()
            cls._service = factory.getInstance(loci.formats.services.OMEXMLService)
        return cls._service.createOMEXMLMetadata()


def _pixtype2dtype(pixeltype: int, little_endian: bool) -> str:
    from bioformats_jar import loci

    FT = loci.formats.FormatTools
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
    return ("<" if little_endian else ">") + fmt2type[pixeltype]


class ArrayMethodProxy:
    def __init__(self, method, reader) -> None:
        self.method = method
        self._r = reader

    def __repr__(self):
        return repr(self.method)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        import dask.array as da

        with self._r:
            result = self.method(*args, **kwds)
        if isinstance(result, da.Array):
            return DaskArrayProxy(result, self._r)
        return result


class DaskArrayProxy(ObjectProxy):
    def __init__(self, wrapped, reader):
        super().__init__(wrapped)
        self._r = reader

    def __getitem__(self, key):
        return DaskArrayProxy(self.__wrapped__.__getitem__(key), self._r)

    def __getattr__(self, key):
        attr = getattr(self.__wrapped__, key)
        if callable(attr):
            return ArrayMethodProxy(attr, self._r)
        return attr

    def __repr__(self):
        return repr(self.__wrapped__)

    def compute(self, **kwargs):
        with self._r:
            return self.__wrapped__.compute(**kwargs)

    def __array__(self, dtype=None, **kwargs):
        with self._r:
            return self.__wrapped__.__array__(dtype, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        with self._r:
            return self.__wrapped__.__array_function__(func, types, args, kwargs)


def _slice2width(slc, length):
    if slc.stop is not None or slc.start is not None:
        # NOTE: we're ignoring step != 1 here
        start, stop, _ = slc.indices(length)
        return min(start, stop), abs(stop - start)
    return 0, length


@lru_cache
def _hide_memoization_warning():
    import jpype

    # hack: this silences a warning about memoization for now
    # "An illegal reflective access operation has occurred"
    # https://github.com/ome/bioformats/issues/3659
    System = jpype.JPackage("java").lang.System
    System.err.close()
