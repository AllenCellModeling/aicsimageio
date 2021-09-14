#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

import dask.array as da
import numpy as np
import xarray as xr

from ..constants import METADATA_PROCESSED, METADATA_UNPROCESSED
from ..dimensions import DimensionNames
from .reader import Reader

if TYPE_CHECKING:
    import jpype
    from fsspec.spec import AbstractFileSystem

    from .. import types


# fmt: off
JAVA_MEMORY = "1024m"
SUPPORTED_FORMATS = (
    '.afm', '.nef', '.lif', '.nhdr', '.ps', '.bmp', '.frm', '.pr3', '.tif',
    '.aim', '.dat', '.fits', '.pcoraw', '.qptiff', '.acff', '.xys', '.mrw',
    '.xml', '.svs', '.arf', '.dm4', '.ome.xml', '.v', '.pds', '.zvi', '.apl',
    '.mrcs', '.i2i', '.mdb', '.ipl', '.oir', '.ali', '.fff', '.vms', '.jpg',
    '.inr', '.pcx', '.vws', '.html', '.al3d', '.ims', '.bif', '.labels',
    '.dicom', '.par', '.map', '.ome.tf2', '.htd', '.tnb', '.mrc',
    '.obf', '.xdce', '.png', '.jpx', '.fli', '.psd', '.pgm', '.obsep',
    '.jpk', '.ome.tif', '.rcpnl', '.pbm', '.grey', '.raw', '.zfr', '.klb',
    '.spc', '.sdt', '.2fl', '.ndpis', '.ipm', '.pict', '.st', '.seq', '.nii',
    '.lsm', '.epsi', '.cr2', '.zfp', '.wat', '.lim', '.1sc', '.ffr', '.liff',
    '.mea', '.nd2', '.tf8', '.naf', '.ch5', '.afi', '.ipw', '.img', '.ids',
    '.mnc', '.crw', '.mtb', '.cxd', '.gel', '.dv', '.jpf', '.tga', '.vff',
    '.ome.tiff', '.ome', '.bin', '.cfg', '.dti', '.ndpi', '.c01', '.avi',
    '.sif', '.flex', '.txt', '.spe', '.ics', '.jp2', '.xv', '.spi', '.lms',
    '.sld', '.vsi', '.lei', '.sm3', '.hx', '.czi', '.nrrd', '.ppm', '.exp',
    '.mov', '.xqd', '.dm3', '.im3', '.pic', '.his', '.j2k', '.rec', '.top',
    '.pnl', '.tf2', '.oif', '.l2d', '.stk', '.fdf', '.mng', '.ome.btf',
    '.tfr', '.res', '.dm2', '.eps', '.hdr', '.am', '.stp', '.sxm',
    '.ome.tf8', '.dib', '.mvd2', '.wlz', '.nd', '.h5', '.cif', '.mod',
    '.nii.gz', '.bip', '.oib', '.csv', '.amiramesh', '.scn', '.gif',
    '.sm2', '.tiff', '.hdf', '.hed', '.r3d', '.wpi', '.dcm', '.btf',
    '.msr', '.xqf'
)
_FMT_2_DTYPE = {
    0: "i1",  # int8
    1: "u1",  # uint8
    2: "i2",  # int16
    3: "u2",  # uint16
    4: "i4",  # int32
    5: "u4",  # uint32
    6: "f4",  # float32
    7: "f8",  # float64
}
# fmt: on


def _get_loci_reader(path: types.PathLike, meta: bool = False) -> Tuple[Any, Any]:
    loci = _load_loci()
    rdr = loci.formats.ChannelSeparator(loci.formats.ChannelFiller())
    _meta = None
    if meta:
        _meta = loci.formats.MetadataTools.createOMEXMLMetadata()
        rdr.setMetadataStore(_meta)
    rdr.setId(str(path))
    return rdr, _meta


def _load_loci() -> jpype.JPackage:
    try:
        import jpype
    except ImportError:
        raise ImportError(
            "jpype is required for this reader. "
            "Install with `pip install aicsimageio[bioformats]`"
        )

    # Start java VM and initialize logger (globally)
    if not jpype.isJVMStarted():
        jpype.startJVM(
            jpype.getDefaultJVMPath(),
            "-ea",
            f"-Djava.class.path={Path(__file__).parent / 'bioformats.jar'}",
            "-Xmx" + JAVA_MEMORY,
            convertStrings=False,
        )
        loci = jpype.JPackage("loci")
        loci.common.DebugTools.setRootLevel("ERROR")

    try:
        java_lang = jpype.JPackage("java").lang
        if not java_lang.Thread.isAttached():
            java_lang.Thread.attach()
    except Exception:
        if not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()

    return jpype.JPackage("loci")


class BioformatsReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        # TODO: perhaps try to read first?
        return Path(path).suffix in SUPPORTED_FORMATS

    def __init__(self, image: types.PathLike):
        self._rdr, self._meta = _get_loci_reader(image, True)
        self._lock = threading.Lock()

    @property
    def scenes(self) -> Tuple[str, ...]:
        # TODO
        return ("Image:0",)

    def _read_delayed(self) -> xr.DataArray:
        return xr.DataArray(
            self._to_dask_array(),
            dims=self._dimensions,
            coords=self._coords,
            attrs=self._attrs,
        )

    def _read_immediate(self) -> xr.DataArray:
        # TODO: can make this faster without dask
        return xr.DataArray(
            self._to_dask_array().compute(),
            dims=self._dimensions,
            coords=self._coords,
            attrs=self._attrs,
        )

    @property
    def _shape(self) -> Tuple[int, int, int, int, int]:
        # currently hardcoded as TCZYX
        r = self._rdr
        return (
            r.getSizeT(),
            r.getRGBChannelCount() if r.isRGB() else r.getSizeC(),
            r.getSizeZ(),
            r.getSizeY(),
            r.getSizeX(),
        )

    @property
    def _dtype(self) -> str:
        """Returns shape (TCZYX) and dtype of rdr"""
        le = self._rdr.isLittleEndian()
        return ("<" if le else ">") + _FMT_2_DTYPE[self._rdr.getPixelType()]

    @property
    def _dimensions(self) -> List[str]:
        # TODO: do better
        return list("TCZYX")

    @property
    def _coords(self) -> dict:
        if not hasattr(self, "_coords_"):
            self._coords_ = {}

            for d, s in zip(self._dimensions, self._shape):
                getpix = getattr(self._meta, f"getPixelsPhysicalSize{d}", None)
                size = 1.0
                if getpix is not None:
                    _size = getpix(0)
                    size = float(_size.value()) if _size is not None else 1
                self._coords_[d] = np.arange(0, s) * size

        return self._coords_

    @property
    def _attrs(self) -> dict:
        if not hasattr(self, "_attrs_"):
            import ome_types

            xml = str(self._meta.dumpXML())
            self._attrs_ = {
                METADATA_UNPROCESSED: xml,
                METADATA_PROCESSED: ome_types.from_xml(xml),
            }
        return self._attrs_

    def _to_dask_array(self) -> da.Array:
        nt, nc, nz, ny, nx = self._shape
        return da.map_blocks(
            self._load_block,
            chunks=((1,) * nt, (1,) * nc, (1,) * nz, (ny,), (nx,)),
            dtype=self._dtype,
        )

    def _load_block(self, block_info: dict) -> np.ndarray:
        """Load a single plane with a loci reader"""
        info = block_info[None]
        idx = self._rdr.getIndex(*info["chunk-location"][2::-1])  # Z, C, T
        with self._lock:
            im = np.frombuffer(self._rdr.openBytes(idx)[:], dtype=info["dtype"]).copy()
        im.shape = info["shape"][-2:]
        return im[np.newaxis, np.newaxis, np.newaxis]
