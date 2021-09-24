from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from fsspec.implementations.local import LocalFileSystem

from .. import constants, exceptions
from ..utils import io_utils
from ..utils.cached_property import cached_property
from ..utils.dask_proxy import DaskArrayProxy
from .reader import Reader

if TYPE_CHECKING:
    import xarray as xr
    from fsspec.spec import AbstractFileSystem
    from ome_types import OME

    from .. import types


try:
    import nd2
except ImportError:
    raise ImportError(
        "The nd2 package is required for this reader. "
        "Install with `pip install aicsimageio[nd2]`"
    )


class ND2Reader(Reader):
    """Read NIS-Elements files using the Nikon nd2 SDK.

    This reader requires `nd2` to be installed in the environment.

    Parameters
    ----------
    image : Path or str
        path to file

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by bioformats.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        from nd2._util import NEW_HEADER_MAGIC_NUM

        with fs.open(path, "rb") as fh:
            return int.from_bytes(fh.read(4), "little") == NEW_HEADER_MAGIC_NUM

    def __init__(self, image: types.PathLike):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read Bioformats from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        return ("Image:0",)  # TODO

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=False)

    def _xarr_reformat(self, delayed: bool) -> xr.DataArray:
        with nd2.ND2File(self._path) as ndfile:
            xarr = ndfile.to_xarray(delayed=delayed)
            if delayed:
                xarr.data = DaskArrayProxy(xarr.data, xarr.data._ctx)
            xarr.attrs[constants.METADATA_UNPROCESSED] = xarr.attrs.pop("metadata")
        return xarr

    @cached_property
    def ome_metadata(self) -> OME:
        """Return OME object parsed by ome_types."""
        from ..metadata.utils import bioformats_ome

        return bioformats_ome(self._path)

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
        with nd2.ND2File(self._path) as nd2file:
            return nd2file.pixel_size()
