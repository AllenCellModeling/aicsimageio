from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

from fsspec.implementations.local import LocalFileSystem

from .. import constants, exceptions, types
from ..utils import io_utils
from ..utils.dask_proxy import DaskArrayProxy
from .reader import Reader

if TYPE_CHECKING:
    import xarray as xr
    from fsspec.spec import AbstractFileSystem


try:
    from mrc import DVFile
except ImportError:
    raise ImportError(
        "The mrc package is required for this reader. "
        "Install with `pip install aicsimageio[dv]`"
    )


class DVReader(Reader):
    """Read DV/Deltavision files.

    This reader requires `mrc` to be installed in the environment.

    Parameters
    ----------
    image : Path or str
        path to file

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not a supported dv file.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        return DVFile.is_supported_file(path)

    def __init__(self, image: types.PathLike):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise NotImplementedError(
                f"dv reader not yet implemented for non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        return ("Image:0",)

    def _read_delayed(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xarr_reformat(delayed=False)

    def _xarr_reformat(self, delayed: bool) -> xr.DataArray:
        with DVFile(self._path) as dv:
            xarr = dv.to_xarray(delayed=delayed, squeeze=False)
            if delayed:
                xarr.data = DaskArrayProxy(xarr.data, dv)
            xarr.attrs[constants.METADATA_UNPROCESSED] = xarr.attrs.pop("metadata")
        return xarr

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
        with DVFile(self._path) as dvfile:
            return types.PhysicalPixelSizes(*dvfile.voxel_size[::-1])
