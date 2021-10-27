from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from fsspec.implementations.local import LocalFileSystem
from tifffile.tifffile import TiffFile, TiffFileError, ZarrTiffStore, xml2dict

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .reader import Reader

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem


class SCNReader(Reader):
    """Read Leica SCN files using tifffile.

    Parameters
    ----------
    image : Path or str
        path to file

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by ND2.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as rdr:
                    if rdr.scn_metadata:
                        return True
                    else:
                        return False

        except (TiffFileError, TypeError):
            return False

    def __init__(self, image: types.PathLike, scene: Optional[Union[int, str]] = None):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read SCN from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        with TiffFile(self._path) as rdr:
            self._scn_metadata = xml2dict(rdr.scn_metadata)

        if scene is None:
            scene = 0

        self.set_scene(scene)

    @property
    def scenes(self) -> Tuple[str, ...]:
        scenes = tuple()
        for idx, im in enumerate(
            self._scn_metadata.get("scn").get("collection").get("image")
        ):
            series_meta = (
                self._scn_metadata.get("scn")
                .get("collection")
                .get("image")[idx]
                .get("pixels")
                .get("dimension")
            )
            res_indices = [d.get("r") for d in series_meta]
            n_res = np.max(res_indices) + 1
            scenes = scenes + tuple(
                f"{im.get('name')} S{idx}-(R{sr})" for sr in range(n_res)
            )

        return scenes

    def _get_series_level_meta(self, series_idx: int, level_idx: int) -> Dict:
        """tifffile organizes data series -> level (sub-resolution)
        The metadata has similar organization but with series -> level -> channel
        This makes sure the correct X and Y pixel lengths are pulled
        """

        series_level_meta = (
            self._scn_metadata.get("scn")
            .get("collection")
            .get("image")[series_idx]
            .get("pixels")
            .get("dimension")
        )

        return [md for md in series_level_meta if md.get("r") == level_idx][0]

    def _get_xy_pixel_sizes(
        self, rdr: TiffFile, series_idx: int, level_idx: int
    ) -> Dict[str, float]:
        tpage = rdr.series[series_idx].levels[level_idx].pages[0]

        x_res = tpage.tags["XResolution"].value
        y_res = tpage.tags["XResolution"].value

        res_unit = tpage.tags["ResolutionUnit"].value

        # convert units to micron
        if res_unit.value == 1:
            res_to_um = 1
        if res_unit.value == 2:
            res_to_um = 25400
        elif res_unit.value == 3:
            res_to_um = 10000

        x_res_um = (1 / (x_res[0] / x_res[1])) * res_to_um
        y_res_um = (1 / (y_res[0] / y_res[1])) * res_to_um

        return {"X": x_res_um, "Y": y_res_um}

    def _get_is_rgb(self, rdr: TiffFile, series_idx: int, level_idx: int) -> bool:

        tpage = rdr.series[series_idx].levels[level_idx].pages[0]
        photometric = tpage.tags["PhotometricInterpretation"].value

        if photometric.name in ["RGB", "YCBCR"]:
            return True
        else:
            return False

    def _get_n_ch(self, series_idx: int) -> int:
        channel_idx_meta = (
            self._scn_metadata.get("scn")
            .get("collection")
            .get("image")[series_idx]
            .get("pixels")
            .get("dimension")
        )
        ch_indices = [d.get("c") for d in channel_idx_meta]
        max_ch_idx = np.max(ch_indices)
        return max_ch_idx + 1

    def _get_ch_names(self, series_idx: int) -> List[str]:
        channel_meta = (
            self._scn_metadata.get("scn")
            .get("collection")
            .get("image")[series_idx]
            .get("scanSettings")
            .get("channelSettings")
            .get("channel")
        )

        ch_names = [d.get("name") for d in channel_meta]
        return ch_names

    def _generate_xarray_coords(
        self, dims: Dict[str, int], x_res: float, y_res: float
    ) -> Dict[str, Union[List[str], np.ndarray]]:

        coords: Dict[str, Union[List[str], np.ndarray]] = {}

        coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
            0, dims[DimensionNames.SpatialY], y_res
        )
        coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
            0, dims[DimensionNames.SpatialX], x_res
        )
        return coords

    def _read_delayed(self) -> xr.DataArray:
        return self._xr_read(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xr_read(delayed=False)

    def _xr_read(self, delayed: bool) -> xr.DataArray:
        with TiffFile(self._path) as rdr:
            current_scene = self.scenes[self.current_scene_index]

            series_idx = int(current_scene.split("S")[1].split("-")[0])
            level_idx = int(current_scene.split("R")[1].split(")")[0])

            # get this series and this subresolution's metadata
            series_level_meta = self._get_series_level_meta(series_idx, level_idx)

            _dims_list = [
                {f"{k.split('size')[1]}": v}
                for k, v in series_level_meta.items()
                if "size" in k
            ]
            _dims = {k: v for d in _dims_list for k, v in d.items()}

            is_rgb = self._get_is_rgb(rdr, series_idx, level_idx)

            xy_res = self._get_xy_pixel_sizes(rdr, series_idx, level_idx)

            coords = self._generate_xarray_coords(
                _dims, x_res=xy_res["X"], y_res=xy_res["Y"]
            )

            zarr_data = zarr.open(
                ZarrTiffStore(rdr.series[series_idx].levels[level_idx])
            )

            if isinstance(zarr_data, zarr.hierarchy.Group):
                data = da.from_zarr(zarr_data[0])
            else:
                data = da.from_zarr(zarr_data)

            if is_rgb:
                _dims.update({"S": data.shape[-1]})
                dims = {"Y": _dims["Y"], "X": _dims["X"], "S": _dims["S"]}
            else:
                n_ch = self._get_n_ch(series_idx)
                ch_names = self._get_ch_names(series_idx)

                coords[DimensionNames.Channel] = ch_names

                _dims.update({"C": n_ch})
                dims = {
                    "C": _dims["C"],
                    "Y": _dims["Y"],
                    "X": _dims["X"],
                }

            if not delayed:
                data = data.compute()

            return xr.DataArray(
                data,
                dims=dims,
                coords=coords,
                attrs={
                    constants.METADATA_UNPROCESSED: rdr.scn_metadata,
                    constants.METADATA_PROCESSED: self._scn_metadata,
                },
            )

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
        current_scene = self.scenes[self.current_scene_index]
        series_idx = int(current_scene.split("S")[1].split("-")[0])
        level_idx = int(current_scene.split("R")[1].split(")")[0])
        with TiffFile(self._path) as rdr:
            xy_res = self._get_xy_pixel_sizes(rdr, series_idx, level_idx)
        return types.PhysicalPixelSizes(X=xy_res["X"], Y=xy_res["Y"], Z=None)
