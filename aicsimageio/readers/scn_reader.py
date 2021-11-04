from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from tifffile.tifffile import TiffFile, TiffFileError, ZarrTiffStore, xml2dict

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..utils import io_utils
from .reader import Reader

if TYPE_CHECKING:
    from fsspec.spec import AbstractFileSystem


class SCNReader(Reader):
    """Read Leica SCN files using tifffile.
    This does not support SCN files containing Z-stacks but works for
    both multi-channel fluorescence, and brightfield RGB SCNs.

    Parameters
    ----------
    image : Path or str
        path to file

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        If the file is not supported by tifffile backed SCNReader.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource) as rdr:
                    return bool(rdr.scn_metadata)

        except (TiffFileError, TypeError):
            return False

    def __init__(self, image: types.PathLike):
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

        with TiffFile(self._path) as rdr:
            self._scn_metadata = xml2dict(rdr.scn_metadata)
            self._image_metadata = self._scn_metadata["scn"]["collection"]["image"]

    @property
    def scenes(self) -> Tuple[str, ...]:
        """Scenes are accessed by their UUID name within metadata
        This name is extended with the Series and Sub-Resolution index
        (S and R, respectively)"""
        scenes: Tuple[str, ...] = tuple()
        for idx, sub_image_meta in enumerate(self._image_metadata):
            series_meta = sub_image_meta.get("pixels").get("dimension")
            res_indices = [d["r"] for d in series_meta]
            n_res = np.max(res_indices) + 1
            scenes = scenes + tuple(
                f"{sub_image_meta.get('name')} S{idx}-(R{sr})" for sr in range(n_res)
            )

        return scenes

    def _get_series_level_meta(self, series_idx: int, level_idx: int) -> Dict:
        """tifffile organizes data series -> levels (sub-resolutions)
        The metadata has similar organization but with series -> level -> channel
        This makes sure the correct X and Y pixel lengths are pulled
        """

        # access metadata by series index which matches
        # tifffiles "series" attribute
        series_level_meta = (
            self._image_metadata[series_idx].get("pixels").get("dimension")
        )

        return [md for md in series_level_meta if md.get("r") == level_idx][0]

    def _get_xy_pixel_sizes(
        self, rdr: TiffFile, series_idx: int, level_idx: int
    ) -> types.PhysicalPixelSizes:
        """Resolution data is stored in the TIFF tags in
        Pixels per cm, this is converted to microns per pixel
        """
        # pages are accessed because they contain the tiff tag
        # subset by series -> level -> first page contains all tags
        current_page = rdr.series[series_idx].levels[level_idx].pages[0]

        x_res = current_page.tags["XResolution"].value
        y_res = current_page.tags["XResolution"].value

        res_unit = current_page.tags["ResolutionUnit"].value

        # convert units to micron
        # res_unit == 1: undefined (px)
        # res_unit == 2 pixels per inch
        # res unit == 3 pixels per cm
        # in all cases we convert to um
        # https://www.awaresystems.be/imaging/tiff/tifftags/resolutionunit.html
        if res_unit.value == 1:
            res_to_um = 1
        if res_unit.value == 2:
            res_to_um = 25400
        elif res_unit.value == 3:
            res_to_um = 10000

        # conversion of pixels / um to um / pixel
        x_res_um = (1 / (x_res[0] / x_res[1])) * res_to_um
        y_res_um = (1 / (y_res[0] / y_res[1])) * res_to_um

        return types.PhysicalPixelSizes(X=x_res_um, Y=y_res_um, Z=None)

    def _get_is_rgb(self, rdr: TiffFile, series_idx: int, level_idx: int) -> bool:
        """Use PhotometricInterpretation TIFF tag to determine if
        RGB (interleaved, multiple samples per pixel) or
        non-interleaved channels
        """
        tpage = rdr.series[series_idx].levels[level_idx].pages[0]
        photometric = tpage.tags["PhotometricInterpretation"].value

        return photometric.name in {"RGB", "YCBCR"}

    def _get_ch_names(self, series_idx: int) -> List[str]:
        """Pull channel names from SCN metadata a given series"""
        channel_meta = (
            self._image_metadata[series_idx]
            .get("scanSettings")
            .get("channelSettings")
            .get("channel")
        )

        ch_names = [d.get("name") for d in channel_meta]
        return ch_names

    def _generate_xarray_coords(
        self, dims: Dict[str, int], xy_res: types.PhysicalPixelSizes
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """Generate xr coord data from metadata and physical pixel size"""

        coords: Dict[str, Union[List[str], np.ndarray]] = {}

        y_res = xy_res.Y
        x_res = xy_res.X

        coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
            0, dims[DimensionNames.SpatialY], y_res
        )
        coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
            0, dims[DimensionNames.SpatialX], x_res
        )
        return coords

    def _get_current_tf_indices(self) -> Tuple[int, int]:
        current_scene = self.scenes[self.current_scene_index]

        # parse series and level indices built into the scene name
        series_idx = int(current_scene.split("S")[1].split("-")[0])
        level_idx = int(current_scene.split("R")[1].split(")")[0])

        return series_idx, level_idx

    def _read_delayed(self) -> xr.DataArray:
        return self._xr_read(delayed=True)

    def _read_immediate(self) -> xr.DataArray:
        return self._xr_read(delayed=False)

    def _xr_read(self, delayed: bool) -> xr.DataArray:
        with TiffFile(self._path) as rdr:

            series_idx, level_idx = self._get_current_tf_indices()

            # get this series and this sub-resolution's metadata
            series_level_meta = self._get_series_level_meta(series_idx, level_idx)

            # dim lengths are provided as size{DIM_NAME}
            _dims_list = [
                {f"{k.split('size')[1]}": v}
                for k, v in series_level_meta.items()
                if "size" in k
            ]
            _dims = {k: v for d in _dims_list for k, v in d.items()}

            is_rgb = self._get_is_rgb(rdr, series_idx, level_idx)

            xy_res = self._get_xy_pixel_sizes(rdr, series_idx, level_idx)

            coords = self._generate_xarray_coords(_dims, xy_res)

            # access zarr data
            zarr_data = zarr.open(
                ZarrTiffStore(rdr.series[series_idx].levels[level_idx])
            )

            # if the base level is accessed, tifffile will always return
            # a zarr group even if level 0 is requested
            # so the first component is accessed to get the zarr.Array
            # sub-resolutions will return a zarr.Array type
            if isinstance(zarr_data, zarr.hierarchy.Group):
                data = da.from_zarr(zarr_data[0])
            else:
                data = da.from_zarr(zarr_data)

            if is_rgb:
                _dims.update({"S": data.shape[-1]})
                dims = {"Y": _dims["Y"], "X": _dims["X"], "S": _dims["S"]}
            else:
                # if fluorescence, get channel names
                ch_names = self._get_ch_names(series_idx)
                n_ch = len(ch_names)

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

        series_idx, level_idx = self._get_current_tf_indices()

        with TiffFile(self._path) as rdr:
            xy_res = self._get_xy_pixel_sizes(rdr, series_idx, level_idx)
        return xy_res
