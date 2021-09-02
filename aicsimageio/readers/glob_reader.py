#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import re
import glob
from pathlib import Path
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError, imread
from tifffile.tifffile import TiffTags

from .. import constants, exceptions, types
from ..dimensions import (
    DEFAULT_CHUNK_DIMS,
    REQUIRED_CHUNK_DIMS,
    DimensionNames,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES,
)
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from .reader import Reader


class GlobReader(Reader):

    """
    Wraps the tifffile imread API to provide the same aicsimageio Reader API but for
    multifile tiff datasets (and other tifffile supported) images.

    Parameters
    ----------
    glob_in: Union[str, List[str]]
        Glob string that identifies all files to be loaded or a list
        of paths to the files as returned by glob.
    indexer: Union[Callable, pandas.DataFrame]
        If callable, should consume each filename and return a pd.Series with index
        corresponding to the dimensions and values corresponding to the index
    chunk_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.
    dim_order: Optional[Union[List[str], str]]
        A string of dimensions to be applied to all array(s) or a
        list of string dimension names to be mapped onto the list of arrays
        provided to image. I.E. "TYX".
        Default: None (guess dimensions for single array or multiple arrays)
    channel_names: Optional[Union[List[str], List[List[str]]]]
        A list of string channel names to be applied to all array(s) or a
        list of lists of string channel names to be mapped onto the list of arrays
        provided to image.
        Default: None (create OME channel IDs for names for single or multiple arrays)
    single_file_shape : Optional[Tuple]
        Expected shape for a single file of the set. If not provided, will attempt to
        determine the shape from the first file found in the glob.
        Default : None
    single_file_dims : Optional[Tuple]
        Dimensions that correspond to the data dimensions of a single file in the glob.
        Default : ('Y', 'X')
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource):
                    return True

        except (TiffFileError, TypeError):
            return False

    def __init__(
        self,
        glob_in: Union[str, List[str]],
        indexer: Union[pd.DataFrame, Callable] = None,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        dim_order: Optional[Union[List[str], str]] = None,
        channel_names: Optional[Union[List[str], List[List[str]]]] = None,
        single_file_shape: Optional[Tuple] = None,
        single_file_dims: Optional[Tuple] = (
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
        ),
        **kwargs: Any,
    ):

        # Assmble glob list if given a string
        if isinstance(glob_in, str):
            file_series = pd.Series(glob.glob(glob_in))
        elif isinstance(glob_in, list):
            file_series = pd.Series(glob_in)  # pd.DataFrame({"filename": glob_in})

        if len(file_series) == 0:
            raise ValueError("No files found matching glob pattern")

        if indexer is None:
            series_idx = [
                DimensionNames.Samples,
                DimensionNames.Time,
                DimensionNames.Channel,
                DimensionNames.SpatialZ,
            ]
            indexer = lambda x: pd.Series(
                re.findall(r"\d+", Path(x).name), index=series_idx
            ).astype(int)

        if callable(indexer):
            self._all_files = file_series.apply(indexer)
            self._all_files["filename"] = file_series
        elif isinstance(indexer, pd.DataFrame):
            self._all_files = indexer
            self._all_files["filename"] = file_series

        sort_order = []
        for dim in DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES:
            if dim not in self._all_files.columns and dim not in single_file_dims:
                self._all_files[dim] = 0
            if dim in self._all_files.columns:
                sort_order.append(dim)

        self._all_files = self._all_files.sort_values(sort_order).reset_index(drop=True)

        # run tests on a single file (?)
        self._fs, self._path = io_utils.pathlike_to_fs(self._all_files.iloc[0].filename)

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        # Run basic checks on dims and channel names
        if isinstance(dim_order, list):
            if len(dim_order) != len(self.scenes):
                raise exceptions.ConflictingArgumentsError(
                    f"Number of dimension strings provided does not match the "
                    f"number of scenes found in the file. "
                    f"Number of scenes: {len(self.scenes)}, "
                    f"Number of provided dimension order strings: {len(dim_order)}"
                )

        # If provided a list
        if isinstance(channel_names, list):
            # If provided a list of lists
            if len(channel_names) > 0 and isinstance(channel_names[0], list):
                # Ensure that the outer list is the number of scenes
                if len(channel_names) != len(self.scenes):
                    raise exceptions.ConflictingArgumentsError(
                        f"Number of channel name lists provided does not match the "
                        f"number of scenes found in the file. "
                        f"Number of scenes: {len(self.scenes)}, "
                        f"Provided channel name lists: {dim_order}"
                    )
            self._channel_names = channel_names

        self.chunk_dims = chunk_dims

        for dim in REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        self._dim_order = dim_order
        self._channel_names = channel_names

        if single_file_shape is None:
            with self._fs.open(self._path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    if tiff.is_shaped:
                        self._single_file_shape = tuple(
                            tiff.shaped_metadata[0]["shape"]
                        )
                    elif len(tiff.series) == 1:
                        self._single_file_shape = tiff.series[0].shape

        else:
            self._single_file_shape = single_file_shape

        if len(single_file_dims) != len(self._single_file_shape):
            raise exceptions.ConflictingArgumentsError(
                f"Number of single file dimensions does not match the"
                f"number of dimensions in a test file. "
                f"Number of dimensions in file: {len(self._single_file_shape)}, "
                f"Provided number of dimensions: {len(single_file_dims)}."
            )

        else:
            self._single_file_dims = list(single_file_dims)

        self._single_file_sizes = dict(
            zip(self._single_file_dims, self._single_file_shape)
        )
        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            self._scenes = tuple(
                metadata_utils.generate_ome_image_id(s)
                for s in range(self._all_files[DimensionNames.Samples].nunique())
            )
        return self._scenes

    def _read_delayed(self) -> xr.DataArray:

        scene_files = self._all_files.loc[
            self._all_files[DimensionNames.Samples] == self.current_scene_index
        ]
        scene_files = scene_files.drop(DimensionNames.Samples, axis=1)

        group_dims = [
            x for x in scene_files.columns if x not in self.chunk_dims + ["filename"]
        ]

        chunks = np.zeros(scene_files[group_dims].nunique().values, dtype="object")

        chunk_shape = self._get_shape_for_scene(scene_files, group_dims)

        ordered_chunk_dims = [d for d in scene_files.columns if d in self.chunk_dims]

        ordered_chunk_dims += [
            d
            for d in self.chunk_dims
            if d not in ordered_chunk_dims + [DimensionNames.Samples]
        ]
        
        #Need to allow rechunk in ways other than -1
        # so that when you want planes of a file in separate chunks e.g. files are ZYX and chunk_dims is TC
        # you can still reshape the dask array but then rechunk appropriately
        image_dims_not_in_chunk = [x for x in self._single_file_dims if x not in ordered_chunk_dims]
        print(image_dims_not_in_chunk)
        print(chunk_shape)
        print(ordered_chunk_dims)

        for i, (idx, val) in enumerate(scene_files.groupby(group_dims)):
            zarr_im = imread(val.filename.tolist(), aszarr=True)
            darr = da.from_zarr(zarr_im).rechunk(-1)
            if i==0: print(darr.shape)
            darr = darr.reshape(chunk_shape)
            chunks[idx] = darr

        chunks = np.expand_dims(
            chunks, tuple(range(-1, -len(ordered_chunk_dims) - 1, -1))
        )

        d_data = da.block(chunks.tolist())

        dims = group_dims + ordered_chunk_dims
        channel_names = self._get_channel_names_for_scene(dims)

        coords = self._get_coords(
            dims, d_data.shape, self.current_scene_index, channel_names
        )
        x_data = xr.DataArray(d_data, dims=dims, coords=coords)
        
        if self._dim_order is not None:
            print(self._dim_order)
            x_data.transpose(*list(self._dim_order))

        return x_data

    def _read_immediate(self) -> xr.DataArray:
        scene_files = self._all_files.loc[
            self._all_files[DimensionNames.Samples] == self.current_scene_index
        ]
        scene_files = scene_files.drop(DimensionNames.Samples, axis=1)

        arr = imread(scene_files.filename.tolist())

        shape = self._get_shape_for_scene(scene_files)

        arr = arr.reshape(shape)

        dims = scene_files.columns.drop("filename").values.tolist()
        file_dims = [x for x in self._single_file_dims if x not in dims]
        dims += file_dims

        channel_names = self._get_channel_names_for_scene(dims)

        coords = self._get_coords(dims, shape, self.current_scene_index, channel_names)

        x_data = xr.DataArray(
            arr,
            dims=dims,
            coords=coords,
        )
        if self._dim_order is not None:
            print(self._dim_order)
            x_data.transpose(*list(self._dim_order))

        return x_data

    def _get_shape_for_scene(
        self, scene_files: pd.DataFrame, group_dims: Union[List[str], type(None)] = None
    ):
        if group_dims is None:
            group_dims = []
        shape = ()
        for i, x in scene_files.nunique().iteritems():
            if i not in group_dims + ["filename"]:
                if i not in self._single_file_dims:
                    shape += (x,)
                else:
                    shape += (self._single_file_sizes[i] * x,)

        for i, x in self._single_file_sizes.items():
            if i not in scene_files.columns:
                shape += (x,)
        return shape

    def _get_channel_names_for_scene(self, dims: List[str]) -> Optional[List[str]]:
        # Fast return in None case
        if self._channel_names is None:
            return None

        # If channels was provided as a list of lists
        if isinstance(self._channel_names[0], list):
            scene_channels = self._channel_names[self.current_scene_index]
        elif all(isinstance(c, str) for c in self._channel_names):
            scene_channels = self._channel_names  # type: ignore
        else:
            return None

        # If scene channels isn't None and no channel dimension raise error
        if DimensionNames.Channel not in dims:
            raise exceptions.ConflictingArgumentsError(
                f"Provided channel names for scene with no channel dimension. "
                f"Scene dims: {dims}, "
                f"Provided channel names: {scene_channels}"
            )

        # If scene channels isn't the same length as the size of channel dim
        if len(scene_channels) != image_shape[dims.index(DimensionNames.Channel)]:
            raise exceptions.ConflictingArgumentsError(
                f"Number of channel names provided does not match the "
                f"size of the channel dimension for this scene. "
                f"Scene shape: {image_shape}, "
                f"Dims: {dims}, "
                f"Provided channel names: {self._channel_names}",
            )

        return scene_channels  # type: ignore

    @staticmethod
    def _get_coords(
        dims: List[str],
        shape: Tuple[int, ...],
        scene_index: int,
        channel_names: Optional[List[str]],
    ) -> Dict[str, Any]:
        # Use dims for coord determination
        coords: Dict[str, Any] = {}

        if channel_names is None:
            # Get ImageId for channel naming
            image_id = metadata_utils.generate_ome_image_id(scene_index)

            # Use range for channel indices
            if DimensionNames.Channel in dims:
                coords[DimensionNames.Channel] = [
                    metadata_utils.generate_ome_channel_id(
                        image_id=image_id, channel_id=i
                    )
                    for i in range(shape[dims.index(DimensionNames.Channel)])
                ]
        else:
            coords[DimensionNames.Channel] = channel_names

        return coords
