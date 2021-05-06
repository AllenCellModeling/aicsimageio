#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from copy import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, metadata, types
from ..dimensions import (
    DEFAULT_CHUNK_DIMS,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES,
    REQUIRED_CHUNK_DIMS,
    DimensionNames,
)
from ..utils import io_utils
from .reader import Reader

try:
    from _aicspylibczi import BBox, TileInfo
    from aicspylibczi import CziFile

except ImportError:
    raise ImportError(
        "aicspylibczi is required for this reader. "
        "Install with `pip install aicsimageio[czi]`"
    )

CZI_SAMPLES_DIM_CHAR = "A"
CZI_BLOCK_DIM_CHAR = "B"
CZI_SCENE_DIM_CHAR = "S"


###############################################################################


def _replace_sample_dim(dims: List[str]) -> List[str]:
    return [
        dim if dim != DimensionNames.Samples else CZI_SAMPLES_DIM_CHAR for dim in dims
    ]


###############################################################################


DEFAULT_CZI_CHUNK_DIMS = _replace_sample_dim(DEFAULT_CHUNK_DIMS)
REQUIRED_CZI_CHUNK_DIMS = _replace_sample_dim(REQUIRED_CHUNK_DIMS)
DEFAULT_CZI_DIMENSION_ORDER_LIST = _replace_sample_dim(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
)
DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES = _replace_sample_dim(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES
)


PIXEL_DICT = {
    "gray8": np.uint8,
    "gray16": np.uint16,
    "gray32": np.uint32,
    "bgr24": np.uint8,
    "bgr48": np.uint16,
}


###############################################################################


class CziReader(Reader):
    """
    Wraps the aicspylibczi API to provide the same aicsimageio Reader API but for
    volumetric Zeiss CZI images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_by_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.

    Notes
    -----
    To use this reader, install with: `pip install aicsimageio[czi]`.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                CziFile(open_resource)
                return True

        except RuntimeError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_by_dims: Union[str, List[str]] = DEFAULT_CZI_CHUNK_DIMS,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Store params
        if isinstance(chunk_by_dims, str):
            chunk_by_dims = list(chunk_by_dims)

        self.chunk_by_dims = chunk_by_dims

        # Delayed storage
        self._px_sizes: Optional[types.PhysicalPixelSizes] = None

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                czi = CziFile(open_resource)
                xpath_str = "./Metadata/Information/Image/Dimensions/S/Scenes/Scene"
                meta_scenes = czi.meta.findall(xpath_str)
                scene_names = [x.get("Name") for x in meta_scenes]
                # if the scene is implicit just assign it name Scene:0
                if len(scene_names) < 1:
                    scene_names = [metadata.utils.generate_ome_image_id(0)]
                self._scenes = tuple(scene_names)

        return self._scenes

    @staticmethod
    def _dims_shape_to_scene_dims_shape(
        dims_shape: List[Dict], scene_index: int, consistent: bool
    ) -> Dict[str, Tuple[int, int]]:
        """
        This function takes the output of `get_dims_shape()` and returns a
        dictionary of dimensions for the selected scene

        Parameters
        ----------
        dims_shape: List[Dict]
            a list of dictionaries, generated by `get_dims_shape()`
        scene_index: int
            the index of the scene being used
        consistent: bool
            true if the dictionaries are consistent could be represented
            compactly (dims_shape with length 1)

        Returns
        -------
        A dictionary of dimensions, ie
        {"T": (0, 1), "C": (0, 3), "Y": (0, 256), "X":(0, 256)}.
        """
        dims_shape_index = 0 if consistent else scene_index
        dims_shape_dict = dims_shape[dims_shape_index]
        dims_shape_dict.pop(CZI_SCENE_DIM_CHAR, None)
        return dims_shape_dict

    @staticmethod
    def _dim_helper(
        dim: str,
        selected: Dict[str, Optional[int]],
        np_index: Tuple,
        retrieve_dims: List,
    ) -> Optional[int]:
        indices = None
        if dim in selected:
            indices = selected[dim]
            if indices is None:
                indices = np_index[retrieve_dims.index(dim)]
        return indices

    @staticmethod
    def _get_just_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None
    ) -> np.ndarray:
        return CziReader._get_image_data(fs=fs, path=path, scene=scene, read_dims=read_dims)[0]

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info
        that was read.
        Parameters
        ----------
        img: Path
            Path to the CZI file to read.
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)
        Returns
        -------
        data: np.ndarray
            The data read for the dimensions provided.
        read_dimensions: List[Tuple[str, int]]]
            The dimension sizes that were returned from the read.
        """
        # Catch optional read dim

        if read_dims is None:
            read_dims = {}

        read_dims["S"] = scene

        # Init czi
        # resource = fs.open(path)
        czi = CziFile(path)

        # Read image
        data, dims = czi.read_image(**read_dims)

        dstr = "".join([d[0] for d in dims])
        if "B" in dstr:
            data = np.squeeze(data, axis=0)
            dims = [item for item in dims if item[0]!="B"]

        # Drop dims that shouldn't be provided back
        ops = []
        real_dims = []
        for i, dim_info in enumerate(dims):
            # Expand dimension info
            dim, size = dim_info

            # If the dim was provided in the read dims we know a single plane for that
            # dimension was requested so remove it
            if dim in read_dims:
                ops.append(0)
            # Otherwise just read the full slice
            else:
                ops.append(slice(None, None, None))
                real_dims.append(dim_info)

        # Convert ops and run getitem
        return data[tuple(ops)] , real_dims

    def _create_dask_array(
        self,
        czi: CziFile,
        selected_scene_dims: List[str]
    ) -> xr.DataArray:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        czi: CziFile
            An open CziFile for processing.
        selected_scene_dims: List[str]
            The dimensions for the scene to create the dask array for

        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in REQUIRED_CZI_CHUNK_DIMS:
            if dim not in self.chunk_by_dims:
                self.chunk_by_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_by_dims = [d.upper() for d in self.chunk_by_dims]

        # Construct the delayed dask array
        dims_shape = CziReader._dims_shape_to_scene_dims_shape(
            czi.get_dims_shape(),
            scene_index=self.current_scene_index,
            consistent=czi.shape_is_consistent,
        )

        if "B" in dims_shape:
            dims_shape.pop("B", None)

        dims_str = czi.dims.replace('B', '').replace('S','')
        if DimensionNames.MosaicTile in dims_str:
            ordered_dims_string = DimensionNames.MosaicTile + dims_str.replace(DimensionNames.MosaicTile, '')
            dims_str = ordered_dims_string
        # Get the shape for the chunk and operating shape for the dask array
        # We also collect the chunk and non chunk dimension ordering so that we can
        # swap the dimensions after we
        # block the dask array together.
        sample_chunk_shape = []
        operating_shape = []
        non_chunk_dimension_ordering = []
        chunk_dimension_ordering = []
        for i, dim in enumerate(dims_str):
            # Unpack dim info
            dim_idx_start, dim_size = dims_shape[dim]

            # If the dim is part of the specified chunk dims then append it to the
            # sample, and, append the dimension
            # to the chunk dimension ordering
            if dim in self.chunk_by_dims:
                sample_chunk_shape.append(dim_size)
                chunk_dimension_ordering.append(dim)

            # Otherwise, append the dimension to the non chunk dimension ordering, and,
            # append the true size of the
            # image at that dimension
            else:
                non_chunk_dimension_ordering.append(dim)
                operating_shape.append( dim_size )

        # Convert shapes to tuples and combine the non and chunked dimension orders as
        # that is the order the data will
        # actually come out of the read data as
        sample_chunk_shape = tuple(sample_chunk_shape)
        blocked_dimension_order = (
            non_chunk_dimension_ordering + chunk_dimension_ordering
        )

        # Fill out the rest of the operating shape with dimension sizes of 1 to match
        # the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to outer-most
        # with the chunks as long as the dimension is size 1
        # Basically, we are adding empty dimensions to the operating shape that will be
        # filled by the chunks from dask
        operating_shape = tuple(operating_shape) + (1,) * len(sample_chunk_shape)

        # Create empty numpy array with the operating shape so that we can iter through
        # and use the multi_index to create the readers.
        lazy_arrays = np.ndarray(operating_shape, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims
        # dictionaries by simply zipping together the ordered dims list and the current
        # multi-index plus the begin index for that plane. We then set the value of the
        # array at the same multi-index to the delayed reader using the constructed
        # read_dims dictionary.
        dims = [d for d in czi.dims if d not in ['B', 'S']]
        begin_indicies = tuple(dims_shape[d][0] for d in dims)
        for i, _ in np.ndenumerate(lazy_arrays):
            # Add the czi file begin index for each dimension to the array dimension
            # index
            this_chunk_read_indicies = (
                current_dim_begin_index + curr_dim_index
                for current_dim_begin_index, curr_dim_index in zip(begin_indicies, i)
            )

            # Zip the dims with the read indices
            this_chunk_read_dims = dict(
                zip(blocked_dimension_order, this_chunk_read_indicies)
            )

            # Remove the dimensions that we want to chunk by from the read dims
            for d in self.chunk_by_dims:
                if d in this_chunk_read_dims:
                    this_chunk_read_dims.pop(d)

            pixel_type = PIXEL_DICT.get(czi.pixel_type)

            # Add delayed array to lazy arrays at index
            lazy_arrays[i] = da.from_delayed(
                delayed(CziReader._get_just_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    read_dims=this_chunk_read_dims),
                shape=sample_chunk_shape,
                dtype=pixel_type,
            )

        # Convert the numpy array of lazy readers into a dask array and fill the inner
        # most empty dimensions with chunks
        merged = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example being, if the original dimension ordering was "SZYX" and we want to
        # chunk by "S", "Y", and "X" we created an array with dimensions ordering "ZSYX"
        transpose_indices = []
        transpose_required = False
        for i, d in enumerate(dims_str):
            new_index = blocked_dimension_order.index(d)
            if new_index != i:
                transpose_required = True
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        # The default case is "Z", "Y", "X", which _usually_ doesn't need to be
        # transposed because that is _usually_ the normal dimension order of the CZI
        # file anyway
        if transpose_required:
            merged = da.transpose(merged, tuple(transpose_indices))

        # Because dimensions outside of Y and X can be in any order and present or not
        # we also return the dimension order string.
        return merged  # , "".join(dims)

        # # Constuct the chunk and non-chunk shapes one dim at a time
        # # We also collect the chunk and non-chunk dimension order so that
        # # we can swap the dimensions after we block out the array
        # non_chunk_dim_order = []
        # non_chunk_shape = []
        # chunk_dim_order = []
        # chunk_shape = []
        # for dim, size in zip(selected_scene_dims_str, selected_scene_shape):
        #     if dim in self.chunk_by_dims:
        #         chunk_dim_order.append(dim)
        #         chunk_shape.append(size)
        #     else:
        #         non_chunk_dim_order.append(dim)
        #         non_chunk_shape.append(size)
        #
        # pixel_type = PIXEL_DICT.get(czi.pixel_type)
        #
        # if pixel_type is None:
        #     raise TypeError(f"Pixel Type: {czi.pixel_type} not supported!")
        #
        # # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # # match the length of the sample chunk
        # # When dask.block happens it fills the dimensions from inner-most to
        # # outer-most with the chunks as long as the dimension is size 1
        # blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        # blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))
        #
        # # Construct the transpose indices that will be used to
        # # transpose the array prior to pulling the chunk dims
        # match_map = {dim: selected_scene_dims_str.find(dim) for dim in selected_scene_dims_str}
        # transposer = []
        # for dim in blocked_dim_order:
        #     transposer.append(match_map[dim])
        #
        # # Make ndarray for lazy arrays to fill
        # lazy_arrays = np.ndarray(blocked_shape, dtype=object)
        # for np_index, _ in np.ndenumerate(lazy_arrays):
        #     # All dimensions get their normal index except for chunk dims
        #     # which get None, which tell the get data func to pull the whole dim
        #     indices_with_slices = np_index[: len(non_chunk_shape)] + (
        #         (None,) * len(chunk_shape)
        #     )
        #
        #     # Fill the numpy array with the delayed arrays
        #     lazy_arrays[np_index] = da.from_delayed(
        #         delayed(CziReader._get_image_data)(
        #             fs=self._fs,
        #             path=self._path,
        #             scene=self.current_scene_index,
        #             get_dims_dict={},
        #         ),
        #         shape=chunk_shape,
        #         dtype=pixel_type,
        #     )
        #
        # # Convert the numpy array of lazy readers into a dask array
        # image_data = da.block(lazy_arrays.tolist())
        #
        # # Because we have set certain dimensions to be chunked and others not
        # # we will need to transpose back to original dimension ordering
        # # Example, if the original dimension ordering was "TZYX" and we
        # # chunked by "T", "Y", and "X"
        # # we created an array with dimensions ordering "ZTYX"
        # transpose_indices = []
        # for i, d in enumerate(selected_scene_dims_str):
        #     new_index = blocked_dim_order.index(d)
        #     if new_index != i:
        #         transpose_indices.append(new_index)
        #     else:
        #         transpose_indices.append(i)
        #
        # # Only run if the transpose is actually required
        # image_data = da.transpose(image_data, tuple(transpose_indices))
        #
        # return image_data

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: ET.Element, scene_index: int, image_short_info: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords = {}

        # Get all images
        img_sets = xml.findall(".//Image/Dimensions/Channels")

        if len(img_sets) != 0:
            # Select the current scene
            img = img_sets[0]
            if scene_index < len(img_sets):
                img = img_sets[scene_index]

            # Construct channel list
            scene_channel_list = []
            channels = img.findall("./Channel")
            for i, channel in enumerate(channels):
                channel_name = channel.attrib["Name"]
                channel_id = channel.attrib["Id"]
                channel_ce = channel.find("./ContrastMethod")
                channel_contrast = channel_ce.text if channel_ce is not None else ""
                scene_channel_list.append(
                    (f"{channel_id}" f"--{channel_name}" f"--{channel_contrast}")
                )

            # Attach channel names to coords
            coords[DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        list_xs = xml.findall(".//Distance[@Id='X']")
        list_ys = xml.findall(".//Distance[@Id='Y']")
        list_zs = xml.findall(".//Distance[@Id='Z']")

        scale_xe = list_xs[0].find("./Value")
        scale_ye = list_ys[0].find("./Value")
        scale_ze = None if len(list_zs) == 0 else list_zs[0].find("./Value")
        scale_x = float(str(scale_xe.text)) if scale_xe is not None else 1.0
        scale_y = float(str(scale_ye.text)) if scale_ye is not None else 1.0
        scale_z = float(str(scale_ze.text)) if scale_ze is not None else 1.0
        scale_t = None

        # Handle Spatial Dimensions
        if scale_ze is not None:
            coords[DimensionNames.SpatialZ] = np.arange(
                0,
                image_short_info[DimensionNames.SpatialZ][1] * scale_z,
                scale_z,
            )
        if scale_y is not None:
            coords[DimensionNames.SpatialY] = np.arange(
                0,
                image_short_info[DimensionNames.SpatialY][1] * scale_y,
                scale_y,
            )
        if scale_x is not None:
            coords[DimensionNames.SpatialX] = np.arange(
                0,
                image_short_info[DimensionNames.SpatialX][1] * scale_x,
                scale_x,
            )

        # Time
        if scale_t is not None:
            coords[DimensionNames.Time] = np.arange(
                0,
                image_short_info["dims"].t * scale_t,
                scale_t,
            )

        # Create physical pixel sizes
        px_sizes = types.PhysicalPixelSizes(scale_z, scale_y, scale_x)

        return coords, px_sizes

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource)

            dims_shape = CziReader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # If there are tiles in the image use mosaic dims
            if czi.is_mosaic():
                ref_dims = DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                ref_dims = DEFAULT_CZI_DIMENSION_ORDER_LIST

            dims = [dim for dim in ref_dims if dim in dims_shape.keys()]
            # Get image data
            image_data = self._create_dask_array(czi, dims)
            # Get metadata
            meta = czi.meta

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                image_short_info=dims_shape,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

            # Map A (aicspylibczi sAmples) back to aicsimageio Samples
            dims = [
                d if d != CZI_SAMPLES_DIM_CHAR else DimensionNames.Samples for d in dims
            ]

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,  # type: ignore
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource)
            # selected_scene, real_dims = czi.read_image(S=self.current_scene_index)

            dims_shape = CziReader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # If there are tiles in the image use mosaic dims
            if czi.is_mosaic():
                ref_dims = DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                ref_dims = DEFAULT_CZI_DIMENSION_ORDER_LIST

            dims = [dim for dim in ref_dims if dim in dims_shape.keys()]

            # Get image data
            image_data, real_dims = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
            )

            read_dims_key_list = [k[0] for k in real_dims]
            if dims != read_dims_key_list:
                new_dim_order = [read_dims_key_list.index(d) for d in dims]
                image_data = np.transpose(image_data, new_dim_order)


            # Get metadata
            meta = czi.meta

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                image_short_info=dims_shape,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,  # type: ignore
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    @staticmethod
    def _dims_list_to_dict(list_in: List[Tuple[str, int]]) -> Dict[str, int]:
        return {x[0]: x[1] for x in list_in}

    @staticmethod
    def _stitch_tiles(
        data: types.ArrayLike,
        data_dims_shape: Dict[str, Tuple[int, int]],
        bboxes: Dict[TileInfo, BBox],
        mosaic_bbox: BBox,
    ) -> types.ArrayLike:
        # Assumptions: 1) docs for ZEISSRAW(CZI) say:
        #   Scene – for clustering items in X/Y direction (data belonging to
        #   contiguous regions of interests in a mosaic image).

        # 'S' in czi is Scene not Samples, 'A' is sAmples

        # Create empty array
        arr_shape_list = []

        ordered_dims_present = [
            dim for dim in DEFAULT_CZI_DIMENSION_ORDER_LIST if dim in data_dims_shape
        ]
        for dim in ordered_dims_present:
            if dim not in REQUIRED_CZI_CHUNK_DIMS:
                arr_shape_list.append(data_dims_shape[dim][1])
            if dim is DimensionNames.SpatialY:
                arr_shape_list.append(mosaic_bbox.h)
            if dim is DimensionNames.SpatialX:
                arr_shape_list.append(mosaic_bbox.w)
            if dim == CZI_SAMPLES_DIM_CHAR:
                arr_shape_list.append(data_dims_shape[dim][1])

        ans = None
        if type(data) is da.Array:
            ans = da.zeros(
                arr_shape_list,
                chunks=data.chunks,  # type: ignore
                dtype=data.dtype,
            )
        else:
            ans = np.zeros(arr_shape_list, dtype=data.dtype)

        for (tile_info, box) in bboxes.items():
            # construct data indexes to use
            tile_dims = tile_info.dimension_coordinates
            tile_dims.pop(CZI_SCENE_DIM_CHAR, None)
            tile_dims.pop(CZI_BLOCK_DIM_CHAR, None)
            data_indexes = [
                tile_dims[t_dim]
                for t_dim in DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES
                if t_dim in tile_dims.keys() and t_dim not in REQUIRED_CZI_CHUNK_DIMS
            ]
            # add Y and X
            data_indexes.append(slice(None))  # Y ":"
            data_indexes.append(slice(None))  # X ":"
            if CZI_SAMPLES_DIM_CHAR in tile_dims.keys():
                data_indexes.append(slice(None))

            # construct data indexes for ans
            ans_indexes = []
            for dim in ordered_dims_present:
                if dim not in [
                    DimensionNames.MosaicTile,
                    DimensionNames.SpatialY,
                    DimensionNames.SpatialX,
                ]:
                    if dim in tile_dims.keys():
                        ans_indexes.append(tile_dims[dim])

                if dim is DimensionNames.SpatialY:
                    start = box.y - mosaic_bbox.y
                    ans_indexes.append(slice(start, start + box.h, 1))
                if dim is DimensionNames.SpatialX:
                    start = box.x - mosaic_bbox.x
                    ans_indexes.append(slice(start, start + box.w, 1))
                if dim == CZI_SAMPLES_DIM_CHAR:
                    ans_indexes.append(slice(None))

            # assign the tiles into ans
            ans[ans_indexes] = data[data_indexes]

        return ans

    def _construct_mosaic_xarray(self, data: types.ArrayLike) -> xr.DataArray:
        # Get max of mosaic positions from lif
        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource)
            dims_shape = CziReader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

        bboxes = czi.get_all_mosaic_tile_bounding_boxes(S=self.current_scene_index)
        mosaic_scene_bbox = czi.get_mosaic_scene_bounding_box(
            index=self.current_scene_index
        )

        # Stitch
        stitched = self._stitch_tiles(
            data=self.data,  # the ndarray
            data_dims_shape=dims_shape,
            bboxes=bboxes,
            mosaic_bbox=mosaic_scene_bbox,
        )

        # Copy metadata
        dims = [
            d for d in self.xarray_dask_data.dims if d is not DimensionNames.MosaicTile
        ]
        coords = {
            d: v
            for d, v in self.xarray_dask_data.coords.items()
            if d
            not in [
                DimensionNames.MosaicTile,
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            ]
        }
        attrs = copy(self.xarray_dask_data.attrs)

        return xr.DataArray(
            data=stitched,
            dims=dims,
            coords=coords,
            attrs=attrs,
        )

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self.dask_data)

    def _get_stitched_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self.data)

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
        if self._px_sizes is None:
            # We get pixel sizes as a part of array construct
            # so simply run array construct
            self.dask_data

        return self._px_sizes  # type: ignore
