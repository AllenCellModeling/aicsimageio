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

from .. import constants, exceptions, types
from ..dimensions import (
    DEFAULT_CHUNK_BY_DIMS,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES,
    REQUIRED_CHUNK_BY_DIMS,
    DimensionNames,
)
from ..utils import io_utils
from .reader import Reader

try:
    from _aicspylibczi import BBox
    from aicspylibczi import CziFile

except ImportError:
    raise ImportError(
        "aicspylibczi is required for this reader. "
        "Install with `pip install aicsimageio[czi]`"
    )

DEFAULT_CZI_CHUNK_BY_DIMS = [
    dim if dim != DimensionNames.Samples else "A" for dim in DEFAULT_CHUNK_BY_DIMS
]
REQUIRED_CZI_CHUNK_BY_DIMS = [
    dim if dim != DimensionNames.Samples else "A" for dim in REQUIRED_CHUNK_BY_DIMS
]
DEFAULT_CZI_DIMENSION_ORDER_LIST = [
    dim if dim != DimensionNames.Samples else "A"
    for dim in DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
]
DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES = [
    dim if dim != DimensionNames.Samples else "A"
    for dim in DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES
]


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
        Default: DEFAULT_CHUNK_BY_DIMS
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
        chunk_by_dims: Union[str, List[str]] = DEFAULT_CHUNK_BY_DIMS,
    ):
        # doesn't do anything in this case but here for completeness
        super(CziReader, self).__init__(image=image)

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
                    scene_names = ["Scene:0"]
                self._scenes = tuple(scene_names)

        return self._scenes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        retrieve_dims: List[str],
        retrieve_indices: List[Optional[int]],
    ) -> np.ndarray:
        """
        Open a file for reading, construct a Zarr store, select data, and compute to
        numpy.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the chunk from.
        retrieve_dims: List[str]
            The order of the retrieve indicies operations
        retrieve_indices: List[Optional[int]],
            The image index operations to retrieve.
            If None, retrieve the whole dimension.

        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        # Open and select the target image
        with fs.open(path) as open_resource:
            czi = CziFile(open_resource)
            dims_shape = CziReader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=scene,
                consistent=czi.shape_is_consistent,
            )

            # Create the fill array shape
            # Drop the YX as we will be pulling the individual YX planes
            retrieve_shape: List[int] = []
            use_selected_or_np_map: Dict[str, Optional[int]] = {}
            for dim, index_op in zip(retrieve_dims, retrieve_indices):
                if (
                    dim not in [DimensionNames.SpatialY, DimensionNames.SpatialX, "A"]
                    and dim in dims_shape.keys()
                ):
                    # Handle slices
                    if index_op is None:
                        # Store the dim for later to inform to use the np index
                        use_selected_or_np_map[dim] = None
                        if dim == DimensionNames.MosaicTile:
                            retrieve_shape.append(
                                dims_shape[DimensionNames.MosaicTile][1]
                            )
                        elif dim == DimensionNames.Time:
                            retrieve_shape.append(dims_shape[DimensionNames.Time][1])
                        elif dim == DimensionNames.Channel:
                            retrieve_shape.append(dims_shape[DimensionNames.Channel][1])
                        elif dim == DimensionNames.SpatialZ:
                            retrieve_shape.append(
                                dims_shape[DimensionNames.SpatialZ][1]
                            )

                    # Handle non-chunk dimensions (specific indices / ints)
                    else:
                        # Store the dim for later to inform to use the provided index
                        use_selected_or_np_map[dim] = index_op
                        retrieve_shape.append(1)

            # Create list of planes that we will add each plane to, later we reshape
            # Create empty arr with the desired shape to enumerate over the np index
            planes: List[np.ndarray] = []
            np_array_for_indices = np.empty(tuple(retrieve_shape), dtype=object)
            for np_index, _ in np.ndenumerate(np_array_for_indices):
                # Get each plane's index selection operations
                # If the dimension is None, use the enumerated np index
                # If the dimension is not None, use the provided value
                plane_indices: Dict[str, int] = {"S": scene}

                # Handle MosaicTile
                if DimensionNames.MosaicTile in use_selected_or_np_map:
                    if use_selected_or_np_map[DimensionNames.MosaicTile] is None:
                        plane_indices["M"] = np_index[
                            retrieve_dims.index(DimensionNames.MosaicTile)
                        ]
                    else:
                        plane_indices["M"] = use_selected_or_np_map[  # type: ignore
                            DimensionNames.MosaicTile
                        ]

                # Handle Time
                if DimensionNames.Time in use_selected_or_np_map:
                    if (
                        DimensionNames.Time in retrieve_dims
                        and use_selected_or_np_map[DimensionNames.Time] is None
                    ):
                        plane_indices["T"] = np_index[
                            retrieve_dims.index(DimensionNames.Time)
                        ]
                    else:
                        plane_indices["T"] = use_selected_or_np_map[  # type: ignore
                            DimensionNames.Time
                        ]

                # Handle Channels
                if DimensionNames.Channel in use_selected_or_np_map:
                    if (
                        DimensionNames.Channel in retrieve_dims
                        and use_selected_or_np_map[DimensionNames.Channel] is None
                    ):
                        plane_indices["C"] = np_index[
                            retrieve_dims.index(DimensionNames.Channel)
                        ]
                    else:
                        plane_indices["C"] = use_selected_or_np_map[  # type: ignore
                            DimensionNames.Channel
                        ]

                # Handle SpatialZ
                if DimensionNames.SpatialZ in use_selected_or_np_map:
                    if use_selected_or_np_map[DimensionNames.SpatialZ] is None:
                        plane_indices["Z"] = np_index[
                            retrieve_dims.index(DimensionNames.SpatialZ)
                        ]
                    else:
                        plane_indices["Z"] = use_selected_or_np_map[  # type: ignore
                            DimensionNames.SpatialZ
                        ]

                # Append the retrieved plane as a numpy array
                plane, scene_dims = czi.read_image(**plane_indices)
                scene_dims_dict = CziReader._dims_list_to_dict(scene_dims)
                planes.append(plane)

            # Stack and reshape to get rid of the array of arrays
            new_chunk_shape = [
                scene_dims_dict[dim] for dim in REQUIRED_CZI_CHUNK_BY_DIMS if
                dim in scene_dims_dict
            ]
            retrieved_chunk = np.stack(planes).reshape(
                np_array_for_indices.shape + tuple(new_chunk_shape)
            )

            # Remove extra dimensions if they were not requested
            remove_dim_ops_list: List[Union[int, slice]] = []
            for index in retrieve_indices:
                if isinstance(index, int):
                    remove_dim_ops_list.append(0)
                else:
                    remove_dim_ops_list.append(slice(None, None, None))

            # Remove extra dimensions by using dim ops
            retrieved_chunk = retrieved_chunk[tuple(remove_dim_ops_list)]

            return retrieved_chunk

    def _create_dask_array(
        self, czi: CziFile, selected_scene_dims: List[str]
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
        for dim in REQUIRED_CZI_CHUNK_BY_DIMS:
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

        valid_dims = []
        selected_scene_shape: List[int] = []
        for dim in selected_scene_dims:
            if dim in dims_shape.keys():
                valid_dims.append(dim)
                if dim == DimensionNames.MosaicTile:
                    selected_scene_shape.append(
                        dims_shape[DimensionNames.MosaicTile][1]
                    )
                elif dim == DimensionNames.Time:
                    selected_scene_shape.append(dims_shape[DimensionNames.Time][1])
                elif dim == DimensionNames.Channel:
                    selected_scene_shape.append(dims_shape[DimensionNames.Channel][1])
                elif dim == DimensionNames.SpatialZ:
                    selected_scene_shape.append(dims_shape[DimensionNames.SpatialZ][1])
                elif dim == DimensionNames.SpatialY:
                    selected_scene_shape.append(dims_shape[DimensionNames.SpatialY][1])
                elif dim == DimensionNames.SpatialX:
                    selected_scene_shape.append(dims_shape[DimensionNames.SpatialX][1])
                elif dim == "A":  # sAmples from aicspylibczi3
                    selected_scene_shape.append(dims_shape["A"][1])

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order = []
        non_chunk_shape = []
        chunk_dim_order = []
        chunk_shape = []
        for dim, size in zip(
            valid_dims, selected_scene_shape
        ):  # zip(selected_scene_dims, selected_scene_shape):
            if dim in self.chunk_by_dims:
                chunk_dim_order.append(dim)
                chunk_shape.append(size)
            else:
                non_chunk_dim_order.append(dim)
                non_chunk_shape.append(size)

        # Get sample for dtype
        czi_pixel_type = czi.pixel_type
        pixel_type: Union[type(np.uint8), type(None)] = None

        if czi_pixel_type == "gray8":
            pixel_type = np.uint8
        elif czi_pixel_type == "gray16":
            pixel_type = np.uint16
        elif czi_pixel_type == "gray32":
            pixel_type = np.uint32
        elif czi_pixel_type == "bgr24":
            pixel_type = np.uint8
        elif czi_pixel_type == "bgr48":
            pixel_type = np.uint16

        if pixel_type is None:
            raise TypeError(f"Pixel Type: {czi_pixel_type} not supported!")

        # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to
        # outer-most with the chunks as long as the dimension is size 1
        blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))

        # Make ndarray for lazy arrays to fill
        lazy_arrays = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get None, which tell the get data func to pull the whole dim
            retrieve_indices = np_index[: len(non_chunk_shape)] + (
                (None,) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(CziReader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    retrieve_dims=blocked_dim_order,
                    retrieve_indices=retrieve_indices,
                ),
                shape=chunk_shape,
                dtype=pixel_type,
            )

        # Convert the numpy array of lazy readers into a dask array
        image_data = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example, if the original dimension ordering was "TZYX" and we
        # chunked by "T", "Y", and "X"
        # we created an array with dimensions ordering "ZTYX"
        transpose_indices = []
        for i, d in enumerate(valid_dims):  # selected_scene_dims):
            new_index = blocked_dim_order.index(d)
            if new_index != i:
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        image_data = da.transpose(image_data, tuple(transpose_indices))

        return image_data

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: ET.Element, scene_index: int, image_short_info: Dict[str, Any] = None
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords = {}

        # Get all images
        img_sets = xml.findall(".//Image/Dimensions/Channels")

        if len(img_sets) != 0:
            # Select the current scene
            img = img_sets[scene_index]

            # Construct channel list
            scene_channel_list = []
            channels = img.findall("./Channel")
            for i, channel in enumerate(channels):
                channel_name = channel.attrib["Name"]
                channel_id = channel.attrib["Id"]
                channel_contrast = channel.find("./ContrastMethod").text
                scene_channel_list.append(
                    (f"{channel_id}" f"--{channel_name}" f"--{channel_contrast}")
                )

            # Attach channel names to coords
            coords[DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        list_xs = xml.findall(".//Distance[@Id='X']")
        list_ys = xml.findall(".//Distance[@Id='Y']")
        list_zs = xml.findall(".//Distance[@Id='Z']")

        # scale_x, scale_y, scale_z, scale_t = image_short_info["scale"]
        scale_x = list_xs[0].find("./Value").text
        scale_y = list_ys[0].find("./Value").text
        scale_z = None if len(list_zs) == 0 else list_zs[0].find("./Value").text
        scale_t = None

        # Handle Spatial Dimensions
        if scale_z is not None:
            scale_z = float(scale_z)
            coords[DimensionNames.SpatialZ] = np.arange(
                0, image_short_info[DimensionNames.SpatialZ][1] * scale_z, scale_z,
            )
        if scale_y is not None:
            scale_y = float(scale_y)
            coords[DimensionNames.SpatialY] = np.arange(
                0, image_short_info[DimensionNames.SpatialY][1] * scale_y, scale_y,
            )
        if scale_x is not None:
            scale_x = float(scale_x)
            coords[DimensionNames.SpatialX] = np.arange(
                0, image_short_info[DimensionNames.SpatialX][1] * scale_x, scale_x,
            )

        # Time
        if scale_t is not None:
            coords[DimensionNames.Time] = np.arange(
                0, image_short_info["dims"].t * scale_t, scale_t,
            )

        # Create physical pixal sizes
        px_sizes = types.PhysicalPixelSizes(
            scale_z if scale_z is not None else 1.0,
            scale_y if scale_y is not None else 1.0,
            scale_x if scale_x is not None else 1.0,
        )

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
            dims = [d if d != "A" else DimensionNames.Samples for d in dims]

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
            image_data = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
                retrieve_dims=dims,
                retrieve_indices=[None] * len(dims),  # Get all planes
            )

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
    def _dims_list_to_dict(list_in: List) -> Dict:
        return {x[0]: x[1] for x in list_in}

    @staticmethod
    def _dims_shape_to_scene_dims_shape(
        dims_shape: List, scene_index: int, consistent: bool
    ) -> Dict:
        dims_shape_index = 0 if consistent else scene_index
        dims_shape = dims_shape[dims_shape_index]
        dims_shape.pop("S", None)
        return dims_shape

    @staticmethod
    def _stitch_tiles(
        data: types.ArrayLike, data_dims_shape: Dict, bboxes: Dict, mosaic_bbox: BBox,
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
            if dim not in REQUIRED_CZI_CHUNK_BY_DIMS:
                arr_shape_list.append(data_dims_shape[dim][1])
            if dim is DimensionNames.SpatialY:
                arr_shape_list.append(mosaic_bbox.h)
            if dim is DimensionNames.SpatialX:
                arr_shape_list.append(mosaic_bbox.w)
            if dim == "A":
                arr_shape_list.append(data_dims_shape[dim][1])

        ans = None
        if type(data) is da.Array:
            ans = da.zeros(arr_shape_list, chunks=data.chunks, dtype=data.dtype)
        else:
            ans = np.zeros(arr_shape_list, dtype=data.dtype)

        for (tile_info, box) in bboxes.items():
            # construct data indexes to use
            tile_dims = tile_info.dimension_coordinates
            tile_dims.pop("S", None)
            tile_dims.pop("B", None)
            data_indexes = [
                tile_dims[t_dim]
                for t_dim in DEFAULT_CZI_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES
                if t_dim in tile_dims.keys() and t_dim not in REQUIRED_CZI_CHUNK_BY_DIMS
            ]
            # add Y and X
            data_indexes.append(slice(None))  # Y ":"
            data_indexes.append(slice(None))  # X ":"
            # if "A" in tile_dims.keys():
            #     data_indexes.append(slice(None))

            # construct data indexes for ans
            ans_indexes = []
            for dim in ordered_dims_present:  # DEFAULT_CZI_DIMENSION_ORDER_LIST:
                if dim not in [
                    "S",
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
                if dim == "A":
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
            if d not in [
                DimensionNames.MosaicTile,
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            ]
        }
        attrs = copy(self.xarray_dask_data.attrs)

        return xr.DataArray(data=stitched, dims=dims, coords=coords, attrs=attrs, )

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
