#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from copy import copy
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, metadata, types
from ..dimensions import DEFAULT_CHUNK_DIMS, REQUIRED_CHUNK_DIMS, DimensionNames
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
        chunk_by_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read CZIs from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        # Store params
        if isinstance(chunk_by_dims, str):
            chunk_by_dims = list(chunk_by_dims)

        self.chunk_by_dims = chunk_by_dims

        # Delayed storage
        self._px_sizes: Optional[types.PhysicalPixelSizes] = None
        self._mapped_dims: Optional[str] = None

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def mapped_dims(self) -> str:
        if self._mapped_dims is None:
            with self._fs.open(self._path) as open_resource:
                czi = CziFile(open_resource)
                self._mapped_dims = CziReader._fix_czi_dims(czi.dims)

        return self._mapped_dims

    @staticmethod
    def _fix_czi_dims(dims: str) -> str:
        return (
            dims.replace(CZI_BLOCK_DIM_CHAR, "")
            .replace(CZI_SCENE_DIM_CHAR, "")
            .replace(CZI_SAMPLES_DIM_CHAR, DimensionNames.Samples)
        )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                czi = CziFile(open_resource)
                xpath_str = "./Metadata/Information/Image/Dimensions/S/Scenes/Scene"
                meta_scenes = czi.meta.findall(xpath_str)
                scene_names: List[str] = []

                # Some "scenes" may have the same name but each scene has a sub-scene
                # "Shape" with a name.
                #
                # An example of this is where someone images a 96 well plate with each
                # well being it's own scene but they name every scene the same value.
                # The sub-scene "Shape" elements have actual names of each well.
                #
                # If we didn't do this, the produced list would have 96 of the same
                # string name making it impossible to switch scenes.
                for meta_scene in meta_scenes:
                    shape = meta_scene.find("Shape")
                    if shape is not None:
                        shape_name = shape.get("Name")
                        scene_name = meta_scene.get("Name")
                        combined_scene_name = f"{scene_name}-{shape_name}"
                    else:
                        combined_scene_name = meta_scene.get("Name")

                    scene_names.append(combined_scene_name)

                # If the scene is implicit just assign it name Scene:0
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
    def _read_chunk_from_image(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        return CziReader._get_image_data(
            fs=fs, path=path, scene=scene, read_dims=read_dims
        )[0]

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info
        that was read.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to the file to read.
        scene: int
            The scene index to pull the chunk from.
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)

        Returns
        -------
        chunk: np.ndarray
            The image chunk read as a numpy array.
        read_dimensions: List[Tuple[str, int]]]
            The dimension sizes that were returned from the read.
        """
        # Catch optional read dim
        if read_dims is None:
            read_dims = {}

        # Get current scene read dims
        read_dims[CZI_SCENE_DIM_CHAR] = scene

        # Init czi
        with fs.open(path) as open_resource:
            czi = CziFile(open_resource)

            # Read image
            data, dims = czi.read_image(**read_dims)

            # Drop dims that shouldn't be provided back
            ops: List[Union[int, slice]] = []
            real_dims = []
            for dim_info in dims:
                # Expand dimension info
                dim, _ = dim_info

                # If the dim was provided in the read dims
                # we know a single plane for that dimension was requested so remove it
                if dim in read_dims or dim is CZI_BLOCK_DIM_CHAR:
                    ops.append(0)

                # Otherwise just read the full slice
                else:
                    ops.append(slice(None, None, None))
                    real_dims.append(dim_info)

            # Convert ops and run getitem
            return data[tuple(ops)], real_dims

    def _create_dask_array(self, czi: CziFile) -> xr.DataArray:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        czi: CziFile
            An open CziFile for processing.

        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in REQUIRED_CHUNK_DIMS:
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

        # Remove block dim as not useful
        dims_shape.pop(CZI_BLOCK_DIM_CHAR, None)

        dims_str = czi.dims
        for remove_dim_char in [CZI_BLOCK_DIM_CHAR, CZI_SCENE_DIM_CHAR]:
            dims_str = dims_str.replace(remove_dim_char, "")

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
            _, dim_size = dims_shape[dim]

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
                operating_shape.append(dim_size)

        # Convert shapes to tuples and combine the non and chunked dimension orders as
        # that is the order the data will
        # actually come out of the read data as
        sample_chunk_shape_tuple = tuple(sample_chunk_shape)
        blocked_dimension_order = (
            non_chunk_dimension_ordering + chunk_dimension_ordering
        )

        # Fill out the rest of the operating shape with dimension sizes of 1 to match
        # the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to outer-most
        # with the chunks as long as the dimension is size 1
        # Basically, we are adding empty dimensions to the operating shape that will be
        # filled by the chunks from dask
        operating_shape_tuple = tuple(operating_shape) + (1,) * len(
            sample_chunk_shape_tuple
        )

        # Create empty numpy array with the operating shape so that we can iter through
        # and use the multi_index to create the readers.
        lazy_arrays: np.ndarray = np.ndarray(operating_shape_tuple, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims
        # dictionaries by simply zipping together the ordered dims list and the current
        # multi-index plus the begin index for that plane. We then set the value of the
        # array at the same multi-index to the delayed reader using the constructed
        # read_dims dictionary.
        dims = [
            d for d in czi.dims if d not in [CZI_BLOCK_DIM_CHAR, CZI_SCENE_DIM_CHAR]
        ]
        begin_indicies = tuple(dims_shape[d][0] for d in dims)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # Add the czi file begin index for each dimension to the array dimension
            # index
            this_chunk_read_indicies = (
                current_dim_begin_index + curr_dim_index
                for current_dim_begin_index, curr_dim_index in zip(
                    begin_indicies, np_index
                )
            )

            # Zip the dims with the read indices
            this_chunk_read_dims = dict(
                zip(blocked_dimension_order, this_chunk_read_indicies)
            )

            # Remove the dimensions that we want to chunk by from the read dims
            for d in self.chunk_by_dims:
                this_chunk_read_dims.pop(d, None)

            # Get pixel type and catch unsupported
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(f"Pixel type: {pixel_type} is not supported.")

            # Add delayed array to lazy arrays at index
            lazy_arrays[np_index] = da.from_delayed(
                delayed(CziReader._read_chunk_from_image)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    read_dims=this_chunk_read_dims,
                ),
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
        return merged

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords: Dict[str, Any] = {}

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
            for channel in channels:
                channel_name = channel.attrib["Name"]
                scene_channel_list.append(channel_name)

            # Attach channel names to coords
            coords[DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        list_xs = xml.findall(".//Distance[@Id='X']")
        list_ys = xml.findall(".//Distance[@Id='Y']")
        list_zs = xml.findall(".//Distance[@Id='Z']")
        scale_xe = list_xs[0].find("./Value")
        scale_ye = list_ys[0].find("./Value")
        scale_ze = None if len(list_zs) == 0 else list_zs[0].find("./Value")

        # Set default scales
        scale_x = None
        scale_y = None
        scale_z = None

        # Unpack the string value to a float
        # Split by "E" and take the first part because the values are stored
        # with E-06 for micrometers, even though the unit is also present in metadata
        # 🤷
        if scale_xe is not None and scale_xe.text is not None:
            scale_x = float(scale_xe.text.split("E")[0])
        if scale_ye is not None and scale_ye.text is not None:
            scale_y = float(scale_ye.text.split("E")[0])
        if scale_ze is not None and scale_ze.text is not None:
            scale_z = float(scale_ze.text.split("E")[0])

        # Handle Spatial Dimensions
        for scale, dim_name in [
            (scale_z, DimensionNames.SpatialZ),
            (scale_y, DimensionNames.SpatialY),
            (scale_x, DimensionNames.SpatialX),
        ]:
            if scale is not None and dim_name in dims_shape:
                dim_size = dims_shape[dim_name][1] - dims_shape[dim_name][0]
                coords[dim_name] = Reader._generate_coord_array(0, dim_size, scale)

        # Time
        # TODO: unpack "TimeSpan" elements
        # I can find a single "TimeSpan" in our data but unsure how multi-scene handles

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

            # Get dims as list for xarray
            img_dims_list = list(self.mapped_dims)

            # Get image data
            image_data = self._create_dask_array(czi)

            # Create coordinate planes
            meta = czi.meta
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                dims_shape=dims_shape,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=img_dims_list,
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
            dims_shape = CziReader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # Get image data
            image_data, _ = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
            )

            # Get metadata
            meta = czi.meta

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                dims_shape=dims_shape,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=[d for d in self.mapped_dims],
                coords=coords,  # type: ignore
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    @staticmethod
    def _stitch_tiles(
        data: types.ArrayLike,
        data_dims: str,
        data_dims_shape: Dict[str, Tuple[int, int]],
        tile_bboxes: Dict[TileInfo, BBox],
        final_bbox: BBox,
    ) -> types.ArrayLike:
        # Assumptions: 1) docs for ZEISSRAW(CZI) say:
        #   Scene – for clustering items in X/Y direction (data belonging to
        #   contiguous regions of interests in a mosaic image).

        # Store the mosaic array shape
        arr_shape_list = []

        ordered_dims_present = [
            dim
            for dim in data_dims
            if dim not in [CZI_BLOCK_DIM_CHAR, DimensionNames.MosaicTile]
        ]
        for dim in ordered_dims_present:
            if dim not in REQUIRED_CHUNK_DIMS:
                arr_shape_list.append(data_dims_shape[dim][1])
            if dim is DimensionNames.SpatialY:
                arr_shape_list.append(final_bbox.h)
            if dim is DimensionNames.SpatialX:
                arr_shape_list.append(final_bbox.w)
            if dim is DimensionNames.Samples:
                arr_shape_list.append(data_dims_shape[CZI_SAMPLES_DIM_CHAR][1])

        ans = None
        if isinstance(data, da.Array):
            ans = da.zeros(
                shape=tuple(arr_shape_list),
                dtype=data.dtype,
            )
        else:
            ans = np.zeros(arr_shape_list, dtype=data.dtype)

        for (tile_info, box) in tile_bboxes.items():
            # Construct data indexes to use
            tile_dims = tile_info.dimension_coordinates
            tile_dims.pop(CZI_SCENE_DIM_CHAR, None)
            tile_dims.pop(CZI_BLOCK_DIM_CHAR, None)
            data_indexes = [
                tile_dims[t_dim]
                for t_dim in data_dims
                if t_dim not in REQUIRED_CHUNK_DIMS
            ]
            # Add Y and X
            data_indexes.append(slice(None))  # Y ":"
            data_indexes.append(slice(None))  # X ":"
            if CZI_SAMPLES_DIM_CHAR in tile_dims.keys():
                data_indexes.append(slice(None))

            # Construct data indexes for ans
            ans_indexes = []
            for dim in ordered_dims_present:
                if dim not in [
                    DimensionNames.MosaicTile,
                    DimensionNames.Samples,
                    DimensionNames.SpatialY,
                    DimensionNames.SpatialX,
                ]:
                    if dim in tile_dims.keys():
                        ans_indexes.append(tile_dims[dim])

                if dim is DimensionNames.SpatialY:
                    start = box.y - final_bbox.y
                    ans_indexes.append(slice(start, start + box.h, 1))
                if dim is DimensionNames.SpatialX:
                    start = box.x - final_bbox.x
                    ans_indexes.append(slice(start, start + box.w, 1))
                if dim is DimensionNames.Samples:
                    ans_indexes.append(slice(None))

            # Assign the tiles into ans
            ans[tuple(ans_indexes)] = data[tuple(data_indexes)]

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
                data=data,
                data_dims=self.mapped_dims,
                data_dims_shape=dims_shape,
                tile_bboxes=bboxes,
                final_bbox=mosaic_scene_bbox,
            )

            # Copy metadata
            dims = [
                d
                for d in self.xarray_dask_data.dims
                if d is not DimensionNames.MosaicTile
            ]
            coords: Dict[Hashable, Any] = {
                d: v
                for d, v in self.xarray_dask_data.coords.items()
                if d
                not in [
                    DimensionNames.MosaicTile,
                    DimensionNames.SpatialY,
                    DimensionNames.SpatialX,
                ]
            }

            # Add expanded Y and X coords
            if self.physical_pixel_sizes.Y is not None:
                dim_y_index = dims.index(DimensionNames.SpatialY)
                coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
                    0, stitched.shape[dim_y_index], self.physical_pixel_sizes.Y
                )
            if self.physical_pixel_sizes.X is not None:
                dim_x_index = dims.index(DimensionNames.SpatialX)
                coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
                    0, stitched.shape[dim_x_index], self.physical_pixel_sizes.X
                )

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

    def get_mosaic_tile_position(self, mosaic_tile_index: int) -> Tuple[int, int]:
        """
        Get the absolute position of the top left point for a single mosaic tile.

        Parameters
        ----------
        mosaic_tile_index: int
            The index for the mosaic tile to retrieve position information for.

        Returns
        -------
        top: int
            The Y coordinate for the tile position.
        left: int
            The X coordinate for the tile position.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        IndexError
            No matching mosaic tile index found.
        """
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        # Get max of mosaic positions from lif
        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource)

            bboxes = czi.get_all_mosaic_tile_bounding_boxes(S=self.current_scene_index)
            bbox = list(bboxes.values())[mosaic_tile_index]
            return bbox.y, bbox.x
