#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import xml.etree.ElementTree as ET
from copy import copy
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types.model.ome import OME

from .. import constants, exceptions, types
from ..dimensions import DEFAULT_CHUNK_DIMS, REQUIRED_CHUNK_DIMS, DimensionNames
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from .reader import Reader

try:
    from _aicspylibczi import BBox, TileInfo
    from aicspylibczi import CziFile

except ImportError:
    raise ImportError(
        "aicspylibczi is required for this reader. "
        "Install with `pip install 'aicspylibczi>=3.1.1' 'fsspec>=2022.7.1'`"
    )

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

CZI_SAMPLES_DIM_CHAR = "A"
CZI_BLOCK_DIM_CHAR = "B"
CZI_SCENE_DIM_CHAR = "S"


###############################################################################

PIXEL_DICT = {
    "gray8": np.uint8,
    "gray16": np.uint16,
    "gray32": np.uint32,
    "gray32float": np.float32,
    "bgr24": np.uint8,
    "bgr48": np.uint16,
    "invalid": np.uint8,
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
    chunk_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: DimensionNames.SpatialY, DimensionNames.SpatialX, and
        DimensionNames.Samples, will always be added to the list if not present during
        dask array construction.
    include_subblock_metadata: bool
        Whether to append metadata from the subblocks to the rest of the embeded
        metadata.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Notes
    -----
    To use this reader, install with: `pip install aicspylibczi>=3.1.1`.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            if not isinstance(fs, LocalFileSystem):
                raise ValueError(
                    f"Cannot read CZIs from non-local file system. "
                    f"Received URI: {path}, which points to {type(fs)}."
                )

            with fs.open(path) as open_resource:
                CziFile(open_resource.f)
                return True

        except RuntimeError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        include_subblock_metadata: bool = False,
        fs_kwargs: Dict[str, Any] = {},
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                f"Cannot read CZIs from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims

        self._include_subblock_metadata = include_subblock_metadata

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
                czi = CziFile(open_resource.f)
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
        """Note: scenes with no name (`None`) will be renamed to
        "filename-<scene index>" to prevent ambiguity. Similarly, scenes with same
        names are automatically appended with occurrence number to distinguish
        between the two.

        Returns:
            Tuple[str, ...]: Scene names/id
        """
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
                czi = CziFile(open_resource.f)
                xpath_str = "./Metadata/Information/Image/Dimensions/S/Scenes/Scene"
                meta_scenes = czi.meta.findall(xpath_str)
                scene_names: List[str] = []

                # mapping of scene name to occurrences, indicating duplication.
                scene_name_frequency = {}
                for scene_idx, meta_scene in enumerate(meta_scenes):
                    shape = meta_scene.find("Shape")
                    if shape is not None:
                        shape_name = shape.get("Name")
                        scene_name = meta_scene.get("Name")
                        combined_scene_name = f"{scene_name}-{shape_name}"
                    else:
                        combined_scene_name = meta_scene.get("Name")
                        # Some scene names can be unpopulated, for those we should fill
                        # with filename-idx
                        if combined_scene_name is None:
                            fname_prefix = Path(self._path).stem
                            combined_scene_name = f"{fname_prefix}-{scene_idx}"
                        # Check for duplicated names
                        # first encounter with a duplicate modify original scene name
                        # to reflect its new duplicate status
                        if combined_scene_name not in scene_name_frequency:
                            scene_name_frequency[combined_scene_name] = [scene_idx, 1]
                        else:
                            if scene_name_frequency[combined_scene_name][1] == 1:
                                scene_names[
                                    scene_name_frequency[combined_scene_name][0]
                                ] += "-1"

                            scene_name_frequency[combined_scene_name][1] += 1

                            combined_scene_name += (
                                f"-{scene_name_frequency[combined_scene_name][1]}"
                            )

                    scene_names.append(combined_scene_name)

                # If the scene is implicit just assign it name Scene:0
                if len(scene_names) < 1:
                    scene_names = [metadata_utils.generate_ome_image_id(0)]
                else:
                    # reconcile scene list against the dims shape
                    dims_shape = czi.get_dims_shape()
                    if len(scene_names) != len(dims_shape) and czi.shape_is_consistent:
                        dims_shape_dict = dims_shape[0]
                        scene_range = dims_shape_dict.get(CZI_SCENE_DIM_CHAR)
                        if scene_range is not None:
                            scene_names = scene_names[scene_range[0] : scene_range[1]]
                        else:
                            # If this is the root node of a split multiscene czi,
                            # then the scene_range could be None because the dims_shape
                            # will be effectively empty.
                            # We do not currently support loading multi-file split
                            # scene CZI files
                            log.warning(
                                "CZI file appears to contain multiple scenes but "
                                "dimension data is not available in this file. "
                                "Root node of split multi-scene CZI files are not "
                                "supported by CziReader."
                            )

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
    def _adjust_scene_index(
        dims_shape: List[Dict], scene_index: int, consistent: bool
    ) -> int:
        """
        This function modifies a scene index to be an offset into the true scene
        indices reported by the czi file

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
        An int representing the scene index to use in a true libCZI dimension
        """
        dims_shape_index = 0 if consistent else scene_index
        dims_shape_dict = dims_shape[dims_shape_index]
        scene_range = dims_shape_dict.get(CZI_SCENE_DIM_CHAR)
        if scene_range is None:
            return scene_index
        if not consistent:
            # we have selected a dims_shape_dict already based on scene index
            # let's make sure the scene index is in the S range
            if scene_index < scene_range[0] or scene_index >= scene_range[1]:
                raise ValueError(
                    f"Scene index {scene_index} is not in the range "
                    f"{scene_range[0]} to {scene_range[1]}"
                )
            return scene_index
        return scene_range[0] + scene_index

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

        # Init czi
        with fs.open(path) as open_resource:
            czi = CziFile(open_resource.f)

            # Get current scene read dims
            adjusted_scene_index = CziReader._adjust_scene_index(
                czi.get_dims_shape(), scene, czi.shape_is_consistent
            )
            read_dims[CZI_SCENE_DIM_CHAR] = adjusted_scene_index

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
                if dim in read_dims or dim == CZI_BLOCK_DIM_CHAR:
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
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

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
            if dim in self.chunk_dims:
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
            for d in self.chunk_dims:
                this_chunk_read_dims.pop(d, None)

            # Get pixel type and catch unsupported
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(f"Pixel type: {czi.pixel_type} is not supported.")

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

            # Construct channel name list
            scene_channel_list = []
            channels = img.findall("./Channel")
            for i, channel in enumerate(channels):
                # Id is required, Name is not.
                # But we prefer to use Name if it is present
                channel_name = channel.attrib.get("Name")
                channel_id = channel.attrib.get("Id")
                if channel_name is None:
                    # TODO: the next best guess is to see if there's a Name in
                    # DisplaySetting/Channels/Channel
                    # xpath_str = "./Metadata/DisplaySetting/Channels"
                    # displaysetting_channels = xml.findall(xpath_str)
                    # ds_channels = displaysetting_channels[0].findall("./Channel")
                    # to find matching channel must match on Id attribute or if Id not
                    # present, just on collection index i
                    # If we didn't find a match this way, just use the Id as the name
                    channel_name = channel_id
                if channel_name is None:
                    # This is actually an error because Id was required by the spec
                    channel_name = metadata_utils.generate_ome_channel_id(
                        str(scene_index), str(i)
                    )

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
        # the values are stored in units of meters always in .czi, so
        # divide by 1E-6 to convert to microns
        if scale_xe is not None and scale_xe.text is not None:
            scale_x = float(scale_xe.text) / (1e-6)
        if scale_ye is not None and scale_ye.text is not None:
            scale_y = float(scale_ye.text) / (1e-6)
        if scale_ze is not None and scale_ze.text is not None:
            scale_z = float(scale_ze.text) / (1e-6)

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
            czi = CziFile(open_resource.f)

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

            # Append subblock metadata to the other metadata if param is True
            if self._include_subblock_metadata:
                subblocks = czi.read_subblock_metadata(unified_xml=True)
                meta.append(subblocks)

            # Store pixel sizes
            self._px_sizes = px_sizes

            # handle edge case where image has 0,0 YX dims:
            if image_data.shape[-2:] == (0, 0):
                return xr.DataArray(
                    dims=coords.keys(),
                    coords=coords,
                    attrs={constants.METADATA_UNPROCESSED: meta},
                )
            else:
                return xr.DataArray(
                    image_data,
                    dims=img_dims_list,
                    coords=coords,
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
            czi = CziFile(open_resource.f)
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
                coords=coords,
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
            czi = CziFile(open_resource.f)
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
    def ome_metadata(self) -> OME:
        return metadata_utils.transform_metadata_with_xslt(
            self.metadata,
            Path(__file__).parent.parent
            / "metadata/czi-to-ome-xslt/xslt/czi-to-ome.xsl",
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
        if self._px_sizes is None:
            # We get pixel sizes as a part of array construct
            # so simply run array construct
            self.dask_data

        if self._px_sizes is None:
            raise ValueError("Pixel sizes weren't created as a part of image reading")

        return self._px_sizes

    def get_mosaic_tile_position(
        self,
        mosaic_tile_index: int,
        **kwargs: int,
    ) -> Tuple[int, int]:
        """
        Get the absolute position of the top left point for a single mosaic tile.

        Parameters
        ----------
        mosaic_tile_index: int
            The index for the mosaic tile to retrieve position information for.
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

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

        Notes
        -----
        Defaults T and C dimensions to 0 if present as dimensions in image
        to avoid reading in massive image stack for large files.
        """
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource.f)

            # Default Channel and Time dimensions to 0 to improve
            # worst case read time for large files **only**
            # when those dimensions are present on the image.
            for dimension_name in [DimensionNames.Channel, DimensionNames.Time]:
                if dimension_name not in kwargs and dimension_name in self.dims.order:
                    kwargs[dimension_name] = 0

            bbox = czi.get_mosaic_tile_bounding_box(
                M=mosaic_tile_index, S=self.current_scene_index, **kwargs
            )
            return bbox.y, bbox.x

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Get the absolute positions of the top left points for each mosaic tile
        matching the specified dimensions and current scene.

        Parameters
        ----------
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        mosaic_tile_positions: List[Tuple[int, int]]
            List of the Y and X coordinate for the tile positions.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        """
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        with self._fs.open(self._path) as open_resource:
            czi = CziFile(open_resource.f)

            tile_info_to_bboxes = czi.get_all_mosaic_tile_bounding_boxes(
                S=self.current_scene_index, **kwargs
            )

            # Convert dictionary of tile info mappings to
            # a list of bounding boxes sorted according to their
            # respective M indexes
            m_indexes_to_mosaic_positions = {
                tile_info.m_index: (bbox.y, bbox.x)
                for tile_info, bbox in tile_info_to_bboxes.items()
            }
            return [
                m_indexes_to_mosaic_positions[m_index]
                for m_index in sorted(m_indexes_to_mosaic_positions.keys())
            ]
