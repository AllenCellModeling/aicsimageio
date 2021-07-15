#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from copy import copy
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, transforms, types
from ..dimensions import (
    DEFAULT_CHUNK_DIMS,
    DEFAULT_DIMENSION_ORDER_LIST,
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES,
    REQUIRED_CHUNK_DIMS,
    DimensionNames,
)
from ..utils import io_utils
from .reader import Reader

try:
    from readlif.reader import LifFile

except ImportError:
    raise ImportError(
        "readlif is required for this reader. "
        "Install with `pip install aicsimageio[lif]`"
    )

###############################################################################


class LifReader(Reader):
    """
    Wraps the readlif API to provide the same aicsimageio Reader API but for
    volumetric LIF images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    chunk_dims: Union[str, List[str]]
        Which dimensions to create chunks for.
        Default: DEFAULT_CHUNK_DIMS
        Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
        will always be added to the list if not present during dask array
        construction.

    Notes
    -----
    To use this reader, install with: `pip install aicsimageio[lif]`.
    """

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        try:
            with fs.open(path) as open_resource:
                LifFile(open_resource)
                return True

        except ValueError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims

        # Delayed storage
        self._scene_short_info: Dict[str, Any] = {}
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
                lif = LifFile(open_resource)
                scene_names = [image["name"] for image in lif.image_list]
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
            selected_scene = LifFile(open_resource).get_image(scene)

            # Create the fill array shape
            # Drop the YX as we will be pulling the individual YX planes
            retrieve_shape: List[int] = []
            use_selected_or_np_map: Dict[str, Optional[int]] = {}
            for dim, index_op in zip(retrieve_dims, retrieve_indices):
                if dim not in [DimensionNames.SpatialY, DimensionNames.SpatialX]:
                    # Handle slices
                    if index_op is None:
                        # Store the dim for later to inform to use the np index
                        use_selected_or_np_map[dim] = None
                        if dim == DimensionNames.MosaicTile:
                            retrieve_shape.append(selected_scene.n_mosaic)
                        elif dim == DimensionNames.Time:
                            retrieve_shape.append(selected_scene.nt)
                        elif dim == DimensionNames.Channel:
                            retrieve_shape.append(selected_scene.channels)
                        elif dim == DimensionNames.SpatialZ:
                            retrieve_shape.append(selected_scene.nz)

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
                plane_indices: Dict[str, int] = {}

                # Handle MosaicTile
                if DimensionNames.MosaicTile in use_selected_or_np_map:
                    if use_selected_or_np_map[DimensionNames.MosaicTile] is None:
                        plane_indices["m"] = np_index[
                            retrieve_dims.index(DimensionNames.MosaicTile)
                        ]
                    else:
                        plane_indices["m"] = use_selected_or_np_map[  # type: ignore
                            DimensionNames.MosaicTile
                        ]

                # Handle Time
                if use_selected_or_np_map[DimensionNames.Time] is None:
                    plane_indices["t"] = np_index[
                        retrieve_dims.index(DimensionNames.Time)
                    ]
                else:
                    plane_indices["t"] = use_selected_or_np_map[  # type: ignore
                        DimensionNames.Time
                    ]

                # Handle Channels
                if use_selected_or_np_map[DimensionNames.Channel] is None:
                    plane_indices["c"] = np_index[
                        retrieve_dims.index(DimensionNames.Channel)
                    ]
                else:
                    plane_indices["c"] = use_selected_or_np_map[  # type: ignore
                        DimensionNames.Channel
                    ]

                # Handle SpatialZ
                if use_selected_or_np_map[DimensionNames.SpatialZ] is None:
                    plane_indices["z"] = np_index[
                        retrieve_dims.index(DimensionNames.SpatialZ)
                    ]
                else:
                    plane_indices["z"] = use_selected_or_np_map[  # type: ignore
                        DimensionNames.SpatialZ
                    ]

                # Append the retrieved plane as a numpy array
                planes.append(np.asarray(selected_scene.get_frame(**plane_indices)))

            # Stack and reshape to get rid of the array of arrays
            scene_dims = selected_scene.info["dims"]
            retrieved_chunk = np.stack(planes).reshape(
                np_array_for_indices.shape + (scene_dims.y, scene_dims.x)
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
        self, lif: LifFile, selected_scene_dims: List[str]
    ) -> xr.DataArray:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        lif: LifFile
            An open LifFile for processing.
        selected_scene_dims: List[str]
            The dimensions for the scene to create the dask array for

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
        selected_scene = lif.get_image(self.current_scene_index)
        selected_scene_shape: List[int] = []
        for dim in selected_scene_dims:
            if dim == DimensionNames.MosaicTile:
                selected_scene_shape.append(selected_scene.n_mosaic)
            elif dim == DimensionNames.Time:
                selected_scene_shape.append(selected_scene.nt)
            elif dim == DimensionNames.Channel:
                selected_scene_shape.append(selected_scene.channels)
            elif dim == DimensionNames.SpatialZ:
                selected_scene_shape.append(selected_scene.nz)
            elif dim == DimensionNames.SpatialY:
                selected_scene_shape.append(selected_scene.info["dims"].y)
            elif dim == DimensionNames.SpatialX:
                selected_scene_shape.append(selected_scene.info["dims"].x)

        # Get sample for dtype
        sample_plane = np.asarray(selected_scene.get_frame())

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order = []
        non_chunk_shape = []
        chunk_dim_order = []
        chunk_shape = []
        for dim, size in zip(selected_scene_dims, selected_scene_shape):
            if dim in self.chunk_dims:
                chunk_dim_order.append(dim)
                chunk_shape.append(size)

            else:
                non_chunk_dim_order.append(dim)
                non_chunk_shape.append(size)

        # Fill out the rest of the blocked shape with dimension sizes of 1 to
        # match the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to
        # outer-most with the chunks as long as the dimension is size 1
        blocked_dim_order = non_chunk_dim_order + chunk_dim_order
        blocked_shape = tuple(non_chunk_shape) + ((1,) * len(chunk_shape))

        # Make ndarray for lazy arrays to fill
        lazy_arrays: np.ndarray = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get None, which tell the get data func to pull the whole dim
            retrieve_indices = np_index[: len(non_chunk_shape)] + (
                (None,) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(LifReader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    retrieve_dims=blocked_dim_order,
                    retrieve_indices=retrieve_indices,
                ),
                shape=chunk_shape,
                dtype=sample_plane.dtype,
            )

        # Convert the numpy array of lazy readers into a dask array
        image_data = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example, if the original dimension ordering was "TZYX" and we
        # chunked by "T", "Y", and "X"
        # we created an array with dimensions ordering "ZTYX"
        transpose_indices = []
        for i, d in enumerate(selected_scene_dims):
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
        xml: ET.Element, image_short_info: Dict[str, Any], scene_index: int
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords: Dict[str, Any] = {}

        # Get all images
        img_sets = xml.findall(".//Image")

        # Select the current scene
        img = img_sets[scene_index]

        # Construct channel list
        scene_channel_list = []
        channels = img.findall(".//ChannelDescription")
        channel_details = img.findall(".//WideFieldChannelInfo")
        for i, channel in enumerate(channels):
            if len(channels) <= len(channel_details):
                channel_detail = channel_details[i]
                scene_channel_list.append(
                    (
                        f"{channel_detail.attrib['LUT']}"
                        f"--{channel_detail.attrib['ContrastingMethodName']}"
                        f"--{channel_detail.attrib['FluoCubeName']}"
                    )
                )
            else:
                scene_channel_list.append(f"{channel.attrib['LUTName']}")

        # Attach channel names to coords
        coords[DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        scale_x, scale_y, scale_z, scale_t = image_short_info["scale"]

        # Handle Spatial Dimensions
        if scale_z is not None:
            coords[DimensionNames.SpatialZ] = Reader._generate_coord_array(
                0, image_short_info["dims"].z, scale_z
            )
        if scale_y is not None:
            coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
                0, image_short_info["dims"].y, scale_y
            )
        if scale_x is not None:
            coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
                0, image_short_info["dims"].x, scale_x
            )

        # Time
        if scale_t is not None:
            coords[DimensionNames.Time] = Reader._generate_coord_array(
                0, image_short_info["dims"].t, scale_t
            )

        # Create physical pixal sizes
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
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)
            self._scene_short_info = selected_scene.info

            # Check for mosaic tiles
            tile_positions = self._scene_short_info["mosaic_position"]

            # If there are tiles in the image use mosaic dims
            if len(tile_positions) > 0:
                dims = DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                dims = DEFAULT_DIMENSION_ORDER_LIST

            # Get image data
            image_data = self._create_dask_array(lif, dims)

            # Get metadata
            meta = lif.xml_root

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                image_short_info=self._scene_short_info,
                scene_index=self.current_scene_index,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

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
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)
            self._scene_short_info = selected_scene.info

            # Check for mosaic tiles
            tile_positions = self._scene_short_info["mosaic_position"]

            # If there are tiles in the image use mosaic dims
            if len(tile_positions) > 0:
                dims = DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES

            # Otherwise use standard dims
            else:
                dims = DEFAULT_DIMENSION_ORDER_LIST

            # Get image data
            image_data = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
                retrieve_dims=dims,
                retrieve_indices=[None] * len(dims),  # Get all planes
            )

            # Get metadata
            meta = lif.xml_root

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                image_short_info=self._scene_short_info,
                scene_index=self.current_scene_index,
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
    def _stitch_tiles(
        data: types.ArrayLike,
        dims: str,
        ny: int,
        nx: int,
    ) -> types.ArrayLike:
        # Fill all tiles
        rows: List[types.ArrayLike] = []
        for row_i in range(ny):
            row: List[types.ArrayLike] = []
            for col_i in range(nx):
                # Calc m_index
                m_index = (row_i * nx) + col_i

                # Get tile by getting all data for specific M
                tile = transforms.reshape_data(
                    data,
                    given_dims=dims,
                    return_dims=dims.replace(DimensionNames.MosaicTile, ""),
                    M=m_index,
                )

                # LIF image stitching has a 1 pixel overlap
                # Take all pixels except the first _except_ if this is the last tile
                # in the row (the X dimension)
                if col_i + 1 < nx:
                    row.insert(0, tile[:, :, :, :, 1:])
                else:
                    row.insert(0, tile)

            # Concat row and append
            # Take all pixels except the first Y dimension pixel _except_ if this is the
            # last row
            np_row = np.concatenate(row, axis=-1)
            if row_i + 1 < ny:
                rows.insert(0, np_row[:, :, :, 1:, :])
            else:
                rows.insert(0, np_row)

        # Concatenate
        mosaic = np.concatenate(rows, axis=-2)

        return mosaic

    def _construct_mosaic_xarray(self, data: types.ArrayLike) -> xr.DataArray:
        # Get max of mosaic positions from lif
        with self._fs.open(self._path) as open_resource:
            lif = LifFile(open_resource)
            selected_scene = lif.get_image(self.current_scene_index)
            last_tile_position = selected_scene.info["mosaic_position"][-1]

        # Stitch
        stitched = self._stitch_tiles(
            data=data,
            dims=self.dims.order,
            ny=last_tile_position[0] + 1,
            nx=last_tile_position[1] + 1,
        )

        # Copy metadata
        dims = [
            d for d in self.xarray_dask_data.dims if d is not DimensionNames.MosaicTile
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
        scale_x, scale_y, _, _ = selected_scene.info["scale"]
        if scale_y is not None:
            coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
                0, stitched.shape[-2], scale_y
            )
        if scale_x is not None:
            coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
                0, stitched.shape[-1], scale_x
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

        # LIFs are packed from bottom right to top left
        # To counter: subtract 1 + M from list index to get from back of list
        index_y, index_x, _, _ = self._scene_short_info["mosaic_position"][
            -(mosaic_tile_index + 1)
        ]

        return (
            (index_y * self.dims.Y) - index_y,  # type: ignore
            (index_x * self.dims.X) - index_x,  # type: ignore
        )
