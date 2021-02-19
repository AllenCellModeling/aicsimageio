#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from readlif.reader import LifFile, LifImage

from .. import dimensions, exceptions, types
from ..utils import io_utils
from .reader import Reader

###############################################################################


class LifReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs) -> bool:
        try:
            with fs.open(path) as open_resource:
                print(type(open_resource))
                LifFile(open_resource)
                return True

        except ValueError:
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_by_dims: List[str] = dimensions.DEFAULT_CHUNK_BY_DIMS,
    ):
        """
        Wraps the readlif API to provide the same aicsimageio Reader API but for
        volumetric LIF images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        chunk_by_dims: List[str]
            Which dimensions to create chunks for.
            Default: DEFAULT_CHUNK_BY_DIMS
            Note: Dimensions.SpatialY, Dimensions.SpatialX, and DimensionNames.Samples,
            will always be added to the list if not present during dask array
            construction.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)

        # Store params
        self.chunk_by_dims = chunk_by_dims

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self.path
            )

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                lif = LifFile(open_resource)
                scene_names = [image["name"] for image in lif.image_list]
                self._scenes = tuple(scene_names)

        return self._scenes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        indices: Tuple[Union[int, slice]],
    ) -> np.ndarray:
        """
        Open a file for reading, pull each plane neccessary to meet the indices
        requested, stack into an array, reshape, and return.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the chunk from.
        indices: Tuple[Union[int, slice]]
            The image indices to retrieve.

        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        # "MTCZYX"
        # chunk TYX
        # (0, slice(None, None, None), 0, 0, slice(...), slice(...))

        # slice == T 100
        # np.product -> [0, 0, 0, 0], [0, 1, 0, 0]
        # Open and select the target image
        with fs.open(path) as open_resource:
            selected_scene = LifFile(open_resource).get_image(scene)

            # Pull all the desired planes
            planes: List[np.ndarray] = []
            for dim, index_operation in zip(
                dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES[:-2], indices[:-2]
            ):
                if isinstance(index_operation, int):
                    # pull plane
                    planes.append(selected_scene.get_frame(dim=index_operation))

                else:
                    for single_slice_iter in index_operation:
                        # pull planes

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
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        # Always add the plane dimensions if not present already
        for dim in dimensions.REQUIRED_CHUNK_BY_DIMS:
            if dim not in self.chunk_by_dims:
                self.chunk_by_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_by_dims = [d.upper() for d in self.chunk_by_dims]

        # Construct the delayed dask array
        with self.fs.open(self.path) as open_resource:
            selected_scene = LifFile(open_resource).get_image(self.current_scene_index)
            selected_scene_dims = (
                dimensions.DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES
            )

            # Constuct the chunk and non-chunk shapes one dim at a time
            # We also collect the chunk and non-chunk dimension order so that
            # we can swap the dimensions after we block out the array
            non_chunk_dim_order = []
            non_chunk_shape = []
            chunk_dim_order = []
            chunk_shape = []
            for dim, size in zip(selected_scene_dims, selected_scene.shape):
                if dim in self.chunk_by_dims:
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
            lazy_arrays = np.ndarray(blocked_shape, dtype=object)
            for plane_index, (np_index, _) in enumerate(np.ndenumerate(lazy_arrays)):
                # All dimensions get their normal index except for chunk dims
                # which get filled with "full" slices
                indices_with_slices = np_index[: len(non_chunk_shape)] + (
                    (slice(None, None, None),) * len(chunk_shape)
                )

                # Fill the numpy array with the delayed arrays
                lazy_arrays[np_index] = da.from_delayed(
                    delayed(LifReader._get_image_data)(
                        fs=self.fs,
                        path=self.path,
                        scene=self.current_scene_index,
                        indices=indices_with_slices,
                    ),
                    shape=chunk_shape,
                    dtype=selected_scene.dtype,
                )

            # Convert the numpy array of lazy readers into a dask array
            image_data = da.block(lazy_arrays.tolist())

            # Because we have set certain dimensions to be chunked and others not
            # we will need to transpose back to original dimension ordering
            # Example, if the original dimension ordering was "TZYX" and we
            # chunked by "T", "Y", and "X"
            # we created an array with dimensions ordering "ZTYX"
            transpose_indices = []
            transpose_required = False
            for i, d in enumerate(selected_scene_dims):
                new_index = blocked_dim_order.index(d)
                if new_index != i:
                    transpose_required = True
                    transpose_indices.append(new_index)
                else:
                    transpose_indices.append(i)

            # Only run if the transpose is actually required
            if transpose_required:
                image_data = da.transpose(image_data, tuple(transpose_indices))

            return image_data

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
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        pass
