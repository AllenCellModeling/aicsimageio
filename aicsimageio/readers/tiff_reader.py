#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError, imread
from tifffile.tifffile import TiffTags

from .. import constants, exceptions, types
from ..dimensions import DEFAULT_CHUNK_BY_DIMS, REQUIRED_CHUNK_BY_DIMS, DimensionNames
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from .reader import Reader

###############################################################################

# "Q" is used by Gohlke to say "unknown dimension"
# https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py#L10840
UNKNOWN_DIM_CHAR = "Q"
TIFF_IMAGE_DESCRIPTION_TAG_INDEX = 270

###############################################################################


class TiffReader(Reader):
    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource):
                    return True

        except (TiffFileError, TypeError):
            return False

    def __init__(
        self,
        image: types.PathLike,
        chunk_by_dims: List[str] = DEFAULT_CHUNK_BY_DIMS,
    ):
        """
        Wraps the tifffile API to provide the same aicsimageio Reader API but for
        volumetric Tiff (and other tifffile supported) images.

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
                with TiffFile(open_resource) as tiff:
                    # This is non-metadata tiff, just use available series indices
                    self._scenes = tuple(
                        metadata_utils.generate_ome_image_id(i)
                        for i in range(len(tiff.series))
                    )

        return self._scenes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        indices: Tuple[Union[int, slice]],
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
        indices: Tuple[Union[int, slice]]
            The image indices to retrieve.

        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        with fs.open(path) as open_resource:
            return da.from_zarr(
                imread(
                    open_resource, aszarr=True, series=scene, level=0, chunkmode="page"
                )
            )[indices].compute()

    def _get_tiff_tags(self) -> TiffTags:
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                return tiff.series[self.current_scene_index].pages[0].tags

    @staticmethod
    def _merge_dim_guesses(dims_from_meta: str, guessed_dims: str) -> str:
        # Construct a "best guess" (super naive)
        best_guess = []
        for dim_from_meta in dims_from_meta:
            # Dim from meta is recognized, add it
            if dim_from_meta != UNKNOWN_DIM_CHAR:
                best_guess.append(dim_from_meta)

            # Dim from meta isn't recognized
            # Find next dim that isn't already in best guess or dims from meta
            else:
                appended_dim = False
                for guessed_dim in guessed_dims:
                    if (
                        guessed_dim not in best_guess
                        and guessed_dim not in dims_from_meta
                    ):
                        best_guess.append(guessed_dim)
                        appended_dim = True
                        break

                # All of our guess dims were already in the best guess list,
                # append the dim read from meta
                if not appended_dim:
                    best_guess.append(dim_from_meta)

        return "".join(best_guess)

    def _guess_dim_order(self) -> List[str]:
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                scene = tiff.series[self.current_scene_index]
                dims_from_meta = scene.pages.axes

                # If all dims are known, simply return as list
                if UNKNOWN_DIM_CHAR not in dims_from_meta:
                    return [d for d in dims_from_meta]

                # Otherwise guess the dimensions and return merge
                else:
                    # Get basic guess from shape size
                    guessed_dims = Reader._guess_dim_order(scene.shape)
                    return [
                        d for d in self._merge_dim_guesses(dims_from_meta, guessed_dims)
                    ]

    @staticmethod
    def _get_coords(
        dims: str,
        shape: Tuple[int],
    ) -> Dict[str, Union[List, types.ArrayLike]]:
        # Use dims for coord determination
        coords = {}

        # Use range for channel indices
        if DimensionNames.Channel in dims:
            coords[DimensionNames.Channel] = [
                f"Channel:{i}" for i in range(shape[dims.index(DimensionNames.Channel)])
            ]

        return coords

    def _create_dask_array(self) -> da.Array:
        """
        Creates a delayed dask array for the file.

        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in REQUIRED_CHUNK_BY_DIMS:
            if dim not in self.chunk_by_dims:
                self.chunk_by_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_by_dims = [d.upper() for d in self.chunk_by_dims]

        # Construct delayed dask array
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                selected_scene = tiff.series[self.current_scene_index]
                selected_scene_dims = selected_scene.axes

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
                for plane_index, (np_index, _) in enumerate(
                    np.ndenumerate(lazy_arrays)
                ):
                    # All dimensions get their normal index except for chunk dims
                    # which get filled with "full" slices
                    indices_with_slices = np_index[: len(non_chunk_shape)] + (
                        (slice(None, None, None),) * len(chunk_shape)
                    )

                    # Fill the numpy array with the delayed arrays
                    lazy_arrays[np_index] = da.from_delayed(
                        delayed(TiffReader._get_image_data)(
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
        # Create the delayed dask array
        image_data = self._create_dask_array()

        # Get unprocessed metadata from tags
        tiff_tags = self._get_tiff_tags()

        # Create dims and coords
        dims = self._guess_dim_order()
        coords = self._get_coords(dims, image_data.shape)

        return xr.DataArray(
            image_data,
            dims=dims,
            coords=coords,
            attrs={
                constants.METADATA_UNPROCESSED: tiff_tags,
                constants.METADATA_PROCESSED: tiff_tags[
                    TIFF_IMAGE_DESCRIPTION_TAG_INDEX
                ].value,
            },
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
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags()

                # Create dims and coords
                dims = self._guess_dim_order()
                coords = self._get_coords(dims, image_data.shape)

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs={
                        constants.METADATA_UNPROCESSED: tiff_tags,
                        constants.METADATA_PROCESSED: tiff_tags[
                            TIFF_IMAGE_DESCRIPTION_TAG_INDEX
                        ].value,
                    },
                )
