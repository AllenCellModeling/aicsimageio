#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError, imread
from tifffile.tifffile import TiffTags

from .. import constants, exceptions, types
from ..dimensions import DEFAULT_CHUNK_DIMS, REQUIRED_CHUNK_DIMS, DimensionNames
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
    """
    Wraps the tifffile API to provide the same aicsimageio Reader API but for
    volumetric Tiff (and other tifffile supported) images.

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
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        dim_order: Optional[Union[List[str], str]] = None,
        channel_names: Optional[Union[List[str], List[List[str]]]] = None,
        **kwargs: Any,
    ):
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(image, enforce_exists=True)

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

        self.chunk_dims = chunk_dims
        self._dim_order = dim_order
        self._channel_names = channel_names

        # Enforce valid image
        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        if self._scenes is None:
            with self._fs.open(self._path) as open_resource:
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
        retrieve_indices: Tuple[Union[int, slice]],
        transpose_indices: List[int],
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
        retrieve_indices: Tuple[Union[int, slice]]
            The image indices to retrieve.
        transpose_indices: List[int]
            The indices to transpose to prior to requesting data.

        Returns
        -------
        chunk: np.ndarray
            The image chunk as a numpy array.
        """
        with fs.open(path) as open_resource:
            arr = da.from_zarr(
                imread(
                    open_resource, aszarr=True, series=scene, level=0, chunkmode="page"
                )
            )
            arr = arr.transpose(transpose_indices)

            # By setting the compute call to always use a "synchronous" scheduler,
            # it informs Dask not to look for an existing scheduler / client
            # and instead simply read the data using the current thread / process.
            #
            # In doing so, we shouldn't run into any worker data transfer and handoff
            # _during_ a read.
            return arr[retrieve_indices].compute(scheduler="synchronous")

    def _get_tiff_tags(self, tiff: TiffFile) -> TiffTags:
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

    def _guess_tiff_dim_order(self, tiff: TiffFile) -> List[str]:
        scene = tiff.series[self.current_scene_index]
        dims_from_meta = scene.pages.axes

        # If all dims are known, simply return as list
        if UNKNOWN_DIM_CHAR not in dims_from_meta:
            return [d for d in dims_from_meta]

        # Otherwise guess the dimensions and return merge
        else:
            # Get basic guess from shape size
            guessed_dims = Reader._guess_dim_order(scene.shape)
            return [d for d in self._merge_dim_guesses(dims_from_meta, guessed_dims)]

    def _get_dims_for_scene(self, tiff: TiffFile) -> List[str]:
        # Get / guess dims
        if self._dim_order is None:
            return self._guess_tiff_dim_order(tiff)

        # Provided list get or guess based
        if isinstance(self._dim_order, list):
            # This list index has a value, use it
            if self._dim_order[self.current_scene_index] is not None:
                return list(self._dim_order[self.current_scene_index])

            # Otherwise guess
            return self._guess_tiff_dim_order(tiff)

        # Provided the same string for all, use
        return list(self._dim_order)

    def _get_channel_names_for_scene(
        self, image_shape: Tuple[int], dims: List[str]
    ) -> Optional[List[str]]:
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

    def _create_dask_array(
        self, tiff: TiffFile, selected_scene_dims_list: List[str]
    ) -> da.Array:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        tiff: TiffFile
            An open TiffFile for processing.
        selected_scene_dims_list: List[str]
            The dimensions to use for constructing the array with.
            Required for managing chunked vs non-chunked dimensions.

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

        # Construct delayed dask array
        selected_scene = tiff.series[self.current_scene_index]
        selected_scene_dims = "".join(selected_scene_dims_list)

        # Raise invalid dims error
        if len(selected_scene.shape) != len(selected_scene_dims):
            raise exceptions.ConflictingArgumentsError(
                f"Dimension string provided does not match the "
                f"number of dimensions found for this scene. "
                f"This scene shape: {selected_scene.shape}, "
                f"Provided dims string: {selected_scene_dims}"
            )

        # Constuct the chunk and non-chunk shapes one dim at a time
        # We also collect the chunk and non-chunk dimension order so that
        # we can swap the dimensions after we block out the array
        non_chunk_dim_order = []
        non_chunk_shape = []
        chunk_dim_order = []
        chunk_shape = []
        for dim, size in zip(selected_scene_dims, selected_scene.shape):
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

        # Construct the transpose indices that will be used to
        # transpose the array prior to pulling the chunk dims
        match_map = {dim: selected_scene_dims.find(dim) for dim in selected_scene_dims}
        transposer = []
        for dim in blocked_dim_order:
            transposer.append(match_map[dim])

        # Make ndarray for lazy arrays to fill
        lazy_arrays = np.ndarray(blocked_shape, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # All dimensions get their normal index except for chunk dims
            # which get filled with "full" slices
            indices_with_slices = np_index[: len(non_chunk_shape)] + (
                (slice(None, None, None),) * len(chunk_shape)
            )

            # Fill the numpy array with the delayed arrays
            lazy_arrays[np_index] = da.from_delayed(
                delayed(TiffReader._get_image_data)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    retrieve_indices=indices_with_slices,
                    transpose_indices=transposer,
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
        for i, d in enumerate(selected_scene_dims):
            new_index = blocked_dim_order.index(d)
            if new_index != i:
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Transpose back to normal
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
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get dims from provided or guess
                dims = self._get_dims_for_scene(tiff)

                # Create the delayed dask array
                image_data = self._create_dask_array(tiff, dims)

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Get channel names for this scene or generate
                channels = self._get_channel_names_for_scene(image_data.shape, dims)

                # Create coords
                coords = self._get_coords(
                    dims,
                    image_data.shape,
                    scene_index=self.current_scene_index,
                    channel_names=channels,
                )

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,  # type: ignore
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
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with self._fs.open(self._path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get dims from provided or guess
                dims = self._get_dims_for_scene(tiff)

                # Read image into memory
                image_data = tiff.series[self.current_scene_index].asarray()

                # Get unprocessed metadata from tags
                tiff_tags = self._get_tiff_tags(tiff)

                # Get channel names for this scene or generate
                channels = self._get_channel_names_for_scene(image_data.shape, dims)

                # Create dims and coords
                coords = self._get_coords(
                    dims,
                    image_data.shape,
                    scene_index=self.current_scene_index,
                    channel_names=channels,
                )

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,  # type: ignore
                    attrs={
                        constants.METADATA_UNPROCESSED: tiff_tags,
                        constants.METADATA_PROCESSED: tiff_tags[
                            TIFF_IMAGE_DESCRIPTION_TAG_INDEX
                        ].value,
                    },
                )
