#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError
from tifffile.tifffile import TiffTags

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
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
    def _is_supported_image(fs: AbstractFileSystem, path: str) -> bool:
        try:
            with fs.open(path) as open_resource:
                with TiffFile(open_resource):
                    return True

        except (TiffFileError, TypeError):
            return False

    def __init__(self, image: types.PathLike):
        """
        Wraps the tifffile API to provide the same aicsimageio Reader API but for
        volumetric Tiff (and other tifffile supported) images.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension = self.path.split(".")[-1]

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self.extension
            )

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # This is non-metadata tiff, just use available series indicies
                    self._scenes = tuple(
                        metadata_utils.generate_ome_image_id(i)
                        for i in range(len(tiff.series))
                    )

        return self._scenes

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem, path: str, scene: int, index: int
    ) -> np.ndarray:
        """
        Open a file for reading, seek to plane index and read as numpy.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
        scene: int
            The scene index to pull the plane from.
        index: int
            The image plane index to seek to and read from the file.

        Returns
        -------
        plane: np.ndarray
            The image plane as a numpy array.
        """
        with fs.open(path) as open_resource:
            with TiffFile(open_resource) as tiff:
                return tiff.series[scene].pages[index].asarray()

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
                str(i) for i in range(shape[dims.index(DimensionNames.Channel)])
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
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get a sample YX plane
                sample = self._get_image_data(
                    fs=self.fs,
                    path=self.path,
                    scene=self.current_scene_index,
                    index=0,
                )

                # Get shape of current scene
                # Replace YX dims with empty dimensions
                operating_shape = tiff.series[self.current_scene_index].shape

                # If the data is RGB we need to pull in the channels as well
                if tiff.series[self.current_scene_index].keyframe.samplesperpixel != 1:
                    operating_shape = operating_shape[:-3] + (1, 1, 1)

                # Otherwise the data is in 2D planes (Y, X)
                else:
                    operating_shape = operating_shape[:-2] + (1, 1)

                # Make ndarray for lazy arrays to fill
                lazy_arrays = np.ndarray(operating_shape, dtype=object)
                for plane_index, (np_index, _) in enumerate(
                    np.ndenumerate(lazy_arrays)
                ):
                    # Fill the numpy array with the delayed arrays
                    lazy_arrays[np_index] = da.from_delayed(
                        delayed(TiffReader._get_image_data)(
                            fs=self.fs,
                            path=self.path,
                            scene=self.current_scene_index,
                            index=plane_index,
                        ),
                        shape=sample.shape,
                        dtype=sample.dtype,
                    )

                # Convert the numpy array of lazy readers into a dask array
                image_data = da.block(lazy_arrays.tolist())

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
