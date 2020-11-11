#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, Tuple

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem
from tifffile import TiffFile, TiffFileError

from .. import exceptions, types
from ..dimensions import DEFAULT_DIMENSION_ORDER, DimensionNames
from ..utils import io_utils
from .reader import Reader

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
            raise exceptions.UnsupportedFileFormatError(self.extension)

    @property
    def scenes(self) -> Tuple[int]:
        if self._scenes is None:
            with self.fs.open(self.path) as open_resource:
                with TiffFile(open_resource) as tiff:
                    # This is non-metadata tiff, just use available series indicies
                    self._scenes = tuple(range(tiff.scenes))

        return self._scenes

    @property
    def current_scene(self) -> int:
        if self._current_scene is None:
            self._current_scene = self.scenes[0]

        return self._current_scene

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

    @staticmethod
    def _get_metadata(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
    ) -> Dict:
        with fs.open(path) as open_resource:
            with TiffFile(open_resource) as tiff:
                return tiff.scenes[scene].pages[0].tags

    @staticmethod
    def _merge_dim_guesses(dims_from_meta: str, guessed_dims: str):
        # Construct a "best guess" (super naive)
        best_guess = []
        for dim_from_meta in dims_from_meta:
            # Dim from meta is recognized, add it
            if dim_from_meta in DEFAULT_DIMENSION_ORDER:
                best_guess.append(dim_from_meta)

            # Dim from meta isn't recognized
            # Find next dim that isn't already in best guess
            else:
                appended_dim = False
                for guessed_dim in guessed_dims:
                    if guessed_dim not in best_guess:
                        best_guess.append(guessed_dim)
                        appended_dim = True
                        break

                # All of our guess dims were already in the best guess list,
                # append the dim read from meta
                if not appended_dim:
                    best_guess.append(dim_from_meta)

        return "".join(best_guess)

    def _guess_dim_order(self):
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                scene = tiff.series[self.current_scene]
                dims_from_meta = scene.pages.axes

                # We can sometimes trust the dimension info in the image
                if all([d in DEFAULT_DIMENSION_ORDER for d in dims_from_meta]):
                    return dims_from_meta

                # Sometimes the dimension info is wrong in certain dimensions,
                # so guess that dimension
                else:
                    # Get basic guess from shape size
                    guessed_dims = Reader._guess_dim_order(scene.shape)
                    return self._merge_dim_guesses(dims_from_meta, guessed_dims)

    @staticmethod
    def _get_coords(dims: str, shape: Tuple[int]):
        # Use dims for coord determination
        coords = {}

        # Use range for channel indices
        if DimensionNames.Channel in dims:
            coords[DimensionNames.Channel] = list(
                range(shape[dims.index(DimensionNames.Channel)])
            )

        return coords

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray  object.
            Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.fs.open(self.path) as open_resource:
            with TiffFile(open_resource) as tiff:
                # Get a sample YX plane
                sample = self._get_image_data(
                    fs=self.fs,
                    path=self.path,
                    scene=self.current_scene,
                    index=0,
                )

                # Get shape of current scene
                # Replace YX dims with empty dimensions
                operating_shape = tiff.scenes[self.current_scene].shape
                operating_shape = operating_shape[:-2] + (1, 1)

                # Make ndarray for lazy arrays to fill
                lazy_arrays = np.ndarray(operating_shape, dtype=object)
                for plane_index, (np_index, _) in enumerate(
                    np.ndenumerate(lazy_arrays)
                ):
                    # Fill the numpy array with the delayed arrays
                    lazy_arrays[np_index] = da.from_delayed(
                        delayed(TiffReader._imread)(
                            fs=self.fs,
                            path=self.path,
                            scene=self.current_scene,
                            index=plane_index,
                        ),
                        shape=sample.shape,
                        dtype=sample.dtype,
                    )

                # Convert the numpy array of lazy readers into a dask array
                image_data = da.block(lazy_arrays.tolist())

                # Get metadata from tags
                metadata = self._get_metadata(
                    fs=self.fs,
                    path=self.path,
                    scene=self.current_scene,
                )

                # Create dims and coords
                dims = self._guess_dim_order()
                coords = self._get_coords(dims, image_data.shape)

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs=metadata,
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
                image_data = tiff.scenes[self.current_scene].asarray()

                # Get metadata from tags
                metadata = self._get_metadata(
                    fs=self.fs,
                    path=self.path,
                    scene=self.current_scene,
                )

                # Create dims and coords
                dims = self._guess_dim_order()
                coords = self._get_coords(dims, image_data.shape)

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,
                    attrs=metadata,
                )

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.xarray_dask_data.attrs

        return self._metadata
