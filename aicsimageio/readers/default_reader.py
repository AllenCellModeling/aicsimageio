#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from typing import Dict, Set

import dask.array as da
import fsspec
import imageio
import numpy as np
import xarray as xr
from dask import delayed

from .. import exceptions, types
from ..dimensions import Dimensions
from ..utils import io_utils
from .reader import Reader

###############################################################################

class DefaultReader(Reader):
    """
    A catch all for image file reading that defaults to using imageio implementations.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    """

    FFMPEG_FORMATS = ["mov", "avi", "mpg", "mpeg", "mp4", "mkv", "wmv", "ogg"]

    def __init__(self, image: types.PathLike):
        # Expand details of provided image
        self.abstract_file = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension = self.abstract_file.path.split(".")[-1]

        # Set mode to many-image reading if FFMPEG format was provided
        if self.extension in DefaultReader.FFMPEG_FORMATS:
            self.imageio_read_mode = "I"
        # Otherwise, have imageio infer the mode
        else:
            self.imageio_read_mode = "?"

    @staticmethod
    def _assert_reader_supports_image(image: types.FSSpecBased) -> bool:
        # Select extension to handle special formats
        extension = image.path.split(".")[-1]

        # Set mode to many-image reading if FFMPEG format was provided
        if extension in DefaultReader.FFMPEG_FORMATS:
            mode = "I"
        # Otherwise, have imageio infer the mode
        else:
            mode = "?"

        # Use imageio to check if they have a reader for this file
        try:
            with image.fs.open(image.path) as open_resource:
                with imageio.get_reader(open_resource, format=extension, mode=mode):
                    return True

        except ValueError:
            return False


    @property
    def scenes(self) -> Set[int]:
        return [0]

    @property
    def current_scene(self) -> int:
        return self.scenes[0]

    def set_scene(self, id: int):
        # We don't need to do any value setting here, we simply need to keep to the
        # Reader spec and enforce that the scene id provided is valid
        if id not in self.scenes:
            raise IndexError(
                f"Scene id: {id} "
                f"is not present in available image scenes: {self.scenes}"
            )

    @staticmethod
    def _get_image_data(
        abstract_file: types.FSSpecBased, mode: str, format: str, index: int
    ) -> np.ndarray:
        with abstract_file.fs.open(abstract_file.path) as open_resource:
            with imageio.get_reader(open_resource, format=format, mode=mode) as reader:
                return np.asarray(reader.get_data(index))

    @staticmethod
    def _get_image_metadata(
        abstract_file: types.FSSpecBased, mode: str, format: str,
    ) -> Dict:
        with abstract_file.fs.open(abstract_file.path) as open_resource:
            with imageio.get_reader(open_resource, format=format, mode=mode) as reader:
                return reader.get_meta_data()


    def _read_delayed(self) -> xr.DataArray:
        # Read image
        with self.abstract_file.fs.open(self.abstract_file.path) as open_resource:
            reader = imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            )

            # Store image length
            try:
                if self.extension in DefaultReader.FFMPEG_FORMATS:
                    image_length = reader.count_frames()
                else:
                    image_length = reader.get_length()
            except ValueError:
                raise exception.IOHandlingError(
                    "This reader cannot read the provided buffer. "
                    "Please download the file locally before continuing your work."
                )

            # Handle single image formats like png, jpeg, etc
            if image_length == 1:
                image_data = da.from_array(self._get_image_data(
                    abstract_file=self.abstract_file,
                    mode=self.imageio_read_mode,
                    format=self.extension,
                    index=0
                ))

            # Handle many image formats like gif, mp4, etc
            elif image_length > 1:
                # Get a sample image
                sample = self._get_image_data(
                    abstract_file=self.abstract_file,
                    mode=self.imageio_read_mode,
                    format=self.extension,
                    index=0
                )

                # Create operating shape for the final dask array by prepending
                # image length to a tuple of ones that is the same length as
                # the sample shape
                operating_shape = (image_length,) + ((1,) * len(sample.shape))
                # Create numpy array of empty arrays for delayed get data
                # functions
                lazy_arrays = np.ndarray(operating_shape, dtype=object)
                for indicies, _ in np.ndenumerate(lazy_arrays):
                    lazy_arrays[indicies] = da.from_delayed(
                        delayed(self._get_image_data)(
                            abstract_file=self.abstract_file,
                            mode=self.imageio_read_mode,
                            format=self.extension,
                            index=indicies[0],
                        ),
                        shape=sample.shape,
                        dtype=sample.dtype,
                    )

                # Block them into a single dask array
                image_data = da.block(lazy_arrays.tolist())

            # Catch all other image types as unsupported
            # https://imageio.readthedocs.io/en/stable/userapi.html#imageio.core.format.Reader.get_length
            else:
                raise exceptions.UnsupportedFileFormatError(self.extension)

            return xr.DataArray(
                image_data,
                dims=[c for c in self.guess_dim_order(image_data.shape)],
                # TODO:
                # Solve this dask bug w/ delayed
                # 
                # attrs=delayed(self._get_image_metadata)(
                #     abstract_file=self.abstract_file,
                #     mode=self.imageio_read_mode,
                #     format=self.extension,
                # ),
            )


    def _read_immediate(self) -> xr.DataArray:
        # Read image
        with self.abstract_file.fs.open(self.abstract_file.path) as open_resource:
            reader = imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            )

            # Store image length
            try:
                if self.extension in DefaultReader.FFMPEG_FORMATS:
                    image_length = reader.count_frames()
                else:
                    image_length = reader.get_length()
            except ValueError:
                raise exceptions.IOHandlingError(
                    "This reader cannot read the provided buffer. "
                    "Please download the file locally before continuing your work."
                )

            # Handle single-image formats like png, jpeg, etc
            if image_length == 1:
                image_data = reader.get_data(0)

            # Handle many image formats like gif, mp4, etc
            elif image_length > 1:
                # Read and stack all frames
                frames = []
                for i, frame in enumerate(reader.iter_data()):
                    frames.append(frame)

                image_data = np.stack(frames)

            return xr.DataArray(
                image_data,
                dims=[c for c in self.guess_dim_order(image_data.shape)],
                attrs=reader.get_meta_data(),
            )

            raise TypeError(path)

    def dims(self):
        pass

    def metadata(self):
        pass

# BIG BUCK BUNNY 15 SECONDS:
# "https://archive.org/embed/archive-video-files/test.mp4"

# AICS TEST RESOURCES
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.png
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.jpg
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.gif
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/s_1_t_1_c_1_z_1.tiff
