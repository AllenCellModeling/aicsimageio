#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Dict, List, Set, Tuple

import dask.array as da
import imageio
import numpy as np
import xarray as xr
from dask import delayed

from .. import exceptions, types
from ..dimensions import DimensionNames
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

    @staticmethod
    def _get_extension_and_mode(abstract_file: types.FSSpecBased) -> Tuple[str, str]:
        """
        Provided an FSSpecBased file, proved back the extension (format) of the file
        and the imageio read mode to use for reading.

        Parameters
        ----------
        abstract_file: types.FSSpecBased
            The file to provide extension and mode info for.

        Returns
        -------
        extension: str
            The extension (a naive guess at the format) of the file.
        mode: str
            The imageio read mode to use for image reading.
        """
        # Select extension to handle special formats
        extension = abstract_file.path.split(".")[-1]

        # Set mode to many-image reading if FFMPEG format was provided
        # https://imageio.readthedocs.io/en/stable/userapi.html#imageio.get_reader
        if extension in DefaultReader.FFMPEG_FORMATS:
            mode = "I"
        # Otherwise, have imageio infer the mode
        else:
            mode = "?"

        return extension, mode

    def __init__(self, image: types.PathLike):
        # Expand details of provided image
        self.abstract_file = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension, self.imageio_read_mode = self._get_extension_and_mode(
            self.abstract_file
        )

    @staticmethod
    def guess_dim_order(shape: Tuple[int]) -> str:
        if len(shape) == 2:
            return f"{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
        elif len(shape) == 3:
            return (
                f"{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
                f"{DimensionNames.Channel}"
            )
        elif len(shape) == 4:
            return (
                f"{DimensionNames.Time}{DimensionNames.SpatialY}"
                f"{DimensionNames.SpatialX}{DimensionNames.Channel}"
            )

        return Reader.guess_dim_order(shape)

    @staticmethod
    def _assert_reader_supports_image(image: types.FSSpecBased) -> bool:
        # Get extension and mode for reading the file
        extension, mode = DefaultReader._get_extension_and_mode(image)

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
        abstract_file: types.FSSpecBased, format: str, mode: str, index: int
    ) -> np.ndarray:
        """
        Open a file for reading, seek to plane index and read as numpy.

        Parameters
        ----------
        abstract_file: types.FSSpecBased
            The file to open and read.
        format: str
            The format to use to read the file.
            For our use case this is primarily the file extension.
        mode: str
            The read mode to use for opening and reading.
            See mode parameter on imageio.get_reader
            https://imageio.readthedocs.io/en/stable/userapi.html#imageio.get_reader
        index: int
            The image plane index to seek to and read from the file.

        Returns
        -------
        plane: np.ndarray
            The image plane as a numpy array.
        """
        with abstract_file.fs.open(abstract_file.path) as open_resource:
            with imageio.get_reader(open_resource, format=format, mode=mode) as reader:
                return np.asarray(reader.get_data(index))

    @staticmethod
    def _get_image_length(
        abstract_file: types.FSSpecBased,
        format: str,
        mode: str,
    ) -> int:
        """
        Parameters
        ----------
        abstract_file: types.FSSpecBased
            The file to open and read.
        format: str
            The format to use to read the file.
            For our use case this is primarily the file extension.
        mode: str
            The read mode to use for opening and reading.
            See mode parameter on imageio.get_reader
            https://imageio.readthedocs.io/en/stable/userapi.html#imageio.get_reader

        Returns
        -------
        length: int
            The length of the image (number of planes).

        Raises
        ------
        exceptions.IOHandlingError: The provided file cannot be read from a remote
            source.
        """
        try:
            with abstract_file.fs.open(abstract_file.path) as open_resource:
                with imageio.get_reader(
                    open_resource, format=format, mode=mode
                ) as reader:
                    # Handle FFMPEG formats
                    if format in DefaultReader.FFMPEG_FORMATS:
                        return reader.count_frames()

                    # Default get_length call for all others
                    return reader.get_length()

        except ValueError:
            raise exceptions.IOHandlingError(
                f"Cannot read the provided file ({abstract_file.path}) remotely. "
                f"Please download the file locally before continuing your work."
            )

    @staticmethod
    def _unpack_dims_and_coords(
        image_data: types.ImageLike, metadata: Dict
    ) -> Tuple[List[str], Dict]:
        """
        Unpack image data into assumed dims and coords.

        Parameters
        ----------
        image_data: types.ImageLike
            The image data to unpack dims and coords for.
        metadata: Dict
            The EXIF, XMP, etc metadata dictionary.

        Returns
        -------
        dims: List[str]
            The dimension names for each dimension in the image data.
        coords: Dict
            If possible, the coordinates for dimensions in the image data.
        """
        # Guess dims
        dims = [c for c in DefaultReader.guess_dim_order(image_data.shape)]

        # Use dims for coord determination
        coords = {}

        # Handle typical RGB and RGBA from channels
        if DimensionNames.Channel in dims:
            if image_data.shape[dims.index(DimensionNames.Channel)] == 3:
                coords[DimensionNames.Channel] = ["R", "G", "B"]
            elif image_data.shape[dims.index(DimensionNames.Channel)] == 4:
                coords[DimensionNames.Channel] = ["R", "G", "B", "A"]

        # Handle time when duration is present in metadata
        if DimensionNames.Time in dims:
            if "duration" in metadata:
                coords[DimensionNames.Time] = np.linspace(
                    0,
                    metadata["duration"],
                    image_data.shape[dims.index(DimensionNames.Time)],
                )

        return dims, coords

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is both attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.abstract_file.fs.open(self.abstract_file.path) as open_resource:
            with imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            ) as reader:
                # Store image length
                image_length = self._get_image_length(
                    self.abstract_file,
                    format=self.extension,
                    mode=self.imageio_read_mode,
                )

                # Handle single image formats like png, jpeg, etc
                if image_length == 1:
                    image_data = da.from_array(
                        self._get_image_data(
                            abstract_file=self.abstract_file,
                            format=self.extension,
                            mode=self.imageio_read_mode,
                            index=0,
                        )
                    )

                # Handle many image formats like gif, mp4, etc
                elif image_length > 1:
                    # Get a sample image
                    sample = self._get_image_data(
                        abstract_file=self.abstract_file,
                        format=self.extension,
                        mode=self.imageio_read_mode,
                        index=0,
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
                                format=self.extension,
                                mode=self.imageio_read_mode,
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

                # Get basic metadata
                metadata = reader.get_meta_data()

                # Create extra metadata from assumptions based off image data
                dims, coords = self._unpack_dims_and_coords(image_data, metadata)

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
            object. Metadata is both attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        # Read image
        with self.abstract_file.fs.open(self.abstract_file.path) as open_resource:
            reader = imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            )

            # Store image length
            image_length = self._get_image_length(
                self.abstract_file, format=self.extension, mode=self.imageio_read_mode
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

            # Get basic metadata
            metadata = reader.get_meta_data()

            # Create extra metadata from assumptions based off image data
            dims, coords = self._unpack_dims_and_coords(image_data, metadata)

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,
                attrs=metadata,
            )

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
