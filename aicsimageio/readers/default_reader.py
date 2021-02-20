#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple, Union

import dask.array as da
import imageio
import numpy as np
import xarray as xr
from dask import delayed
from fsspec.spec import AbstractFileSystem

from .. import constants, exceptions, types
from ..dimensions import DimensionNames
from ..metadata import utils as metadata_utils
from ..utils import io_utils
from .reader import Reader

###############################################################################

REMOTE_READ_FAIL_MESSAGE = (
    "Cannot read the provided file ({path}) remotely. "
    "Please download the file locally before continuing your work."
)

###############################################################################


class DefaultReader(Reader):

    FFMPEG_FORMATS = ["mov", "avi", "mpg", "mpeg", "mp4", "mkv", "wmv", "ogg"]

    @staticmethod
    def _get_extension_and_mode(path: str) -> Tuple[str, str]:
        """
        Provided a path to a file, provided back the extension (format) of the file
        and the imageio read mode.

        Parameters
        ----------
        path: str
            The file to provide extension and mode info for.

        Returns
        -------
        extension: str
            The extension (a naive guess at the format) of the file.
        mode: str
            The imageio read mode to use for image reading.
        """
        # Select extension to handle special formats
        extension = path.split(".")[-1]

        # Set mode to many-image reading if FFMPEG format was provided
        # https://imageio.readthedocs.io/en/stable/userapi.html#imageio.get_reader
        if extension in DefaultReader.FFMPEG_FORMATS:
            mode = "I"
        # Otherwise, have imageio infer the mode
        else:
            mode = "?"

        return extension, mode

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        # Get extension and mode for reading the file
        extension, mode = DefaultReader._get_extension_and_mode(path)

        # Use imageio to check if they have a reader for this file
        try:
            with fs.open(path) as open_resource:
                with imageio.get_reader(open_resource, format=extension, mode=mode):
                    return True

        # Exceptions that are raised by imageio for unsupported file types
        except (ValueError, IndexError):
            return False

        # Some FFMPEG formats and reading just suck
        # If they can't get metadata remotely they throw an OSError because ffmpeg is
        # ran through subprocess (I believe)
        # If we let the stack trace go, user would receive:
        #
        # OSError: Could not load meta information
        # === stderr ===
        #
        # ffmpeg version 4.2.2-static https://johnvansickle.com/ffmpeg/
        # Copyright (c) # 2000-2019 the FFmpeg developers
        # ...
        # /tmp/imageio_cbof2u37: Invalid data found when processing input
        except OSError:
            raise IOError(REMOTE_READ_FAIL_MESSAGE.format(path=path))

    def __init__(self, image: types.PathLike):
        """
        A catch all for image file reading that defaults to using imageio
        implementations.

        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for.
        """
        # Expand details of provided image
        self.fs, self.path = io_utils.pathlike_to_fs(image, enforce_exists=True)
        self.extension, self.imageio_read_mode = self._get_extension_and_mode(self.path)

        # Enforce valid image
        if not self._is_supported_image(self.fs, self.path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self.path
            )

    @staticmethod
    def _guess_dim_order(shape: Tuple[int, ...]) -> str:
        if len(shape) == 2:
            return f"{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
        elif len(shape) == 3:
            # Handle greyscale timeseries
            if shape[-1] > 4:
                return (
                    f"{DimensionNames.Time}"
                    f"{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
                )

            # Else, return normal RGB / RGBA dims
            return (
                f"{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
                f"{DimensionNames.Samples}"
            )
        elif len(shape) == 4:
            return (
                f"{DimensionNames.Time}{DimensionNames.SpatialY}"
                f"{DimensionNames.SpatialX}{DimensionNames.Samples}"
            )

        return Reader._guess_dim_order(shape)

    @property
    def scenes(self) -> Tuple[str]:
        # There is currently an assumption that DefaultReader will not encounter
        # files with multiple scenes. But, if we do encounter a file that DefaultReader
        # hits and a user wants scene management from that file type, we can update
        # this property then.
        return (metadata_utils.generate_ome_image_id(0),)

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem, path: str, format: str, mode: str, index: int
    ) -> np.ndarray:
        """
        Open a file for reading, seek to plane index and read as numpy.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
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
        with fs.open(path) as open_resource:
            with imageio.get_reader(open_resource, format=format, mode=mode) as reader:
                return np.asarray(reader.get_data(index))

    @staticmethod
    def _get_image_length(
        fs: AbstractFileSystem,
        path: str,
        format: str,
        mode: str,
    ) -> int:
        """
        Open a file for reading, using the format, determine the image length
        (the number of planes).

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to file to read.
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
            The length of the image (number of YX planes).

        Notes
        -----
        In the case this file is an FFMPEG format, this function will attempt to seek
        and retrieve the last frame of data as reported by traditional imageio methods
        to verify that the frame count is correct.

        This is to check for FFMPEG off-by-one errors in frame indexing.
        See here for more details: https://github.com/imageio/imageio/issues/168
        """
        with fs.open(path) as open_resource:
            with imageio.get_reader(open_resource, format=format, mode=mode) as reader:
                # Handle FFMPEG formats
                if format in DefaultReader.FFMPEG_FORMATS:
                    # A reminder, this is the _total_ frame count, not the last index
                    reported_frames = reader.count_frames()

                    # As a safety measure against FFMPEG off-by-one
                    # try and get the last frame (by index)
                    try:
                        reader.get_data(reported_frames - 1)
                        return reported_frames

                    # Couldn't get the last frame by index, FFMPEG must have off-by-oned
                    # So return the _total_ frame count minus the one frame correction
                    except IndexError:
                        return reported_frames - 1

                # Default get_length call for all others
                return reader.get_length()

    @staticmethod
    def _unpack_dims_and_coords(
        image_data: types.ArrayLike, metadata: Dict
    ) -> Tuple[List[str], Dict[str, Union[List[str], types.ArrayLike]]]:
        """
        Unpack image data into assumed dims and coords.

        Parameters
        ----------
        image_data: types.ArrayLike
            The image data to unpack dims and coords for.
        metadata: Dict
            The EXIF, XMP, etc metadata dictionary.

        Returns
        -------
        dims: List[str]
            The dimension names for each dimension in the image data.
        coords: Dict[str, Union[List[str], types.ArrayLike]]
            If possible, the coordinates for dimensions in the image data.
        """
        # Guess dims
        dims = [c for c in DefaultReader._guess_dim_order(image_data.shape)]

        # Use dims for coord determination
        coords: Dict[str, Union[List[str], np.ndarray]] = {}

        # Handle typical RGB and RGBA from Samples
        if DimensionNames.Samples in dims:
            if image_data.shape[dims.index(DimensionNames.Samples)] == 3:
                coords[DimensionNames.Samples] = ["R", "G", "B"]
            elif image_data.shape[dims.index(DimensionNames.Samples)] == 4:
                coords[DimensionNames.Samples] = ["R", "G", "B", "A"]

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
            The fully constructed and fully delayed image as a DataArray  object.
            Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError: The file could not be read or is not
            supported.
        """
        with self.fs.open(self.path) as open_resource:
            with imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            ) as reader:
                # Store image length
                image_length = self._get_image_length(
                    fs=self.fs,
                    path=self.path,
                    format=self.extension,
                    mode=self.imageio_read_mode,
                )

                # Handle single image formats like png, jpeg, etc
                if image_length == 1:
                    image_data = da.from_array(
                        self._get_image_data(
                            fs=self.fs,
                            path=self.path,
                            format=self.extension,
                            mode=self.imageio_read_mode,
                            index=0,
                        )
                    )

                # Handle many image formats like gif, mp4, etc
                elif image_length > 1:
                    # Get a sample image
                    sample = self._get_image_data(
                        fs=self.fs,
                        path=self.path,
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
                    for indices, _ in np.ndenumerate(lazy_arrays):
                        lazy_arrays[indices] = da.from_delayed(
                            delayed(self._get_image_data)(
                                fs=self.fs,
                                path=self.path,
                                format=self.extension,
                                mode=self.imageio_read_mode,
                                index=indices[0],
                            ),
                            shape=sample.shape,
                            dtype=sample.dtype,
                        )

                    # Block them into a single dask array
                    image_data = da.block(lazy_arrays.tolist())

                # Catch all other image types as unsupported
                # https://imageio.readthedocs.io/en/stable/userapi.html#imageio.core.format.Reader.get_length
                else:
                    raise exceptions.UnsupportedFileFormatError(
                        self.__class__.__name__, self.extension
                    )

                # Get basic metadata
                metadata = reader.get_meta_data()

                # Create extra metadata from assumptions based off image data
                dims, coords = self._unpack_dims_and_coords(image_data, metadata)

                return xr.DataArray(
                    image_data,
                    dims=dims,
                    coords=coords,  # type: ignore
                    attrs={constants.METADATA_UNPROCESSED: metadata},
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
        # Read image
        with self.fs.open(self.path) as open_resource:
            reader = imageio.get_reader(
                open_resource, format=self.extension, mode=self.imageio_read_mode
            )

            # Store image length
            image_length = self._get_image_length(
                fs=self.fs,
                path=self.path,
                format=self.extension,
                mode=self.imageio_read_mode,
            )

            # Handle single-image formats like png, jpeg, etc
            if image_length == 1:
                image_data = reader.get_data(0)

            # Handle many image formats like gif, mp4, etc
            elif image_length > 1:
                # Read and stack all frames
                frames = []
                for frame in reader:
                    frames.append(frame)

                image_data = np.stack(frames)

            # Get basic metadata
            metadata = reader.get_meta_data()

            # Create extra metadata from assumptions based off image data
            dims, coords = self._unpack_dims_and_coords(image_data, metadata)

            return xr.DataArray(
                image_data,
                dims=dims,
                coords=coords,  # type: ignore
                attrs={constants.METADATA_UNPROCESSED: metadata},
            )
