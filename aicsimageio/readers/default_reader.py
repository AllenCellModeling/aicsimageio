#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import fsspec
import imageio
import numpy as np
from dask import delayed

from .. import types
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

    def __init__(self, image: types.PathLike):
        # Expand details of provided image
        self.abstract_file = io_utils.pathlike_to_fs(image, enforce_exists=True)


FFMPEG_FORMATS = ["mov", "avi", "mpg", "mpeg", "mp4", "mkv", "wmv", "ogg"]

# BIG BUCK BUNNY 15 SECONDS:
# "https://archive.org/embed/archive-video-files/test.mp4"

# AICS TEST RESOURCES
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.png
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.jpg
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/example.gif
# s3://aics-modeling-packages-test-resources/aicsimageio/test_resources/resources/s_1_t_1_c_1_z_1.tiff

class UnsupportedIOException(Exception):
    pass

def read_image(image: types.PathLike):
    abstract_file = io_utils.pathlike_to_fs(image, enforce_exists=True)

    # Get / unpack file info
    # Select extension to handle special formats
    extension = abstract_file.path.split(".")[-1]

    # Set mode to many-image reading if FFMPEG format was provided
    if extension in FFMPEG_FORMATS:
        mode = "I"
    # Otherwise, have imageio infer the mode
    else:
        mode = "?"

    # Read image
    slices = []
    with abstract_file.fs.open(abstract_file.path) as open_resource:
        reader = imageio.get_reader(open_resource, format=extension, mode=mode)

        # Store image length
        try:
            if extension in FFMPEG_FORMATS:
                image_length = reader.count_frames()
            else:
                image_length = reader.get_length()
        except ValueError:
            raise UnsupportedIOException(
                "This reader cannot read the provided buffer. "
                "Please download the file locally before continuing your work."
            )

        # Handle single-image formats like png, jpeg, etc
        if image_length == 1:
            return reader.get_data(0)

        # Handle many image formats like gif, mp4, etc
        elif image_length > 1:
            frames = []
            for i, frame in enumerate(reader.iter_data()):
                frames.append(frame)

            return np.stack(frames)

        raise TypeError(path)
