#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from . import AICSImage, constants, exceptions, types

log = logging.getLogger(__name__)

###############################################################################


class AICSSeries:
    """
    AICSSeries takes an ordered iterable of image data types (files / bytestreams) of consistent dimensions and forces
    them to act like they are a single image. The data for each image in the series is lazy loaded as are the images
    themselves. This is beneficial when the images stacked together may be too large to fit in memory.

    Simple Example
    --------------
    # Force multiple OME-Tiff files to act as if they are all the same image but each image is a different timepoint
    series = AICSSeries(
        [
            "timepoint_0.ome.tiff",
            "timepoint_1.ome.tiff",
            "timepoint_2.ome.tiff",
        ],
        dim="T"
    )

    # Get the ZYX data for timepoint 1
    zstack_1 = series.get_image_data("ZYX", S=0, C=0, T=1)

    Directory Example
    -----------------
    # Force multiple OME-Tiff files to act as if they are all the same image but each image is a different timepoint
    # All images are in the same directory with nothing else in the directory, sorted by filename
    series = AICSSeries("timeseries_tiffs/", dim="T")

    # Get the ZYX data for timepoint 1
    zstack_1 = series.get_image_data("ZYX", S=0, C=0, T=1)

    Notes
    -----
    * Because each image is lazy loaded, we validate on every image read. If the image has different dimensions and
    sizes from the prior read image, an InvalidDimensionOrderingError is raised.
    * Each image in the iterable must meet either of two conditions, have less than six dimensions, or, be a single
    plane in the dimension you are trying to override.
    * This class may be a temporary stop gap until chunked image reading is added to the core `AICSImage` class and if
    and when that occurs, this class will be depreciated.
    * No metadata is checked for ensuring that the data in each image is consistent across them all. Example being,
    channels in different indicies in different files across the series.
    """

    def __init__(
        self,
        images: Union[types.PathLike, Iterable[types.PathLike]],
        series_dim: str
    ):
        """
        Constructor for AICSSeries class intended for providing an interface for image reading of a list of file like
        references.

        Parameters
        ----------
        images: Union[types.PathLike, Iterable[types.PathLike]]
            Either a pathlike value to a directory with images or the ordered iterable set of pathlike objects to read
            from for the series.
        series_dim: str
            Which dimension to override with the iterable images.
        """
        # Check if provided a directory, if so, get the directory contents
        if isinstance(images, (str, Path)):
            # Expand and handle path
            images = Path(images).expanduser().resolve(strict=True)

            # Ensure that it is actually a directory and not a single file
            if not images.is_dir():
                raise NotADirectoryError(
                    f"Provided a single file: {images}. Did you mean to use `AICSImage`?"
                )

            # Set images to the contents of the directory
            images = sorted(images.iterdir())

            # Drop any files that start with `.`
            # TODO: Is this actually a valid operation to do for the user or way to specific?
            # TODO: Different option, find the most common suffix and use it as the base?
            predrop_count = len(images)
            images = [img for img in images if img.name[0] != "."]
            log.debug(f"Dropped {predrop_count - len(images)} files that began with a `.`.")

        # At this point, images should have either been converted to a list of files if provided a directory
        # Or was given an iterable of file like
        if not isinstance(images, Iterable):
            raise TypeError(
                f"AICSSeries requires either a path to a directory or an iterable of files."
            )

        # Check that the iterable is more than one image
        if len(images) <= 1:
            raise ValueError(f"Provided a single file: {images}. Did you mean to use `AICSImage`?")

        # Now check that everything in the iterable is file like
        # This will also raise an error if any of the file like paths was not found
        images = [Path(img).expanduser().resolve(strict=True) for img in images]

        # Ensure that the series dim provided is part of the normal set and that only one dim is provided
        if (
            not isinstance(series_dim, str)
            or series_dim not in constants.DEFAULT_DIMENSION_ORDER
            or len(series_dim) != 1
        ):
            raise ValueError(
                f"The series dimension must be a single character from the "
                f"standard {constants.DEFAULT_DIMENSION_ORDER}. Received: '{series_dim}'."
            )

        # All easy checks are now complete, store the image list and the series dim
        self._images = images
        self._series_dim = series_dim.upper()

        # Set the initial state of the series for future validation
        self._shape = None

    # def get_image_data(self, out_orientation: str = None, **kwargs) -> np.ndarray:
    #     """
    #     Get a specific slice of the data out of the series
    #     """

    def __getitem__(self, selections: Tuple[Union[slice, int]]) -> np.ndarray:
        if len(selections) != len(constants.DEFAULT_DIMENSION_ORDER):
            raise ValueError(f"Not enough dimensions provided to slice properly.")

        # Find get series dimension operating index
        operating_index = constants.DEFAULT_DIMENSION_ORDER.index(self.series_dim)

        # Handle single operating index slice get
        if isinstance(selections[operating_index], int):
            pass

    @property
    def series_dim(self) -> str:
        return self._series_dim

    @property
    def operating_index(self) -> int:
        return constants.DEFAULT_DIMENSION_ORDER.index(self.series_dim)

    @property
    def images(self) -> List[Path]:
        return self._images

    @staticmethod
    def _ensure_valid_data_shape(
        data_shape: Tuple[int],
        operating_index: int,
        prior_shape: Optional[Tuple[int]] = None
    ):
        """
        Used to ensure that the data shape is `1` in whichever index is the operating index for the series.
        Additionally, if provided a prior shape, will check for shape consistenting between the pair.

        Raises an InvalidDimensionOrderingError or InconsitentDataShapeException.

        Parameters
        ----------
        data_shape: Tuple[int]
            A tuple of six dimension sizes produced by `AICSImage.size()` for an image in the series.
        operating_index: int
            The operating index for the series. This is constants.DEFAULT_DIMENSION_ORDER.index(self.series_dim).
        prior_shape: Optional[Tuple[int]] = None
            An optional prior shape to ensure that the provided data shape is consistent with the prior.
        """
        if data_shape[operating_index] != 1:
            raise exceptions.InvalidDimensionOrderingError(
                f"The read data shape is invalid for the current operating series dimension. Read shape: {data_shape}."
            )

        if prior_shape:
            if data_shape != prior_shape:
                raise exceptions.InconsitentDataShapeException(
                    f"The read data shape is inconsitent with the prior data shape. "
                    f"Read shape: {data_shape}, prior shape: {prior_shape}"
                )

    def size(self, dims: str = constants.DEFAULT_DIMENSION_ORDER):
        """
        Parameters
        ----------
        dims: A string containing a list of dimensions being requested.  The default is to return the 6 standard dims

        Returns
        -------
        Returns a tuple with the requested dimensions filled in
        """
        # Only run if shape has never been retrieved before
        if self._shape is None:
            with AICSImage(self.images[0]) as img:
                # Get size
                shape = img.size()

                # Check data shape
                self._ensure_valid_data_shape(shape, self.operating_index)

                # Replace the retrieved size at operating index with the length of the series
                formatted_shape = []
                for i, val in enumerate(shape):
                    if i == self.operating_index:
                        formatted_shape.append(len(self.images))
                    else:
                        formatted_shape.append(val)

                # Set shape state
                self._shape = tuple(formatted_shape)

        return tuple([self._shape[constants.DEFAULT_DIMENSION_ORDER.index(c)] for c in dims])
