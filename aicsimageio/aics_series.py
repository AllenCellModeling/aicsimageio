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

    Examples
    --------
    series = AICSSeries(["img1.tiff", "img2.tiff"], "T")
    series[0, :, :, 3, :, :]  # returns a 4D array of TCYX

    series = AICSImage("directory_full_of_tiffs/", "Z")
    series[0, 0, 0, 3, :, :]  # returns a 2D array of YX

    Notes
    -----
    * Because each image is lazy loaded, we validate on every image read. If the image has different dimensions or
    sizes from the prior read image, an InvalidDimensionOrderingError or InconsitentDataShapeException is raised.
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
        # _shape is for ndarray shape consistency
        # _size is for fast access after read
        self._shape = None
        self._size = None

    @property
    def series_dim(self) -> str:
        """
        Returns
        -------
        series_dim: str
            Which dimension is supposed to act like a normal array axis but is actually reading a series of images.
        """
        return self._series_dim

    @property
    def operating_index(self) -> int:
        """
        Returns
        -------
        operating_index: int
            The dimension index of the series dimension.
        """
        return constants.DEFAULT_DIMENSION_ORDER.index(self.series_dim)

    @property
    def images(self) -> List[Path]:
        """
        Returns
        -------
        images: List[Path]:
            The list of filepaths to images to be used for the series.
        """
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
        if self._size is None:
            with AICSImage(self.images[0]) as img:
                # Get size
                shape = img.size()

                # Check data shape
                self._ensure_valid_data_shape(shape, self.operating_index)

                # Store shape for future checks
                self._shape = shape
                log.debug(f"Will hold all images to shape of: {self._shape}")

                # Replace the retrieved size at operating index with the length of the series
                size = []
                for i, val in enumerate(shape):
                    if i == self.operating_index:
                        size.append(len(self.images))
                    else:
                        size.append(val)

                # Set shape state
                self._size = tuple(size)

        return tuple([self._size[constants.DEFAULT_DIMENSION_ORDER.index(c)] for c in dims])

    @property
    def size_x(self) -> int:
        """
        Returns
        -------
        size_x: int
            The size of the x dimension.
        """
        return self.size("X")[0]

    @property
    def size_y(self) -> int:
        """
        Returns
        -------
        size_y: int
            The size of the x dimension.
        """
        return self.size("Y")[0]

    @property
    def size_z(self) -> int:
        """
        Returns
        -------
        size_z: int
            The size of the x dimension.
        """
        return self.size("Z")[0]

    @property
    def size_c(self) -> int:
        """
        Returns
        -------
        size_c: int
            The size of the x dimension.
        """
        return self.size("C")[0]

    @property
    def size_t(self) -> int:
        """
        Returns
        -------
        size_t: int
            The size of the x dimension.
        """
        return self.size("T")[0]

    @property
    def size_s(self) -> int:
        """
        Returns
        -------
        size_s: int
            The size of the x dimension.
        """
        return self.size("S")[0]

    def __getitem__(self, selections: Tuple[Union[slice, int]]) -> np.ndarray:
        """
        Apply slice operations to the image series like a normal numpy.ndarray.
        """
        # Easy check to make sure that length of selections is at most the length of dims (6)
        if len(selections) > len(constants.DEFAULT_DIMENSION_ORDER):
            raise IndexError(f"More operations provided than dimensions available.")

        # To maintain consistent behavior with numpy ndarray slicing behavior, if there are less operations than dims
        # pad the selections with slice(None, None, None)
        # Ex: series[0, 1, ] should pad to series[0, 1, :, :, :, :]
        if len(selections) < len(constants.DEFAULT_DIMENSION_ORDER):
            formatted_selections = [op for op in selections]
            while len(formatted_selections) < len(constants.DEFAULT_DIMENSION_ORDER):
                formatted_selections.append(slice(None, None, None))

            # Cast to tuple and save as selections
            selections = tuple(formatted_selections)

        # Final check that every operation is either an integer or a slice
        for op in selections:
            if not isinstance(op, (int, slice)):
                raise TypeError(
                    f"Operations on __getitem__ must be a single value or a slice to get from the data. Received: {op}."
                )

        # Get the operations required for the operating index
        to_read = self.images[selections[self.operating_index]]

        # Always convert to a list
        if isinstance(to_read, Path):
            to_read = [to_read]

        # Get the other operations required on each image by selection all operations except the one occuring on the
        # operating index, additionally, keep track of what we expect the dims to be out of these operations
        ops = []
        expected_dims = []
        for i, op in enumerate(selections):
            if i != self.operating_index:
                ops.append(op)

            if isinstance(op, slice):
                expected_dims.append(constants.DEFAULT_DIMENSION_ORDER[i])

        # Convert to Tuple
        ops = tuple(ops)

        # Read and apply operations across series
        read_data = []
        for img_to_read in to_read:
            log.debug(f"Reading {img_to_read}...")
            with AICSImage(img_to_read) as img:
                # Set self._shape if not already set
                if self._shape is None:
                    self._shape = img.size()

                # Ensure data consistency
                self._ensure_valid_data_shape(img.size(), self.operating_index, self._shape)

                # Read and append
                read_data.append(
                    # This will get us the other five dimensions
                    img.get_image_data(
                        constants.DEFAULT_DIMENSION_ORDER.replace(self.series_dim, ""),
                        copy=True
                    )[ops]
                )

            # Clean up after reading to save on memory
            # TODO: Fix aicsimageio to do this clean up for us? I am not sure why it is being held onto
            del img

        # Stack data on series dim axis if expected and return
        if self.series_dim in expected_dims:
            data = np.stack(read_data, axis=expected_dims.index(self.series_dim))

        # Otherwise we know it is just a single image because operating axis didn't matter so pull that image data out
        else:
            data = read_data[0]

        # Clean up read data to save on memory
        del read_data

        return data

    def __str__(self) -> str:
        return (
            f"<AICSSeries ["
            f"dimensions: {constants.DEFAULT_DIMENSION_ORDER}, series_dim: {self.series_dim}, size: {self.size()}"
            f"]>"
        )

    def __repr__(self) -> str:
        return str(self)
