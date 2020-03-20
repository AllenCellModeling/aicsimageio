#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import dask.array as da
import imageio
import numpy as np
from dask import delayed

from .. import exceptions
from ..constants import Dimensions
from .reader import Reader

###############################################################################


class DefaultReader(Reader):
    """
    A catch all for image file reading that uses imageio for reading.

    Parameters
    ----------
    file: types.ImageLike
        String with path to file.
    """

    @staticmethod
    def _get_data(file: Path, index: int) -> np.ndarray:
        with imageio.get_reader(file) as reader:
            return np.asarray(reader.get_data(index))

    @property
    def dask_data(self) -> da.core.Array:
        # Construct delayed many image reads
        if self._dask_data is None:
            try:
                with imageio.get_reader(self._file) as reader:
                    # Store length as it is used a bunch
                    image_length = reader.get_length()

                    # Handle single image formats like png, jpeg, etc
                    if image_length == 1:
                        self._dask_data = da.from_array(self._get_data(self._file, 0))

                    # Handle many image formats like gif, mp4, etc
                    elif image_length > 1:
                        # Get a sample image
                        sample = self._get_data(self._file, 0)

                        # Create operating shape for the final dask array by prepending image length to a tuple of
                        # ones that is the same length as the sample shape
                        operating_shape = (image_length, ) + ((1, ) * len(sample.shape))
                        # Create numpy array of empty arrays for delayed get data functions
                        lazy_arrays = np.ndarray(operating_shape, dtype=object)
                        for indicies, _ in np.ndenumerate(lazy_arrays):
                            lazy_arrays[indicies] = da.from_delayed(
                                delayed(self._get_data)(self._file, indicies[0]),
                                shape=sample.shape,
                                dtype=sample.dtype
                            )

                        # Block them into a single dask array
                        self._dask_data = da.block(lazy_arrays.tolist())

                    # Catch all other image types as unsupported
                    # https://imageio.readthedocs.io/en/stable/userapi.html#imageio.core.format.Reader.get_length
                    else:
                        exceptions.UnsupportedFileFormatError(self._file)

            # Reraise unsupported file format
            except exceptions.UnsupportedFileFormatError:
                raise exceptions.UnsupportedFileFormatError(self._file)

        return self._dask_data

    @property
    def dims(self) -> str:
        # Set dims if not set
        if self._dims is None:
            if len(self.dask_data.shape) == 2:
                self._dims = "YX"
            elif len(self.dask_data.shape) == 3:
                self._dims = "YXC"
            elif len(self.dask_data.shape) == 4:
                self._dims = "TYXC"
            else:
                self._dims = self.guess_dim_order(self.dask_data.shape)

        return self._dims

    @dims.setter
    def dims(self, dims: str):
        # Check amount of provided dims against data shape
        if len(dims) != len(self.dask_data.shape):
            raise exceptions.InvalidDimensionOrderingError(
                f"Provided too many dimensions for the associated file. "
                f"Received {len(dims)} dimensions [dims: {dims}] "
                f"for image with {len(self.data.shape)} dimensions [shape: {self.data.shape}]."
            )

        # Set the dims
        self._dims = dims

    @property
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            with imageio.get_reader(self._file) as reader:
                self._metadata = reader.get_meta_data()

        return self._metadata

    def get_channel_names(self, scene: int = 0) -> Optional[List[str]]:
        # Check for channel in dims
        if Dimensions.Channel in self.dims:
            channel_index = self.dims.index(Dimensions.Channel)
            channel_dim_size = self.dask_data.shape[channel_index]

            # RGB vs RGBA vs other
            if channel_dim_size == 3:
                return ["Red", "Green", "Blue"]
            elif channel_dim_size == 4:
                return ["Red", "Green", "Blue", "Alpha"]
            else:
                return [str(i) for i in range(channel_dim_size)]

        return None

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        # Use imageio to check if they have a reader for this file
        try:
            with imageio.get_reader(buffer):
                return True
        except ValueError:
            return False
