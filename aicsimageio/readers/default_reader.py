#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io

import dask.array as da
import imageio

from .. import exceptions
from .reader import Reader


class DefaultReader(Reader):
    """
    A catch all for image file reading that uses imageio for reading.

    Parameters
    ----------
    data: types.FileLike
        A path or bytes object to read from.
    """

    @property
    def data(self) -> da.core.Array:
        # TODO:
        # try catch mimread then imread from imageio
        # wrap the results in dask array
        if self._data is None:
            self._data = imageio.imread(self._bytes)

        return self._data

    @property
    def dims(self) -> str:
        """
        `imageio.imread` returns the first YX plane of an image, except in the case where the image is RGB / RGBA, in
        which case it returns the first YXC plane of an image.

        This dims property handles those cases by making assumptions about the dimension order based off the shape of
        the data stored in the reader. In the case where more data than an YXC plane is returned, it uses the
        `guess_dim_order` function. However, I can't tell when that should ever get run and is really just used as a
        safety catch all.
        """
        if self._dims is None:
            if len(self.data.shape) == 2:
                self._dims = "YX"
            elif len(self.data.shape) == 3:
                self._dims = "YXC"
            else:
                self._dims = self.guess_dim_order(self.data.shape)

        return self._dims

    @dims.setter
    def dims(self, dims: str):
        # Check amount of provided dims against data shape
        if len(dims) != len(self.data.shape):
            raise exceptions.InvalidDimensionOrderingError(
                f"Provided too many dimensions for the associated file. "
                f"Received {len(dims)} dimensions [dims: {dims}] "
                f"for image with {len(self.data.shape)} dimensions [shape: {self.data.shape}]."
            )

        # Set the dims
        self._dims = dims

    @property
    def metadata(self) -> None:
        return None

    @staticmethod
    def _is_this_type(buffer: io.BytesIO) -> bool:
        # Use imageio to check if they have a reader for this file
        try:
            imageio.get_reader(buffer)
            return True
        except ValueError:
            return False
