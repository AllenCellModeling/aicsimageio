#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io

import imageio
import numpy as np

from .. import exceptions
from .reader import Reader


class DefaultReader(Reader):
    """
    DefaultReader('file.jpg')

    A catch all for image file reading that uses imageio as its back end.

    Parameters
    ----------
    file: types.FileLike
        A path or bytes object to read from.

    Notes
    -----
    Because this is simply a wrapper around imageio, no metadata is returned. However, dimension order is returned with
    dimensions assumed in order but with extra dimensions removed depending on image shape.
    """

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = imageio.imread(self._bytes)

        return self._data

    @property
    def dims(self) -> str:
        """
        Remove n number of characters from dimension order where n is number of dimensions in image data.

        That said these dimensions are probably incorrect because jpg for example is actually 'XYC' dimension order,
        but it's our assumption to make. You can override the dims by using the property setter.
        """
        if self._dims is None:
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
    def _is_this_type(buffer: io.BufferedIOBase) -> bool:
        return True
