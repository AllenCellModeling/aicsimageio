#!/usr/bin/env python
# -*- coding: utf-8 -*-

import imageio
import io
import numpy as np

from .reader import Reader
from .. import constants
from .. import types


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

    def __init__(self, file: types.FileLike):
        self._bytes = self.convert_to_bytes_io(file)

    @property
    def data(self) -> np.ndarray:
        if self._data:
            return self._data

        self._data = imageio.imread(self._bytes)

    @property
    def dims(self) -> str:
        """
        Remove n number of characters from dimension order where n is number of dimensions in image data.

        That said this is pro
        """
        return constants.DEFAULT_DIMENSION_ORDER[len(constants)-len(self.data.shape):]

    @property
    def metadata(self) -> None:
        return None

    @staticmethod
    def _is_this_type(byte_io: io.BytesIO) -> bool:
        return True
