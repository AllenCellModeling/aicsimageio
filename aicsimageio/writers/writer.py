#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import dask.array as da
import numpy as np

from ..constants import Dimensions

###############################################################################


class Writer(ABC):
    """
    Write a data array to file, with accompanying metadata
    This will overwrite existing files of same name.

    Parameters
    ----------
    filepath: types.PathLike
        Path to image output location

    Examples
    --------
    Construct and use as object
    >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
    ... writer = DerivedWriter("file.ome.tif")
    ... writer.save(image)
    ... writer.close()

    Construct with a context manager
    >>> image2 = numpy.ndarray([5, 486, 210])
    ... with DerivedWriter("file2.ome.tif") as writer2:
    ...     writer2.save(image2, metadata=some_metadata_object)
    """
    filepath = None
    _buffer = None

    @abstractmethod
    def open(self):
        self._buffer = open(self.filepath, "w")

    def __enter__(self):
        # Open the file
        self.open()
        return self

    @abstractmethod
    def close(self):
        self._buffer.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @abstractmethod
    def save(
        self,
        data: Union[da.Array, np.ndarray],
        dim_order: str = Dimensions.DefaultOrder,
        metadata: Optional[Any] = None,
        **kwargs
    ):
        """
        Write a data array to an open file.

        Parameters
        ----------
        data: Union[da.Array, np.ndarray]
            The array of data to store.
        dim_order: str
            The dimension order of the data.
        metadata: Optional[Any]
            An optional metadata object to store with the data. Metadata object type is dependent on writer.
        """
        # There are no requirements for n-dimensions of data. The data provided can be 2D - ND.
        pass
