#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from typing import Any

from .. import types
from ..dimensions import DEFAULT_DIMENSION_ORDER

###############################################################################


class Writer(ABC):
    """
    A small class to build standardized image writer functions.
    """

    @staticmethod
    @abstractmethod
    def save(
        data: types.ArrayLike,
        uri: types.PathLike,
        dim_order: str = DEFAULT_DIMENSION_ORDER,
        **kwargs: Any
    ) -> None:
        """
        Write a data array to a file.

        Parameters
        ----------
        data: types.ArrayLike
            The array of data to store.
        uri: types.PathLike
            The URI or local path for where to save the data.
        dim_order: str
            The dimension order of the data.

        Examples
        --------
        >>> image = numpy.ndarray([1, 10, 3, 1024, 2048])
        ... DerivedWriter.save(image, "file.ome.tif", "TCZYX")

        >>> image = dask.array.ones((4, 100, 100))
        ... DerivedWriter.save(image, "file.png", "CYX")
        """
        # There are no requirements for n-dimensions of data.
        # The data provided can be 2D - ND.
        pass
