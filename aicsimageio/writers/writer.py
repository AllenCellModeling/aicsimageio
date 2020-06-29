#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import dask.array as da
import numpy as np

from ..constants import Dimensions

###############################################################################


class Writer(ABC):
    """
    A small class to build standardized image writer functions.
    """

    @staticmethod
    @abstractmethod
    def save(
        data: Union[da.Array, np.ndarray],
        filepath: Union[str, Path],
        dim_order: str = Dimensions.DefaultOrder,
        **kwargs
    ):
        """
        Write a data array to a file.

        Parameters
        ----------
        data: Union[da.Array, np.ndarray]
            The array of data to store.
        filepath: Union[str, Path]
            The path to save the data at.
        dim_order: str
            The dimension order of the data.

        Examples
        --------
        >>> image = numpy.ndarray([1, 1, 10, 3, 1024, 2048])
        ... DerivedWriter.save(image, "file.ome.tif", "STCZYX")

        >>> image = dask.array.ones((4, 100, 100))
        ... DerivedWriter.save(image, "file.png", "CYX")
        """
        # There are no requirements for n-dimensions of data.
        # The data provided can be 2D - ND.
        pass
