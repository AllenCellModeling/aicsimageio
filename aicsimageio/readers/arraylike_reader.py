#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import dask.array as da
import numpy as np

from .reader import Reader

###############################################################################


class ArrayLikeReader(Reader):
    """
    A catch all for numpy ndarray and dask array reading.

    Parameters
    ----------
    data: Union[numpy.ndarray, da.core.Array]
        An in memory numpy ndarray or preconfigured dask array.

    Notes
    -----
    Because this is simply a wrapper around numpy ndarray, no metadata is returned. However, dimension order is
    returned with dimensions assumed in order but with extra dimensions removed depending on image shape.
    """

    def __init__(self, data: Union[np.ndarray, da.core.Array], **kwargs):
        # Store data as dask array
        if isinstance(data, np.ndarray):
            self._dask_data = da.from_array(data)
        elif isinstance(data, da.core.Array):
            self._dask_data = data
        else:
            raise TypeError(data)

        # Guess dims
        self._dims = self.guess_dim_order(self.dask_data.shape)

    @property
    def dask_data(self) -> da.core.Array:
        return self._dask_data

    @property
    def dims(self) -> str:
        return self._dims

    @property
    def metadata(self) -> None:
        return None

    @staticmethod
    def _is_this_type(arr: Union[np.ndarray, da.core.Array]) -> bool:
        return isinstance(arr, (np.ndarray, da.core.Array))
