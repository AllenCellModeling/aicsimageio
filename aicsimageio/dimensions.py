#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

###############################################################################


class DimensionNames:
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"


DEFAULT_DIMENSION_ORDER = (
    f"{DimensionNames.Time}{DimensionNames.Channel}"
    f"{DimensionNames.SpatialZ}{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
)

###############################################################################


class Dimensions:
    def __init__(self, dims: str, shape: Tuple[int]):
        """
        A general object for managing the pairing of dimension name and dimension size.

        Parameters
        ----------
        dims: str
            An ordered string of the dimensions to pair with their sizes.
        shape: Tuple[int]
            An ordered tuple of the dimensions sizes to pair with their names.

        Examples
        --------
        >>> dims = Dimensions("TCZYX", (1, 4, 75, 624, 924))
        ... dims.X
        """
        self._order = dims
        self._shape = shape
        self._dims_shape = dict(zip(dims, shape))
        for dim, size in self._dims_shape.items():
            setattr(self, dim, size)

    @property
    def order(self) -> str:
        """
        Returns
        -------
        order: str
            The natural order of the dimensions as a single string.
        """
        return self._order

    @property
    def shape(self) -> Tuple[int]:
        """
        Returns
        -------
        shape: Tuple[int]
            The dimension sizes in their natural order.
        """
        return self._shape

    def __str__(self):
        dims_string = ", ".join(
            [f"{dim}: {size}" for dim, size in self._dims_shape.items()]
        )
        return f"<Dimensions [{dims_string}]>"

    def __repr__(self):
        return str(self)
