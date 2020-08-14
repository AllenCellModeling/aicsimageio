#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

###############################################################################

class DimensionNames:
    Scene = "S"
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"

DEFAULT_DIMENSION_ORDER = (
    f"{DimensionNames.Scene}{DimensionNames.Time}{DimensionNames.Channel}"
    f"{DimensionNames.SpatialZ}{DimensionNames.SpatialY}{DimensionNames.SpatialX}"
)

###############################################################################

# We don't currently have this problem but I don't really like that I have a
# DimensionNames and a Dimensions object
#
# We could potentially use a "Dimension" NamedTuple to track "name" and "size"
# This would alleviate the problem of two lookup tables
# Dimensions.Scene vs Dimensions.S
# Dimensions.Scene.short == "S"
# Dimensions.Scene.size == int
# vs
# Dimensions.S.name == "Scene"
# Dimensions.S.size == int
#
# This is more verbose than current Dimensions object
# Dimensions.S == int
# but may be valuable
#
# Just thinking out loud here

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
        >>> dims = Dimensions("STCZYX", (1, 1, 4, 75, 624, 924))
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
        Return the natural order of the dimensions as a single string.
        """
        return self._order

    @property
    def shape(self) -> Tuple[int]:
        """
        Return the dimension sizes in their natural order.
        """
        return self._shape

    def __str__(self):
        dims_string = ", ".join([
            f"{dim}: {size}" for dim, size in self._dims_shape.items()
        ])
        return f"<Dimensions [{dims_string}]>"

    def __repr__(self):
        return str(self)
