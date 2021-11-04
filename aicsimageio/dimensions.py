#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections.abc import Sequence as seq
from typing import ItemsView, Iterable, Sequence, Tuple, Union

###############################################################################


class DimensionNames:
    Time = "T"
    Channel = "C"
    SpatialZ = "Z"
    SpatialY = "Y"
    SpatialX = "X"
    Samples = "S"
    MosaicTile = "M"


DEFAULT_DIMENSION_ORDER_LIST = [
    DimensionNames.Time,
    DimensionNames.Channel,
    DimensionNames.SpatialZ,
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
]
DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES = DEFAULT_DIMENSION_ORDER_LIST + [
    DimensionNames.Samples
]
DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES = [
    DimensionNames.MosaicTile
] + DEFAULT_DIMENSION_ORDER_LIST
DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES = (
    [DimensionNames.MosaicTile]
    + DEFAULT_DIMENSION_ORDER_LIST
    + [DimensionNames.Samples]
)

DEFAULT_DIMENSION_ORDER = "".join(DEFAULT_DIMENSION_ORDER_LIST)
DEFAULT_DIMENSION_ORDER_WITH_SAMPLES = "".join(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_SAMPLES
)
DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES = "".join(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES
)
DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES_AND_SAMPLES = "".join(
    DEFAULT_DIMENSION_ORDER_LIST_WITH_MOSAIC_TILES_AND_SAMPLES
)

DEFAULT_CHUNK_DIMS = [
    DimensionNames.SpatialZ,
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
    DimensionNames.Samples,
]

REQUIRED_CHUNK_DIMS = [
    DimensionNames.SpatialY,
    DimensionNames.SpatialX,
    DimensionNames.Samples,
]

###############################################################################


class Dimensions:
    def __init__(self, dims: Union[str, Iterable], shape: Tuple[int, ...]):
        """
        A general object for managing the pairing of dimension name and dimension size.

        Parameters
        ----------
        dims: Union[str, Iterable]
            An ordered string or iterable of the dimensions to pair with their sizes.
        shape: Tuple[int, ...]
            An ordered tuple of the dimensions sizes to pair with their names.

        Examples
        --------
        >>> dims = Dimensions("TCZYX", (1, 4, 75, 624, 924))
        ... dims.X
        ... dims['T', 'X']
        """
        # Make dims a string
        if not isinstance(dims, str):
            dims = "".join(dims)

        # Store order and shape
        self._order = dims
        self._shape = shape

        # Create attributes
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
    def shape(self) -> Tuple[int, ...]:
        """
        Returns
        -------
        shape: Tuple[int, ...]
            The dimension sizes in their natural order.
        """
        return self._shape

    def items(self) -> ItemsView[str, int]:
        return self._dims_shape.items()

    def __str__(self) -> str:
        dims_string = ", ".join([f"{dim}: {size}" for dim, size in self.items()])
        return f"<Dimensions [{dims_string}]>"

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, key: Union[str, Sequence[str]]) -> Tuple[int, ...]:
        if isinstance(key, str):
            if key not in self._order:
                raise IndexError(f"{key} not in {self._order}")
            return (self._dims_shape[key],)
        elif isinstance(key, seq) and all(isinstance(k, str) for k in key):
            invalid_dims = []
            for k in key:
                if k not in self._order:
                    invalid_dims.append(k)
            if len(invalid_dims) == 0:
                return tuple(self._dims_shape[k] for k in key)
            else:
                raise IndexError(f"{', '.join(invalid_dims)} not in {self._order}")
        else:
            raise TypeError(
                f"Key must be a string or list of strings but got type {type(key)}"
            )
