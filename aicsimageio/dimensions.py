#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple

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
        # THIS WORKS FOR SIMPLE CASES BUT NEEDS BUG TESTING
        self._order = dims
        self._sizes = shape
        self._dims_shape = dict(zip(dims, shape))
        for dim, size in self._dims_shape.items():
            setattr(self, dim, size)

    def order(self) -> str:
        """
        Return the natural order of the dimensions as a single string.
        """
        return self._order

    def sizes(self) -> Tuple[int]:
        """
        Return the dimension sizes in their natural order.
        """
        return self._sizes

    def __str__(self):
        dims_string = ", ".join([
            f"{dim}: {size}" for dim, size in self._dims_shape.items()
        ])
        return f"<Dimensions [{dims_string}]>"

    def __repr__(self):
        return str(self)
