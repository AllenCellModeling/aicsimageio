#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from collections import Counter

import dask.array as da
import numpy as np

from . import types
from .constants import Dimensions
from .exceptions import ConflictingArgumentsError

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def reshape_data(
    data: types.ArrayLike, given_dims: str, return_dims: str, **kwargs
) -> types.ArrayLike:
    """
    Reshape the data into return_dims, pad missing dimensions, and prune extra
    dimensions. Warns the user to use the base reader if the depth of the Dimension
    being removed is not 1.

    Parameters
    ----------
    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions
        specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data
    kwargs:
        C=1 => desired specific channel, if C in the input data has depth 3 then C=1
        returns the 2nd slice (0 indexed)
        Z=10 => desired specific channel, if Z in the input data has depth 20 then Z=10
        returns the 11th slice

    Returns
    -------
    data: types.ArrayLike
        An array in return_dims order, if return_dims=DEFAULT_DIMS then the return
        would have order "STCZYX"

    """
    # Check for parameter conflicts
    for dim in Dimensions.DefaultOrderList:
        # return_dims='TCZYX' and fixed dimensions 'C=1'
        # Dimension is in kwargs
        # Dimension is an integer
        # Dimension is in return dimensions
        if isinstance(kwargs.get(dim), int) and dim in return_dims:
            raise ConflictingArgumentsError(
                f"Argument return_dims={return_dims} and "
                f"argument {dim}={kwargs.get(dim)} conflict. "
                f"Check usage."
            )

        # return_dims='CZYX' and iterable dimensions 'T=range(10)'
        # Dimension is in kwargs
        # Dimension is an iterable
        # Dimension is not in return dimensions
        if (
            isinstance(kwargs.get(dim), (list, tuple, range, slice))
            and dim not in return_dims
        ):
            raise ConflictingArgumentsError(
                f"Argument return_dims={return_dims} and "
                f"argument {dim}={kwargs.get(dim)} conflict. "
                f"Check usage."
            )

    # Process each dimension available
    new_dims = given_dims
    getitem_ops = []
    for dim in given_dims:
        # Store index of the dim as it is in given data
        dim_index = given_dims.index(dim)

        # Handle dim in return dims which means that it is
        # an iterable or None selection
        if dim in return_dims:
            # Specific iterable requested
            if dim in kwargs:
                # Actual dim operator
                dim_operator = kwargs.get(dim)

                # Convert operator to standard list or slice
                # dask.Array and numpy.ndarray both natively support
                # List[int] and slices being passed to getitem so no need to cast them
                # to anything different
                if isinstance(dim_operator, (tuple, range)):
                    dim_operator = list(dim_operator)

                # Check max of iterables isn't out of range of index
                # "min" of iterables can be below zero and in array index terms that is
                # just "from the reverse order". Useful in cases where you may want the
                # first and last slices of an image [0, -1]
                if isinstance(dim_operator, list):
                    check_selection_max = max(dim_operator)

                # No need to check slices as slices work with array syntax and will
                # ignore out-of-bounds indicies
                if isinstance(dim_operator, slice):
                    check_selection_max = 0
            else:
                # Nothing was requested from this dimension
                dim_operator = slice(None, None, None)

                # No op means that it doesn't matter how much data is in this dimension
                check_selection_max = 0

        # Not in given dims means that it is a fixed integer selection
        else:
            if dim in kwargs:
                # Integer requested
                dim_operator = kwargs.get(dim)

                # Check that integer
                check_selection_max = dim_operator
            else:
                # Dimension wasn't included in kwargs, default to zero
                log.warning(
                    f"Data has dimension {dim} with depth {data.shape[dim_index]}, "
                    f"assuming {dim}=0 is the desired value, "
                    f"if not the case specify {dim}=x where "
                    f"x is an integer in [0, {data.shape[dim_index]}])."
                )
                dim_operator = 0
                check_selection_max = 0

            # Remove dim from new dims as it is fixed size
            new_dims = new_dims.replace(dim, "")

        # Check that fixed integer request isn't outside of request
        if check_selection_max > data.shape[dim_index]:
            raise IndexError(
                f"Dimension specified with {dim}={dim_operator} "
                f"but Dimension shape is {data.shape[dim_index]}."
            )

        # All checks and operations passed, append dim operation to getitem ops
        getitem_ops.append(dim_operator)

    # Run getitems
    data = data[tuple(getitem_ops)]

    # Add empty dims where dimensions were requested but data doesn't exist
    # Add dimensions to new dims where empty dims are added
    for i, dim in enumerate(return_dims):
        # This dimension wasn't processed
        if dim not in given_dims:
            new_dims = f"{new_dims[:i]}{dim}{new_dims[i:]}"
            data = data.reshape(*data.shape[:i], 1, *data.shape[i:])

    # Any extra dimensions have been removed, only a problem if the depth is > 1
    return transpose_to_dims(
        data, given_dims=new_dims, return_dims=return_dims
    )  # don't pass kwargs or 2 copies


def transpose_to_dims(
    data: types.ArrayLike, given_dims: str, return_dims: str,
) -> types.ArrayLike:
    """
    This shuffles the data dimensions from given_dims to return_dims. Each dimension
    must be present in given_dims must be used in return_dims

    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions
        specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data

    Returns
    -------
    data: types.ArrayLike
        An array in return_dims order, if return_dims=DEFAULT_DIMS then the return
        would have order "STCZYX"
    """
    # Use a counter to track that the contents are composed of the same letters
    # and that no letter is repeated
    if (
        Counter(given_dims) != Counter(return_dims)
        or max(Counter(given_dims).values()) > 1
    ):
        raise ConflictingArgumentsError(
            f"given_dims={given_dims} and return_dims={return_dims} are incompatible."
        )
    # Resort the data into return_dims order
    match_map = {dim: given_dims.find(dim) for dim in given_dims}
    transposer = []
    for dim in return_dims:
        transposer.append(match_map[dim])
    data = data.transpose(transposer)
    return data
