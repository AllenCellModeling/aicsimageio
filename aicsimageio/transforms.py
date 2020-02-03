#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from collections import Counter

import dask.array as da
import numpy as np

from . import types
from .exceptions import ConflictingArgumentsError

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _array_split(operator, ary, indices_or_sections, axis=0):
    """
    A reimplementation of array split that doesn't cast to numpy and returns a list of
    whichever object type it is currently operating on.

    https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/shape_base.py#L723
    """
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError("number sections must be larger than 0.")
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section+1] + (Nsections-extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=ary.dtype).cumsum(axis=None)

    sub_arys = []
    sary = operator.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]

        # Make sure start and end are integers
        st = int(st)
        end = int(end)

        sub_arys.append(operator.swapaxes(sary[st:end], axis, 0))

    return sub_arys


def _split(operator, ary, indices_or_sections, axis=0):
    """
    A reimplementation of split that doesn't cast to numpy and returns a list of
    whichever object type it is currently operating on.

    https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/shape_base.py#L782
    """
    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = ary.shape[axis]
        if N % sections:
            raise ValueError("array split does not result in an equal division")
    return _array_split(operator, ary, indices_or_sections, axis)


def reshape_data(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
    **kwargs
) -> types.ArrayLike:
    """
    Reshape the data into return_dims, pad missing dimensions, and prune extra dimensions.
    Warns the user to use the base reader if the depth of the Dimension being removed is not 1.

    Parameters
    ----------
    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data
    kwargs:
        C=1 => desired specific channel, if C in the input data has depth 3 then C=1 returns the 2nd slice (0 indexed)
        Z=10 => desired specific channel, if Z in the input data has depth 20 then Z=10 returns the 11th slice

    Returns
    -------
    data: types.ArrayLike
        An array in return_dims order, if return_dims=DEFAULT_DIMS then the return would have order "STCZYX"

    """
    # Get operator
    if isinstance(data, da.core.Array):
        operator = da
    else:
        operator = np

    # Check for conflicts like return_dims='TCZYX' and fixed channels 'C=1'
    for dim in return_dims:
        if kwargs.get(dim) is not None:
            msg = f"Argument return_dims={return_dims} and argument {dim}={kwargs.get(dim)} conflict. Check usage."
            raise ConflictingArgumentsError(msg)

    # Add each dimension not included in original data
    new_dims = given_dims
    excluded_dims = "".join(set(return_dims) - set(given_dims))
    for dim in excluded_dims:
        # Dask doesn't have an explicit expand dims so we simply reshape with an extra dim
        data = operator.reshape(data, (1, *data.shape))
        new_dims = dim + new_dims  # add the missing Dimension to the front

    # If given dims contains a Dimension not in return dims and its depth is 1 remove it
    # If it's larger than 1 give a warning and suggest interfacing with the Reader object
    extra_dims = "".join(set(given_dims) - set(return_dims))
    for dim in extra_dims:
        index = new_dims.find(dim)
        if data.shape[index] > 1:
            index_depth = kwargs.get(dim)
            if index_depth is None:
                log.warning(
                    f"Data has dimension {dim} with depth {data.shape[index]}, assuming {dim}=0 is "
                    f"the desired value, if not the case specify {dim}=x where "
                    f"x is an integer in [0, {data.shape[index]})."
                )
                index_depth = 0
            if index_depth >= data.shape[index]:
                raise IndexError(f"Dimension specified with {dim}={index_depth} "
                                 f"but Dimension shape is {data.shape[index]}.")
            planes = _split(operator, data, data.shape[index], axis=index)  # split dim into list of arrays
            data = planes[index_depth]  # take the specified value of the dim
        data = operator.squeeze(data, axis=index)  # remove the dim from ndarray
        new_dims = new_dims[0:index] + new_dims[index + 1:]  # clip out the Dimension from new_dims
    # Any extra dimensions have been removed, only a problem if the depth is > 1
    return transpose_to_dims(data, given_dims=new_dims, return_dims=return_dims)  # don't pass kwargs or 2 copies


def transpose_to_dims(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
) -> types.ArrayLike:
    """
    This shuffles the data dimensions from given_dims to return_dims. Each dimension must be present in
    given_dims must be used in return_dims

    data: types.ArrayLike
        Either a dask array or numpy.ndarray of arbitrary shape but with the dimensions specified in given_dims
    given_dims: str
        The dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: str
        The dimension ordering of the return data

    Returns
    -------
    data: types.ArrayLike
        An array in return_dims order, if return_dims=DEFAULT_DIMS then the return would have order "STCZYX"
    """
    # Use a counter to track that the contents are composed of the same letters and that no letter is repeated
    if Counter(given_dims) != Counter(return_dims) or max(Counter(given_dims).values()) > 1:
        raise ConflictingArgumentsError(f"given_dims={given_dims} and return_dims={return_dims} are incompatible.")
    # Resort the data into return_dims order
    match_map = {dim: given_dims.find(dim) for dim in given_dims}
    transposer = []
    for dim in return_dims:
        transposer.append(match_map[dim])
    data = data.transpose(transposer)
    return data
