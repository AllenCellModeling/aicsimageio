import logging
import re
from collections import Counter

import numpy as np

from .exceptions import ConflictingArgsError

log = logging.getLogger(__name__)


def reshape_data(data: np.ndarray, given_dims: str, return_dims: str, **kwargs) -> np.ndarray:
    """
    Reshape the data into return_dims, pad missing dimensions, and prune extra dimensions.
    Warns the user to use the base reader if the depth of the Dimension being removed is not 1.

    Parameters
    ----------
    data: a numpy.ndarray of arbitrary shape but with the dimensions specified in given_dims
    given_dims: the dimension ordering of data, "CZYX", "VBTCXZY" etc
    return_dims: the dimension ordering of the return data
    kwargs:
        C=1 => desired specific channel, if C in the input data has depth 3 then C=1 takes index 1
        copy=True => copy the data object passed in and return a new object
    Returns
    -------
    a numpy.ndarray in return_dims order, if return_dims=DEFAULT_DIMS then the return would have order "STCZYX"

    """
    # copy the data object if copy=True is in kwargs
    data = data.copy() if kwargs.get('copy', False) else data
    # check for conflicts like return_dims='TCZYX' and fixed channels 'C=1'
    for dim in return_dims:
        if kwargs.get(dim, None) is not None:
            msg = f"argument return_dims={return_dims} and argument {dim}={kwargs.get(dim)} conflict. Check usage."
            raise ConflictingArgsError(msg)

    # add each dimension not included in original data
    new_dims = given_dims
    excluded_dims = re.sub('|'.join(given_dims), '', return_dims)
    for dim in excluded_dims:
        data = np.expand_dims(data, axis=0)
        new_dims = dim + new_dims  # add the missing Dimension to the front

    # if given dims contains a Dimension not in DEFAULT_DIMS and its depth is 1 remove it
    # if it's larger than 1 give a warning and suggest interfacing with the Reader object
    extra_dims = re.sub('|'.join(return_dims), '', given_dims)
    for dim in extra_dims:
        index = new_dims.find(dim)
        if data.shape[index] > 1:
            index_depth = kwargs.get(dim, None)
            if index_depth is None:
                msg = (f'data has dimension {dim} with depth {data.shape[index]}, assuming {dim}=0 is  '
                       f'the desired value, if not the case specify {dim}=x where '
                       f'x is an integer in [0, {data.shape[index]}).')
                log.warning(msg)
                index_depth = 0
            if index_depth >= data.shape[index]:
                raise IndexError(f'Dimension specified with {dim}={index_depth} '
                                 f'but Dimension shape is {data.shape[index]}.')
            planes = np.split(data, data.shape[index], axis=index)  # split dim into list of arrays
            data = planes[index_depth]  # take the specified value of the dim
        data = np.squeeze(data, axis=index)  # remove the dim from ndarray
        new_dims = new_dims[0:index:] + new_dims[index + 1::]  # clip out the Dimension from new_dims
    # any extra dimensions have been removed, only a problem if the depth is > 1
    return transpose_to_dims(data, given_dims=new_dims, return_dims=return_dims)


def transpose_to_dims(data: np.ndarray, given_dims: str, return_dims: str) -> np.ndarray:
    """
    This shuffles the data dimensions from know_dims to return_dims, return_dims can be and subset
    of known_dims in any order.

    Parameters
    ----------
    data: the input data with dimensions known_dims
    given_dims: the dimensions of data
    return_dims: the subset of known_dims to return

    Returns
    -------
    a numpy.ndarray with known_dims

    """
    if Counter(given_dims) != Counter(return_dims) or max(Counter(given_dims).values()) > 1:
        raise ConflictingArgsError(f"given_dims={given_dims} and return_dims={return_dims} are incompatible.")
    # resort the data into return_dims order
    match_map = {dim: given_dims.find(dim) for dim in given_dims}
    transposer = []
    for dim in return_dims:
        if match_map[dim] == -1:
            msg = f'Dimension {dim} requested but not present in given_dims={given_dims}.'
            raise ConflictingArgsError(msg)
        transposer.append(match_map[dim])
    data = data.transpose(transposer)
    return data