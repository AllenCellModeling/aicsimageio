#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import Counter
from typing import Any, Literal, Optional, Union

import dask.array as da
import numpy as np
import xarray as xr

from . import types
from .exceptions import ConflictingArgumentsError, UnexpectedShapeError
from .image_container import ImageContainer

###############################################################################


def transpose_to_dims(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
) -> types.ArrayLike:
    """
    This shuffles the data dimensions from given_dims to return_dims. Each dimension
    must be present in given_dims must be used in return_dims

    Parameters
    ----------
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
        The data with the specified dimension ordering.

    Raises
    ------
    ConflictingArgumentsError
        given_dims and return_dims are incompatible.
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


def reshape_data(
    data: types.ArrayLike, given_dims: str, return_dims: str, **kwargs: Any
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
        * C=1 => desired specific channel, if C in the input data has depth 3 then C=1
          returns the 2nd slice (0 indexed)
        * Z=10 => desired specific channel, if Z in the input data has depth 20 then
          Z=10 returns the 11th slice
        * T=[0, 1] => desired specific timepoints, if T in the input data has depth 100
          then T=[0, 1] returns the 1st and 2nd slice (0 indexed)
        * T=(0, 1) => desired specific timepoints, if T in the input data has depth 100
          then T=(0, 1) returns the 1st and 2nd slice (0 indexed)
        * T=(0, -1) => desired specific timepoints, if T in the input data has depth 100
          then T=(0, -1) returns the first and last slice
        * T=range(10) => desired specific timepoints, if T in the input data has depth
          100 then T=range(10) returns the first ten slices
        * T=slice(0, -1, 5) => desired specific timepoints, T=slice(0, -1, 5) returns
          every fifth timepoint

    Returns
    -------
    data: types.ArrayLike
        The data with the specified dimension ordering.

    Raises
    ------
    ConflictingArgumentsError
        Missing dimension in return dims when using range, slice, or multi-index
        dimension selection for the requested dimension.

    IndexError
        Requested dimension index not present in data.

    Examples
    --------
    Specific index selection

    >>> data = np.random.rand((10, 100, 100))
    ... z1 = reshape_data(data, "ZYX", "YX", Z=1)

    List of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_and_second = reshape_data(data, "ZYX", "YX", Z=[0, 1])

    Tuple of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_and_last = reshape_data(data, "ZYX", "YX", Z=(0, -1))

    Range of index selection

    >>> data = np.random.rand((10, 100, 100))
    ... first_three = reshape_data(data, "ZYX", "YX", Z=range(3))

    Slice selection

    >>> data = np.random.rand((10, 100, 100))
    ... every_other = reshape_data(data, "ZYX", "YX", Z=slice(0, -1, 2))

    Empty dimension expansion

    >>> data = np.random.rand((10, 100, 100))
    ... with_time = reshape_data(data, "ZYX", "TZYX")

    Dimension order shuffle

    >>> data = np.random.rand((10, 100, 100))
    ... as_zx_base = reshape_data(data, "ZYX", "YZX")

    Selections, empty dimension expansions, and dimension order shuffle

    >>> data = np.random.rand((10, 100, 100))
    ... example = reshape_data(data, "CYX", "BSTCZYX", C=slice(0, -1, 3))
    """
    # Check for parameter conflicts
    for dim in given_dims:
        # return_dims='TCZYX' and fixed dimensions 'C=1'
        # Dimension is in kwargs
        # Dimension is an integer
        # Dimension is in return dimensions
        if isinstance(kwargs.get(dim), int) and dim in return_dims:
            raise ConflictingArgumentsError(
                f"When selecting a single dimension index, the specified dimension can "
                f"not be provided in return_dims. "
                f"return_dims={return_dims}, dimension {dim} = {kwargs.get(dim)}"
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
                f"When selecting a multiple dimension indices, the specified "
                f"dimension must be provided in return_dims. "
                f"return_dims={return_dims}, dimension {dim} = {kwargs.get(dim)}"
            )

    # Process each dimension available
    new_dims = given_dims
    dim_specs = []
    for dim in given_dims:
        # Store index of the dim as it is in given data
        dim_index = given_dims.index(dim)

        # Handle dim in return dims which means that it is
        # an iterable or None selection
        if dim in return_dims:
            # Specific iterable requested
            if dim in kwargs:
                # Actual dim specification
                # The specification provided for this dimension in the kwargs
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec

                # Convert operator to standard list or slice
                # dask.Array and numpy.ndarray both natively support
                # List[int] and slices being passed to getitem so no need to cast them
                # to anything different
                if isinstance(dim_spec, (tuple, range)):
                    dim_spec = list(dim_spec)

                # Get the largest absolute value index in the list using min and max
                if isinstance(dim_spec, list):
                    check_selection_max = max([abs(min(dim_spec)), max(dim_spec)])

                # Get the largest absolute value index from start and stop of slice
                if isinstance(dim_spec, slice):
                    check_selection_max = max([abs(dim_spec.stop), abs(dim_spec.start)])
            else:
                # Nothing was requested from this dimension
                dim_spec = slice(None, None, None)
                display_dim_spec = dim_spec

                # No op means that it doesn't matter how much data is in this dimension
                check_selection_max = 0

        # Not in given dims means that it is a fixed integer selection
        else:
            if dim in kwargs:
                # Integer requested
                dim_spec = kwargs.get(dim)
                display_dim_spec = dim_spec

                # Check that integer
                check_selection_max = dim_spec
            else:
                dim_spec = 0
                display_dim_spec = dim_spec
                check_selection_max = 0

            # Remove dim from new dims as it is fixed size
            new_dims = new_dims.replace(dim, "")

        # Check that fixed integer request isn't outside of request
        if check_selection_max > data.shape[dim_index]:
            raise IndexError(
                f"Dimension specified with {dim}={display_dim_spec} "
                f"but Dimension shape is {data.shape[dim_index]}."
            )

        # All checks and operations passed, append dim operation to getitem ops
        dim_specs.append(dim_spec)

    # Run getitems
    data = data[tuple(dim_specs)]

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


def generate_stack(
    image_container: ImageContainer,
    mode: Literal["data", "dask_data", "xarray_data", "xarray_dask_data"],
    drop_non_matching_scenes: bool = False,
    select_scenes: Optional[
        Union[list[Union[str, int]], tuple[Union[str, int], ...]]
    ] = None,
    scene_character: str = "I",
    scene_coord_values: str = "index",
) -> types.MetaArrayLike:
    """
    Stack each scene contained in the reader into a
    single array. This method handles the logic of determining which
    stack function to use (dask or numpy) and whether or not to return a
    labelled array (xr.DataArray). Users should prefer
    to use one of get_stack, get_dask_stack, get_xarray_stack, or
    get_xarray_dask_stack.

    Parameters
    ----------
    mode: Literal["data", "dask_data", "xarray_data", "xarray_dask_data"]
        String describing the style of data to return. Should be one of:
        "data", "dask_data", "xarray_data", "xarray_dask_data".
    drop_non_matching_scenes: bool
        During the scene iteration process, if the next scene to be added
        to the stack has different shape or dtype, should it be dropped or
        raise an error.  Default: False (raise an error)
    select_scenes: Optional[
            Union[List[Union[str, int]], Tuple[Union[str, int], ...]]]
        Which scenes to stack into a single array. Scenes can be provided
        as a list or tuple of scene indices or names. It is recommended to
        use the scene integer index instead of the scene name to avoid
        duplicate scene name lookup issues.
        Default: None (stack all scenes)
    scene_character: str
        Character to use as the name of the scene dimension on the output
        array. Default "I"
    scene_coord_values : str
        How to assign coordinates to the scene dimension of the final
        array. If scene_coord_values="names" use the scene name from
        the reader object. If scene_coord_values="index" don't attach any
        coordinates and fall back to integer values.
        Default: "index"

    Returns
    -------
    stack: types.MetaArrayLike
        The fully stacked array. This can be 6+ dimensions with Scene being
        the first dimension.

    """

    mode_check = ["data", "dask_data", "xarray_data", "xarray_dask_data"]
    if mode not in mode_check:
        raise ValueError(
            f"Invalid mode kwarg. Found {mode} but should be one of:"
            f"{', '.join(mode_check)}."
        )

    scene_stacks = []
    scene_names = []

    if select_scenes is None:
        select_scenes = list(range(len(image_container.scenes)))

    for i, s in enumerate(select_scenes):
        image_container.set_scene(s)
        data = getattr(image_container, mode)

        # Store the shape and dtype of the first scenes data
        # to check against later scenes. If returning a DataArray
        # store the coords and dims to use in the final output
        if i == 0:
            shape = data.shape
            dtype = data.dtype

            if "xarray" in mode:
                coords = dict(data.coords)
                dims = data.dims

                if scene_character in dims:
                    raise ValueError(
                        f"Provided scene dimension character '{scene_character}' "
                        f"was found in the existing dimensions of the data {dims}"
                    )

        # Check other scenes against the first scene
        else:
            if data.shape != shape:
                if not drop_non_matching_scenes:
                    raise UnexpectedShapeError(
                        f"All scenes must have same shape. Found shape"
                        f"{data.shape} in scene {s} but expected"
                        f"{shape} based on scene {select_scenes[0]}"
                    )
                else:
                    continue
            if data.dtype != dtype:
                if not drop_non_matching_scenes:
                    raise TypeError(
                        f"All scenes must have the same dtype. Found data"
                        f"with dtype {data.dtype} in scene {s} but expected"
                        f"dtype {dtype} based on scene {select_scenes[0]}"
                    )
                else:
                    continue

        scene_stacks.append(data)
        scene_names.append(image_container.current_scene)

    stack = da.stack if "dask" in mode else np.stack

    if "xarray" in mode:
        all_data = stack([x.data for x in scene_stacks])
        if scene_coord_values == "names":
            coords = {scene_character: scene_names, **coords}

        return xr.DataArray(
            all_data,
            dims=(scene_character, *dims),
            coords=coords,
        )

    else:
        return stack(scene_stacks)
