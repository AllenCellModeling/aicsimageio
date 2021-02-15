#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr

from .. import constants, exceptions
from ..dimensions import DimensionNames
from ..metadata import utils as metadata_utils
from ..types import MetaArrayLike
from .reader import Reader

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################


class ArrayLikeReader(Reader):
    """
    A catch all for numpy, dask, or xarray to Reader interface.

    Parameters
    ----------
    image: Union[List[MetaArrayLike], MetaArrayLike]
        A single, numpy ndarray, dask Array, or xarray DataArray, or list of many.
        If provided a list, each item in the list will be exposed through the scene API.
        If provided an xarray DataArray alone or as an element of the list, the
        known_dims and known_channel_names, kwargs are ignored if there is dimension
        (or channel coordinate) information attached the xarray object. If the provided
        xarray object is missing these pieces, the AICSImageIO defaults will be added.

    known_dims: Optional[Union[List[str], str]]
        A string of known dimensions to be applied to all array(s) or a
        list of string dimension names to be mapped onto the list of arrays
        provided to image.
        Default: None (guess dimensions for single array or multiple arrays)

    known_channel_names: Optional[Union[List[str], List[List[str]]]]
        A list of string channel names to be applied to all array(s) or a
        list of lists of string channel names to be mapped onto the list of arrays
        provided to image.
        Default: None (create fake channel names for single or multiple arrays)

    Raises
    ------
    exceptions.ConflictingArgumentsError: Raised when the number of scenes provided is
    different from the number of items provided to the metadata parameters if as a list.

    exceptions.ConflictingArgumentsError: Raised when known_channel_names is provided
    but the channel dimension was either not guessed or not provided in known_dims.

    ValueError: Provided known_dims string or known_channel_names are not the same
    length as the number of dimensions or the size of the channel dimensions for the
    array at the matching index.

    Notes
    -----
    If you want to combine multiple numpy ndarrays or dask arrays with xarray DataArrays
    and attach metadata to all non-xarray.DataArrays, you can do so, the metadata for
    the xarray DataArray scenes will simply be ignored if it the non-xarray defaults.

    Other way of saying this is:
    If there are dimension names besides "dim_N" and there are coordinates
    for the channel dimension, then the known_dims and known_channel_names will be
    ignored. If the provided xarray object IS using the xarray defaults, we will
    override their defaults.

    In such cases, it is recommended that you provided metadata for those scenes as
    None. I.E.

    >>> some_xr = ...
    ... some_np = ...
    ... some_dask = ...
    ... reader = ArrayLikeReader(
    ...     image=[some_xr, some_np, some_dask],
    ...     known_dims=[None, "CTYX", "ZTYX"],
    ...     known_channel_names=[None, ["A", "B", C"], None],
    ... )

    Will create a three scene ArrayLikeReader with the raw xarray DataArray as the first
    scene, a numpy array with CTYX as the dimensions and ["A", "B", "C"] as the
    channel names, and a dask array with ZTYX as the dimensions and no channel names.
    """

    @staticmethod
    def _is_supported_image(
        image: Union[List[MetaArrayLike], MetaArrayLike], *args, **kwargs
    ) -> bool:
        if isinstance(image, list):
            return all(
                isinstance(scene, (np.ndarray, da.Array, xr.DataArray))
                for scene in image
            )

        return isinstance(image, (np.ndarray, da.Array, xr.DataArray))

    def __init__(
        self,
        image: Union[List[MetaArrayLike], MetaArrayLike],
        known_dims: Optional[Union[List[str], str]] = None,
        known_channel_names: Optional[Union[List[str], List[List[str]]]] = None,
        **kwargs,
    ):
        # Enforce valid image
        if not self._is_supported_image(image):
            raise exceptions.UnsupportedFileFormatError(self.__class__.__name__, image)

        # General note
        # Any time we do a `known_channel_names[0]` it's because the type check for
        # channel names is a List[List[str]], so by checking the first element we should
        # be getting back a list or a string. Anything else will error.

        # The result of all of this validation and pre-compute work is that at the end
        # of this init, we should have n-number of constructed xarray objects created
        # from the parameters provided that we then just pass-through to for all other
        # standard Reader operations

        # If metadata is attached as lists
        # Enforcing matching shape
        if isinstance(image, list):
            if isinstance(known_dims, list):
                # Check known dims
                if len(known_dims) != len(image):
                    raise exceptions.ConflictingArgumentsError(
                        f"ArrayLikeReader received a list of arrays to use as scenes "
                        f"but the provided list of known dimensions is of different "
                        f"length. "
                        f"Number of provided scenes: {len(image)}, "
                        f"Number of provided known dimension strings: {len(known_dims)}"
                    )

            # Check known channel names
            if known_channel_names is not None:
                if isinstance(known_channel_names[0], list):
                    if len(known_channel_names) != len(image):
                        raise exceptions.ConflictingArgumentsError(
                            f"ArrayLikeReader received a list of arrays to use as "
                            f"scenes but the provided list of known channel names is "
                            f"of different length. "
                            f"Number of provided scenes: {len(image)}, "
                            f"Number of provided known channel names: "
                            f"{len(known_channel_names)}"
                        )

        # If metadata is attached as singles
        # but many scenes provided, expand
        if isinstance(image, list):
            if known_dims is None or isinstance(known_dims, str):
                known_dims = [known_dims for i in range(len(image))]
            if known_channel_names is None or isinstance(known_channel_names[0], str):
                known_channel_names = [known_channel_names for i in range(len(image))]

        # Set all kwargs to lists for standard interface
        if not isinstance(image, list):
            image = [image]
        if not isinstance(known_dims, list):
            known_dims = [known_dims]
        if known_channel_names is None:
            known_channel_names = [known_channel_names]
        # Also wrap the channel names list if they were provided
        # but only a single scene was
        elif len(image) == 1 and not isinstance(known_channel_names[0], list):
            known_channel_names = [known_channel_names]

        # Store image(s)
        self._all_scenes = image

        # Validate and store dims
        self._scene_dims_list = []
        for i, dims_string in enumerate(known_dims):
            this_scene = self._all_scenes[i]
            # Provided None, guess
            if dims_string is None:
                # Get dims from xarray
                if isinstance(this_scene, xr.DataArray):
                    # Guess for xarray when they guessed
                    # http://xarray.pydata.org/en/stable/data-structures.html?highlight=dim_n#creating-a-dataarray
                    # See "dim_N"
                    if this_scene.dims[0] == "dim_0":
                        log.debug(
                            "Overrode the default xarray dimensions for AICSImage "
                            "guessed dimensions."
                        )
                        # Rename the dimensions from "dim_N" to just the guess dim
                        # Update scene list in place
                        self._all_scenes[i] = this_scene.rename(
                            {
                                f"dim_{d_index}": d
                                for d_index, d in enumerate(
                                    self._guess_dim_order(this_scene.shape)
                                )
                            }
                        )

                # Guess for non xarray data
                self._scene_dims_list.append(self._guess_dim_order(this_scene.shape))

            # Provided Some, validate
            else:
                if len(dims_string) == len(this_scene.shape):
                    # Update dims for xarray
                    if isinstance(this_scene, xr.DataArray):
                        # Guess for xarray when they guessed
                        # http://xarray.pydata.org/en/stable/data-structures.html?highlight=dim_n#creating-a-dataarray
                        # See "dim_N"
                        if this_scene.dims[0] == "dim_0":
                            log.debug(
                                "Overrode the default xarray dimensions for AICSImage "
                                "provided dimensions."
                            )
                            # Rename the dimensions from "dim_N" to just the guess dim
                            # Update scene list in place
                            self._all_scenes[i] = this_scene.rename(
                                {
                                    f"dim_{d_index}": d
                                    for d_index, d in enumerate(dims_string)
                                }
                            )

                    self._scene_dims_list.append(dims_string)
                else:
                    raise ValueError(
                        f"Provided dimension string does not have the same amount of "
                        f"dimensions as the matching provided array. "
                        f"Provided array shape: {this_scene.shape}, "
                        f"Provided dimension string: {dims_string}"
                    )

        # Validate and store channel_names
        self._scene_channel_names = []
        for s_index, channel_names in enumerate(known_channel_names):
            this_scene = self._all_scenes[s_index]
            this_scene_dims = self._scene_dims_list[s_index]

            # Create channel names is needed
            if DimensionNames.Channel in this_scene_dims:
                channel_dim_index = this_scene_dims.index(DimensionNames.Channel)

                # Handle xarray missing channel names
                if isinstance(this_scene, xr.DataArray):
                    if DimensionNames.Channel not in this_scene.coords:
                        # Use provided
                        if channel_names is not None:
                            this_scene.coords[DimensionNames.Channel] = channel_names

                        # Generate
                        else:
                            set_channel_names = []
                            for c_index in range(this_scene.shape[channel_dim_index]):
                                image_id = metadata_utils.generate_ome_image_id(s_index)
                                set_channel_names.append(
                                    metadata_utils.generate_ome_channel_id(
                                        image_id=image_id, channel_id=c_index
                                    )
                                )

                            this_scene.coords[
                                DimensionNames.Channel
                            ] = set_channel_names

                # Provided None, generate
                if channel_names is None:
                    this_scene_channels = []
                    for c_index in range(this_scene.shape[channel_dim_index]):
                        image_id = metadata_utils.generate_ome_image_id(s_index)
                        this_scene_channels.append(
                            metadata_utils.generate_ome_channel_id(
                                image_id=image_id, channel_id=c_index
                            )
                        )

                    self._scene_channel_names.append(this_scene_channels)

                # Provided Some, validate
                else:
                    if len(channel_names) != this_scene.shape[channel_dim_index]:
                        raise ValueError(
                            f"Provided channel names list does not match the size of "
                            f"channel dimension for the provided array. "
                            f"Provided array shape: {this_scene.shape}, "
                            f"Channel dimension size: "
                            f"{this_scene.shape[channel_dim_index]}, "
                            f"Provided channel names: {channel_names}"
                        )
                    else:
                        self._scene_channel_names.append(channel_names)

            # Raise error when channel names were provided when they shouldn't have been
            else:
                if channel_names is not None:
                    raise ValueError(
                        f"Received channel names for array without channel dimension. "
                        f"Provided array shape: {this_scene.shape}, "
                        f"Provided (or guessed) dimensions: {this_scene_dims}, "
                        f"Provided channel names: {channel_names}"
                    )
                else:
                    self._scene_channel_names.append(channel_names)

        # Construct full xarrays
        # All data arrays in this list are dask Arrays
        self._xr_darrays = []
        for scene_data, dims, channel_names in zip(
            self._all_scenes, self._scene_dims_list, self._scene_channel_names
        ):
            # Handle simple case of provided a DataArray
            if isinstance(scene_data, xr.DataArray):
                # Set metadata to point at existing attrs
                scene_data.attrs[constants.METADATA_UNPROCESSED] = scene_data.attrs

                # If the data backing the xarray is dask
                # just append
                if isinstance(scene_data.data, da.Array):
                    self._xr_darrays.append(scene_data)

                # If the data backing the xarray is numpy
                # copy the array but use dask from numpy
                else:
                    self._xr_darrays.append(
                        scene_data.copy(data=da.from_array(scene_data.data))
                    )

            # Handle non-xarray cases
            else:
                dims = list(dims)
                coords = {}
                if DimensionNames.Channel in dims:
                    coords[DimensionNames.Channel] = channel_names

                # Handle dask
                if isinstance(scene_data, np.ndarray):
                    scene_data = da.from_array(scene_data)

                # Append the dask backed array
                self._xr_darrays.append(
                    xr.DataArray(
                        data=scene_data,
                        dims=dims,
                        coords=coords,
                        attrs={constants.METADATA_UNPROCESSED: None},
                    )
                )

    @property
    def scenes(self) -> Tuple[str]:
        if self._scenes is None:
            self._scenes = tuple(
                metadata_utils.generate_ome_image_id(i)
                for i in range(len(self._all_scenes))
            )

        return self._scenes

    def _read_delayed(self) -> xr.DataArray:
        return self._xr_darrays[self.current_scene_index]

    def _read_immediate(self) -> xr.DataArray:
        return self._xr_darrays[self.current_scene_index].copy(
            data=self._xr_darrays[self.current_scene_index].data.compute()
        )
