#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from typing import (Any, Callable, Dict, List, NamedTuple, Optional, Tuple,
                    Union)

import dask.array as da
import numpy as np
from pluggy import HookimplMarker

from aicsimageio import AICSImage, dask_utils, exceptions
from aicsimageio.constants import Dimensions
from aicsimageio.readers.reader import Reader

###############################################################################

LayerData = Union[Tuple[Any], Tuple[Any, Dict], Tuple[Any, Dict, str]]
PathLike = Union[str, List[str]]
ReaderFunction = Callable[[PathLike], List[LayerData]]

napari_hook_implementation = HookimplMarker("napari")

###############################################################################


class LoadResult(NamedTuple):
    data: np.ndarray
    index: int
    channel_axis: Optional[int]
    channel_names: Optional[List[str]]

###############################################################################


def _load_image(path: str, ReaderClass: Reader, index: int) -> LoadResult:
    # Initialize reader
    reader = ReaderClass(path)

    # Set channel_axis
    if Dimensions.Channel in reader.dims:
        channel_axis = reader.dims.index(Dimensions.Channel)
    else:
        channel_axis = None

    # Set channel names
    if channel_axis is not None:
        channel_names = reader.get_channel_names()
    else:
        channel_names = None

    # Finalize data and metadata to send to napari viewer
    return LoadResult(
        data=np.squeeze(reader.dask_data.astype(np.float16)),
        index=index,
        channel_axis=channel_axis,
        channel_names=channel_names,
    )


def reader_function(path: PathLike, compute: bool) -> List[LayerData]:
    """
    Given a single path return a list of LayerData tuples.
    """
    # Alert console of how we are loading the image
    print(f"Reader will load image in-memory: {compute}")

    # Standardize path to list of paths
    paths = [path] if isinstance(path, str) else path

    # Determine reader for all
    ReaderClass = AICSImage.determine_reader(paths[0])

    # Spawn dask cluster for parallel read
    with dask_utils.cluster_and_client() as (cluster, client):
        # Map each file read
        futures = client.map(
            _load_image,
            paths,
            [ReaderClass for i in range(len(paths))],
            [i for i in range(len(paths))],
        )

        # Block until done
        results = client.gather(futures)

        # Sort results by index
        results = sorted(results, key=lambda result: result.index)

        # Stack all arrays and configure metadata
        data = da.stack([result.data for result in results])

        # Determine whether or not to read in full first
        if compute:
            data = data.compute()

        # Construct metadata using any of the returns as there is an assumption it is all the same
        meta = {
            "name": results[0].channel_names,
            "channel_axis": results[0].channel_axis + 1,  # Add 1 to offset the new axis from the array stack
            "is_pyramid": False,
            "visible": False,
        }

    return [(data, meta)]


def get_reader(path: PathLike, compute: bool) -> Optional[ReaderFunction]:
    """
    Given a single path or list of paths, return the appropriate aicsimageio reader.
    """
    # Standardize path to list of paths
    paths = [path] if isinstance(path, str) else path

    # See if there is a supported reader for the file(s) provided
    try:
        # There is an assumption that the images are stackable and
        # I think it is also safe to assume that if stackable, they are of the same type
        # So only determine reader for the first one
        AICSImage.determine_reader(paths[0])

        # The above line didn't error so we know we have a supported reader
        # Return a partial function with compute determined
        return partial(reader_function, compute=compute)

    # No supported reader, return None
    except exceptions.UnsupportedFileFormatError:
        return None
