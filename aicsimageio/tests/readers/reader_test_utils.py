#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from typing import List, Optional, Set, Tuple

import numpy as np
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from psutil import Process
from xarray.testing import assert_equal

from aicsimageio import types
from aicsimageio.readers.reader import Reader

###############################################################################


def check_local_file_not_open(fs: AbstractFileSystem, path: str):
    # Check that there are no open file pointers
    if isinstance(fs, LocalFileSystem):
        proc = Process()
        assert str(path) not in [f.path for f in proc.open_files()]


def check_can_serialize_reader(reader: Reader):
    # Dump and reconstruct
    _bytes = pickle.dumps(reader)
    reconstructed = pickle.loads(_bytes)

    # Assert primary attrs are equal
    if reader._xarray_data is None:
        assert reconstructed._xarray_data is None
    else:
        assert_equal(reader._xarray_data, reconstructed._xarray_data)

    if reader._xarray_dask_data is None:
        assert reconstructed._xarray_dask_data is None
    else:
        assert_equal(reader._xarray_dask_data, reconstructed._xarray_dask_data)


def run_image_read_checks(
    ReaderClass: Reader,
    uri: types.PathLike,
    set_scene: int,
    expected_scenes: Set[int],
    expected_current_scene: int,
    expected_shape: Tuple[int],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[Optional[float]],
) -> Reader:
    # Read file
    reader = ReaderClass(uri)

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Set scene
    reader.set_scene(set_scene)

    # Check scene info
    assert reader.scenes == expected_scenes
    assert reader.current_scene == expected_current_scene

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Check basics
    assert reader.shape == expected_shape
    assert reader.dtype == expected_dtype
    assert reader.dims.order == expected_dims_order
    assert reader.dims.shape == expected_shape
    assert reader.metadata
    assert reader.channel_names == expected_channel_names
    assert reader.physical_pixel_sizes == expected_physical_pixel_sizes

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Read only a chunk, then read a chunk from the in-memory, compare
    np.testing.assert_array_equal(
        reader.get_image_dask_data("YX").compute(),
        reader.get_image_data("YX"),
    )

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Check that the shape and dtype are expected after reading in full
    assert reader.data.shape == expected_shape
    assert reader.data.dtype == expected_dtype

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    return reader
