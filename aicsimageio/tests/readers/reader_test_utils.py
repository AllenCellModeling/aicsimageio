#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import numpy as np
from aicsimageio import types
from aicsimageio.readers.reader import Reader
from distributed.protocol import deserialize, serialize
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from psutil import Process
from xarray.testing import assert_equal

###############################################################################


def check_local_file_not_open(fs: AbstractFileSystem, path: str):
    # Check that there are no open file pointers
    if isinstance(fs, LocalFileSystem):
        proc = Process()
        assert str(path) not in [f.path for f in proc.open_files()]


def check_can_serialize_reader(reader: Reader):
    # Dump and reconstruct
    reconstructed = deserialize(*serialize(reader))

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
    expected_scenes: Tuple[int],
    expected_current_scene: int,
    expected_shape: Tuple[int],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[float],
) -> Reader:
    """
    A general suite of tests to run against every Reader.
    """
    # Read file
    reader = ReaderClass(uri)

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Set scene
    reader.set_scene(set_scene)

    # Check scene info
    assert reader.scenes == expected_scenes
    assert reader.current_scene == expected_current_scene

    # Check basics
    assert reader.shape == expected_shape
    assert reader.dtype == expected_dtype
    assert reader.dims.order == expected_dims_order
    assert reader.dims.shape == expected_shape
    assert reader.metadata
    assert reader.channel_names == expected_channel_names
    assert reader.physical_pixel_sizes == expected_physical_pixel_sizes

    # Read only a chunk, then read a chunk from the in-memory, compare
    np.testing.assert_array_equal(
        reader.get_image_dask_data("YX").compute(),
        reader.get_image_data("YX"),
    )

    # Check that the shape and dtype are expected after reading in full
    assert reader.data.shape == expected_shape
    assert reader.data.dtype == expected_dtype

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    return reader


def run_multi_scene_image_read_checks(
    ReaderClass: Reader,
    uri: types.PathLike,
    first_scene_id: int,
    first_scene_shape: Tuple[int],
    first_scene_dtype: np.dtype,
    second_scene_id: int,
    second_scene_shape: Tuple[int],
    second_scene_dtype: np.dtype,
) -> Reader:
    """
    A suite of tests to ensure that data is reset when switching scenes.
    """
    # Read file
    reader = ReaderClass(uri)

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Set scene
    reader.set_scene(first_scene_id)

    # Check basics
    assert reader.shape == first_scene_shape
    assert reader.dtype == first_scene_dtype

    # Check that the shape and dtype are expected after reading in full
    assert reader.data.shape == first_scene_shape
    assert reader.data.dtype == first_scene_dtype

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    # Change scene
    reader.set_scene(second_scene_id)

    # Check data was reset
    assert reader._xarray_dask_data is None
    assert reader._xarray_data is None
    assert reader._dims is None
    assert reader._metadata is None

    # Check basics
    assert reader.shape == second_scene_shape
    assert reader.dtype == second_scene_dtype

    # Check that the shape and dtype are expected after reading in full
    assert reader.data.shape == first_scene_shape
    assert reader.data.dtype == first_scene_dtype

    check_local_file_not_open(reader.fs, reader.path)
    check_can_serialize_reader(reader)

    return reader
