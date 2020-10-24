#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from typing import List, Optional, Tuple

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileOpener
from psutil import Process
from xarray.testing import assert_equal

from aicsimageio import types
from aicsimageio.readers.reader import Reader

###############################################################################


def check_local_file_not_open(abstract_file: types.FSSpecBased):
    # Check that there are no open file pointers
    if isinstance(abstract_file, LocalFileOpener):
        proc = Process()
        assert str(abstract_file.path) not in [f.path for f in proc.open_files()]


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
    can_read_chunks: bool,
    expected_shape: Tuple[int],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: Optional[List[str]],
    expected_physical_pixel_sizes: Tuple[Optional[float]],
) -> Reader:
    # Read file
    reader = ReaderClass(uri)

    # check_local_file_not_open(reader.abstract_file)
    check_can_serialize_reader(reader)

    # Check basics
    # Due to how data is routed through the _xarray_dask_data prop,
    # any call to metadata will be initially routed through it
    # thus the first call and test needs additional warnings check handling
    if can_read_chunks:
        assert reader.dims.order == expected_dims_order
    else:
        with pytest.warns(UserWarning):
            assert reader.dims.order == expected_dims_order

    # Finish checking basics
    assert reader.metadata
    assert reader.shape == expected_shape
    assert reader.dtype == expected_dtype
    assert reader.channel_names == expected_channel_names
    assert reader.physical_pixel_sizes == expected_physical_pixel_sizes

    # check_local_file_not_open(reader.abstract_file)
    check_can_serialize_reader(reader)

    # Try reading remote chunks
    if can_read_chunks:
        assert reader.get_image_dask_data("YX").compute() is not None
    else:
        with pytest.raises(AttributeError):
            assert reader.get_image_dask_data("YX").compute() is not None

    # check_local_file_not_open(reader.abstract_file)
    check_can_serialize_reader(reader)

    # Read the image in full
    # All readers should be able to do this unless there is some weird format
    # We can try our best but some implementations just don't exist for reading buffers
    # I am looking at you FFMPEG formats
    assert reader.data.shape == expected_shape
    assert reader.data.dtype == expected_dtype

    # check_local_file_not_open(reader.abstract_file)
    check_can_serialize_reader(reader)

    return reader
