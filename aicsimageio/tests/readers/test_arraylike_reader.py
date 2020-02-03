#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest
from dask.diagnostics import Profiler

from aicsimageio import exceptions
from aicsimageio.readers import ArrayLikeReader


@pytest.mark.parametrize("arr, expected_shape, expected_dims, expected_chunksize, expected_task_count", [
    (np.ones((1, 1)), (1, 1), "YX", (1, 1), 0),
    (np.ones((1, 1, 1)), (1, 1, 1), "ZYX", (1, 1, 1), 0),
    (np.ones((1, 1, 1, 1)), (1, 1, 1, 1), "CZYX", (1, 1, 1, 1), 0),
    (da.ones((1, 1)), (1, 1), "YX", (1, 1), 1),
    (da.ones((1, 1, 1)), (1, 1, 1), "ZYX", (1, 1, 1), 1),
    (da.ones((1, 1, 1, 1)), (1, 1, 1, 1), "CZYX", (1, 1, 1, 1), 1),
    pytest.param(
        "hello_word",
        None,
        None,
        None,
        None,
        marks=pytest.mark.raises(exceptions=TypeError)
    )
])
def test_arraylike_reader(arr, expected_shape, expected_dims, expected_chunksize, expected_task_count):
    # Init
    reader = ArrayLikeReader(arr)

    # Check basics
    with Profiler() as prof:
        assert reader.dims == expected_dims
        assert reader.metadata is None
        assert reader.dask_data.shape == expected_shape
        assert reader.dask_data.chunksize == expected_chunksize
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        assert isinstance(reader.data, np.ndarray)
        assert reader.data.shape == expected_shape
        assert len(prof.results) == expected_task_count


@pytest.mark.parametrize("expected_starting_dims, set_dims, expected_ending_dims", [
    ("ZYX", "YXC", "YXC"),
    ("ZYX", "TYX", "TYX"),
    ("ZYX", "ABC", "ABC"),
    pytest.param("ZYX", "ABCDE", None, marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError))
])
def test_dims_setting(expected_starting_dims, set_dims, expected_ending_dims):
    # Read file
    img = ArrayLikeReader(da.ones((2, 2, 2)))

    # Check basics
    with Profiler() as prof:
        assert img.dims == expected_starting_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check no tasks happen during dims setting
    with Profiler() as prof:
        img.dims = set_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check no tasks happen during dims getting
    with Profiler() as prof:
        assert img.dims == expected_ending_dims
        # Check that basic details don't require task computation
        assert len(prof.results) == 0
