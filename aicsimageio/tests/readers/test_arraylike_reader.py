#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest
from dask.diagnostics import Profiler

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
    assert reader.data.shape == expected_shape
    assert reader.dims == expected_dims
    assert reader.data.chunksize == expected_chunksize

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        in_mem = reader.data.compute()
        assert isinstance(in_mem, np.ndarray)
        assert in_mem.shape == expected_shape
        assert len(prof.results) == expected_task_count
