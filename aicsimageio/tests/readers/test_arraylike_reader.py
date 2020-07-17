#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import ArrayLikeReader


@pytest.mark.parametrize(
    "arr, expected_shape, expected_dims",
    [
        (np.ones((1, 1)), (1, 1), "YX"),
        (np.ones((1, 1, 1)), (1, 1, 1), "ZYX"),
        (np.ones((1, 1, 1, 1)), (1, 1, 1, 1), "CZYX"),
        (da.ones((1, 1)), (1, 1), "YX"),
        (da.ones((1, 1, 1)), (1, 1, 1), "ZYX"),
        (da.ones((1, 1, 1, 1)), (1, 1, 1, 1), "CZYX"),
        pytest.param(
            "hello_word", None, None, marks=pytest.mark.raises(exceptions=TypeError),
        ),
    ],
)
def test_arraylike_reader(arr, expected_shape, expected_dims):
    # Init
    reader = ArrayLikeReader(arr)

    # Check basics
    assert reader.dims == expected_dims
    assert reader.metadata is None
    assert reader.shape == expected_shape
    assert reader.dask_data.shape == expected_shape
    assert reader.get_size(expected_dims) == expected_shape

    # Will error because those dimensions don't exist in the file
    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        assert reader.get_size("ABCDEFG") == expected_shape

    # Check array
    assert isinstance(reader.data, np.ndarray)
    assert reader.data.shape == expected_shape
