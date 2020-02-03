#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from aicsimageio.exceptions import ConflictingArgumentsError
from aicsimageio.transforms import reshape_data, transpose_to_dims


@pytest.mark.parametrize("data, given_dims, return_dims, other_args, expected_shape", [
    (np.zeros((10, 1, 5, 6, 200, 400)), "STCZYX", "CSZYX", {}, (5, 10, 6, 200, 400)),
    (da.zeros((10, 1, 5, 6, 200, 400)), "STCZYX", "CSZYX", {}, (5, 10, 6, 200, 400)),
    (np.zeros((6, 200, 400)), "ZYX", "STCZYX", {}, (1, 1, 1, 6, 200, 400)),
    (da.zeros((6, 200, 400)), "ZYX", "STCZYX", {}, (1, 1, 1, 6, 200, 400)),
    (np.zeros((6, 200, 400)), "ZYX", "ZCYSXT", {}, (6, 1, 200, 1, 400, 1)),
    (da.zeros((6, 200, 400)), "ZYX", "ZCYSXT", {}, (6, 1, 200, 1, 400, 1)),
    (np.zeros((2, 2, 2)), "ABI", "ZCYSXT", {}, (1, 1, 1, 1, 1, 1)),
    (da.zeros((2, 2, 2)), "ABI", "ZCYSXT", {}, (1, 1, 1, 1, 1, 1)),
    pytest.param(
        np.zeros((6, 200, 400)),
        "ZYX",
        "TYXC",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=IndexError)
    ),
    pytest.param(
        da.zeros((6, 200, 400)),
        "ZYX",
        "TYXC",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=IndexError)
    ),
    pytest.param(
        np.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZ",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        da.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZ",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        np.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZX",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        da.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZX",
        {"Z": 7},
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
])
def test_reshape_data_shape(data, given_dims, return_dims, other_args, expected_shape):
    actual = reshape_data(data=data, given_dims=given_dims, return_dims=return_dims, **other_args)
    assert actual.shape == expected_shape

    # Check that the output data is the same type as the input
    assert type(actual) == type(data)


@pytest.mark.parametrize("data, given_dims, return_dims, idx_in, idx_out", [
    (
        np.random.rand(10, 1, 5, 6, 200, 400),
        "STCZYX",
        "CSZYX",
        (5, 0, 3, 3, ...),
        (3, 5, 3, ...)
    ),
    (
        da.random.random((10, 1, 5, 6, 200, 400)),
        "STCZYX",
        "CSZYX",
        (5, 0, 3, 3, ...),
        (3, 5, 3, ...)
    ),
    (
        np.zeros((6, 200, 400)),
        "ZYX",
        "STCZYX",
        (..., 100, 200),
        (0, 0, 0, ..., 100, 200)
    ),
    (
        da.zeros((6, 200, 400)),
        "ZYX",
        "STCZYX",
        (..., 100, 200),
        (0, 0, 0, ..., 100, 200)
    ),
    (
        np.zeros((6, 200, 400)),
        "ZYX",
        "ZCYSXT",
        (3, 100, ...),
        (3, 0, 100, 0, ..., 0)
    ),
    (
        da.zeros((6, 200, 400)),
        "ZYX",
        "ZCYSXT",
        (3, 100, ...),
        (3, 0, 100, 0, ..., 0)
    ),
])
def test_reshape_data_values(data, given_dims, return_dims, idx_in, idx_out):
    slice_in = data[idx_in]
    actual = reshape_data(data=data, given_dims=given_dims, return_dims=return_dims)
    if isinstance(data, da.core.Array):
        slice_in = slice_in.compute()
        actual = actual.compute()
    np.testing.assert_array_equal(slice_in, actual[idx_out])

    # Check that the output data is the same type as the input
    assert type(actual) == type(slice_in)


@pytest.mark.parametrize("data, given_dims, return_dims, expected_shape", [
    (np.zeros((1, 2, 3, 4, 5, 6)), "STCZYX", "XYZCTS", (6, 5, 4, 3, 2, 1)),
    (da.zeros((1, 2, 3, 4, 5, 6)), "STCZYX", "XYZCTS", (6, 5, 4, 3, 2, 1)),
    (np.zeros((1, 2, 3)), "ZYX", "ZXY", (1, 3, 2)),
    (da.zeros((1, 2, 3)), "ZYX", "ZXY", (1, 3, 2)),
    pytest.param(
        np.zeros((6, 200, 400)),
        "ZYX",
        "TYXC",
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        da.zeros((6, 200, 400)),
        "ZYX",
        "TYXC",
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        np.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZ",
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
    pytest.param(
        da.zeros((6, 200, 400)),
        "ZYX",
        "TYXCZ",
        None,
        marks=pytest.mark.raises(exception=ConflictingArgumentsError)
    ),
])
def test_transpose_to_dims(data, given_dims, return_dims, expected_shape):
    actual = transpose_to_dims(data=data, given_dims=given_dims, return_dims=return_dims)
    assert actual.shape == expected_shape

    # Check that the output data is the same type as the input
    assert type(actual) == type(data)
