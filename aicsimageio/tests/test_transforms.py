#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, List, Mapping, Tuple, Union

import dask.array as da
import numpy as np
import pytest

from aicsimageio import types
from aicsimageio.exceptions import ConflictingArgumentsError
from aicsimageio.transforms import reshape_data, transpose_to_dims


@pytest.mark.parametrize("array_maker", [np.zeros, da.zeros])
@pytest.mark.parametrize(
    "data_shape, given_dims, return_dims, other_args, expected_shape",
    [
        (
            (10, 1, 5, 6, 200, 400),
            "STCZYX",
            "CSZYX",
            {},
            (5, 10, 6, 200, 400),
        ),
        ((6, 200, 400), "ZYX", "STCZYX", {}, (1, 1, 1, 6, 200, 400)),
        ((6, 200, 400), "ZYX", "ZCYSXT", {}, (6, 1, 200, 1, 400, 1)),
        ((6, 200, 400), "ZYX", "CYSXT", {"Z": 2}, (1, 200, 1, 400, 1)),
        (
            (6, 200, 400),
            "ZYX",
            "ZCYSXT",
            {"Z": [0, 1]},
            (2, 1, 200, 1, 400, 1),
        ),
        (
            (6, 200, 400),
            "ZYX",
            "ZCYSXT",
            {"Z": (0, 1)},
            (2, 1, 200, 1, 400, 1),
        ),
        (
            (6, 200, 400),
            "ZYX",
            "ZCYSXT",
            {"Z": range(2)},
            (2, 1, 200, 1, 400, 1),
        ),
        (
            (6, 200, 400),
            "ZYX",
            "ZCYSXT",
            {"Z": slice(0, 2, 1)},
            (2, 1, 200, 1, 400, 1),
        ),
        (
            (6, 200, 200, 3),
            "ZYXS",
            "YX",
            {"Z": 0, "S": 0},
            (200, 200),
        ),
        ((2, 2, 2), "ABI", "ZCYSXT", {}, (1, 1, 1, 1, 1, 1)),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXC",
            {"Z": 7},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZ",
            {"Z": 7},
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": 7},
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCX",
            {"Z": [0, 1, 4]},
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": [0, 1, 7]},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": (0, 1, 7)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": range(7)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": slice(0, 7, 2)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": [0, 1, -7]},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": (0, 1, -7)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": range(0, -8, -1)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": slice(-7, 0, 2)},
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_reshape_data_shape(
    array_maker: Callable,
    data_shape: Tuple[int, ...],
    given_dims: str,
    return_dims: str,
    other_args: Any,
    expected_shape: Tuple[int, ...],
) -> None:
    data = array_maker(data_shape)

    actual = reshape_data(
        data=data, given_dims=given_dims, return_dims=return_dims, **other_args
    )
    assert actual.shape == expected_shape

    # Check that the output data is the same type as the input
    assert type(actual) == type(data)


@pytest.mark.parametrize(
    "data, given_dims, return_dims, idx_in, idx_out",
    [
        (
            np.random.rand(10, 1, 5, 6, 200, 400),
            "STCZYX",
            "CSZYX",
            (5, 0, 3, 3, ...),
            (3, 5, 3, ...),
        ),
        (
            da.random.random((10, 1, 5, 6, 200, 400)),
            "STCZYX",
            "CSZYX",
            (5, 0, 3, 3, ...),
            (3, 5, 3, ...),
        ),
        (
            np.zeros((6, 200, 400)),
            "ZYX",
            "STCZYX",
            (..., 100, 200),
            (0, 0, 0, ..., 100, 200),
        ),
        (
            da.zeros((6, 200, 400)),
            "ZYX",
            "STCZYX",
            (..., 100, 200),
            (0, 0, 0, ..., 100, 200),
        ),
        (
            np.zeros((6, 200, 400)),
            "ZYX",
            "ZCYSXT",
            (3, 100, ...),
            (3, 0, 100, 0, ..., 0),
        ),
        (
            da.zeros((6, 200, 400)),
            "ZYX",
            "ZCYSXT",
            (3, 100, ...),
            (3, 0, 100, 0, ..., 0),
        ),
    ],
)
def test_reshape_data_values(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
    idx_in: Tuple[int, ...],
    idx_out: Tuple[int, ...],
) -> None:
    slice_in = data[idx_in]
    actual = reshape_data(data=data, given_dims=given_dims, return_dims=return_dims)

    # Handle dask vs numpy
    if isinstance(actual, da.core.Array):
        slice_in_computed = slice_in.compute()
        actual_computed = actual.compute()
    else:
        slice_in_computed = slice_in
        actual_computed = actual

    np.testing.assert_array_equal(slice_in_computed, actual_computed[idx_out])

    # Check that the output data is the same type as the input
    assert type(actual) == type(slice_in)


# Arrays used for value checking on kwarg provided reshape_data
NP_ONES = np.ones((10, 10))
TEST_NDARRAY = np.stack([NP_ONES * i for i in range(7)])
DA_ONES = da.ones((10, 10))
TEST_DARRAY = da.stack([DA_ONES * i for i in range(7)])


@pytest.mark.parametrize("data", [TEST_NDARRAY, TEST_DARRAY])
@pytest.mark.parametrize(
    "given_dims, return_dims, other_args, getitem_ops_for_expected, transposer",
    [
        # Just dimension selection
        ("ZYX", "YX", {}, 0, None),
        ("ZYX", "YX", {"Z": 1}, 1, None),
        ("ZYX", "ZYX", {"Z": [0, 1]}, [0, 1], None),
        ("ZYX", "ZYX", {"Z": (0, 1)}, [0, 1], None),
        ("ZYX", "ZYX", {"Z": [0, -1]}, [0, -1], None),
        ("ZYX", "ZYX", {"Z": (0, -1)}, [0, -1], None),
        ("ZYX", "ZYX", {"Z": range(2)}, [0, 1], None),
        ("ZYX", "ZYX", {"Z": range(0, 6, 2)}, [0, 2, 4], None),
        ("ZYX", "ZYX", {"Z": slice(0, 6, 2)}, [0, 2, 4], None),
        ("ZYX", "ZYX", {"Z": slice(6, 3, -1)}, [6, 5, 4], None),
        ("ZYX", "ZYX", {"Z": slice(-1, 3, -1)}, [6, 5, 4], None),
        # Dimension selection and order swap
        (
            "ZYX",
            "YXZ",
            {"Z": (0, -1)},
            [0, -1],
            (1, 2, 0),
        ),
        (
            "ZYX",
            "YXZ",
            {"Z": range(0, 6, 2)},
            [0, 2, 4],
            (1, 2, 0),
        ),
    ],
)
def test_reshape_data_kwargs_values(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
    other_args: Mapping[str, Union[int, List[int], range, slice]],
    getitem_ops_for_expected: List[int],
    transposer: Tuple[int],
) -> None:
    actual = reshape_data(
        data=data,
        given_dims=given_dims,
        return_dims=return_dims,
        **other_args,
    )

    expected = data[getitem_ops_for_expected]

    if transposer is not None:
        if isinstance(data, np.ndarray):
            expected = np.transpose(expected, transposer)
        else:
            expected = da.transpose(expected, transposer)

    # Check that the output data is the same type as the input
    assert type(actual) == type(expected)

    if isinstance(actual, da.core.Array):
        actual = actual.compute()
        expected = expected.compute()

    # Check actual data
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "data, given_dims, return_dims, expected_shape",
    [
        (np.zeros((1, 2, 3, 4, 5, 6)), "STCZYX", "XYZCTS", (6, 5, 4, 3, 2, 1)),
        (da.zeros((1, 2, 3, 4, 5, 6)), "STCZYX", "XYZCTS", (6, 5, 4, 3, 2, 1)),
        (np.zeros((1, 2, 3)), "ZYX", "ZXY", (1, 3, 2)),
        (da.zeros((1, 2, 3)), "ZYX", "ZXY", (1, 3, 2)),
        pytest.param(
            np.zeros((6, 200, 400)),
            "ZYX",
            "TYXC",
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            da.zeros((6, 200, 400)),
            "ZYX",
            "TYXC",
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            np.zeros((6, 200, 400)),
            "ZYX",
            "TYXCZ",
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
        pytest.param(
            da.zeros((6, 200, 400)),
            "ZYX",
            "TYXCZ",
            None,
            marks=pytest.mark.raises(exception=ConflictingArgumentsError),
        ),
    ],
)
def test_transpose_to_dims(
    data: types.ArrayLike,
    given_dims: str,
    return_dims: str,
    expected_shape: Tuple[int, ...],
) -> None:
    actual = transpose_to_dims(
        data=data, given_dims=given_dims, return_dims=return_dims
    )
    assert actual.shape == expected_shape

    # Check that the output data is the same type as the input
    assert type(actual) == type(data)
