#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Callable, List, Literal, Mapping, Optional, Tuple, Union

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from aicsimageio import AICSImage, types
from aicsimageio.exceptions import ConflictingArgumentsError, UnexpectedShapeError
from aicsimageio.readers import ArrayLikeReader
from aicsimageio.transforms import (
    generate_stack,
    reduce_to_slice,
    reshape_data,
    transpose_to_dims,
)


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
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            (6, 200, 400),
            "ZYX",
            "TYXCZX",
            {"Z": 7},
            None,
            marks=pytest.mark.raises(exception=IndexError),
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


def get_data_reference(
    mode: str,
    mismatch: str,
    scene_character: str,
    scene_coord_values: str,
    select_scenes: Optional[tuple[Union[int, str], ...]] = None,
) -> tuple[list[types.MetaArrayLike], types.MetaArrayLike]:

    # set up the core values for the dataset
    shape = (3, 4, 5, 6, 7, 8)
    scene_name_to_idx = dict(
        zip([f"Image:{i}" for i in range(shape[0])], range(shape[0]))
    )
    data = np.arange(np.prod(shape), dtype="uint16").reshape(shape)

    # convert to dask if necessary
    if "dask" in mode:
        data = da.from_array(data)

    # split into list for image container construction
    data = [_ for _ in data]

    # mess with a scene for tests
    if mismatch == "shape":
        data[1] = data[1][:-1]
    elif mismatch == "dtype":
        data[1] = data[1].astype(int)

    # select scenes from argumnet or to make valid call to stack
    if select_scenes is None:
        reference = (
            np.stack(data) if mismatch == "none" else np.stack([data[0], data[2]])
        )
    else:
        if isinstance(select_scenes[0], int):
            reference = np.stack([data[i] for i in select_scenes])
        elif isinstance(select_scenes[0], str):
            inds = (scene_name_to_idx[str(s)] for s in select_scenes)
            reference = np.stack([data[i] for i in inds])
    # Assign coords and dims for xarray types
    if "xarray" in mode:
        data = [xr.DataArray(x, dims=list("TCZYX")) for x in data]
        coords = {"C": [f"Channel:0:{i}" for i in range(shape[2])]}
        if scene_coord_values == "names":
            if mismatch == "none":
                coords[scene_character] = [f"Image:{i}" for i in range(3)]
            else:
                coords[scene_character] = [f"Image:{i}" for i in (0, 2)]
        reference = xr.DataArray(
            reference, dims=list((scene_character, *"TCZYX")), coords=coords
        )

    return data, reference


@pytest.mark.parametrize("image_container", [AICSImage, ArrayLikeReader])
@pytest.mark.parametrize(
    "mode, scene_character, scene_coord_values",
    [
        ("data", "I", "index"),
        ("data", "U", "names"),
        ("dask_data", "I", "index"),
        ("xarray_data", "I", "index"),
        ("xarray_data", "U", "index"),
        ("xarray_data", "U", "names"),
        pytest.param(
            "xarray_data", "T", "index", marks=pytest.mark.raises(exception=ValueError)
        ),
        ("xarray_dask_data", "I", "index"),
        ("xarray_dask_data", "U", "index"),
        ("xarray_dask_data", "U", "names"),
        pytest.param(
            "xarray_dask_data",
            "T",
            "index",
            marks=pytest.mark.raises(exception=ValueError),
        ),
    ],
)
def test_generate_stack_stacking(
    image_container: Any,
    mode: Literal["data", "dask_data", "xarray_data", "xarray_dask_data"],
    scene_character: str,
    scene_coord_values: str,
) -> None:
    data, reference = get_data_reference(
        mode, "none", scene_character, scene_coord_values
    )
    container = image_container(data)
    stack = generate_stack(
        container, mode, False, None, scene_character, scene_coord_values
    )
    if "xarray" in mode:
        xr.testing.assert_allclose(stack, reference)
    else:
        np.testing.assert_allclose(stack, reference)


@pytest.mark.parametrize("image_container", [AICSImage, ArrayLikeReader])
@pytest.mark.parametrize(
    "mode", ["data", "xarray_data", "dask_data", "xarray_dask_data"]
)
@pytest.mark.parametrize(
    "mismatch, drop_non_matching_scenes, select_scenes",
    [
        ("none", True, None),
        ("shape", True, None),
        ("dtype", True, None),
        ("none", False, (0, 2)),
        ("shape", False, (0, 2)),
        ("dtype", False, (0, 2)),
        ("none", False, tuple(f"Image:{i}" for i in [0, 2])),
        ("shape", False, tuple(f"Image:{i}" for i in [0, 2])),
        ("dtype", False, tuple(f"Image:{i}" for i in [0, 2])),
        pytest.param(
            "shape",
            False,
            None,
            marks=pytest.mark.raises(exception=UnexpectedShapeError),
        ),
        pytest.param(
            "dtype", False, None, marks=pytest.mark.raises(exception=TypeError)
        ),
    ],
)
def test_generate_stack_mismatch_and_drop(
    # Test dropping and catching mismatched images
    image_container: Any,
    mode: Literal["data", "dask_data", "xarray_data", "xarray_dask_data"],
    mismatch: str,
    drop_non_matching_scenes: bool,
    select_scenes: Optional[tuple[Union[int, str], ...]],
) -> None:
    data, reference = get_data_reference(mode, mismatch, "I", "index", select_scenes)
    container = image_container(data)

    stack = generate_stack(
        container, mode, drop_non_matching_scenes, select_scenes, "I", "index"
    )
    if "xarray" in mode:
        xr.testing.assert_allclose(stack, reference)
    else:
        np.testing.assert_allclose(stack, reference)


@pytest.mark.parametrize(
    "list_to_test, expected",
    [
        ([0], slice(0, 1, None)),
        ([4], slice(4, 5, None)),
        ([0, 1], slice(0, 2, 1)),
        ([1, 0], [1, 0]),
        ([3, 5], slice(3, 6, 2)),
        ([8, 9, 11], [8, 9, 11]),
        ([15, 20, 25], slice(15, 26, 5)),
        ((0,), slice(0, 1, None)),
        ((0, 1), slice(0, 2, 1)),
        ((1, 0), (1, 0)),
        ((3, 5), slice(3, 6, 2)),
        ((8, 9, 11), (8, 9, 11)),
        ((15, 20, 25), slice(15, 26, 5)),
    ],
)
def test_convert_list_to_slice(
    list_to_test: Union[List, Tuple], expected: Union[int, List, slice, Tuple]
) -> None:
    assert reduce_to_slice(list_to_test) == expected
