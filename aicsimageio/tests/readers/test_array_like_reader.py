#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import ArrayLikeReader

from ..image_container_test_utils import run_image_container_checks


@pytest.mark.parametrize(
    "array_constructor",
    [
        np.ones,
        da.ones,
    ],
)
@pytest.mark.parametrize(
    "arr_shape, known_dims, expected_shape, expected_dims, expected_channel_names",
    [
        ((1, 1), None, (1, 1), "YX", None),
        ((1, 1, 1), None, (1, 1, 1), "ZYX", None),
        (
            (1, 1, 1, 1),
            None,
            (1, 1, 1, 1),
            "CZYX",
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1, 1),
            None,
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        ((1, 1, 1), "AYX", (1, 1, 1), "AYX", None),
        (
            (1, 1, 1, 1),
            "ZCYX",
            (1, 1, 1, 1),
            "ZCYX",
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1, 1),
            "ABCYX",
            (1, 1, 1, 1, 1),
            "ABCYX",
            ["Channel:0"],
        ),
        (
            (3, 2, 5, 4, 10, 10),
            "RTCZYX",
            (3, 2, 5, 4, 10, 10),
            "RTCZYX",
            ["Channel:0", "Channel:1", "Channel:2", "Channel:3", "Channel:4"],
        ),
        # Test that without known dims and with more than five dims, it raises an error
        # Our guess dim order only support up to five dims
        pytest.param(
            (1, 2, 3, 4, 5, 6),
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            "hello world",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_arraylike_reader(
    array_constructor,
    arr_shape,
    known_dims,
    expected_shape,
    expected_dims,
    expected_channel_names,
):
    # Construct array
    arr = array_constructor(arr_shape)

    # Init
    image_container = ArrayLikeReader(arr, known_dims=known_dims)

    run_image_container_checks(
        image_container=image_container,
        set_scene="Image:0",
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=expected_shape,
        expected_dtype=np.float64,
        expected_dims_order=expected_dims,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
        expected_metadata_type=type(None),
    )


@pytest.mark.parametrize(
    "known_dims, expected_dims, expected_channels",
    [
        ("ZYX", "ZYX", None),
        ("CYX", "CYX", ["Channel:0", "Channel:1"]),
        ("ABD", "ABD", None),
        pytest.param(
            "ABCDEFG",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
def test_dims_setting(known_dims, expected_dims, expected_channels):
    # Read file
    image_container = ArrayLikeReader(da.ones((2, 2, 2)), known_dims=known_dims)

    # Check basics
    assert image_container.dims.order == expected_dims
    assert image_container.channel_names == expected_channels
