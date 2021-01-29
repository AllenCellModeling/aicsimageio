#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import ArrayLikeReader

from ..image_container_test_utils import check_can_serialize_image_container


@pytest.mark.parametrize(
    "array_constructor",
    [
        np.ones,
        da.ones,
    ],
)
@pytest.mark.parametrize(
    "arr_shape, expected_shape, expected_dims, expected_channel_names",
    [
        ((1, 1), (1, 1), "YX", None),
        ((1, 1, 1), (1, 1, 1), "ZYX", None),
        ((1, 1, 1, 1), (1, 1, 1, 1), "CZYX", ["Channel:0"]),
    ],
)
def test_arraylike_reader(
    array_constructor, arr_shape, expected_shape, expected_dims, expected_channel_names
):
    # Construct array
    arr = array_constructor(arr_shape)

    # Init
    image_container = ArrayLikeReader(arr)

    check_can_serialize_image_container(image_container)

    # Set scene
    image_container.set_scene("Image:0")

    # Check scene info
    assert image_container.scenes == ("Image:0",)
    assert image_container.current_scene == "Image:0"

    # Check basics
    assert image_container.shape == expected_shape
    assert image_container.dtype == np.float64
    assert image_container.dims.order == expected_dims
    assert image_container.dims.shape == expected_shape
    assert image_container.metadata is None
    assert image_container.channel_names == expected_channel_names
    assert image_container.physical_pixel_sizes == (1.0, 1.0, 1.0)

    # Read only a chunk, then read a chunk from the in-memory, compare
    np.testing.assert_array_equal(
        image_container.get_image_dask_data("YX").compute(),
        image_container.get_image_data("YX"),
    )

    # Check that the shape and dtype are expected after reading in full
    assert image_container.data.shape == expected_shape
    assert image_container.data.dtype == np.float64

    check_can_serialize_image_container(image_container)

    return image_container


def test_invalid_type():
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        ArrayLikeReader("hello world")


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
