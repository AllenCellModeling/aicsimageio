#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader
from aicsimageio.writers.two_d_writer import TwoDWriter

from ...conftest import LOCAL, array_constructor, get_resource_write_full_path


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape, read_dim_order",
    [
        ((100, 100, 3), None, (100, 100, 3), "YXS"),
        ((100, 100), None, (100, 100), "YX"),
        ((100, 100), "XY", (100, 100), "YX"),
        ((3, 100, 100), "SYX", (100, 100, 3), "YXS"),
        ((100, 3, 100), "XSY", (100, 100, 3), "YXS"),
        pytest.param(
            (1, 1, 1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        ),
        pytest.param(
            (1, 1),
            "AB",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["a.png", "d.bmp"])
def test_two_d_writer(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: str,
    read_shape: Tuple[int, ...],
    read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)

    # Save
    TwoDWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    reader = DefaultReader(save_uri)

    # Check basics
    assert reader.shape == read_shape
    assert reader.dims.order == read_dim_order

    # We want to check the arrays equal
    # but remember, the reader returns data in standard read order
    # so we need to get it back as write order
    np.testing.assert_array_equal(arr, reader.get_image_data(write_dim_order))
