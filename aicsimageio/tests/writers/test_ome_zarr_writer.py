#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import shutil

from ome_zarr.reader import Reader
from ome_zarr.io import parse_url

from aicsimageio import exceptions
from aicsimageio.writers import OmeZarrWriter

from ..conftest import LOCAL, array_constructor, get_resource_write_full_path


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, expected_read_shape, expected_read_dim_order",
    [
        ((5, 16, 16), None, (5, 16, 16), "ZYX"),
        ((5, 16, 16), "ZYX", (5, 16, 16), "ZYX"),
        ((5, 16, 16), "CYX", (5, 16, 16), "CYX"),
        pytest.param(
            (10, 5, 16, 16),
            "ZCYX",
            (10, 5, 16, 16),
            "ZCYX",
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((5, 10, 16, 16), "CZYX", (5, 10, 16, 16), "CZYX"),
        ((16, 16), "YX", (16, 16), "YX"),
        pytest.param(
            (2, 3, 3),
            "AYX",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((2, 3, 3), "YXZ", (2, 3, 3), "YXZ"),
        pytest.param(
            (2, 5, 16, 16),
            "CYX",
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
        ((1, 2, 3, 4, 5), None, (1, 2, 3, 4, 5), "TCZYX"),
        ((2, 3, 4, 5, 6), "TCZYX", (2, 3, 4, 5, 6), "TCZYX"),
        ((2, 3, 4, 5, 6), None, (2, 3, 4, 5, 6), "TCZYX"),
        # error 6D data doesn't work yet
        pytest.param(
            (1, 2, 3, 4, 5, 3),
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["e.zarr"])
def test_ome_zarr_writer_no_meta(
    array_constructor: Callable,
    write_shape: Tuple[int, ...],
    write_dim_order: Optional[str],
    expected_read_shape: Tuple[int, ...],
    expected_read_dim_order: str,
    filename: str,
) -> None:
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, LOCAL)
    # clear out anything left over
    shutil.rmtree(save_uri, ignore_errors=True)

    # Normal save
    writer = OmeZarrWriter(save_uri)
    writer.write_image(arr, "", None, None, None, dimension_order=write_dim_order)

    # Read written result and check basics
    reader = Reader(parse_url(save_uri))
    node = list(reader())[0]
    num_levels = len(node.data)
    assert num_levels == 1
    level = 0
    shape = node.data[level].shape
    assert shape == expected_read_shape
    axes = node.metadata["axes"]
    dims = "".join([a["name"] for a in axes]).upper()
    assert dims == expected_read_dim_order
