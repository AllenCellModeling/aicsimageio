#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter

from ..conftest import array_constructor, get_resource_write_full_path, host


@array_constructor
@pytest.mark.parametrize(
    "write_shape, write_dim_order, read_shape, read_dim_order",
    [
        ((5, 16, 16), None, (5, 16, 16), "ZYX"),
        # ((5, 16, 16), "ZYX", (5, 16, 16), "ZYX"),
        # ((5, 16, 16), "CYX", (5, 16, 16), "CYX"),
        # ((5, 16, 16), "TYX", (5, 16, 16), "TYX"),
        # # Note that files get saved out with RGBA, instead of just RGB
        # ((30, 100, 100, 3), None, (30, 100, 100, 4), "TYXS"),
        # ((100, 30, 100), "XTY", (30, 100, 100), "TYX"),
        # # Note that files get saved out with RGBA, instead of just RGB
        # ((3, 100, 30, 100), "SYTX", (30, 100, 100, 4), "TYXS"),
        # pytest.param(
        #     (1, 1),
        #     None,
        #     None,
        #     None,
        #     marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        # ),
        # pytest.param(
        #     (1, 1, 1, 1, 1),
        #     None,
        #     None,
        #     None,
        #     marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        # ),
        # pytest.param(
        #     (1, 1, 1, 1, 1, 1),
        #     "STCZYX",
        #     None,
        #     None,
        #     marks=pytest.mark.raises(exception=exceptions.UnexpectedShapeError),
        # ),
        # pytest.param(
        #     (1, 1, 1, 1),
        #     "ABCD",
        #     None,
        #     None,
        #     marks=pytest.mark.raises(
        #         exception=exceptions.InvalidDimensionOrderingError
        #     ),
        # ),
    ],
)
@host
@pytest.mark.parametrize("filename", ["e.ome.tiff"])
def test_ome_tiff_writer(
    array_constructor,
    host,
    write_shape,
    write_dim_order,
    read_shape,
    read_dim_order,
    filename,
):
    # Create array
    arr = array_constructor(write_shape, dtype=np.uint8)

    # Construct save end point
    save_uri = get_resource_write_full_path(filename, host)

    # Normal save
    OmeTiffWriter.save(arr, save_uri, write_dim_order)

    # Read written result and check basics
    reader = OmeTiffReader(save_uri)

    # Check basics
    assert reader.shape == read_shape
    assert reader.dims.order == read_dim_order

    # Can't do "easy" testing because compression + shape mismatches on RGB data
