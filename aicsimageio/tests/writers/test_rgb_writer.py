#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.writers import RGBWriter


@pytest.mark.parametrize(
    "array_creator", [lambda shape: np.random.rand(*shape), da.random.random]
)
@pytest.mark.parametrize(
    "shape, dim_order",
    [
        ((100, 100, 3), None),
        ((100, 100), None),
        ((100, 100), "XY"),
        ((3, 100, 100), "CYX"),
        ((100, 3, 100), "XCY"),
        pytest.param(
            (1, 1, 1, 1),
            None,
            marks=pytest.mark.raises(exception=exceptions.InconsistentShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1),
            None,
            marks=pytest.mark.raises(exception=exceptions.InconsistentShapeError),
        ),
        pytest.param(
            (1, 1, 1, 1, 1, 1),
            "STCZYX",
            marks=pytest.mark.raises(exception=exceptions.InconsistentShapeError),
        ),
        pytest.param(
            (1, 1),
            "AB",
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["a.png", "b.jpg", "c.jpeg", "d.bmp"])
def test_rgb_writer(tmpdir, array_creator, shape, dim_order, filename):
    # Create array
    arr = array_creator(shape)

    # Get save location
    save_path = Path(tmpdir) / filename

    # Save
    RGBWriter.save(arr, save_path, dim_order)
