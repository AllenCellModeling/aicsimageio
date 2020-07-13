#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.writers import TimeseriesWriter


@pytest.mark.parametrize(
    "array_creator", [lambda shape: np.random.rand(*shape), da.random.random]
)
@pytest.mark.parametrize(
    "shape, dim_order",
    [
        ((30, 100, 100), None),
        ((30, 100, 100, 3), None),
        ((30, 100, 100, 4), None),
        ((100, 30, 100), "XTY"),
        ((100, 4, 100, 30), "YCXT"),
        pytest.param(
            (1, 1),
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
            (1, 1, 1, 1),
            "ABCD",
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
@pytest.mark.parametrize("filename", ["a.gif", "b.mp4", "c.mkv", "d.avi"])
def test_timeseries_writer(tmpdir, array_creator, shape, dim_order, filename):
    # Create array
    arr = array_creator(shape)

    # Get save location
    save_path = Path(tmpdir) / filename

    # Save
    TimeseriesWriter.save(arr.astype(np.uint8), save_path, dim_order)

    # TODO:
    # Uncomment this test after reader rework

    # Reread
    # img = AICSImage(save_path)
    # if dim_order is None:
    #     data = img.reader.data
    # else:
    #     data = img.reader.get_image_data(dim_order)
    #
    # # Assert that the data is allclose (not exact due to compression)
    # assert da.allclose(arr.astype(np.uint8), data)
