#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader


@pytest.mark.parametrize(
    "filename, expected_shape, expected_dims",
    [
        ("example.bmp", (480, 640, 4), "YXC"),
        ("example.png", (800, 537, 4), "YXC"),
        ("example.jpg", (452, 400, 3), "YXC"),
        ("example.gif", (72, 268, 268, 4), "TYXC"),
        pytest.param(
            "example.txt",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_default_reader(
    resources_dir, filename, expected_shape, expected_dims,
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = DefaultReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.dims == expected_dims
    assert img.metadata
    assert img.dask_data.shape == expected_shape

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check array
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == expected_shape

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "expected_starting_dims, set_dims, expected_ending_dims",
    [
        ("YXC", "ZXC", "ZXC"),
        ("YXC", "YXZ", "YXZ"),
        ("YXC", "ABC", "ABC"),
        pytest.param(
            "YXC",
            "ABCDE",
            None,
            marks=pytest.mark.raises(
                exception=exceptions.InvalidDimensionOrderingError
            ),
        ),
    ],
)
def test_dims_setting(
    resources_dir, expected_starting_dims, set_dims, expected_ending_dims
):
    # Get file
    f = resources_dir / "example.png"

    # Read file
    img = DefaultReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.dims == expected_starting_dims

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Update dims
    img.dims = set_dims

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check expected
    assert img.dims == expected_ending_dims

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]
