#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.tiff_reader import TiffReader


@pytest.mark.parametrize(
    "filename, " "expected_shape, " "expected_dims, " "expected_dtype, " "select_scene",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            (325, 475),
            "YX",
            np.uint16,
            0,
        ),
        (
            "s_1_t_1_c_1_z_1.tiff",
            (325, 475),
            "YX",
            np.uint16,
            0,
        ),
        (
            "s_1_t_1_c_3_z_1_RGB.tiff",
            (7548, 7548, 3),
            "YXC",
            np.uint16,
            0,
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            (10, 1736, 1776),
            "CYX",
            np.uint16,
            0,
        ),
        (
            "s_1_t_10_c_3_z_1.tiff",
            (10, 3, 325, 475),
            "TCYX",
            np.uint16,
            0,
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            (3, 5, 3, 325, 475),
            "SZCYX",
            np.uint16,
            0,
        ),
        pytest.param(
            "s_1_t_1_c_6_z_1_RGB.tiff",
            (2, 32, 32, 3),
            "CYXC",
            np.uint8,
            0,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_tiff_reader(
    resources_dir,
    filename,
    expected_shape,
    expected_dims,
    expected_dtype,
    select_scene,
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = TiffReader(f, S=select_scene)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.dims == expected_dims
    assert img.dtype() == expected_dtype
    assert img.metadata
    assert img.shape == expected_shape
    assert img.size(expected_dims) == expected_shape

    # Will error because those dimensions don't exist in the file
    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        assert img.size("ABCDEFG") == expected_shape

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check arrays
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == expected_shape

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "expected_starting_dims, set_dims, expected_ending_dims",
    [
        ("YX", "XY", "XY"),
        pytest.param(
            "YX",
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
    f = resources_dir / "s_1_t_1_c_1_z_1.tiff"

    # Read file
    img = TiffReader(f)

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

    # Check expected dims
    assert img.dims == expected_ending_dims

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]
