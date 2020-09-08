#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO

import numpy as np
import pytest
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.czi_reader import CziReader


@pytest.mark.parametrize(
    "filename, "
    "expected_shape, "
    "expected_dims, "
    "expected_dtype, "
    "select_scene, "
    "chunk_dims",
    [
        (
            "s_1_t_1_c_1_z_1.czi",
            (1, 1, 325, 475),
            "BCYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            (1, 3, 3, 5, 325, 475),
            "BSCZYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            (1, 3, 3, 5, 325, 475),
            "BSCZYX",
            np.uint16,
            0,
            ("Y", "X"),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            (1, 3, 3, 5, 325, 475),
            "BSCZYX",
            np.uint16,
            0,
            ("C", "Y", "X"),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            (1, 3, 3, 5, 325, 475),
            "BSCZYX",
            np.uint16,
            0,
            ("S", "Y", "X"),
        ),
        # Check that Spatial Y and Spatial X dims are always added to chunk dims
        (
            "s_3_t_1_c_3_z_5.czi",
            (1, 3, 3, 5, 325, 475),
            "BSCZYX",
            np.uint16,
            0,
            ("S"),
        ),
        (
            "variable_per_scene_dims.czi",
            (1, 1, 2, 1, 2, 1248, 1848),
            "BSTCZYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
        ),
        (
            "variable_per_scene_dims.czi",
            (1, 1, 1, 1, 2, 1248, 1848),
            "BSTCZYX",
            np.uint16,
            1,
            ("Z", "Y", "X"),
        ),
        pytest.param(
            "variable_per_scene_dims.czi",
            (1, 1, 1, 1, 2, 1248, 1848),
            "BSTCZYX",
            np.uint16,
            2,
            ("Z", "Y", "X"),
            marks=pytest.mark.raises(exception=exceptions.InconsistentShapeError),
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_czi_reader(
    resources_dir,
    filename,
    expected_shape,
    expected_dims,
    expected_dtype,
    select_scene,
    chunk_dims,
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = CziReader(f, chunk_by_dims=chunk_dims, S=select_scene)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.dims == expected_dims
    assert img.metadata is not None
    assert img.shape == expected_shape
    assert img.dask_data.shape == expected_shape
    assert img.size(expected_dims) == expected_shape
    assert img.dtype() == expected_dtype

    # Will error because those dimensions don't exist in the file
    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        assert img.size("ABCDEFG") == expected_shape

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check array
    assert isinstance(img.data, np.ndarray)
    assert img.data.shape == expected_shape

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "raw_bytes, expected",
    [
        (BytesIO(b"ZInotaczinope"), False),
        (BytesIO(b"ZISRAWFILE"), True),
        (BytesIO(b"ZI"), False),
        (BytesIO(b""), False),
    ],
)
def test_is_this_type(raw_bytes, expected):
    res = CziReader._is_this_type(raw_bytes)
    assert res == expected


@pytest.mark.parametrize(
    "filename, scene, expected",
    [
        ("s_1_t_1_c_1_z_1.czi", 0, ["Bright"]),
        ("s_3_t_1_c_3_z_5.czi", 0, ["EGFP", "TaRFP", "Bright"]),
        ("s_3_t_1_c_3_z_5.czi", 1, ["EGFP", "TaRFP", "Bright"]),
        ("s_3_t_1_c_3_z_5.czi", 2, ["EGFP", "TaRFP", "Bright"]),
        # Our current get channel names doesn't take scene into account
        # pytest.param(
        #     "s_3_t_1_c_3_z_5.czi",
        #     3,
        #     None,
        #     marks=pytest.mark.raises(exception=IndexError)
        # ),
    ],
)
def test_get_channel_names(resources_dir, filename, scene, expected):
    assert CziReader(resources_dir / filename).get_channel_names(scene) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("s_1_t_1_c_1_z_1.czi", (1.0833333333333333e-06, 1.0833333333333333e-06, 1.0)),
        (
            "s_3_t_1_c_3_z_5.czi",
            (1.0833333333333333e-06, 1.0833333333333333e-06, 1e-06),
        ),
    ],
)
def test_get_physical_pixel_size(resources_dir, filename, expected):
    assert CziReader(resources_dir / filename).get_physical_pixel_size() == expected


@pytest.mark.parametrize(
    "filename, s, t, c, z, y, x",
    [
        ("s_1_t_1_c_1_z_1.czi", 1, 1, 1, 1, 325, 475),
        ("s_3_t_1_c_3_z_5.czi", 3, 1, 3, 5, 325, 475),
    ],
)
def test_size_functions(resources_dir, filename, s, t, c, z, y, x):
    # Get file
    f = resources_dir / filename

    # Init reader
    img = CziReader(f)

    # Check sizes
    assert img.size_s() == s
    assert img.size_t() == t
    assert img.size_c() == c
    assert img.size_z() == z
    assert img.size_y() == y
    assert img.size_x() == x
