#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO

import numpy as np
import pytest
from psutil import Process

from aicsimageio.readers.lif_reader import LifReader


@pytest.mark.parametrize(
    "filename, "
    "expected_shape, "
    "expected_dims, "
    "expected_dtype, "
    "select_scene, "
    "chunk_dims",
    [
        (
            "s_1_t_1_c_2_z_1.lif",
            (1, 1, 2, 1, 2048, 2048),
            "STCZYX",
            np.uint16,
            0,
            ["Z", "Y", "X"],
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            (1, 4, 2, 1, 614, 614),
            "STCZYX",
            np.uint16,
            0,
            ["Z", "Y", "X"],
        ),
        # To be added back in when rebased off jackson's S3 pr
        (
            "s_14_t_1_c_2_variable_dims.lif",
            (1, 1, 2, 38, 2048, 2048),
            "STCZYX",
            np.uint16,
            0,
            ["Z", "Y", "X"],
        ),
        (
            "s_14_t_1_c_2_variable_dims.lif",
            (1, 1, 2, 52, 2048, 2048),
            "STCZYX",
            np.uint16,
            1,
            ["C", "Y", "X"],
        ),
    ],
)
def test_lif_reader(
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
    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Read file
    img = LifReader(f, chunk_by_dims=chunk_dims, S=select_scene)

    # Check that there are no open file pointers after init
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    assert img.dims == expected_dims
    assert img.metadata
    assert img.dask_data.shape == expected_shape
    assert img.dtype() == expected_dtype

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
        (BytesIO(b"ZInotalifnope"), False),
        (BytesIO(b"ZISRAWFILE"), False),
        (BytesIO(bytearray.fromhex("70000000759c07002a38")), True),
        (BytesIO(b""), False),
        (BytesIO(bytearray.fromhex("700000009f7500002acd3a00003c004c004d0053")), True),
        (BytesIO(bytearray.fromhex("70000000b1c700002ad66300003c004c004d0053")), True),
    ],
)
def test_is_this_type(raw_bytes, expected):
    res = LifReader._is_this_type(raw_bytes)
    assert res == expected


@pytest.mark.parametrize(
    "filename, scene, expected",
    [
        ("s_1_t_1_c_2_z_1.lif", 0, ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"]),
        ("s_1_t_4_c_2_z_1.lif", 0, ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"]),
        (
            "s_14_t_1_c_2_variable_dims.lif",
            0,
            ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"],
        ),
        pytest.param(
            "s_1_t_1_c_2_z_1.lif",
            2,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_get_channel_names(resources_dir, filename, scene, expected):
    f = resources_dir / filename

    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    assert LifReader(resources_dir / filename).get_channel_names(scene) == expected

    # Check that there are no open file pointers
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, scene, expected",
    [
        ("s_1_t_1_c_2_z_1.lif", 0, (3.25e-07, 3.25e-07, 1.0)),
        (
            "s_1_t_4_c_2_z_1.lif",
            0,
            (3.3914910277324634e-07, 3.3914910277324634e-07, 1.0),
        ),
        ("s_14_t_1_c_2_variable_dims.lif", 0, (1.625e-07, 1.625e-07, 1.000715e-06)),
    ],
)
def test_get_physical_pixel_size(resources_dir, filename, scene, expected):
    f = resources_dir / filename

    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    assert LifReader(resources_dir / filename).get_physical_pixel_size(
        scene
    ) == pytest.approx(expected, rel=0.001)

    # Check that there are no open file pointers
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, s, t, c, z, y, x",
    [
        ("s_1_t_1_c_2_z_1.lif", 1, 1, 2, 1, 2048, 2048),
        ("s_1_t_4_c_2_z_1.lif", 1, 4, 2, 1, 614, 614),
    ],
)
def test_size_functions(resources_dir, filename, s, t, c, z, y, x):
    # Get file
    f = resources_dir / filename

    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Init reader
    img = LifReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check sizes
    assert img.size_s() == s
    assert img.size_t() == t
    assert img.size_c() == c
    assert img.size_z() == z
    assert img.size_y() == y
    assert img.size_x() == x

    # Check that there are no open file pointers
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, scene, expected", [("s_1_t_4_c_2_z_1.lif", 0, 51221)]
)
def test_lif_image_data_two(resources_dir, filename, scene, expected):
    f = resources_dir / filename

    # Check that there are no open file pointers
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]
    img = LifReader(f)

    assert img._chunk_offsets[0][0, 0, 0] == expected

    # Check that there are no open file pointers
    assert str(f) not in [f.path for f in proc.open_files()]
