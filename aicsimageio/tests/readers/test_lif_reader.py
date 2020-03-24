#!/usr/bin/env python
# -*- coding: utf-8 -*-

from io import BytesIO

import numpy as np
import pytest
from dask.diagnostics import Profiler
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.lif_reader import LifReader


@pytest.mark.parametrize(
    "filename, "
    "expected_shape, "
    "expected_dims, "
    "expected_dtype, "
    "select_scene, "
    "chunk_dims, "
    "expected_chunksize, "
    "expected_task_count",
    [
        # Expected task counts should be each non chunk dimension size multiplied againest each other * 2
        (
            "s1_t1_c2_z1.lif",
            (1, 1, 2, 1, 2048, 2048),
            "STCZYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
            (1, 1, 1, 1, 2048, 2048),
            2  # 1 * 1 * 2 = 2
        ),
        (
            "s1_t4_c2_z1.lif",
            (1, 4, 2, 1, 614, 614),
            "SCZYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
            (1, 1, 1, 5, 325, 475),
            18  # 1 * 3 * 3 * 2 = 18
        ),
        (
            "s14_t1_c2_z52_inconsistent.lif",
            (1, 3, 3, 5, 2048, 2048),
            "STCZYX",
            np.uint16,
            0,
            ("Z", "Y", "X"),
            (1, 1, 1, 38, 2048, 2048),
            90  # 1 * 3 * 3 * 5 * 2 = 90
        ),
        (
            "s14_t1_c2_z52_inconsistent.lif",
            (14, 1, 2, 52, 2048, 2048),
            "STCZYX",
            np.uint16,
            1,
            ("C", "Y", "X"),
            (1, 1, 3, 1, 2048, 2048),
            30  # 1 * 3 * 5 * 2 = 30
        ),
    ]
)
def test_lif_reader(
    resources_dir,
    filename,
    expected_shape,
    expected_dims,
    expected_dtype,
    select_scene,
    chunk_dims,
    expected_chunksize,
    expected_task_count
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = LifReader(f, chunk_by_dims=chunk_dims, S=select_scene)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check basics
    with Profiler() as prof:
        assert img.dims == expected_dims
        assert img.metadata
        shp = img.dask_data.shape
        assert img.dask_data.shape == expected_shape
        assert img.dask_data.chunksize == expected_chunksize
        assert img.dtype() == expected_dtype
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after retrieval
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize("raw_bytes, expected", [
    (BytesIO(b"ZInotalifnope"), False),
    (BytesIO(b"ZISRAWFILE"), False),
    (BytesIO(bytearray.fromhex('70000000759c07002a38')), True),
    (BytesIO(b""), False),
    (BytesIO(bytearray.fromhex('700000009f7500002acd3a00003c004c004d0053')), True),
    (BytesIO(bytearray.fromhex('70000000b1c700002ad66300003c004c004d0053')), True),
])
def test_is_this_type(raw_bytes, expected):
    res = LifReader._is_this_type(raw_bytes)
    assert res == expected


@pytest.mark.parametrize("filename, scene, expected", [
    ("s1_t1_c2_z1.lif", 0, ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"]),
    ("temp_example.lif", 0, ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"])
    # Our current get channel names doesn't take scene into account
    # pytest.param("s_3_t_1_c_3_z_5.czi", 3, None, marks=pytest.mark.raises(exception=IndexError))
])
def test_get_channel_names(resources_dir, filename, scene, expected):
    assert LifReader(resources_dir / filename).get_channel_names(scene) == expected


@pytest.mark.parametrize("filename, scene, expected", [
    ("s1_t1_c2_z1.lif", 0, (0.325, 0.325, 1.0)),
    ("temp_example.lif", 0, (0.1625, 0.1625, 1.000715))
])
def test_get_physical_pixel_size(resources_dir, filename, scene, expected):
    assert pytest.approx(expected, 0.00001) == LifReader(resources_dir / filename).get_physical_pixel_size(scene)


@pytest.mark.parametrize("filename, s, t, c, z, y, x", [
    ("s_1_t_1_c_1_z_1.czi", 1, 1, 1, 1, 325, 475),
    ("s_3_t_1_c_3_z_5.czi", 3, 1, 3, 5, 325, 475)
])
def test_size_functions(resources_dir, filename, s, t, c, z, y, x):
    # Get file
    f = resources_dir / filename

    # Init reader
    img = LifReader(f)

    # Check sizes
    assert img.size_s() == s
    assert img.size_t() == t
    assert img.size_c() == c
    assert img.size_z() == z
    assert img.size_y() == y
    assert img.size_x() == x
