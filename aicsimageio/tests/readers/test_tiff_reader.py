#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from dask.diagnostics import Profiler
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.readers.tiff_reader import TiffReader


@pytest.mark.parametrize(
    "filename, "
    "expected_shape, "
    "expected_dims, "
    "select_scene, "
    "expected_chunksize, "
    "expected_task_count",
    [
        # Expected task counts should be each non chunk dimension size multiplied againest each other * 2
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            (325, 475),
            "YX",
            0,
            (325, 475),
            2  # 2 = 2
        ),
        (
            "s_1_t_1_c_1_z_1.tiff",
            (325, 475),
            "YX",
            0,
            (325, 475),
            2  # 2 = 2
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            (10, 1736, 1776),
            "CYX",
            0,
            (1, 1736, 1776),
            20  # 2 = 2
        ),
        (
            "s_1_t_10_c_3_z_1.tiff",
            (10, 3, 325, 475),
            "TCYX",
            0,
            (1, 1, 325, 475),
            60  # 10 * 3 * 2 = 60
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            (3, 5, 3, 325, 475),
            "SZCYX",
            0,
            (1, 1, 1, 325, 475),
            90  # 3 * 5 * 3 * 2 = 90
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError)
        )
    ]
)
def test_tiff_reader(
    resources_dir,
    filename,
    expected_shape,
    expected_dims,
    select_scene,
    expected_chunksize,
    expected_task_count
):
    # Get file
    f = resources_dir / filename

    # Read file
    img = TiffReader(f, S=select_scene)

    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        assert img.dims == expected_dims
        assert img.metadata
        assert img.dask_data.shape == expected_shape
        assert img.dask_data.chunksize == expected_chunksize
        # Check that basic details don't require task computation
        assert len(prof.results) == 0

    # Check that there are no open file pointers after basics
    assert len(proc.open_files()) == 0

    # Check computed type is numpy array, computed shape is expected shape, and task count is expected
    with Profiler() as prof:
        assert isinstance(img.data, np.ndarray)
        assert img.data.shape == expected_shape
        assert len(prof.results) == expected_task_count

    # Check that there are no open file pointers after retrieval
    assert len(proc.open_files()) == 0
