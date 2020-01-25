#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from dask.diagnostics import Profiler
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.constants import Dimensions
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader


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
            "s_1_t_1_c_10_z_1.ome.tiff",
            (10, 1736, 1776),
            "CYX",
            0,
            (1, 1736, 1776),
            20  # 10 * 2 = 2
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
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.tiff",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError)
        )
    ]
)
def test_ome_tiff_reader(
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
    img = OmeTiffReader(f, S=select_scene)

    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check basics
    with Profiler() as prof:
        # Check that OME Metadata matches the dask data array shape and dims order
        dim_size_getters = {
            Dimensions.Scene: img.size_s,
            Dimensions.Time: img.size_t,
            Dimensions.Channel: img.size_c,
            Dimensions.SpatialZ: img.size_z,
            Dimensions.SpatialY: img.size_y,
            Dimensions.SpatialX: img.size_x
        }
        for d, getter in dim_size_getters.items():
            if d in expected_dims:
                assert getter() == img.dask_data.shape[img.dims.index(d)]

        assert img.dims == expected_dims
        assert img.is_ome()
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


@pytest.mark.parametrize("filename, scene, expected_channel_names", [
    ("s_1_t_1_c_1_z_1.ome.tiff", 0, ["Bright"]),
    ("s_1_t_1_c_10_z_1.ome.tiff", 0, [f"C:{i}" for i in range(10)]),
    ("s_3_t_1_c_3_z_5.ome.tiff", 0, ["EGFP", "TaRFP", "Bright"]),
    ("s_3_t_1_c_3_z_5.ome.tiff", 1, ["EGFP", "TaRFP", "Bright"]),
    ("s_3_t_1_c_3_z_5.ome.tiff", 2, ["EGFP", "TaRFP", "Bright"]),
    pytest.param("s_1_t_1_c_1_z_1.ome.tiff", 1, None, marks=pytest.mark.raises(exception=IndexError))
])
def test_get_channel_names(resources_dir, filename, scene, expected_channel_names):
    # Get file
    f = resources_dir / filename

    # Read file
    img = OmeTiffReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check channel names
    assert img.get_channel_names(scene) == expected_channel_names

    # Check that there are no open file pointers after check
    assert len(proc.open_files()) == 0


@pytest.mark.parametrize("filename, scene, expected_pixel_sizes", [
    ("s_1_t_1_c_1_z_1.ome.tiff", 0, (1.0833333333333333, 1.0833333333333333, 1.0)),
    ("s_1_t_1_c_10_z_1.ome.tiff", 0, (1.0, 1.0, 1.0)),
    ("s_3_t_1_c_3_z_5.ome.tiff", 0, (1.0833333333333333, 1.0833333333333333, 1.0)),
    ("s_3_t_1_c_3_z_5.ome.tiff", 1, (1.0833333333333333, 1.0833333333333333, 1.0)),
    ("s_3_t_1_c_3_z_5.ome.tiff", 2, (1.0833333333333333, 1.0833333333333333, 1.0)),
    pytest.param("s_1_t_1_c_1_z_1.ome.tiff", 1, None, marks=pytest.mark.raises(exception=IndexError))
])
def test_get_physical_pixel_size(resources_dir, filename, scene, expected_pixel_sizes):
    # Get file
    f = resources_dir / filename

    # Read file
    img = OmeTiffReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert len(proc.open_files()) == 0

    # Check physical pixel sizes
    assert img.get_physical_pixel_size(scene) == expected_pixel_sizes

    # Check that there are no open file pointers after check
    assert len(proc.open_files()) == 0
