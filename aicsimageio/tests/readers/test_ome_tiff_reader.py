#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from psutil import Process

from aicsimageio import exceptions
from aicsimageio.constants import Dimensions
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader

from .utils import run_image_read_checks


@pytest.mark.parametrize(
    "filename, " "expected_shape, " "expected_dims, " "select_scene",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", (325, 475), "YX", 0),
        ("s_1_t_1_c_10_z_1.ome.tiff", (10, 1736, 1776), "CYX", 0,),
        ("s_3_t_1_c_3_z_5.ome.tiff", (3, 5, 3, 325, 475), "SZCYX", 0,),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.tiff",
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_ome_tiff_reader(
    resources_dir, filename, expected_shape, expected_dims, select_scene,
):
    reader = run_image_read_checks(
        ReaderClass=OmeTiffReader,
        resources_dir=resources_dir,
        filename=filename,
        chunk_dims=None,
        select_scene=select_scene,
        expected_shape=expected_shape,
        expected_dims=expected_dims,
        expected_dtype=np.uint16,
    )

    # Check that OME Metadata matches the dask data array shape and dims order
    dim_sizes = {
        Dimensions.Scene: reader.size_s,
        Dimensions.Time: reader.size_t,
        Dimensions.Channel: reader.size_c,
        Dimensions.SpatialZ: reader.size_z,
        Dimensions.SpatialY: reader.size_y,
        Dimensions.SpatialX: reader.size_x,
    }
    for d, val in dim_sizes.items():
        if d in expected_dims:
            assert val == reader.dask_data.shape[reader.dims.index(d)]

    assert reader.is_ome()


@pytest.mark.parametrize(
    "filename, scene, expected_channel_names",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0, ["Bright"]),
        ("s_1_t_1_c_10_z_1.ome.tiff", 0, [f"C:{i}" for i in range(10)]),
        ("s_3_t_1_c_3_z_5.ome.tiff", 0, ["EGFP", "TaRFP", "Bright"]),
        ("s_3_t_1_c_3_z_5.ome.tiff", 1, ["EGFP", "TaRFP", "Bright"]),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2, ["EGFP", "TaRFP", "Bright"]),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            1,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_get_channel_names(resources_dir, filename, scene, expected_channel_names):
    # Get file
    f = resources_dir / filename

    # Read file
    img = OmeTiffReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check channel names
    assert img.get_channel_names(scene) == expected_channel_names

    # Check that there are no open file pointers after check
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, scene, expected_pixel_sizes",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0, (1.0833333333333333, 1.0833333333333333, 1.0)),
        ("s_1_t_1_c_10_z_1.ome.tiff", 0, (1.0, 1.0, 1.0)),
        ("s_3_t_1_c_3_z_5.ome.tiff", 0, (1.0833333333333333, 1.0833333333333333, 1.0)),
        ("s_3_t_1_c_3_z_5.ome.tiff", 1, (1.0833333333333333, 1.0833333333333333, 1.0)),
        ("s_3_t_1_c_3_z_5.ome.tiff", 2, (1.0833333333333333, 1.0833333333333333, 1.0)),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            1,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_get_physical_pixel_size(resources_dir, filename, scene, expected_pixel_sizes):
    # Get file
    f = resources_dir / filename

    # Read file
    img = OmeTiffReader(f)

    # Check that there are no open file pointers after init
    proc = Process()
    assert str(f) not in [f.path for f in proc.open_files()]

    # Check physical pixel sizes
    assert img.get_physical_pixel_size(scene) == expected_pixel_sizes

    # Check that there are no open file pointers after check
    assert str(f) not in [f.path for f in proc.open_files()]


@pytest.mark.parametrize(
    "filename, s, t, c, z, y, x",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 1, 1, 1, 1, 325, 475),
        ("s_1_t_1_c_10_z_1.ome.tiff", 1, 1, 10, 1, 1736, 1776),
        ("s_3_t_1_c_3_z_5.ome.tiff", 3, 1, 3, 5, 325, 475),
    ],
)
def test_size_functions(resources_dir, filename, s, t, c, z, y, x):
    # Get file
    f = resources_dir / filename

    # Init reader
    img = OmeTiffReader(f)

    # Check sizes
    assert img.size_s == s
    assert img.size_t == t
    assert img.size_c == c
    assert img.size_z == z
    assert img.size_y == y
    assert img.size_x == x
