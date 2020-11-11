#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import TiffReader

from ..conftest import LOCAL, REMOTE, get_resource_full_path
from .reader_test_utils import run_image_read_checks


@pytest.mark.parametrize("host", [LOCAL, REMOTE])
@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dims_order, "
    "expected_channel_names",
    [
        ("s_1_t_1_c_1_z_1.ome.tiff", 0, (0,), (325, 475), "YX", None),
        ("s_1_t_1_c_1_z_1.tiff", 0, (0,), (325, 475), "YX", None),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            0,
            (0,),
            (10, 1736, 1776),
            "CYX",
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        ),
        ("s_1_t_10_c_3_z_1.tiff", 0, (0,), (10, 3, 325, 475), "TCYX", ["0", "1", "2"]),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            0,
            (0, 1, 2),
            (5, 3, 325, 475),
            "ZCYX",
            ["0", "1", "2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            1,
            (0, 1, 2),
            (5, 3, 325, 475),
            "ZCYX",
            ["0", "1", "2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            2,
            (0, 1, 2),
            (5, 3, 325, 475),
            "ZCYX",
            ["0", "1", "2"],
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
        pytest.param(
            "s_1_t_1_c_2_z_1.lif",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            1,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.ome.tiff",
            3,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_tiff_reader(
    filename,
    host,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dims_order,
    expected_channel_names,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_read_checks(
        ReaderClass=TiffReader,
        uri=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=np.uint16,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
    )
