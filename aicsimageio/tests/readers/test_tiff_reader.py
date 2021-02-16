#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import TiffReader

from ..conftest import LOCAL, get_resource_full_path, host
from ..image_container_test_utils import (
    run_image_read_checks,
    run_multi_scene_image_read_checks,
)


@host
@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
        ),
        (
            "s_1_t_1_c_1_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (325, 475),
            np.uint16,
            "YX",
            None,
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (10, 1736, 1776),
            np.uint16,
            "CYX",
            [f"Channel:{i}" for i in range(10)],
        ),
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 325, 475),
            np.uint16,
            "TCYX",
            ["Channel:0", "Channel:1", "Channel:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:0", "Channel:1", "Channel:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:0", "Channel:1", "Channel:2"],
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (5, 3, 325, 475),
            np.uint16,
            "ZCYX",
            ["Channel:0", "Channel:1", "Channel:2"],
        ),
        (
            "s_1_t_1_c_1_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (7548, 7548, 3),
            np.uint16,
            "YXS",  # S stands for samples dimension
            None,
        ),
        (
            # Doesn't affect this test but this is actually an OME-TIFF file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (2, 32, 32, 3),
            np.uint8,
            "CYXS",  # S stands for samples dimension
            ["Channel:0", "Channel:1"],
        ),
        pytest.param(
            "example.txt",
            None,
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
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:1",
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:3",
            None,
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
    expected_dtype,
    expected_dims_order,
    expected_channel_names,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_read_checks(
        ImageContainer=TiffReader,
        uri=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
    )


@host
@pytest.mark.parametrize(
    "filename, "
    "first_scene_id, "
    "first_scene_shape, "
    "second_scene_id, "
    "second_scene_shape",
    [
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            (5, 3, 325, 475),
            "Image:1",
            (5, 3, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (5, 3, 325, 475),
            "Image:2",
            (5, 3, 325, 475),
        ),
    ],
)
def test_multi_scene_tiff_reader(
    filename,
    host,
    first_scene_id,
    first_scene_shape,
    second_scene_id,
    second_scene_shape,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_multi_scene_image_read_checks(
        ImageContainer=TiffReader,
        uri=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.uint16,
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.uint16,
    )


@pytest.mark.parametrize(
    "dims_from_meta, guessed_dims, expected",
    [
        ("QZYX", "CZYX", "CZYX"),
        ("ZQYX", "CZYX", "ZCYX"),
        ("ZYXC", "CZYX", "ZYXC"),
        ("TQQYX", "TCZYX", "TCZYX"),
        ("QTQYX", "TCZYX", "CTZYX"),
        # testing that nothing happens when Q isn't present
        ("LTCYX", "DIMOK", "LTCYX"),
    ],
)
def test_merge_dim_guesses(dims_from_meta, guessed_dims, expected):
    assert TiffReader._merge_dim_guesses(dims_from_meta, guessed_dims) == expected


def test_micromanager_ome_tiff_binary_file():
    # Construct full filepath
    uri = get_resource_full_path(
        "image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos001_000.ome.tif",
        LOCAL,
    )

    # Even though the file name says it is an OME TIFF, this is
    # a binary TIFF file where the actual metadata for all scenes
    # lives in a different image file.
    # (image_stack_tpzc_50tp_2p_5z_3c_512k_1_MMStack_2-Pos000_000.ome.tif)
    # Because of this, we will read "non-main" micromanager files as just
    # normal TIFFs

    # Run image read checks on the first scene
    # (this files binary data)
    run_image_read_checks(
        ImageContainer=TiffReader,
        uri=uri,
        set_scene="Image:0",
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=(50, 5, 3, 256, 256),
        expected_dtype=np.uint16,
        # Note this dimension order is correct but is different from OmeTiffReader
        # because we swap the dimensions into "standard" order
        expected_dims_order="TZCYX",
        expected_channel_names=["Channel:0", "Channel:1", "Channel:2"],
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
    )
