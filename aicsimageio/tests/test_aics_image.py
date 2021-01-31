#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dask.array as da
import numpy as np
import pytest
from ome_types import OME

from aicsimageio import AICSImage, dimensions, exceptions

from .conftest import LOCAL, get_resource_full_path
from .image_container_test_utils import (
    run_image_container_checks,
    run_image_file_checks,
    run_multi_scene_image_read_checks,
)

###############################################################################

# We only run checks on a subset of local files
# The base reader unit tests show that we can handle both local or remote
# If we handled them here, tests would just take longer than they already do

###############################################################################


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes, "
    "expected_metadata_type",
    [
        #######################################################################
        # DefaultReader
        (
            "example.png",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 800, 537, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        (
            "example.gif",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 268, 268, 4),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        (
            "example_valid_frame_count.mp4",
            "Image:0",
            ("Image:0",),
            (72, 1, 1, 272, 272, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
            (1.0, 1.0, 1.0),
            dict,
        ),
        #######################################################################
        # TiffReader
        (
            "s_1_t_10_c_3_z_1.tiff",
            "Image:0",
            ("Image:0",),
            (10, 3, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0", "Channel:1", "Channel:2"],
            (1.0, 1.0, 1.0),
            str,
        ),
        (
            "s_1_t_1_c_1_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 7548, 7548, 3),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
            (1.0, 1.0, 1.0),
            str,
        ),
        #######################################################################
        # OmeTiffReader
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 10, 1, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [f"C:{i}" for i in range(10)],  # This is the actual metadata
            (1.0, 1.0, 1.0),
            OME,
        ),
        (
            # This is actually an OME-TIFF file
            # Shows we don't just work off of extensions
            # But the content of the file
            "s_1_t_1_c_2_z_1_RGB.tiff",
            "Image:0",
            ("Image:0",),
            (1, 2, 1, 32, 32, 3),
            np.uint8,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0:0", "Channel:0:1"],
            (1.0, 1.0, 1.0),
            OME,
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:0",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
            OME,
        ),
        (
            "actk.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 6, 65, 233, 345),
            np.float64,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "nucleus_segmentation",
                "membrane_segmentation",
                "dna",
                "membrane",
                "structure",
                "brightfield",
            ],
            (0.29, 0.29, 0.29),
            OME,
        ),
        #######################################################################
        # Errors
        pytest.param(
            "example.txt",
            None,
            None,
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
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_aicsimage(
    filename,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dtype,
    expected_dims_order,
    expected_channel_names,
    expected_physical_pixel_sizes,
    expected_metadata_type,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_image_file_checks(
        ImageContainer=AICSImage,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=expected_metadata_type,
    )


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
            (1, 3, 5, 325, 475),
            "Image:1",
            (1, 3, 5, 325, 475),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            (1, 3, 5, 325, 475),
            "Image:2",
            (1, 3, 5, 325, 475),
        ),
    ],
)
def test_multi_scene_aicsimage(
    filename,
    first_scene_id,
    first_scene_shape,
    second_scene_id,
    second_scene_shape,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_multi_scene_image_read_checks(
        ImageContainer=AICSImage,
        image=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.uint16,
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.uint16,
    )


@pytest.mark.parametrize(
    "array_constructor",
    [
        # np.ones,
        da.ones,
    ],
)
@pytest.mark.parametrize(
    "arr_shape, known_dims, expected_shape, expected_dims, expected_channel_names",
    [
        (
            (1, 1),
            None,
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        (
            (1, 1, 1),
            None,
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1),
            None,
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1, 1),
            None,
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        (
            (1, 1, 1),
            "SYX",
            (1, 1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1),
            "ZCYX",
            (1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0"],
        ),
        (
            (1, 1, 1, 1, 1),
            "SZCYX",
            (1, 1, 1, 1, 1, 1),
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0"],
        ),
        (
            (3, 2, 5, 4, 10, 10),
            "STCZYX",
            (2, 5, 4, 10, 10, 3),
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_SAMPLES,
            ["Channel:0", "Channel:1", "Channel:2", "Channel:3", "Channel:4"],
        ),
        # Test that without known dims and with more than five dims, it raises an error
        # Our guess dim order only support up to five dims
        pytest.param(
            (1, 2, 3, 4, 5, 6),
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            (1, 1),
            "AB",
            None,
            None,
            None,
            marks=pytest.mark.raises(
                exceptions=exceptions.InvalidDimensionOrderingError
            ),
        ),
        pytest.param(
            "hello world",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exceptions=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_aicsimage_from_array(
    array_constructor,
    arr_shape,
    known_dims,
    expected_shape,
    expected_dims,
    expected_channel_names,
):
    # Construct array
    arr = array_constructor(arr_shape)

    # Init
    image_container = AICSImage(arr, known_dims=known_dims)

    run_image_container_checks(
        image_container=image_container,
        set_scene="Image:0",
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=expected_shape,
        expected_dtype=np.float64,
        expected_dims_order=expected_dims,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
        expected_metadata_type=type(None),
    )
