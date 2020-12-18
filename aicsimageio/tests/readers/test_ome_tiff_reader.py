#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import OmeTiffReader
import numpy as np
from urllib.error import HTTPError
from xmlschema.validators import (
    XMLSchemaChildrenValidationError,
    XMLSchemaValidationError,
)

from ..conftest import LOCAL, REMOTE, get_resource_full_path
from .reader_test_utils import run_image_read_checks, run_multi_scene_image_read_checks


@pytest.mark.parametrize(
    "host",
    [LOCAL, REMOTE],
)
@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_1_t_1_c_1_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 1, 1, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_1_t_1_c_10_z_1.ome.tiff",
            "Image:0",
            ("Image:0",),
            (1, 10, 1, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [f"C:{i}" for i in range(10)],  # This is the actual metadata
            (1.0, 1.0, 1.0),
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
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:1",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.ome.tiff",
            "Image:2",
            ("Image:0", "Image:1", "Image:2"),
            (1, 3, 5, 325, 475),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["EGFP", "TaRFP", "Bright"],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        pytest.param(
            "example.txt",
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
            "s_1_t_1_c_2_z_1.lif",
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
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_ome_tiff_reader(
    filename,
    host,
    set_scene,
    expected_scenes,
    expected_shape,
    expected_dtype,
    expected_dims_order,
    expected_channel_names,
    expected_physical_pixel_sizes,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_read_checks(
        ReaderClass=OmeTiffReader,
        uri=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
    )


@pytest.mark.parametrize("host", [LOCAL, REMOTE])
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
def test_multi_scene_ome_tiff_reader(
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
        ReaderClass=OmeTiffReader,
        uri=uri,
        first_scene_id=first_scene_id,
        first_scene_shape=first_scene_shape,
        first_scene_dtype=np.uint16,
        second_scene_id=second_scene_id,
        second_scene_shape=second_scene_shape,
        second_scene_dtype=np.uint16,
    )


@pytest.mark.parametrize(
    "filename",
    [
        # Pipline 4 is valid, :tada:
        "3500000719_100X_20170317_F08_P06.ome.tiff",
        # Some of our test files are valid, :tada:
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        # A lot of our files aren't valid, :upside-down-smiley:
        # These files have invalid schema / layout
        pytest.param(
            "new-cfe-file.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaChildrenValidationError),
        ),
        pytest.param(
            "old-cfe-file.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaChildrenValidationError),
        ),
        pytest.param(
            "actk.ome.tiff",
            marks=pytest.mark.raises(exception=XMLSchemaValidationError),
        ),
        # This file has a namespace that doesn't exist
        pytest.param(
            "s_1_t_1_c_10_z_1.ome.tiff", marks=pytest.mark.raises(exception=HTTPError)
        ),
    ],
)
def test_known_errors_without_cleaning(filename):
    filepath = get_resource_full_path(filename, "LOCAL")
    OmeTiffReader(filepath, clean_metadata=False)


@pytest.mark.parametrize(
    "filename",
    [
        "3500000719_100X_20170317_F08_P06.ome.tiff",
        "new-cfe-file.ome.tiff",
        "old-cfe-file.ome.tiff",
        "s_1_t_1_c_1_z_1.ome.tiff",
        "s_1_t_1_c_10_z_1.ome.tiff",
        "s_3_t_1_c_3_z_5.ome.tiff",
        "actk.ome.tiff",
    ],
)
def test_ome_validates_after_cleaning(filename):
    filepath = get_resource_full_path(filename, "LOCAL")
    OmeTiffReader(filepath)
