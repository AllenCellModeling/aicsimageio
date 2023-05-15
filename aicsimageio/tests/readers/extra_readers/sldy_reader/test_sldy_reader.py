#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Tuple, Union

import numpy as np
import pytest

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers import SldyReader
from aicsimageio.tests.image_container_test_utils import run_image_file_checks

from ....conftest import LOCAL, get_resource_full_path, host


@host
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
        pytest.param(
            "example.png",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        (
            "s1_t10_c1_z5.dir",
            "Image:20220726 endo diff1658874976",
            ("Image:20220726 endo diff1658874976",),
            (10, 1, 5, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:20220726 endo diff1658874976:0"],
            (None, 0.38388850322622897, 0.38388850322622897),
        ),
        (
            "s1_t1_c2_z40.dir",
            "Image:3500005564_20X_timelapse_202304201682033857",
            ("Image:3500005564_20X_timelapse_202304201682033857",),
            (1, 2, 40, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            [
                "Channel:3500005564_20X_timelapse_202304201682033857:0",
                "Channel:3500005564_20X_timelapse_202304201682033857:1",
            ],
            (None, 0.3820158766750814, 0.3820158766750814),
        ),
    ],
)
def test_sldy_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=SldyReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=dict,
    )


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
        (
            "s1_t10_c1_z5.dir",
            "Image:20220726 endo diff1658874976",
            ("Image:20220726 endo diff1658874976",),
            (10, 1, 5, 1736, 1776),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:20220726 endo diff1658874976:0"],
            (None, 0.38388850322622897, 0.38388850322622897),
            dict,
        ),
    ],
)
def test_aicsimage(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
    expected_metadata_type: Union[type, Tuple[Union[type, Tuple[Any, ...]], ...]],
) -> None:
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
