#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Tuple, Union

import numpy as np
import pytest

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers import DVReader
from aicsimageio.tests.image_container_test_utils import run_image_file_checks

from ...conftest import LOCAL, get_resource_full_path, host


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
        (
            "DV_siRNAi-HeLa_IN_02.r3d_D3D.dv",
            "Image:0",
            ("Image:0",),
            (4, 1, 40, 512, 512),
            np.int16,
            "CTZYX",
            ["360/457", "490/528", "555/617", "640/685"],
            (0.20000000298023224, 0.06502940505743027, 0.06502940505743027),
        ),
        (
            "DV_siRNAi-HeLa_IN_02.r3d",
            "Image:0",
            ("Image:0",),
            (1, 4, 40, 512, 512),
            np.dtype(">i2"),
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["0/0", "0/0", "0/0", "0/0"],
            (0.20000000298023224, 0.06502940505743027, 0.06502940505743027),
        ),
    ],
)
def test_dv_reader(
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
        ImageContainer=DVReader,
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
            "DV_siRNAi-HeLa_IN_02.r3d_D3D.dv",
            "Image:0",
            ("Image:0",),
            (1, 4, 40, 512, 512),
            np.int16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["360/457", "490/528", "555/617", "640/685"],
            (0.20000000298023224, 0.06502940505743027, 0.06502940505743027),
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
