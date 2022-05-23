#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, List, Tuple, Union

import numpy as np
import pytest

from aicsimageio import AICSImage, dimensions, exceptions
from aicsimageio.readers.nd2_reader import ND2Reader
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
        pytest.param(
            "ND2_aryeh_but3_cont200-1.nd2",
            "XYPos:0",
            ("XYPos:0", "XYPos:1", "XYPos:2", "XYPos:3", "XYPos:4"),
            (1, 2, 1040, 1392),
            np.uint16,
            "TCYX",
            ["20phase", "20xDiO"],
            (1, 50, 50),
        ),
        (
            "ND2_jonas_header_test2.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 4, 5, 520, 696),
            np.uint16,
            "CTZYX",
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
        ),
        (
            "ND2_maxime_BF007.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 156, 164),
            np.uint16,
            "CYX",
            ["405/488/561/633nm"],
            (1.0, 0.158389678930686, 0.158389678930686),
        ),
        (
            "ND2_dims_p4z5t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0", "XYPos:1", "XYPos:2", "XYPos:3"),
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
        ),
        (
            "ND2_dims_c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (2, 32, 32),
            np.uint16,
            "CYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
        ),
        (
            "ND2_dims_p1z5t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 5, 2, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
        ),
        (
            "ND2_dims_p2z5t3-2c4y32x32.nd2",
            "XYPos:1",
            ("XYPos:0", "XYPos:1"),
            (5, 5, 4, 32, 32),
            np.uint16,
            "TZCYX",
            ["Widefield Green", "Widefield Red", "Widefield Far-Red", "Brightfield"],
            (1.0, 0.652452890023035, 0.652452890023035),
        ),
        (
            "ND2_dims_t3c2y32x32.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (3, 2, 32, 32),
            np.uint16,
            "TCYX",
            ["Widefield Green", "Widefield Red"],
            (1.0, 0.652452890023035, 0.652452890023035),
        ),
        (
            "ND2_dims_rgb_t3p2c2z3x64y64.nd2",
            "XYPos:1",
            ("XYPos:0", "XYPos:1"),
            (3, 3, 2, 32, 32, 3),
            np.uint8,
            "TZCYXS",
            ["Brightfield", "Brightfield"],
            (0.01, 0.34285714285714286, 0.34285714285714286),
        ),
        (
            "ND2_dims_rgb.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (1, 64, 64, 3),
            np.uint8,
            "CYXS",
            ["Brightfield"],
            (1.0, 0.34285714285714286, 0.34285714285714286),
        ),
    ],
)
def test_nd2_reader(
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
        ImageContainer=ND2Reader,
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
            "ND2_jonas_header_test2.nd2",
            "XYPos:0",
            ("XYPos:0",),
            (4, 1, 5, 520, 696),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER,
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
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
