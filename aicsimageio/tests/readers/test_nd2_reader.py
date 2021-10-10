#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers.nd2_reader import ND2Reader
from aicsimageio.tests.image_container_test_utils import run_image_file_checks

from ..conftest import get_resource_full_path, host


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
            "Image:0",
            ("Image:0",),
            (1, 5, 2, 1040, 1392),
            np.uint16,
            "TSCYX",
            ["20phase", "20xDiO"],
            (1, 50, 50),
        ),
        (
            "ND2_jonas_header_test2.nd2",
            "Image:0",
            ("Image:0",),
            (1, 4, 5, 520, 696),
            np.uint16,
            "CTZYX",
            ["Jonas_DIC"],
            (0.5, 0.12863494437945, 0.12863494437945),
        ),
        (
            "ND2_maxime_BF007.nd2",
            "Image:0",
            ("Image:0",),
            (1, 156, 164),
            np.uint16,
            "CYX",
            ["405/488/561/633nm"],
            (1.0, 0.158389678930686, 0.158389678930686),
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
