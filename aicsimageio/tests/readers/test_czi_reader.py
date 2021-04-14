#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions, dimensions
from aicsimageio.readers import CziReader
from lxml import etree as ET

from ..conftest import LOCAL, get_resource_full_path, host
from ..image_container_test_utils import (
    run_image_file_checks,
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
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",  # dimensions.DEFAULT_DIMENSION_ORDER,
            ["Channel:0--EGFP--Fluorescence",
             "Channel:1--TaRFP--Fluorescence",
             "Channel:2--Bright--Brightfield"],
            (1e-06, 1.0833333333333333e-06, 1.0833333333333333e-06),
        ),
    ],
)
def test_czi_reader(
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
        ImageContainer=CziReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=ET.Element,
    )
