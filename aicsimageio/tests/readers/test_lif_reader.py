#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import LifReader

from ..conftest import get_resource_full_path, host
from ..image_container_test_utils import run_image_file_checks


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
            "s_1_t_1_c_2_z_1.lif",
            "PEI_laminin_35k",
            ("PEI_laminin_35k",),
            (1, 1, 2, 1, 2048, 2048),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray--TL-BF--EMP_BF", "Green--FLUO--GFP"],
            (1.0, 3.076923076923077, 3.076923076923077),
        ),
        (
            "s_1_t_4_c_2_z_1.lif",
            "b2_001_Crop001_Resize001",
            ("b2_001_Crop001_Resize001",),
            (1, 4, 2, 1, 614, 614),
            np.uint16,
            dimensions.DEFAULT_DIMENSION_ORDER_WITH_MOSAIC_TILES,
            ["Gray--TL-PH--EMP_BF", "Green--FLUO--GFP"],
            (1.0, 2.9485556406398508, 2.9485556406398508),
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
            "3d-cell-viewer.ome.tiff",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_lif_reader(
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
        ImageContainer=LifReader,
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
