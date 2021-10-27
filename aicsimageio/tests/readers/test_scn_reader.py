#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers.scn_reader import SCNReader
from aicsimageio.tests.image_container_test_utils import run_image_file_checks

from ..conftest import LOCAL, get_resource_full_path


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_current_scene, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "Leica-RGB.scn",
            "image_0000027734 S0-(R0)",
            (
                "image_0000027734 S0-(R0)",
                "image_0000027734 S0-(R1)",
                "image_0000027734 S0-(R2)",
                "image_0000027735 S1-(R0)",
                "image_0000027735 S1-(R1)",
                "image_0000027735 S1-(R2)",
                "image_0000027735 S1-(R3)",
                "image_0000027735 S1-(R4)",
            ),
            "image_0000027734 S0-(R0)",
            (4668, 1616, 3),
            np.uint8,
            "YXS",
            None,
            (None, 16.43844182054476, 16.43844182054476),
        ),
        (
            "Leica-RGB.scn",
            "image_0000027735 S1-(R2)",
            (
                "image_0000027734 S0-(R0)",
                "image_0000027734 S0-(R1)",
                "image_0000027734 S0-(R2)",
                "image_0000027735 S1-(R0)",
                "image_0000027735 S1-(R1)",
                "image_0000027735 S1-(R2)",
                "image_0000027735 S1-(R3)",
                "image_0000027735 S1-(R4)",
            ),
            "image_0000027735 S1-(R2)",
            (1520, 1666, 3),
            np.uint8,
            "YXS",
            None,
            (None, 8.0, 8.0),
        ),
        (
            "Leica-Fluorescence-1.scn",
            "image_0000000586 S0-(R0)",
            (
                "image_0000000586 S0-(R0)",
                "image_0000000586 S0-(R1)",
                "image_0000000586 S0-(R2)",
                "image_0000000590 S1-(R0)",
                "image_0000000590 S1-(R1)",
                "image_0000000590 S1-(R2)",
                "image_0000000591 S2-(R0)",
                "image_0000000591 S2-(R1)",
                "image_0000000591 S2-(R2)",
                "image_0000000591 S2-(R3)",
            ),
            "image_0000000586 S0-(R0)",
            (4668, 1616, 3),
            np.uint8,
            "YXS",
            None,
            (None, 16.43844182054476, 16.43844182054476),
        ),
        (
            "Leica-Fluorescence-1.scn",
            "image_0000000591 S2-(R1)",
            (
                "image_0000000586 S0-(R0)",
                "image_0000000586 S0-(R1)",
                "image_0000000586 S0-(R2)",
                "image_0000000590 S1-(R0)",
                "image_0000000590 S1-(R1)",
                "image_0000000590 S1-(R2)",
                "image_0000000591 S2-(R0)",
                "image_0000000591 S2-(R1)",
                "image_0000000591 S2-(R2)",
                "image_0000000591 S2-(R3)",
            ),
            "image_0000000591 S2-(R1)",
            (3, 1584, 1184),
            np.uint8,
            "CYX",
            ["405|Empty", "L5|Empty", "TX2|Empty"],
            (None, 2.0, 2.0),
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
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "Leica-RGB.scn",
            9,
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
def test_scn_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_current_scene: str,
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, LOCAL)

    # Run checks
    run_image_file_checks(
        ImageContainer=SCNReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=expected_current_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=dict,
    )
