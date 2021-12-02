#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Tuple

import numpy as np
import pytest

from aicsimageio import dimensions, exceptions
from aicsimageio.readers import MrcReader
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
    "expected_physical_pixel_sizes",
    [
        (
            "image.mrc",
            "Image:0",
            ("Image:0",),
            (256, 512),
            np.float32,
            "YX",
            (None, 1, 1),
        ),
        (
            "image_stack.mrc",
            "Image:0",
            tuple(f"Image:{i}" for i in range(20)),
            (256, 256),
            np.float32,
            "YX",
            (None, 1.1449999809265137, 1.1449999809265137),
        ),
        (
            "volume.mrc.new",
            "Image:0",
            ("Image:0",),
            (128, 128, 128),
            np.float32,
            "ZYX",
            (1.600000023841858, 1.600000023841858, 1.600000023841858),
        ),
        (
            "volume_stack.mrc",
            "Image:0",
            tuple(f"Image:{i}" for i in range(2)),
            (128, 128, 128),
            np.float32,
            "ZYX",
            (1.600000023841858, 1.600000023841858, 1.600000023841858),
        ),
    ],
)
def test_mrc_reader(
    filename: str,
    host: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_file_checks(
        ImageContainer=MrcReader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=None,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=np.ndarray,
    )
