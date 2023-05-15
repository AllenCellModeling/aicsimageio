#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pytest

from aicsimageio.readers.sldy_reader.sldy_image import SldyImage
from aicsimageio.utils import io_utils

from ....conftest import LOCAL, get_resource_full_path

METADATA_FILES = {
    "annotation_record",
    "aux_data",
    "channel_record",
    "elapsed_times",
    "image_record",
    "mask_record",
    "sa_position_data",
    "stage_position_data",
}


@pytest.mark.parametrize(
    "filename, "
    "expected_physical_pixel_size_x, "
    "expected_physical_pixel_size_y, "
    "expected_physical_pixel_size_z, "
    "expected_metadata_files, "
    "expected_shape, ",
    [
        (
            "s1_t10_c1_z5.dir/20220726 endo diff1658874976.imgdir",
            0.38388850322622897,
            0.38388850322622897,
            None,
            METADATA_FILES,
            (5, 1736, 1776),
        ),
        (
            "s1_t2_c2_z40.dir/3500005564_20X_timelapse_202304201682033857.imgdir",
            0.3820158766750814,
            0.3820158766750814,
            None,
            METADATA_FILES,
            (40, 1736, 1776),
        ),
    ],
)
def test_sldy_image(
    filename: str,
    expected_physical_pixel_size_x: float,
    expected_physical_pixel_size_y: float,
    expected_physical_pixel_size_z: Optional[float],
    expected_metadata_files: Set[str],
    expected_shape: Tuple[int, ...],
) -> None:
    # Determine path to file
    uri = get_resource_full_path(filename, LOCAL)
    fs, path = io_utils.pathlike_to_fs(
        uri,
        enforce_exists=True,
    )

    # Construct image
    image = SldyImage(fs, Path(path), data_file_prefix="ImageData")

    # Assert image properties match expectation
    assert expected_physical_pixel_size_x == image.physical_pixel_size_x
    assert expected_physical_pixel_size_y == image.physical_pixel_size_y
    assert expected_physical_pixel_size_z == image.physical_pixel_size_z

    # Ensure all metadata files are present then ensure all expected
    # metadata files have values
    assert METADATA_FILES.issubset(image.metadata.keys())
    for key in image.metadata.keys():
        if key in expected_metadata_files:
            assert image.metadata[key], f"Metadata file {key} not found"

    # Assert the data retrieved from the image matches the expectation
    data_at_t0_c0 = image.get_data(timepoint=0, channel=0, delayed=True)
    assert expected_shape == data_at_t0_c0.shape
    assert not np.array_equal(
        data_at_t0_c0, image.get_data(timepoint=1, channel=0, delayed=True)
    )
