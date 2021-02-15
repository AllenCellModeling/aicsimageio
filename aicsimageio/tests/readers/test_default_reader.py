#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import DefaultReader

from ..conftest import get_resource_full_path, host
from ..image_container_test_utils import run_image_read_checks


@host
@pytest.mark.parametrize(
    "filename, set_scene, expected_shape, expected_dims_order",
    [
        ("example.bmp", "Image:0", (480, 640, 4), "YXS"),
        ("example.png", "Image:0", (800, 537, 4), "YXS"),
        ("example.jpg", "Image:0", (452, 400, 3), "YXS"),
        ("example.gif", "Image:0", (72, 268, 268, 4), "TYXS"),
        (
            "example_invalid_frame_count.mp4",
            "Image:0",
            (55, 1080, 1920, 3),
            "TYXS",
        ),
        (
            "example_valid_frame_count.mp4",
            "Image:0",
            (72, 272, 272, 3),
            "TYXS",
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.png",
            "Image:1",
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
    ],
)
def test_default_reader(
    filename,
    host,
    set_scene,
    expected_shape,
    expected_dims_order,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_read_checks(
        ImageContainer=DefaultReader,
        uri=uri,
        set_scene=set_scene,
        expected_scenes=("Image:0",),
        expected_current_scene="Image:0",
        expected_shape=expected_shape,
        expected_dtype=np.uint8,
        expected_dims_order=expected_dims_order,
        expected_channel_names=None,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
    )


def test_ffmpeg_header_fail():
    with pytest.raises(IOError):
        # Big Buck Bunny
        DefaultReader("https://archive.org/embed/archive-video-files/test.mp4")
