#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import DefaultReader

from ..conftest import LOCAL, REMOTE, get_resource_full_path
from .reader_test_utils import run_image_read_checks


@pytest.mark.parametrize("host", [LOCAL, REMOTE])
@pytest.mark.parametrize(
    "filename, set_scene, expected_shape, expected_dims_order, expected_channel_names",
    [
        ("example.bmp", 0, (480, 640, 4), "YXC", ["R", "G", "B", "A"]),
        ("example.png", 0, (800, 537, 4), "YXC", ["R", "G", "B", "A"]),
        ("example.jpg", 0, (452, 400, 3), "YXC", ["R", "G", "B"]),
        ("example.gif", 0, (72, 268, 268, 4), "TYXC", ["R", "G", "B", "A"]),
        (
            "example_invalid_frame_count.mp4",
            0,
            (55, 1080, 1920, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        (
            "example_valid_frame_count.mp4",
            0,
            (72, 272, 272, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.png",
            1,
            None,
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
    expected_channel_names,
):
    # Construct full filepath
    uri = get_resource_full_path(filename, host)

    # Run checks
    run_image_read_checks(
        ReaderClass=DefaultReader,
        uri=uri,
        set_scene=set_scene,
        expected_scenes=(0,),
        expected_current_scene=0,
        expected_shape=expected_shape,
        expected_dtype=np.uint8,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(1.0, 1.0, 1.0),
    )


def test_ffmpeg_header_fail():
    with pytest.raises(IOError):
        # Big Buck Bunny
        DefaultReader("https://archive.org/embed/archive-video-files/test.mp4")
