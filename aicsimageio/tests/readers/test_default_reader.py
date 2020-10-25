#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import DefaultReader

from ..conftest import LOCAL, REMOTE, get_resource_full_path
from .reader_test_utils import run_image_read_checks


@pytest.mark.parametrize(
    (
        "filename, "
        "host, "
        "set_scene, "
        "expected_shape, "
        "expected_dims_order, "
        "expected_channel_names, "
    ),
    [
        (
            "example.bmp",
            LOCAL,
            0,
            (480, 640, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.bmp",
            REMOTE,
            0,
            (480, 640, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.png",
            LOCAL,
            0,
            (800, 537, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.png",
            REMOTE,
            0,
            (800, 537, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        ("example.jpg", LOCAL, 0, (452, 400, 3), "YXC", ["R", "G", "B"]),
        ("example.jpg", REMOTE, 0, (452, 400, 3), "YXC", ["R", "G", "B"]),
        (
            "example.gif",
            LOCAL,
            0,
            (72, 268, 268, 4),
            "TYXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.gif",
            REMOTE,
            0,
            (72, 268, 268, 4),
            "TYXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.mp4",
            LOCAL,
            0,
            (183, 1080, 1920, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        (
            "example.mp4",
            REMOTE,
            0,
            (183, 1080, 1920, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        pytest.param(
            "example.txt",
            LOCAL,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.txt",
            REMOTE,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.png",
            LOCAL,
            1,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=IndexError),
        ),
        pytest.param(
            "example.png",
            REMOTE,
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
        expected_physical_pixel_sizes=(None, 1.0, 1.0),
    )
