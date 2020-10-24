#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers import DefaultReader

from .reader_test_utils import run_image_read_checks


@pytest.mark.parametrize(
    (
        "filename, "
        "from_local, "
        "can_read_chunks, "
        "expected_shape, "
        "expected_dims_order, "
        "expected_channel_names, "
    ),
    [
        (
            "example.bmp",
            True,
            True,
            (480, 640, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.bmp",
            False,
            True,
            (480, 640, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.png",
            True,
            True,
            (800, 537, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.png",
            False,
            True,
            (800, 537, 4),
            "YXC",
            ["R", "G", "B", "A"],
        ),
        ("example.jpg", True, True, (452, 400, 3), "YXC", ["R", "G", "B"]),
        ("example.jpg", False, True, (452, 400, 3), "YXC", ["R", "G", "B"]),
        (
            "example.gif",
            True,
            True,
            (72, 268, 268, 4),
            "TYXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.gif",
            False,
            False,
            (72, 268, 268, 4),
            "TYXC",
            ["R", "G", "B", "A"],
        ),
        (
            "example.mp4",
            True,
            True,
            (183, 1080, 1920, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        (
            "example.mp4",
            False,
            False,
            (183, 1080, 1920, 3),
            "TYXC",
            ["R", "G", "B"],
        ),
        pytest.param(
            "example.txt",
            True,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "example.txt",
            False,
            None,
            None,
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_default_reader(
    local_resources_dir,
    remote_resources_dir,
    filename,
    from_local,
    can_read_chunks,
    expected_shape,
    expected_dims_order,
    expected_channel_names,
):
    # Construct full filepath
    if from_local:
        uri = local_resources_dir / filename
    else:
        uri = f"{remote_resources_dir}/{filename}"

    # Run checks
    run_image_read_checks(
        ReaderClass=DefaultReader,
        uri=uri,
        can_read_chunks=can_read_chunks,
        expected_shape=expected_shape,
        expected_dtype=np.uint8,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=(None, 1.0, 1.0),
    )
