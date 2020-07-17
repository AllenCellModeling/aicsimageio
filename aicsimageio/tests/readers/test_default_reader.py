#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader

from .utils import run_image_read_checks


@pytest.mark.parametrize(
    "filename, expected_shape, expected_dims",
    [
        ("example.bmp", (480, 640, 4), "YXC"),
        ("example.png", (800, 537, 4), "YXC"),
        ("example.jpg", (452, 400, 3), "YXC"),
        ("example.gif", (72, 268, 268, 4), "TYXC"),
        pytest.param(
            "example.txt",
            None,
            None,
            marks=pytest.mark.raises(exception=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_default_reader(
    resources_dir, filename, expected_shape, expected_dims,
):
    run_image_read_checks(
        ReaderClass=DefaultReader,
        resources_dir=resources_dir,
        filename=filename,
        chunk_dims=None,
        select_scene=None,
        expected_shape=expected_shape,
        expected_dims=expected_dims,
        expected_dtype=np.uint8,
    )
