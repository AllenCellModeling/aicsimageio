#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import pytest

from .test_aics_image import CZI_FILE, OME_FILE, PNG_FILE, TIF_FILE


@pytest.mark.parametrize(
    "filename, expected_shape",
    [
        (PNG_FILE, (1, 1, 4, 1, 800, 537)),
        (TIF_FILE, (1, 1, 1, 1, 325, 475)),
        (OME_FILE, (1, 1, 1, 1, 325, 475)),
        (CZI_FILE, (1, 1, 1, 1, 325, 475)),
    ],
)
def test_aicsimageio_no_networking(resources_dir, filename, expected_shape):
    # This should test and make sure that distributed isn't imported when aicsimageio is
    # Importing distributed on a machine (or container) that doesn't have any
    # networking capabilities results in socket errors, _during the import_
    # See: https://github.com/AllenCellModeling/aicsimageio/issues/82
    if "distributed" in sys.modules:
        del sys.modules["distributed"]

    # Re import
    import aicsimageio  # noqa: F401

    # Some basic operation to ensure that distributed is not imported
    # anywhere down the line
    img = aicsimageio.AICSImage(resources_dir / filename)
    assert img.data.shape == expected_shape

    # Assert not imported
    assert "distributed" not in sys.modules
