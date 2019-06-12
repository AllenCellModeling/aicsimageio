#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from aicsimageio.readers.default_reader import DefaultReader
from aicsimageio import exceptions


@pytest.mark.parametrize("filename", [
    ("example.bmp"),
    ("example.png"),
    ("example.jpg"),
    ("very_good_seattle_boi.gif")
])
def test_default_reader_get_default_dims(image_dir, filename):
    # Get file
    f = image_dir / filename

    # Open
    with DefaultReader(f) as r:
        # Dims should be set to 3D for all of these images
        assert r.dims == "ZYX"


@pytest.mark.parametrize("expected", [
    ("XYC"),
    ("STC"),
    pytest.param("HELLOWORLD", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError)),
    pytest.param("NO", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError))
])
def test_default_reader_set_dims(image_dir, expected):
    with DefaultReader(image_dir / "example.png") as r:
        r.dims = expected
        assert r.dims == expected
