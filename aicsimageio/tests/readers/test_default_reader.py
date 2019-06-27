#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from aicsimageio import exceptions
from aicsimageio.readers.default_reader import DefaultReader


@pytest.mark.parametrize("filename", [
    "example.bmp",
    "example.png",
    "example.jpg",
    "example.gif"
])
def test_default_reader_get_default_dims(resources_dir, filename):
    # Get file
    f = resources_dir / filename

    # Open
    with DefaultReader(f) as r:
        # Dims should be set to 3D for all of these images
        assert r.dims == "ZYX"
        assert r.metadata is None
        assert DefaultReader.is_this_type(f)


@pytest.mark.parametrize("expected", [
    "XYC",
    "STC",
    pytest.param("HELLOWORLD", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError)),
    pytest.param("NO", marks=pytest.mark.raises(exception=exceptions.InvalidDimensionOrderingError))
])
def test_default_reader_set_dims(resources_dir, expected):
    with DefaultReader(resources_dir / "example.png") as r:
        r.dims = expected
        assert r.dims == expected
