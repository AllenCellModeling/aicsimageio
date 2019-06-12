#!/usr/bin/env python
import math
import os
import unittest
from pathlib import Path

import numpy as np
import pytest

from aicsimageio.readers import TiffReader


RESOURCES = Path(__file__).parent / 'img'


@pytest.mark.parametrize('name, expected', [
    ('img40_1_dna.tif', True),
    ('TestXYCZ_imagej.tif', True),
    ('img40_1.png', False),
    ('single-channel.ome.tif', True)
])
def test_type_check(name, expected):
    tiff_image = RESOURCES / name
    assert TiffReader.is_this_type(tiff_image) == expected


@pytest.mark.parametrize('name, shape, dims, metadata', [
    ('img40_1_dna.tif', (43, 410, 286), 'ZYX', ''),
    ('TestXYCZ_imagej.tif', (4, 3, 300, 400), 'CZYX', 'ImageJ'),
    ('single-channel.ome.tif', (167, 439), 'YX', 'OME-XML'),
    ('4D-series.ome.tif', (7, 5, 167, 439), 'CZYX', 'OME-XML'),
    ('z-series.ome.tif', (5, 167, 439), 'ZYX', 'OME-XML'),
    ('multi-channel-4D-series.ome.tif', (7, 3, 5, 167, 439), 'TCZYX', 'OME-XML'),
    ('BigTIFF.tif', (64, 64, 3), 'ZYX', '')  # This dimension ordering is actually wrong, but nothing we can do about it
])
def test_load(name, shape, dims, metadata):
    with TiffReader(RESOURCES / name) as reader:
        data, actual_dims, actual_metadata = reader.load()
        assert data.shape == shape
        assert actual_dims == dims
        if metadata == '':
            assert metadata == actual_metadata
        else:
            assert metadata in actual_metadata
