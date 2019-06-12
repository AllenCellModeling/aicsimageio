#!/usr/bin/env python
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


@pytest.mark.parametrize('name, shape, dims, metadata, dtype', [
    ('img40_1_dna.tif', (43, 410, 286), 'ZYX', None, np.uint8),
    ('TestXYCZ_imagej.tif', (4, 3, 300, 400), 'CZYX', 'ImageJ', np.uint16),
    ('single-channel.ome.tif', (167, 439), 'YX', 'OME-XML', np.int8),
    ('4D-series.ome.tif', (7, 5, 167, 439), 'CZYX', 'OME-XML', np.int8),
    ('z-series.ome.tif', (5, 167, 439), 'ZYX', 'OME-XML', np.int8),
    ('multi-channel-4D-series.ome.tif', (7, 3, 5, 167, 439), 'TCZYX', 'OME-XML', np.int8),
    ('BigTIFF.tif', (64, 64, 3), 'ZYX', None, np.uint8),  # This dimension ordering is wrong, but nothing we can do
    ('BigTIFFMotorola.tif', (64, 64, 3), 'ZYX', None, np.uint8)  # Same here
])
def test_load(name, shape, dims, metadata, dtype):
    with TiffReader(RESOURCES / name) as reader:
        data, actual_dims, actual_metadata = reader.load()
        assert data.shape == shape
        assert actual_dims == dims
        if metadata is None:
            assert actual_metadata == ''
        else:
            assert metadata in actual_metadata
        assert dtype == reader.dtype()
