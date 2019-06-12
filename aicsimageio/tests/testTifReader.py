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


def test_load():
    with TiffReader(RESOURCES / 'img40_1_dna.tif') as reader:
        print(reader.data)


def test_metadata():
    with TiffReader(RESOURCES / 'single-channel.ome.tif') as reader:
        print(reader.metadata)
