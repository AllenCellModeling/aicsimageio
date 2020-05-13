#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

from aicsimageio import AICSImage
from aicsimageio.exceptions import InvalidDimensionOrderingError
from aicsimageio.readers.ome_tiff_reader import OmeTiffReader
from aicsimageio.writers import OmeTiffWriter, convert_to_ome_tiff

filename = "ometif_test_output.ome.tif"

image = np.random.rand(1, 40, 3, 128, 256).astype(np.uint16)


def test_writerShapeComparison(resources_dir):
    """
    Test to check that OmeTiffWriter saves arrays that are reflexive with OmeTiffReader
    """
    with OmeTiffWriter(resources_dir / filename, overwrite_file=True) as writer:
        writer.save(image)

    output = OmeTiffReader(resources_dir / filename).data

    assert output.shape == image.shape[1:]


def test_loadAssertionError(resources_dir):
    """
    Test to check if save() will only accept 3, 4, 5 dimensions for data
    """
    image_to_save = np.ones((1, 2, 3, 4, 5, 6))
    with pytest.raises(Exception):
        with OmeTiffWriter(resources_dir / filename, overwrite_file=True) as writer:
            writer.save(image_to_save)


def test_overwriteFile(resources_dir):
    """
    Test to check if save() can overwrite a file
    """
    with OmeTiffWriter(resources_dir / filename, overwrite_file=True) as writer:
        writer.save(image)


def test_dontOverwriteFile(resources_dir):
    """
    Test to check if save() will raise error when user does not want to overwrite a
    file that exists
    """
    with pytest.raises(Exception):
        with OmeTiffWriter(resources_dir / filename) as writer:
            writer.save(image)


def test_noopOverwriteFile(resources_dir):
    """
    Test to check if save() silently no-ops when user does not want to overwrite
    exiting file
    """
    with open(resources_dir / filename, "w") as f:
        f.write("test")
    with OmeTiffWriter(resources_dir / filename, overwrite_file=False) as writer:
        writer.save(image)
    with open(resources_dir / filename, "r") as f:
        line = f.readline().strip()
        assert "test" == line


def test_big_tiff():
    x = np.zeros((10, 10))
    assert OmeTiffWriter._size_of_ndarray(data=x) == 10 * 10 * x.itemsize


@pytest.mark.parametrize(
    "dims, expected_t, expected_c, expected_z, expected_y, expected_x",
    [
        ("TCZYX", 1, 40, 3, 128, 256),
        ("TZCYX", 1, 3, 40, 128, 256),
        ("CZTYX", 3, 1, 40, 128, 256),
        ("CTZYX", 40, 1, 3, 128, 256),
        ("ZCTYX", 3, 40, 1, 128, 256),
        ("ZTCYX", 40, 3, 1, 128, 256),
        ("STCZYX", 1, 40, 3, 128, 256),
        (
            pytest.param(
                "XY",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
        (
            pytest.param(
                "ABCD",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
        (
            pytest.param(
                "XXXXX",
                1,
                40,
                3,
                128,
                256,
                marks=pytest.mark.raises(exception=InvalidDimensionOrderingError),
            )
        ),
    ],
)
def test_dimensionOrder(
    resources_dir, dims, expected_t, expected_c, expected_z, expected_y, expected_x
):
    with OmeTiffWriter(resources_dir / filename, overwrite_file=True) as writer:
        writer.save(image, dimension_order=dims)

    reader = OmeTiffReader(resources_dir / filename)
    output = reader.data
    t = reader.size_t()
    c = reader.size_c()
    z = reader.size_z()
    y = reader.size_y()
    x = reader.size_x()

    os.remove(resources_dir / filename)

    assert output.shape == image.shape[1:]
    assert x == expected_x
    assert y == expected_y
    assert z == expected_z
    assert c == expected_c
    assert t == expected_t


@pytest.mark.parametrize(
    "czi_file, ome_tiff_file",
    [
        ("s_3_t_1_c_3_z_5.czi", "s_3_t_1_c_3_z_5_4DN.ome.tif"),
        ("s_1_t_1_c_1_z_1.czi", "s_1_t_1_c_1_z_1_4DN.ome.tif"),
    ],
)
def test_ome_conversion(resources_dir, czi_file, ome_tiff_file):
    """
    Test to check serialization of OME generated metadata
    """
    czi_file = resources_dir / czi_file
    ome_tiff_file = resources_dir / ome_tiff_file

    # Convert
    convert_to_ome_tiff(original_file=czi_file, save_path=ome_tiff_file)

    # Check metadata values
    czi_file = AICSImage(czi_file)
    ome_tiff_file = AICSImage(ome_tiff_file)

    assert czi_file.get_channel_names() == ome_tiff_file.get_channel_names()
    assert czi_file.get_physical_pixel_size() == ome_tiff_file.get_physical_pixel_size()
    assert czi_file.reader.size_s() == ome_tiff_file.reader.size_s()
    assert czi_file.reader.size_t() == ome_tiff_file.reader.size_t()
    assert czi_file.reader.size_c() == ome_tiff_file.reader.size_c()
    assert czi_file.reader.size_z() == ome_tiff_file.reader.size_z()
    assert czi_file.reader.size_y() == ome_tiff_file.reader.size_y()
    assert czi_file.reader.size_x() == ome_tiff_file.reader.size_x()

    os.remove(resources_dir / ome_tiff_file)
